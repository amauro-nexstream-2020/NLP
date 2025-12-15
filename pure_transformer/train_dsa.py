#!/usr/bin/env python
"""
DSA (DeepSeek Sparse Attention) Training Script

Train the DSATransformer with Lightning Indexer for O(L*k) attention.

Features:
- Lightning Indexer with KL alignment training
- Top-k sparse attention for long sequences
- Combined LM + Indexer loss optimization
- Full W&B integration

Usage:
    # Single GPU (testing)
    python train_dsa.py --model xlarge --devices 1

    # 8 GPUs (production)  
    python train_dsa.py --model xlarge --devices 8 --strategy ddp
    
    # With custom sparse settings
    python train_dsa.py --model xlarge --index-topk 1024 --indexer-loss-weight 0.05
"""

import os
import argparse
from pathlib import Path
import torch

try:
    import lightning as L
    from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
    from lightning.pytorch.loggers import TensorBoardLogger
    from lightning.pytorch.strategies import DDPStrategy
except ImportError:
    print("ERROR: PyTorch Lightning not installed.")
    print("Install with: pip install lightning")
    exit(1)

from pure_transformer.configs import get_model_config
from pure_transformer.training.dsa_lightning_module import create_dsa_lightning_module
from pure_transformer.data.streaming import StreamingConfig, create_pretraining_dataloader
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description='DSA Transformer Training')
    
    # Model configuration
    parser.add_argument('--model', type=str, default='xlarge',
                       choices=['tiny', 'small', 'medium', 'medium-large', 'large', 'xlarge'],
                       help='Base model size')
    
    # DSA-specific configuration
    parser.add_argument('--index-topk', type=int, default=2048,
                       help='Top-k tokens for sparse attention (default: 2048)')
    parser.add_argument('--index-n-heads', type=int, default=4,
                       help='Number of indexer heads (default: 4)')
    parser.add_argument('--index-head-dim', type=int, default=32,
                       help='Indexer head dimension (default: 32)')
    parser.add_argument('--train-indexer', action='store_true', default=True,
                       help='Train indexer with KL alignment loss')
    parser.add_argument('--no-train-indexer', dest='train_indexer', action='store_false')
    parser.add_argument('--indexer-loss-weight', type=float, default=0.1,
                       help='Weight for indexer alignment loss (default: 0.1)')
    parser.add_argument('--sparse-threshold', type=int, default=4096,
                       help='Use sparse attention for seq > threshold')
    
    # Training configuration
    parser.add_argument('--total-tokens', type=int, default=35_000_000_000,
                       help='Total training tokens (default: 35B)')
    parser.add_argument('--global-batch-size', type=int, default=524_288,
                       help='Global batch size in tokens')
    parser.add_argument('--micro-batch-size', type=int, default=16,
                       help='Micro batch size per GPU')
    parser.add_argument('--seq-length', type=int, default=2048,
                       help='Sequence length')
    
    # Optimizer configuration
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                       help='Peak learning rate')
    parser.add_argument('--min-lr', type=float, default=3e-5,
                       help='Minimum learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.1,
                       help='Weight decay')
    parser.add_argument('--warmup-tokens', type=int, default=100_000_000,
                       help='Warmup tokens')
    
    # Hardware configuration
    parser.add_argument('--devices', type=int, default=8,
                       help='Number of GPUs')
    parser.add_argument('--nodes', type=int, default=1,
                       help='Number of nodes')
    parser.add_argument('--strategy', type=str, default='ddp',
                       choices=['ddp', 'ddp_find_unused_parameters_true', 'fsdp'],
                       help='Distributed strategy')
    parser.add_argument('--precision', type=str, default='bf16-mixed',
                       choices=['32', '16-mixed', 'bf16-mixed'],
                       help='Training precision')
    
    # Data configuration
    parser.add_argument('--tokenizer', type=str, default='gpt2',
                       help='Tokenizer name')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Data loading workers (keep low for streaming)')
    parser.add_argument('--fineweb-subset', type=str, default='sample-100BT',
                       help='FineWeb-Edu subset')
    parser.add_argument('--fineweb-prob', type=float, default=0.65)
    parser.add_argument('--finepdf-prob', type=float, default=0.34)
    parser.add_argument('--usmle-prob', type=float, default=0.01)
    
    # Checkpointing
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--save-every-n-steps', type=int, default=500,
                       help='Save checkpoint every N steps')
    parser.add_argument('--resume-from', type=str, default=None,
                       help='Resume from checkpoint')
    
    # Logging
    parser.add_argument('--log-every-n-steps', type=int, default=10)
    parser.add_argument('--use-wandb', action='store_true',
                       help='Use Weights & Biases')
    parser.add_argument('--wandb-project', type=str, default='pure-transformer-dsa')
    parser.add_argument('--wandb-entity', type=str, default=None)
    
    # Other
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--compile', action='store_true',
                       help='Use torch.compile')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seed
    L.seed_everything(args.seed)
    torch.set_float32_matmul_precision('medium')
    
    print('='*80)
    print('DSA (DeepSeek Sparse Attention) TRAINING')
    print('='*80)
    
    # Load base model config
    base_config = get_model_config(args.model)
    print(f'\nBase Model: {base_config.model_name}')
    print(f'  Layers: {base_config.num_layers}')
    print(f'  Hidden: {base_config.hidden_size}')
    print(f'  Heads: {base_config.num_heads} ({base_config.num_kv_heads} KV)')
    
    print(f'\nDSA Configuration:')
    print(f'  Index Top-K: {args.index_topk}')
    print(f'  Indexer Heads: {args.index_n_heads}')
    print(f'  Train Indexer: {args.train_indexer}')
    print(f'  Indexer Loss Weight: {args.indexer_loss_weight}')
    print(f'  Sparse Threshold: {args.sparse_threshold}')
    
    # Create DSA Lightning module
    lightning_model = create_dsa_lightning_module(
        base_config=base_config,
        total_tokens=args.total_tokens,
        global_batch_size=args.global_batch_size,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_lr,
        weight_decay=args.weight_decay,
        warmup_tokens=args.warmup_tokens,
        index_topk=args.index_topk,
        train_indexer=args.train_indexer,
        indexer_loss_weight=args.indexer_loss_weight,
    )
    
    total_params = lightning_model.count_parameters()
    indexer_params = lightning_model.count_indexer_parameters()
    print(f'\nParameters:')
    print(f'  Total: {total_params:,}')
    print(f'  Indexer: {indexer_params:,} ({100*indexer_params/total_params:.2f}%)')
    print(f'Training tokens: {args.total_tokens/1e9:.1f}B')
    print(f'Max steps: {lightning_model.max_steps:,}')
    
    # Compile model
    if args.compile and torch.__version__ >= '2.0.0':
        print('\nCompiling model with torch.compile...')
        lightning_model.model = torch.compile(lightning_model.model)
    
    # Setup data
    print(f'\nData Configuration:')
    print(f'  FineWeb-Edu: {args.fineweb_prob*100:.0f}%')
    print(f'  FinePDFs: {args.finepdf_prob*100:.0f}%')
    print(f'  USMLE: {args.usmle_prob*100:.0f}%')
    
    streaming_config = StreamingConfig(
        fineweb_subset=args.fineweb_subset,
        fineweb_probability=args.fineweb_prob,
        finepdf_probability=args.finepdf_prob,
        usmle_probability=args.usmle_prob,
        max_seq_length=args.seq_length,
        shuffle_buffer_size=10_000,
        seed=args.seed,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = args.seq_length
    
    train_dataloader = create_pretraining_dataloader(
        tokenizer=tokenizer,
        config=streaming_config,
        batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )
    
    # Gradient accumulation
    total_gpus = args.devices * args.nodes
    tokens_per_micro_batch = args.micro_batch_size * args.seq_length
    accumulation_steps = args.global_batch_size // (tokens_per_micro_batch * total_gpus)
    accumulation_steps = max(1, accumulation_steps)
    
    print(f'\nTraining Configuration:')
    print(f'  GPUs: {total_gpus}')
    print(f'  Micro batch: {args.micro_batch_size}')
    print(f'  Gradient accumulation: {accumulation_steps}')
    print(f'  Effective batch (tokens): {tokens_per_micro_batch * total_gpus * accumulation_steps:,}')
    
    # Setup callbacks
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename='dsa-transformer-{epoch:02d}-{step:06d}',
            save_top_k=3,
            save_last=True,
            every_n_train_steps=args.save_every_n_steps,
            save_on_train_epoch_end=True,
        ),
        LearningRateMonitor(logging_interval='step'),
    ]
    
    # Setup loggers
    loggers = [
        TensorBoardLogger(save_dir=str(checkpoint_dir), name='tensorboard')
    ]
    
    if args.use_wandb:
        try:
            from lightning.pytorch.loggers import WandbLogger
            wandb_logger = WandbLogger(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=f'dsa-{base_config.model_name}-{args.devices}gpu',
                save_dir=str(checkpoint_dir),
                config={
                    'model': args.model,
                    'index_topk': args.index_topk,
                    'train_indexer': args.train_indexer,
                    'indexer_loss_weight': args.indexer_loss_weight,
                    'total_tokens': args.total_tokens,
                    'global_batch_size': args.global_batch_size,
                    'learning_rate': args.learning_rate,
                    'parameters': total_params,
                    'indexer_parameters': indexer_params,
                },
            )
            loggers.append(wandb_logger)
            print('\nW&B logging enabled')
        except ImportError:
            print('Warning: wandb not installed, skipping W&B logging')
    
    # Setup strategy
    if args.strategy == 'ddp':
        strategy = DDPStrategy(find_unused_parameters=False)
    elif args.strategy == 'ddp_find_unused_parameters_true':
        strategy = DDPStrategy(find_unused_parameters=True)
    else:
        strategy = args.strategy
    
    # Create trainer
    trainer = L.Trainer(
        accelerator='gpu',
        devices=args.devices,
        num_nodes=args.nodes,
        strategy=strategy,
        precision=args.precision,
        max_steps=lightning_model.max_steps,
        accumulate_grad_batches=accumulation_steps,
        gradient_clip_val=1.0,
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=args.log_every_n_steps,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
    )
    
    # Train
    print('\n' + '='*80)
    print('STARTING DSA TRAINING')
    print('='*80 + '\n')
    
    trainer.fit(
        lightning_model,
        train_dataloaders=train_dataloader,
        ckpt_path=args.resume_from,
    )
    
    print('\nTraining complete!')


if __name__ == '__main__':
    main()
