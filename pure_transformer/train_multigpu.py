#!/usr/bin/env python
"""
Multi-GPU Training Script for Pure Transformer

Optimized for 8x A100 GPUs with PyTorch Lightning DDP
Target: 1.3B model with 11B tokens in 48 hours

Usage:
    # Single GPU (testing)
    python train_multigpu.py --model xlarge --devices 1

    # 8 GPUs (production)
    python train_multigpu.py --model xlarge --devices 8 --nodes 1
    
    # On Kubernetes with 8 A100s
    python train_multigpu.py --model xlarge --devices 8 --strategy ddp
"""

import os
import argparse
from pathlib import Path
import torch

# Check for lightning
try:
    import lightning as L
    from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
    from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
    from lightning.pytorch.strategies import DDPStrategy
except ImportError:
    print("ERROR: PyTorch Lightning not installed.")
    print("Install with: pip install lightning")
    exit(1)

from pure_transformer.configs import get_model_config
from pure_transformer.training.lightning_module import create_lightning_module
from pure_transformer.data.streaming import StreamingConfig, create_pretraining_dataloader
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description='Multi-GPU Training for Pure Transformer')
    
    # Model configuration
    parser.add_argument('--model', type=str, default='xlarge',
                       choices=['tiny', 'small', 'medium', 'medium-large', 'large', 'xlarge'],
                       help='Model size')
    
    # Training configuration (optimized for 8xA100 80GB)
    parser.add_argument('--total-tokens', type=int, default=35_000_000_000,
                       help='Total training tokens (default: 35B)')
    parser.add_argument('--global-batch-size', type=int, default=524_288,
                       help='Global batch size in tokens (default: 512K for maximum A100 throughput)')
    parser.add_argument('--micro-batch-size', type=int, default=16,
                       help='Micro batch size per GPU (default: 16 for A100 80GB)')
    parser.add_argument('--seq-length', type=int, default=2048,
                       help='Sequence length (default: 2048)')
    
    # Optimizer configuration
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                       help='Peak learning rate')
    parser.add_argument('--min-lr', type=float, default=3e-5,
                       help='Minimum learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.1,
                       help='Weight decay')
    parser.add_argument('--warmup-tokens', type=int, default=100_000_000,
                       help='Warmup tokens (default: 100M)')
    
    # Hardware configuration
    parser.add_argument('--devices', type=int, default=8,
                       help='Number of GPUs to use')
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
    parser.add_argument('--num-workers', type=int, default=12,
                       help='Number of data loading workers per GPU (12 for maximum I/O throughput)')
    parser.add_argument('--fineweb-subset', type=str, default='sample-100BT',
                       help='FineWeb-Edu subset (sample-10BT, sample-100BT, or CC-MAIN-2024-10)')
    parser.add_argument('--fineweb-prob', type=float, default=0.65,
                       help='FineWeb-Edu probability (65%% - high-quality web content)')
    parser.add_argument('--finepdf-prob', type=float, default=0.34,
                       help='FinePDFs probability (34%% - long-context PDF documents)')
    parser.add_argument('--usmle-prob', type=float, default=0.01,
                       help='USMLE QA probability (1%% - small dataset, will fine-tune with GRPO later)')
    
    # Checkpointing
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--save-every-n-steps', type=int, default=1000,
                       help='Save checkpoint every N steps')
    parser.add_argument('--resume-from', type=str, default=None,
                       help='Resume from checkpoint path')
    
    # Logging
    parser.add_argument('--log-every-n-steps', type=int, default=10,
                       help='Log every N steps')
    parser.add_argument('--use-wandb', action='store_true',
                       help='Use Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='pure-transformer',
                       help='W&B project name')
    parser.add_argument('--wandb-entity', type=str, default=None,
                       help='W&B entity/team name')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--compile', action='store_true',
                       help='Use torch.compile for speedup')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seed
    L.seed_everything(args.seed)
    
    print('='*80)
    print('MULTI-GPU TRAINING SETUP')
    print('='*80)
    
    # Load model config
    model_config = get_model_config(args.model)
    print(f'\nModel: {model_config.model_name}')
    print(f'  Layers: {model_config.num_layers}')
    print(f'  Hidden: {model_config.hidden_size}')
    print(f'  Intermediate: {model_config.intermediate_size}')
    print(f'  Heads: {model_config.num_heads} ({model_config.num_kv_heads} KV)')
    
    # Create Lightning module
    lightning_model = create_lightning_module(
        config=model_config,
        total_tokens=args.total_tokens,
        global_batch_size=args.global_batch_size,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_lr,
        weight_decay=args.weight_decay,
        warmup_tokens=args.warmup_tokens,
    )
    
    print(f'\nParameters: {lightning_model.count_parameters():,}')
    print(f'Training tokens: {args.total_tokens/1e9:.1f}B')
    print(f'Max steps: {lightning_model.max_steps:,}')
    print(f'Warmup steps: {lightning_model.warmup_steps:,}')
    
    # Compile model for speedup (optional)
    if args.compile and torch.__version__ >= '2.0.0':
        print('\nCompiling model with torch.compile...')
        lightning_model.model = torch.compile(lightning_model.model)
    
    # Setup data
    print(f'\nData Configuration:')
    print(f'  FineWeb-Edu subset: {args.fineweb_subset}')
    print(f'  FineWeb-Edu: {args.fineweb_prob*100:.0f}%')
    print(f'  FinePDFs: {args.finepdf_prob*100:.0f}% (high-quality structured content)')
    print(f'  USMLE QA: {args.usmle_prob*100:.0f}% (medical domain knowledge)')
    print(f'  Target: {args.total_tokens/1e9:.1f}B tokens')
    
    streaming_config = StreamingConfig(
        fineweb_subset=args.fineweb_subset,
        fineweb_probability=args.fineweb_prob,
        finepdf_probability=args.finepdf_prob,
        usmle_probability=args.usmle_prob,
        max_seq_length=args.seq_length,
        shuffle_buffer_size=10_000,
        seed=args.seed,
    )
    
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataloader
    train_dataloader = create_pretraining_dataloader(
        tokenizer=tokenizer,
        config=streaming_config,
        batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )
    
    # Calculate gradient accumulation steps
    tokens_per_batch = args.micro_batch_size * args.seq_length
    tokens_per_device_step = tokens_per_batch * args.devices * args.nodes
    grad_accum_steps = max(1, args.global_batch_size // tokens_per_device_step)
    
    print(f'\nBatch Configuration:')
    print(f'  Micro batch: {args.micro_batch_size}')
    print(f'  Sequence length: {args.seq_length}')
    print(f'  Tokens/micro-batch: {tokens_per_batch:,}')
    print(f'  Devices: {args.devices}')
    print(f'  Gradient accumulation: {grad_accum_steps}')
    print(f'  Effective global batch: {tokens_per_device_step * grad_accum_steps:,} tokens')
    
    # Setup callbacks
    callbacks = []
    
    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename='pure-transformer-{epoch}-{step}',
        save_top_k=3,
        save_last=True,
        every_n_train_steps=args.save_every_n_steps,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # Setup loggers
    loggers = []
    
    # TensorBoard
    tb_logger = TensorBoardLogger(
        save_dir=args.checkpoint_dir,
        name='lightning_logs',
    )
    loggers.append(tb_logger)
    
    # Weights & Biases (optional)
    if args.use_wandb:
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f'{model_config.model_name}-{args.devices}gpu',
            config=vars(args),
        )
        loggers.append(wandb_logger)
    
    # Setup distributed strategy
    if args.strategy == 'ddp':
        strategy = DDPStrategy(
            find_unused_parameters=False,
            gradient_as_bucket_view=True,  # More efficient
            static_graph=True,  # Faster with static computational graph
        )
    elif args.strategy == 'ddp_find_unused_parameters_true':
        strategy = DDPStrategy(find_unused_parameters=True)
    else:
        strategy = args.strategy
    
    print(f'\nHardware Configuration:')
    print(f'  GPUs: {args.devices}')
    print(f'  Nodes: {args.nodes}')
    print(f'  Strategy: {args.strategy}')
    print(f'  Precision: {args.precision}')
    
    # Create trainer
    trainer = L.Trainer(
        max_steps=lightning_model.max_steps,
        devices=args.devices,
        num_nodes=args.nodes,
        strategy=strategy,
        precision=args.precision,
        accumulate_grad_batches=grad_accum_steps,
        gradient_clip_val=1.0,
        log_every_n_steps=args.log_every_n_steps,
        callbacks=callbacks,
        logger=loggers,
        enable_checkpointing=True,
        enable_model_summary=True,
        deterministic=False,  # Faster
        benchmark=True,  # cudnn benchmark for speed
    )
    
    print('\n' + '='*80)
    print('STARTING TRAINING')
    print('='*80)
    # Optimized: 8xA100 with bf16, larger batches, more workers
    # Target: 1.8M-2.2M tokens/sec with optimized settings
    # 35B tokens / 2M tokens/sec = ~4.9 hours (well within 2 day target!)
    throughput_estimate = 200_000 * args.devices  # Optimized: 200K tokens/sec/GPU
    expected_hours = args.total_tokens / throughput_estimate / 3600
    print(f'Expected training time: ~{expected_hours:.1f} hours ({expected_hours/24:.2f} days)')
    print(f'Target throughput: {throughput_estimate:,} tokens/sec ({throughput_estimate/1e6:.1f}M tokens/sec)')
    print(f'Checkpoint directory: {args.checkpoint_dir}')
    print(f'\n⚡ OPTIMIZED FOR MAXIMUM A100 THROUGHPUT ⚡')
    print('='*80 + '\n')
    
    # Train
    trainer.fit(
        model=lightning_model,
        train_dataloaders=train_dataloader,
        ckpt_path=args.resume_from,
    )
    
    print('\n' + '='*80)
    print('TRAINING COMPLETE')
    print('='*80)
    print(f'Final checkpoint: {checkpoint_callback.best_model_path}')


if __name__ == '__main__':
    main()
