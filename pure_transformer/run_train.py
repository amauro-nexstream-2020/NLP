#!/usr/bin/env python3
"""
Run Pretraining for Pure Transformer with DeepSeek-V3.2 Optimizations

Usage:
    python run_pretrain.py --config a100_1day --model medium
    
For RL training:
    python run_pretrain.py --mode rl --config grpo_medqa --checkpoint pretrain.pt
    
Features:
- FineWeb-Edu + USMLE/MedQA streaming
- DeepSeek Sparse Attention (optional)
- Enhanced GRPO with DeepSeek optimizations
- A100 optimized (bf16, gradient checkpointing)
"""

import argparse
import os
import sys
import time
from datetime import datetime
from typing import Optional

import torch
import torch.distributed as dist


def setup_distributed():
    """Setup distributed training if available."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    return 0, 1, 0


def get_tokenizer():
    """Get tokenizer (GPT-2 based)."""
    try:
        from transformers import GPT2TokenizerFast
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    except ImportError:
        print("WARNING: transformers not installed. Using dummy tokenizer.")
        
        class DummyTokenizer:
            vocab_size = 50304
            eos_token_id = 0
            pad_token_id = 1
            pad_token = "<pad>"
            eos_token = "<eos>"
            
            def encode(self, text):
                return [ord(c) % self.vocab_size for c in text]
            
            def decode(self, ids):
                return "".join(chr(i % 128) for i in ids if i < 128)
        
        return DummyTokenizer()


def run_pretraining(args):
    """Run pretraining pipeline."""
    from pure_transformer.model import TransformerLM
    from pure_transformer.configs import get_model_config, get_training_config
    from pure_transformer.data import create_pretraining_dataloader, StreamingConfig
    from pure_transformer.training.pretrain import get_lr
    
    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    is_main = rank == 0
    
    # Get device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        device = torch.device("cpu")
    
    if is_main:
        print(f"Using device: {device}")
        print(f"World size: {world_size}")
    
    # Load configs
    model_config = get_model_config(args.model)
    train_config = get_training_config(args.config)
    
    if args.checkpoint_dir:
        train_config.checkpoint_dir = args.checkpoint_dir
    
    # Create model
    if is_main:
        print(f"Creating model: {model_config.model_name}")
    
    model = TransformerLM(model_config).to(device)
    
    if is_main:
        print(f"Model parameters: {model.count_parameters():,}")
    
    # Resume if specified
    start_step = 0
    if args.resume and os.path.exists(args.resume):
        if is_main:
            print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_step = checkpoint.get("step", 0)
    
    # Get tokenizer
    tokenizer = get_tokenizer()
    
    # Setup streaming data
    data_config = StreamingConfig(
        fineweb_subset=args.fineweb_subset,
        fineweb_probability=args.fineweb_prob,
        medqa_probability=1.0 - args.fineweb_prob,
        max_seq_length=train_config.max_seq_length,
    )
    
    dataloader = create_pretraining_dataloader(
        tokenizer=tokenizer,
        config=data_config,
        batch_size=train_config.micro_batch_size,
        num_workers=args.num_workers,
    )
    
    # Compute training parameters
    tokens_per_step = train_config.global_batch_size
    total_steps = train_config.total_tokens // tokens_per_step
    warmup_steps = train_config.warmup_tokens // tokens_per_step
    
    micro_batch_tokens = train_config.micro_batch_size * train_config.max_seq_length
    grad_accum_steps = train_config.global_batch_size // (micro_batch_tokens * world_size)
    
    if is_main:
        print(f"\nTraining configuration:")
        print(f"  Total tokens: {train_config.total_tokens:,}")
        print(f"  Total steps: {total_steps:,}")
        print(f"  Warmup steps: {warmup_steps:,}")
        print(f"  Gradient accumulation: {grad_accum_steps}")
        print(f"  Micro batch size: {train_config.micro_batch_size}")
        print(f"  Global batch size: {train_config.global_batch_size:,}")
    
    # Setup optimizer
    optimizer = model.setup_optimizers(
        learning_rate=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    
    # DDP wrapper
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank]
        )
    
    # Mixed precision
    autocast_dtype = torch.bfloat16 if train_config.mixed_precision == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler() if train_config.mixed_precision == "fp16" else None
    
    # Training loop
    model.train()
    data_iter = iter(dataloader)
    total_tokens = 0
    running_loss = 0.0
    start_time = time.time()
    
    for step in range(start_step, total_steps):
        # Update learning rate
        lr = get_lr(step, warmup_steps, total_steps, train_config.learning_rate, train_config.min_learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Gradient accumulation
        optimizer.zero_grad()
        accum_loss = 0.0
        
        for micro_step in range(grad_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward with autocast
            with torch.cuda.amp.autocast(dtype=autocast_dtype):
                raw_model = model.module if hasattr(model, 'module') else model
                loss, _ = raw_model(input_ids, labels=labels)
                loss = loss / grad_accum_steps
            
            # Backward
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            accum_loss += loss.item()
            total_tokens += input_ids.numel() * world_size
        
        # Gradient clipping
        if scaler:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.max_grad_norm)
        
        # Optimizer step
        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        
        running_loss += accum_loss
        
        # Logging
        if step % train_config.log_every_n_steps == 0 and is_main:
            elapsed = time.time() - start_time
            tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0
            
            print(f"Step {step}/{total_steps} | "
                  f"Loss: {accum_loss:.4f} | "
                  f"LR: {lr:.2e} | "
                  f"Tokens: {total_tokens:,} | "
                  f"Tok/s: {tokens_per_sec:,.0f}")
        
        # Save checkpoint
        if step % train_config.save_every_n_steps == 0 and step > 0 and is_main:
            os.makedirs(train_config.checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(train_config.checkpoint_dir, f"step_{step}.pt")
            
            raw_model = model.module if hasattr(model, 'module') else model
            checkpoint = {
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": step,
                "total_tokens": total_tokens,
                "config": model_config,
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
            
            # Also save as latest
            latest_path = os.path.join(train_config.checkpoint_dir, "latest.pt")
            torch.save(checkpoint, latest_path)
    
    # Final checkpoint
    if is_main:
        final_path = os.path.join(train_config.checkpoint_dir, "final.pt")
        raw_model = model.module if hasattr(model, 'module') else model
        torch.save({
            "model_state_dict": raw_model.state_dict(),
            "step": total_steps,
            "total_tokens": total_tokens,
        }, final_path)
        print(f"Training complete! Final checkpoint: {final_path}")


def run_rl_training(args):
    """Run RL (GRPO) training pipeline."""
    from pure_transformer.model import TransformerLM
    from pure_transformer.configs import get_model_config, get_rl_config
    from pure_transformer.data import RLPromptDataset, StreamingConfig
    from pure_transformer.training.enhanced_grpo import EnhancedGRPOTrainer, GRPOConfig
    from pure_transformer.training.grpo import medqa_reward, gsm8k_reward
    
    # Setup
    rank, world_size, local_rank = setup_distributed()
    is_main = rank == 0
    
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    if is_main:
        print(f"Using device: {device}")
        print(f"Starting RL training with config: {args.config}")
    
    # Load model config (use same as pretrain)
    model_config = get_model_config(args.model)
    
    # Create model
    model = TransformerLM(model_config).to(device)
    
    # Load pretrained checkpoint
    if args.checkpoint and os.path.exists(args.checkpoint):
        if is_main:
            print(f"Loading pretrained checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    
    # Create reference model (frozen copy)
    ref_model = TransformerLM(model_config).to(device)
    ref_model.load_state_dict(model.state_dict())
    for param in ref_model.parameters():
        param.requires_grad = False
    
    # Get tokenizer
    tokenizer = get_tokenizer()
    
    # Setup GRPO config
    grpo_config = GRPOConfig(
        num_samples_per_prompt=args.num_samples,
        prompts_per_step=args.prompts_per_step,
        device_batch_size=args.device_batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        use_unbiased_kl=True,
        use_off_policy_masking=True,
        kl_coef=args.kl_coef,
    )
    
    # Create trainer
    trainer = EnhancedGRPOTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        config=grpo_config,
        device=device,
        rank=rank,
        world_size=world_size,
    )
    
    # Setup data
    data_config = StreamingConfig(max_seq_length=512)
    
    # Select reward function
    task = args.task
    if task == "medqa":
        reward_fn = medqa_reward
    elif task == "gsm8k":
        reward_fn = gsm8k_reward
    else:
        raise ValueError(f"Unknown task: {task}")
    
    if is_main:
        print(f"Training on task: {task}")
        print(f"Samples per prompt: {grpo_config.num_samples_per_prompt}")
        print(f"Learning rate: {grpo_config.learning_rate}")
    
    # Create prompt dataset
    prompt_dataset = RLPromptDataset(tokenizer, data_config, task=task)
    
    # Training loop
    prompts_buffer = []
    for epoch in range(grpo_config.num_epochs):
        if is_main:
            print(f"\nEpoch {epoch + 1}/{grpo_config.num_epochs}")
        
        for i, prompt in enumerate(prompt_dataset):
            prompts_buffer.append(prompt)
            
            if len(prompts_buffer) >= grpo_config.prompts_per_step:
                # Training step
                metrics = trainer.train_step(prompts_buffer, reward_fn)
                prompts_buffer = []
                
                if trainer.global_step % grpo_config.log_every_n_steps == 0 and is_main:
                    print(f"Step {trainer.global_step} | "
                          f"Loss: {metrics['loss']:.4f} | "
                          f"Reward: {metrics['mean_reward']:.4f} | "
                          f"Max: {metrics['max_reward']:.4f}")
                
                # Save checkpoint
                if trainer.global_step % grpo_config.save_every_n_steps == 0 and is_main:
                    save_path = os.path.join(
                        args.checkpoint_dir, f"rl_step_{trainer.global_step}.pt"
                    )
                    os.makedirs(args.checkpoint_dir, exist_ok=True)
                    trainer.save_checkpoint(save_path)
            
            # Early stop for debugging
            if args.max_steps > 0 and trainer.global_step >= args.max_steps:
                break
        
        if args.max_steps > 0 and trainer.global_step >= args.max_steps:
            break
    
    # Final save
    if is_main:
        final_path = os.path.join(args.checkpoint_dir, "rl_final.pt")
        trainer.save_checkpoint(final_path)
        print(f"RL training complete! Final checkpoint: {final_path}")


def run_tests(args):
    """Run all tests."""
    print("Running comprehensive tests...")
    
    from pure_transformer.tests import run_all_tests
    success = run_all_tests()
    
    if success:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed!")
        sys.exit(1)


def test_streaming(args):
    """Test streaming datasets."""
    print("Testing streaming datasets...")
    
    from pure_transformer.data.streaming import (
        test_fineweb_streaming,
        test_medqa_streaming,
        test_interleaved_streaming,
        test_rl_prompts,
    )
    
    results = []
    results.append(("FineWeb-Edu", test_fineweb_streaming()))
    results.append(("MedQA", test_medqa_streaming()))
    results.append(("Interleaved", test_interleaved_streaming()))
    results.append(("RL Prompts", test_rl_prompts()))
    
    print("\n" + "=" * 60)
    print("Streaming Test Summary:")
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
    
    if all(r[1] for r in results):
        print("\n✓ All streaming tests passed!")
    else:
        print("\n✗ Some streaming tests failed!")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Pure Transformer Training with DeepSeek-V3.2 Optimizations"
    )
    
    subparsers = parser.add_subparsers(dest="mode", help="Training mode")
    
    # Pretrain subparser
    pretrain_parser = subparsers.add_parser("pretrain", help="Run pretraining")
    pretrain_parser.add_argument("--config", type=str, default="a100_1day",
                                 choices=["debug", "a100_1day", "a100_3day"])
    pretrain_parser.add_argument("--model", type=str, default="medium",
                                 choices=["tiny", "small", "medium"])
    pretrain_parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/pretrain")
    pretrain_parser.add_argument("--resume", type=str, default=None)
    pretrain_parser.add_argument("--fineweb-subset", type=str, default="sample-10BT")
    pretrain_parser.add_argument("--fineweb-prob", type=float, default=0.85)
    pretrain_parser.add_argument("--num-workers", type=int, default=2)
    
    # RL subparser
    rl_parser = subparsers.add_parser("rl", help="Run RL training (GRPO)")
    rl_parser.add_argument("--config", type=str, default="grpo_medqa")
    rl_parser.add_argument("--model", type=str, default="medium",
                           choices=["tiny", "small", "medium"])
    rl_parser.add_argument("--checkpoint", type=str, required=True,
                           help="Pretrained checkpoint to start from")
    rl_parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/rl")
    rl_parser.add_argument("--task", type=str, default="medqa",
                           choices=["medqa", "gsm8k"])
    rl_parser.add_argument("--num-samples", type=int, default=8)
    rl_parser.add_argument("--prompts-per-step", type=int, default=4)
    rl_parser.add_argument("--device-batch-size", type=int, default=2)
    rl_parser.add_argument("--lr", type=float, default=1e-5)
    rl_parser.add_argument("--kl-coef", type=float, default=0.01)
    rl_parser.add_argument("--epochs", type=int, default=3)
    rl_parser.add_argument("--max-steps", type=int, default=-1)
    
    # Test subparser
    test_parser = subparsers.add_parser("test", help="Run tests")
    test_parser.add_argument("--streaming", action="store_true",
                             help="Test streaming datasets only")
    
    args = parser.parse_args()
    
    if args.mode == "pretrain":
        run_pretraining(args)
    elif args.mode == "rl":
        run_rl_training(args)
    elif args.mode == "test":
        if args.streaming:
            test_streaming(args)
        else:
            run_tests(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
