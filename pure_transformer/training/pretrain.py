"""
Pretraining Script for Pure Transformer

Streaming pretraining on FineWeb-Edu + MedQA mixture.
Optimized for 3-day A100 training.
"""

import os
import math
from typing import Optional, Iterator, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from torch import Tensor

from pure_transformer.model import TransformerLM, TransformerConfig
from pure_transformer.configs import TrainingConfig, DataConfig


# =============================================================================
# Streaming Dataset (reuse from hybrid_llm)
# =============================================================================

def get_streaming_dataloader(
    tokenizer,
    config: DataConfig,
    batch_size: int,
    num_workers: int = 2,
) -> DataLoader:
    """
    Create streaming dataloader for pretraining.
    
    Uses FineWeb-Edu + MedQA mixture from HuggingFace Hub.
    """
    try:
        from datasets import load_dataset, interleave_datasets
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")
    
    # Load datasets in streaming mode
    fineweb = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name=config.fineweb_subset,
        split="train",
        streaming=True,
    )
    
    medqa = load_dataset(
        "GBaker/MedQA-USMLE-4-options",
        split="train",
        streaming=True,
    )
    
    # Format MedQA
    def format_medqa(example):
        question = example.get("question", "")
        options = example.get("options", [])
        answer = example.get("answer", "")
        text = f"Question: {question}\n"
        for i, opt in enumerate(options):
            text += f"{chr(65+i)}) {opt}\n"
        text += f"Answer: {answer}"
        return {"text": text}
    
    medqa = medqa.map(format_medqa)
    
    # Interleave
    mixed = interleave_datasets(
        [fineweb, medqa],
        probabilities=[config.fineweb_probability, config.medqa_probability],
        seed=config.seed,
    )
    
    # Shuffle
    mixed = mixed.shuffle(buffer_size=config.shuffle_buffer_size, seed=config.seed)
    
    # Create PyTorch dataset
    class StreamingLMDataset(IterableDataset):
        def __init__(self, hf_dataset, tokenizer, max_length):
            self.hf_dataset = hf_dataset
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.buffer = []
        
        def __iter__(self):
            for example in self.hf_dataset:
                text = example.get("text", "")
                if not text:
                    continue
                
                # Tokenize
                tokens = self.tokenizer.encode(text)
                self.buffer.extend(tokens)
                
                # Yield complete sequences
                while len(self.buffer) >= self.max_length + 1:
                    seq = self.buffer[:self.max_length + 1]
                    self.buffer = self.buffer[self.max_length:]
                    
                    input_ids = torch.tensor(seq[:-1], dtype=torch.long)
                    labels = torch.tensor(seq[1:], dtype=torch.long)
                    
                    yield {"input_ids": input_ids, "labels": labels}
    
    dataset = StreamingLMDataset(mixed, tokenizer, config.max_seq_length)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )


# =============================================================================
# Learning Rate Scheduler
# =============================================================================

def get_lr(
    step: int,
    warmup_steps: int,
    total_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """Cosine learning rate schedule with warmup."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step >= total_steps:
        return min_lr
    
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


# =============================================================================
# Training Loop
# =============================================================================

def run_pretraining(
    model: TransformerLM,
    tokenizer,
    config: TrainingConfig,
    device: torch.device,
    logger=None,
):
    """
    Run pretraining loop.
    
    Args:
        model: TransformerLM model
        tokenizer: Tokenizer
        config: TrainingConfig
        device: Device to train on
        logger: Optional logger (wandb, clearml, etc.)
    """
    model = model.to(device)
    
    # Compute steps
    tokens_per_step = config.global_batch_size
    total_steps = config.total_tokens // tokens_per_step
    warmup_steps = config.warmup_tokens // tokens_per_step
    
    # Compute gradient accumulation
    micro_batch_tokens = config.micro_batch_size * config.max_seq_length
    grad_accum_steps = config.global_batch_size // micro_batch_tokens
    
    print(f"Training configuration:")
    print(f"  Total tokens: {config.total_tokens:,}")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Warmup steps: {warmup_steps:,}")
    print(f"  Gradient accumulation: {grad_accum_steps}")
    print(f"  Micro batch size: {config.micro_batch_size}")
    print(f"  Model parameters: {model.count_parameters():,}")
    
    # Setup optimizer
    optimizer = model.setup_optimizers(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    
    # Setup dataloader
    dataloader = get_streaming_dataloader(
        tokenizer=tokenizer,
        config=config.data,
        batch_size=config.micro_batch_size,
    )
    data_iter = iter(dataloader)
    
    # Mixed precision
    scaler = torch.cuda.amp.GradScaler() if config.mixed_precision == "fp16" else None
    autocast_dtype = torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16
    
    # Training loop
    model.train()
    total_loss = 0.0
    tokens_processed = 0
    
    for step in range(total_steps):
        # Update learning rate
        lr = get_lr(step, warmup_steps, total_steps, config.learning_rate, config.min_learning_rate)
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
            
            # Forward pass with autocast
            with torch.cuda.amp.autocast(dtype=autocast_dtype):
                loss, logits = model(input_ids, labels=labels)
                loss = loss / grad_accum_steps
            
            # Backward pass
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            accum_loss += loss.item()
            tokens_processed += input_ids.numel()
        
        # Gradient clipping
        if scaler:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        
        # Optimizer step
        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        
        total_loss += accum_loss
        
        # Logging
        if step % config.log_every_n_steps == 0:
            avg_loss = total_loss / (step + 1)
            tokens_per_sec = tokens_processed / ((step + 1) * grad_accum_steps)  # Rough estimate
            
            print(f"Step {step}/{total_steps} | "
                  f"Loss: {accum_loss:.4f} | "
                  f"LR: {lr:.2e} | "
                  f"Tokens: {tokens_processed:,}")
            
            if logger:
                logger.log({
                    "step": step,
                    "loss": accum_loss,
                    "lr": lr,
                    "tokens": tokens_processed,
                })
        
        # Checkpointing
        if step % config.save_every_n_steps == 0 and step > 0:
            save_path = os.path.join(config.checkpoint_dir, f"step_{step}.pt")
            os.makedirs(config.checkpoint_dir, exist_ok=True)
            
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": step,
                "tokens": tokens_processed,
                "config": config,
            }
            torch.save(checkpoint, save_path)
            print(f"Saved checkpoint to {save_path}")
    
    # Final checkpoint
    final_path = os.path.join(config.checkpoint_dir, "final.pt")
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": total_steps,
        "tokens": tokens_processed,
        "config": config,
    }
    torch.save(checkpoint, final_path)
    print(f"Training complete! Final checkpoint: {final_path}")
    
    return model
