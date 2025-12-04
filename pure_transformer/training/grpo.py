"""
GRPO (Group Relative Policy Optimization) Training

Implementation based on:
- DeepSeekMath: "GRPO: Group Relative Policy Optimization"
- Simplified REINFORCE with baseline subtraction
- Token-level advantage normalization (GAPO style)

Key features:
- No KL regularization (simpler than PPO)
- On-policy sampling
- Advantage = reward - mean(rewards)
- Multiple samples per prompt for stable gradients
"""

import os
import math
import itertools
from typing import Optional, List, Dict, Callable, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor

from pure_transformer.model import TransformerLM
from pure_transformer.configs import RLConfig


# =============================================================================
# Reward Functions
# =============================================================================

def extract_answer_gsm8k(text: str) -> Optional[str]:
    """Extract numerical answer from GSM8K response."""
    import re
    # Look for "#### <number>" pattern
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    if match:
        return match.group(1).replace(',', '')
    # Fallback: look for last number
    numbers = re.findall(r'-?[\d,]+\.?\d*', text)
    return numbers[-1].replace(',', '') if numbers else None


def gsm8k_reward(generated_text: str, ground_truth: str) -> float:
    """
    Compute reward for GSM8K task.
    
    Returns:
        1.0 if correct, 0.0 if incorrect
    """
    pred = extract_answer_gsm8k(generated_text)
    true = extract_answer_gsm8k(ground_truth)
    if pred is None or true is None:
        return 0.0
    try:
        return 1.0 if float(pred) == float(true) else 0.0
    except ValueError:
        return 0.0


def extract_answer_medqa(text: str) -> Optional[str]:
    """Extract answer letter from MedQA response."""
    import re
    # Look for "Answer: X" pattern
    match = re.search(r'Answer:\s*([A-E])', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    # Fallback: first letter A-E
    match = re.search(r'\b([A-E])\b', text)
    return match.group(1).upper() if match else None


def medqa_reward(generated_text: str, ground_truth: str) -> float:
    """
    Compute reward for MedQA task.
    
    Returns:
        1.0 if correct, 0.0 if incorrect
    """
    pred = extract_answer_medqa(generated_text)
    true = extract_answer_medqa(ground_truth)
    if pred is None or true is None:
        return 0.0
    return 1.0 if pred == true else 0.0


REWARD_FUNCTIONS = {
    "gsm8k": gsm8k_reward,
    "medqa": medqa_reward,
}


# =============================================================================
# GRPO Trainer
# =============================================================================

class GRPOTrainer:
    """
    Group Relative Policy Optimization (GRPO) Trainer.
    
    Algorithm:
    1. For each prompt, generate G samples
    2. Compute rewards for each sample
    3. Compute advantages: A = r - mean(r) for the group
    4. Policy gradient: -log(pi) * A
    """
    
    def __init__(
        self,
        model: TransformerLM,
        tokenizer,
        config: RLConfig,
        device: torch.device,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.is_main = rank == 0
        
        # Setup optimizer
        self.optimizer = model.setup_optimizers(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            embedding_lr_scale=config.embedding_lr / config.learning_rate if config.embedding_lr > 0 else 1.0,
        )
        
        # Reward functions
        self.reward_fns = {task: REWARD_FUNCTIONS[task] for task in config.tasks}
        
        # Training state
        self.global_step = 0
        self.total_rewards = []
    
    @torch.no_grad()
    def generate_samples(
        self,
        prompt_ids: Tensor,
        num_samples: int,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Generate multiple samples for a prompt.
        
        Returns:
            sequences: List of generated token sequences
            masks: List of generation masks (1 for generated, 0 for prompt)
        """
        self.model.eval()
        
        B = prompt_ids.shape[0]
        prompt_len = prompt_ids.shape[1]
        
        sequences = []
        masks = []
        
        # Generate in batches to avoid OOM
        batch_size = self.config.device_batch_size
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        for _ in range(num_batches):
            curr_batch = min(batch_size, num_samples - len(sequences))
            
            # Expand prompt for batch generation
            expanded_prompt = prompt_ids.repeat(curr_batch, 1)
            
            # Generate
            generated = self.model.generate(
                expanded_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
            
            # Create masks
            for i in range(curr_batch):
                seq = generated[i]
                mask = torch.zeros_like(seq)
                mask[prompt_len:] = 1  # Mark generated tokens
                sequences.append(seq)
                masks.append(mask)
        
        return sequences, masks
    
    def compute_log_probs(
        self,
        input_ids: Tensor,
        target_ids: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """
        Compute log probabilities for target tokens.
        
        Args:
            input_ids: (B, T) input token IDs
            target_ids: (B, T) target token IDs
            mask: (B, T) mask for generated tokens
            
        Returns:
            log_probs: (B,) sum of log probs per sequence
        """
        self.model.train()
        
        # Forward pass
        logits = self.model(input_ids[:, :-1])
        
        # Compute log probs
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Gather target log probs
        target_log_probs = torch.gather(
            log_probs, 
            dim=-1, 
            index=target_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        
        # Mask and sum
        masked_log_probs = target_log_probs * mask[:, 1:]
        seq_log_probs = masked_log_probs.sum(dim=-1)
        
        return seq_log_probs
    
    def grpo_loss(
        self,
        log_probs: Tensor,
        advantages: Tensor,
    ) -> Tensor:
        """
        Compute GRPO loss.
        
        Loss = -E[log(pi) * advantage]
        """
        # Policy gradient loss
        loss = -(log_probs * advantages).mean()
        return loss
    
    def train_step(
        self,
        prompts: List[Dict],
        reward_fn: Callable,
    ) -> Dict[str, float]:
        """
        Single GRPO training step.
        
        Args:
            prompts: List of prompt dicts with 'input_ids' and 'ground_truth'
            reward_fn: Function to compute rewards
            
        Returns:
            Metrics dict
        """
        all_sequences = []
        all_masks = []
        all_rewards = []
        all_prompt_lens = []
        
        # Generate samples for each prompt
        for prompt in prompts:
            prompt_ids = prompt['input_ids'].to(self.device)
            ground_truth = prompt['ground_truth']
            
            sequences, masks = self.generate_samples(
                prompt_ids.unsqueeze(0),
                num_samples=self.config.num_samples_per_prompt,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
            )
            
            # Compute rewards
            for seq in sequences:
                generated_text = self.tokenizer.decode(seq[len(prompt_ids):].tolist())
                reward = reward_fn(generated_text, ground_truth)
                all_rewards.append(reward)
            
            all_sequences.extend(sequences)
            all_masks.extend(masks)
            all_prompt_lens.append(len(prompt_ids))
        
        # Convert to tensors
        rewards = torch.tensor(all_rewards, device=self.device, dtype=torch.float32)
        
        # Compute group advantages (per-prompt normalization)
        advantages = []
        idx = 0
        for _ in prompts:
            group_rewards = rewards[idx:idx + self.config.num_samples_per_prompt]
            mean_reward = group_rewards.mean()
            group_advantages = group_rewards - mean_reward
            if self.config.normalize_advantages:
                std = group_advantages.std() + 1e-8
                group_advantages = group_advantages / std
            advantages.append(group_advantages)
            idx += self.config.num_samples_per_prompt
        advantages = torch.cat(advantages)
        
        # Train in mini-batches
        total_loss = 0.0
        num_batches = 0
        
        batch_size = self.config.device_batch_size
        for i in range(0, len(all_sequences), batch_size):
            batch_seqs = all_sequences[i:i + batch_size]
            batch_masks = all_masks[i:i + batch_size]
            batch_advantages = advantages[i:i + batch_size]
            
            # Pad sequences
            max_len = max(len(s) for s in batch_seqs)
            padded_seqs = torch.zeros(len(batch_seqs), max_len, dtype=torch.long, device=self.device)
            padded_masks = torch.zeros(len(batch_seqs), max_len, dtype=torch.float32, device=self.device)
            
            for j, (seq, mask) in enumerate(zip(batch_seqs, batch_masks)):
                padded_seqs[j, :len(seq)] = seq
                padded_masks[j, :len(mask)] = mask.float()
            
            # Compute log probs
            log_probs = self.compute_log_probs(padded_seqs, padded_seqs, padded_masks)
            
            # Compute loss
            loss = self.grpo_loss(log_probs, batch_advantages)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        # Metrics
        metrics = {
            "loss": total_loss / max(num_batches, 1),
            "mean_reward": rewards.mean().item(),
            "max_reward": rewards.max().item(),
            "reward_std": rewards.std().item(),
        }
        
        self.global_step += 1
        self.total_rewards.extend(all_rewards)
        
        return metrics
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "config": self.config,
        }
        torch.save(checkpoint, path)
        if self.is_main:
            print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint["global_step"]
        if self.is_main:
            print(f"Loaded checkpoint from {path}")


def run_grpo_training(
    model: TransformerLM,
    tokenizer,
    train_dataset,
    config: RLConfig,
    device: torch.device,
    eval_dataset=None,
    logger=None,
):
    """
    Run GRPO training loop.
    
    Args:
        model: TransformerLM model
        tokenizer: Tokenizer
        train_dataset: Training dataset with prompts
        config: RLConfig
        device: Device to train on
        eval_dataset: Optional evaluation dataset
        logger: Optional logger (wandb, clearml, etc.)
    """
    trainer = GRPOTrainer(model, tokenizer, config, device)
    
    # Load pretrained checkpoint if specified
    if config.pretrained_checkpoint and os.path.exists(config.pretrained_checkpoint):
        checkpoint = torch.load(config.pretrained_checkpoint, map_location=device)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        print(f"Loaded pretrained checkpoint from {config.pretrained_checkpoint}")
    
    # Training loop
    num_prompts = len(train_dataset)
    steps_per_epoch = num_prompts // config.prompts_per_step
    total_steps = steps_per_epoch * config.num_epochs
    
    print(f"Starting GRPO training:")
    print(f"  Prompts: {num_prompts}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total steps: {total_steps}")
    
    for epoch in range(config.num_epochs):
        # Shuffle dataset
        indices = torch.randperm(num_prompts).tolist()
        
        for step_in_epoch in range(steps_per_epoch):
            # Get batch of prompts
            start_idx = step_in_epoch * config.prompts_per_step
            batch_indices = indices[start_idx:start_idx + config.prompts_per_step]
            prompts = [train_dataset[i] for i in batch_indices]
            
            # Determine reward function
            task = config.tasks[0]  # Use first task for now
            reward_fn = REWARD_FUNCTIONS[task]
            
            # Training step
            metrics = trainer.train_step(prompts, reward_fn)
            
            global_step = epoch * steps_per_epoch + step_in_epoch
            
            # Logging
            if global_step % config.log_every_n_steps == 0:
                print(f"Step {global_step}/{total_steps} | "
                      f"Loss: {metrics['loss']:.4f} | "
                      f"Reward: {metrics['mean_reward']:.4f}")
                
                if logger:
                    logger.log({"step": global_step, **metrics})
            
            # Save checkpoint
            if global_step % config.save_every_n_steps == 0 and global_step > 0:
                save_path = os.path.join(config.checkpoint_dir, f"step_{global_step}.pt")
                os.makedirs(config.checkpoint_dir, exist_ok=True)
                trainer.save_checkpoint(save_path)
    
    # Final checkpoint
    final_path = os.path.join(config.checkpoint_dir, "final.pt")
    trainer.save_checkpoint(final_path)
    
    return trainer
