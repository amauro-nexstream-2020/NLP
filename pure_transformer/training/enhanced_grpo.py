

import os
import math
from typing import Optional, List, Dict, Callable, Tuple
from dataclasses import dataclass
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class GRPOConfig:
    """Enhanced GRPO configuration."""
    
    # Group sampling
    num_samples_per_prompt: int = 16  # G in GRPO
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    max_new_tokens: int = 512
    
    # Batch sizes
    prompts_per_step: int = 8
    device_batch_size: int = 4
    
    # Optimizer
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    
    # GRPO algorithm settings
    clip_epsilon: float = 0.2  # PPO-style clipping
    use_baseline: bool = True
    normalize_advantages: bool = False
    
    # DeepSeek enhancements
    use_unbiased_kl: bool = True
    kl_coef: float = 0.01  # Beta in the paper
    off_policy_threshold: float = 0.3  # Delta for masking
    use_off_policy_masking: bool = True
    use_keep_sampling_mask: bool = True
    
    # Reward settings
    length_penalty_coef: float = 0.0
    language_consistency_coef: float = 0.0
    
    # Training
    num_epochs: int = 3
    save_every_n_steps: int = 100
    log_every_n_steps: int = 1


class EnhancedGRPOTrainer:
    """
    Enhanced GRPO Trainer with DeepSeek-V3.2 optimizations.
    
    Key improvements:
    1. Unbiased KL estimation
    2. Off-policy sequence masking
    3. Sampling mask preservation
    4. Multi-domain reward support
    """
    
    def __init__(
        self,
        model,
        ref_model,  # Reference model for KL
        tokenizer,
        config: GRPOConfig,
        device: torch.device,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.is_main = rank == 0
        
        # Freeze reference model
        if self.ref_model is not None:
            for param in self.ref_model.parameters():
                param.requires_grad = False
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Training state
        self.global_step = 0
        self.metrics_history = []
    
    def _setup_optimizer(self):
        """Setup AdamW optimizer with proper param groups."""
        param_groups = [
            {'params': self.model.parameters(), 'lr': self.config.learning_rate}
        ]
        return torch.optim.AdamW(
            param_groups,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.95),
        )
    
    @torch.no_grad()
    def generate_samples(
        self,
        prompt_ids: Tensor,
        num_samples: int,
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        """
        Generate samples with top-p/top-k and preserve sampling masks.
        
        Returns:
            sequences: Generated sequences
            masks: Generation masks (1 for generated tokens)
            sampling_masks: Top-p/top-k masks used during sampling
        """
        self.model.eval()
        
        B = prompt_ids.shape[0]
        prompt_len = prompt_ids.shape[1]
        
        sequences = []
        masks = []
        sampling_masks_list = []  # Preserve for keep_sampling_mask
        
        for _ in range(num_samples):
            # Generate with sampling mask tracking
            seq, gen_mask, samp_mask = self._generate_single(
                prompt_ids,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
            )
            sequences.append(seq)
            masks.append(gen_mask)
            sampling_masks_list.append(samp_mask)
        
        return sequences, masks, sampling_masks_list
    
    def _generate_single(
        self,
        prompt_ids: Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> Tuple[Tensor, Tensor, List[Tensor]]:
        """Generate a single sequence with sampling mask tracking."""
        B, T = prompt_ids.shape
        device = prompt_ids.device
        
        generated = prompt_ids.clone()
        sampling_masks = []
        
        for _ in range(max_new_tokens):
            # Forward pass
            logits = self.model(generated)
            next_logits = logits[:, -1, :]  # (B, vocab_size)
            
            # Temperature scaling
            if temperature > 0:
                next_logits = next_logits / temperature
            
            # Create sampling mask
            vocab_size = next_logits.shape[-1]
            mask = torch.ones(B, vocab_size, dtype=torch.bool, device=device)
            
            # Top-k filtering
            if top_k > 0 and top_k < vocab_size:
                top_k_vals, _ = torch.topk(next_logits, top_k)
                threshold = top_k_vals[:, -1].unsqueeze(-1)
                mask = mask & (next_logits >= threshold)
            
            # Top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cumulative_probs <= top_p
                sorted_mask[:, 0] = True  # Always keep at least one token
                # Unsort
                mask_update = torch.zeros_like(mask)
                mask_update.scatter_(1, sorted_indices, sorted_mask)
                mask = mask & mask_update
            
            # Store sampling mask
            sampling_masks.append(mask.clone())
            
            # Apply mask and sample
            filtered_logits = next_logits.masked_fill(~mask, float('-inf'))
            probs = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for EOS
            if hasattr(self.tokenizer, 'eos_token_id'):
                if (next_token == self.tokenizer.eos_token_id).all():
                    break
        
        # Create generation mask
        gen_mask = torch.zeros(generated.shape[1], dtype=torch.float32, device=device)
        gen_mask[T:] = 1.0
        
        return generated.squeeze(0), gen_mask, sampling_masks
    
    def compute_log_probs_with_mask(
        self,
        input_ids: Tensor,
        target_ids: Tensor,
        mask: Tensor,
        sampling_masks: Optional[List[Tensor]] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute log probabilities with optional sampling mask application.
        
        Returns:
            current_log_probs: Log probs under current policy
            ref_log_probs: Log probs under reference policy
        """
        self.model.train()
        
        # Current policy log probs
        logits = self.model(input_ids[:, :-1])
        
        # Apply sampling mask if enabled
        if self.config.use_keep_sampling_mask and sampling_masks is not None:
            for t, samp_mask in enumerate(sampling_masks):
                if t < logits.shape[1]:
                    logits[:, t] = logits[:, t].masked_fill(~samp_mask, float('-inf'))
        
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Gather target log probs
        current_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=target_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        
        # Reference policy log probs
        if self.ref_model is not None:
            with torch.no_grad():
                ref_logits = self.ref_model(input_ids[:, :-1])
                if self.config.use_keep_sampling_mask and sampling_masks is not None:
                    for t, samp_mask in enumerate(sampling_masks):
                        if t < ref_logits.shape[1]:
                            ref_logits[:, t] = ref_logits[:, t].masked_fill(~samp_mask, float('-inf'))
                ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                ref_log_probs = torch.gather(
                    ref_log_probs,
                    dim=-1,
                    index=target_ids[:, 1:].unsqueeze(-1)
                ).squeeze(-1)
        else:
            ref_log_probs = torch.zeros_like(current_log_probs)
        
        # Mask and sum
        current_seq_log_probs = (current_log_probs * mask[:, 1:]).sum(dim=-1)
        ref_seq_log_probs = (ref_log_probs * mask[:, 1:]).sum(dim=-1)
        
        return current_seq_log_probs, ref_seq_log_probs
    
    def compute_unbiased_kl(
        self,
        current_log_probs: Tensor,  # log pi_theta
        ref_log_probs: Tensor,      # log pi_ref  
        old_log_probs: Tensor,      # log pi_old (sampling policy)
    ) -> Tensor:
        """
        Compute unbiased KL estimate using importance sampling.
        
        KL(pi_theta || pi_ref) = E_{pi_old}[
            (pi_theta / pi_old) * (pi_ref/pi_theta - log(pi_ref/pi_theta) - 1)
        ]
        
        This corrects the K3 estimator bias.
        """
        # Importance sampling ratio
        ratio = torch.exp(current_log_probs - old_log_probs)
        
        # Log ratio (pi_ref / pi_theta)
        log_ratio = ref_log_probs - current_log_probs
        
        # Unbiased KL estimate
        kl = ratio * (torch.exp(log_ratio) - log_ratio - 1)
        
        return kl
    
    def compute_off_policy_mask(
        self,
        old_log_probs: Tensor,
        current_log_probs: Tensor,
        advantages: Tensor,
    ) -> Tensor:
        """
        Compute off-policy sequence mask.
        
        Masks negative-advantage sequences with high policy divergence.
        """
        # KL divergence between old and current policy
        kl_div = old_log_probs - current_log_probs  # Approximation
        
        # Mask condition: negative advantage AND high divergence
        mask = torch.ones_like(advantages, dtype=torch.bool)
        if self.config.use_off_policy_masking:
            mask = ~((advantages < 0) & (kl_div > self.config.off_policy_threshold))
        
        return mask.float()
    
    def grpo_loss(
        self,
        current_log_probs: Tensor,
        old_log_probs: Tensor,
        ref_log_probs: Tensor,
        advantages: Tensor,
        off_policy_mask: Tensor,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Compute enhanced GRPO loss with all DeepSeek optimizations.
        
        Loss = -E[clip(ratio, 1-eps, 1+eps) * A * M] + beta * KL
        """
        # Importance sampling ratio
        ratio = torch.exp(current_log_probs - old_log_probs)
        
        # Clipped ratio
        clipped_ratio = torch.clamp(
            ratio,
            1 - self.config.clip_epsilon,
            1 + self.config.clip_epsilon
        )
        
        # Policy gradient loss (with masking)
        pg_loss1 = ratio * advantages
        pg_loss2 = clipped_ratio * advantages
        pg_loss = -torch.min(pg_loss1, pg_loss2) * off_policy_mask
        
        # KL penalty
        if self.config.use_unbiased_kl and self.ref_model is not None:
            kl_loss = self.compute_unbiased_kl(
                current_log_probs, ref_log_probs, old_log_probs
            )
        else:
            # Simple KL approximation
            kl_loss = current_log_probs - ref_log_probs
        
        # Total loss
        total_loss = pg_loss.mean() + self.config.kl_coef * kl_loss.mean()
        
        # Metrics
        metrics = {
            'pg_loss': pg_loss.mean().item(),
            'kl_loss': kl_loss.mean().item(),
            'ratio_mean': ratio.mean().item(),
            'ratio_std': ratio.std().item(),
            'mask_ratio': off_policy_mask.mean().item(),
        }
        
        return total_loss, metrics
    
    def train_step(
        self,
        prompts: List[Dict],
        reward_fn: Callable,
    ) -> Dict[str, float]:
        """
        Single enhanced GRPO training step.
        """
        all_sequences = []
        all_masks = []
        all_sampling_masks = []
        all_rewards = []
        all_old_log_probs = []
        
        # Generate samples for each prompt
        for prompt in prompts:
            prompt_ids = prompt['input_ids'].to(self.device)
            ground_truth = prompt.get('ground_truth', '')
            
            sequences, masks, sampling_masks = self.generate_samples(
                prompt_ids.unsqueeze(0),
                num_samples=self.config.num_samples_per_prompt,
            )
            
            # Compute old log probs (for importance sampling)
            for seq, mask, samp_masks in zip(sequences, masks, sampling_masks):
                with torch.no_grad():
                    seq_tensor = seq.unsqueeze(0)
                    mask_tensor = mask.unsqueeze(0)
                    old_lp, _ = self.compute_log_probs_with_mask(
                        seq_tensor, seq_tensor, mask_tensor, samp_masks
                    )
                    all_old_log_probs.append(old_lp.squeeze())
                
                # Compute reward
                prompt_len = prompt_ids.shape[0]
                generated_text = self.tokenizer.decode(seq[prompt_len:].tolist())
                reward = reward_fn(generated_text, ground_truth)
                
                # Length penalty
                if self.config.length_penalty_coef > 0:
                    gen_len = mask.sum().item()
                    reward -= self.config.length_penalty_coef * gen_len / 100
                
                all_rewards.append(reward)
                all_sequences.append(seq)
                all_masks.append(mask)
                all_sampling_masks.append(samp_masks)
        
        # Convert to tensors
        rewards = torch.tensor(all_rewards, device=self.device, dtype=torch.float32)
        old_log_probs = torch.stack(all_old_log_probs)
        
        # Compute group advantages
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
        
        # Training loop over mini-batches
        total_loss = 0.0
        total_metrics = {}
        num_batches = 0
        
        batch_size = self.config.device_batch_size
        for i in range(0, len(all_sequences), batch_size):
            batch_seqs = all_sequences[i:i + batch_size]
            batch_masks = all_masks[i:i + batch_size]
            batch_samp_masks = all_sampling_masks[i:i + batch_size]
            batch_advantages = advantages[i:i + batch_size]
            batch_old_lp = old_log_probs[i:i + batch_size]
            
            # Pad sequences
            max_len = max(len(s) for s in batch_seqs)
            padded_seqs = torch.zeros(len(batch_seqs), max_len, dtype=torch.long, device=self.device)
            padded_masks = torch.zeros(len(batch_seqs), max_len, dtype=torch.float32, device=self.device)
            
            for j, (seq, mask) in enumerate(zip(batch_seqs, batch_masks)):
                padded_seqs[j, :len(seq)] = seq
                padded_masks[j, :len(mask)] = mask
            
            # Compute current and ref log probs
            current_lp, ref_lp = self.compute_log_probs_with_mask(
                padded_seqs, padded_seqs, padded_masks, batch_samp_masks[0] if batch_samp_masks else None
            )
            
            # Compute off-policy mask
            off_policy_mask = self.compute_off_policy_mask(
                batch_old_lp, current_lp, batch_advantages
            )
            
            # Compute loss
            loss, metrics = self.grpo_loss(
                current_lp, batch_old_lp, ref_lp, batch_advantages, off_policy_mask
            )
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0) + v
            num_batches += 1
        
        # Aggregate metrics
        final_metrics = {
            'loss': total_loss / max(num_batches, 1),
            'mean_reward': rewards.mean().item(),
            'max_reward': rewards.max().item(),
            'reward_std': rewards.std().item(),
        }
        for k, v in total_metrics.items():
            final_metrics[k] = v / max(num_batches, 1)
        
        self.global_step += 1
        self.metrics_history.append(final_metrics)
        
        return final_metrics
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'config': self.config,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)
        if self.is_main:
            print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        if self.is_main:
            print(f"Loaded checkpoint from {path}")
