"""
PyTorch Lightning Module for Multi-GPU Training

Optimized for 8x A100 distributed training with:
- DDP (Distributed Data Parallel)
- Flash Attention
- Gradient Checkpointing
- Mixed Precision (bf16)
- Optimized batch sizing
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
import lightning as L
from lightning.pytorch.utilities import rank_zero_only

from pure_transformer.model import TransformerLM
from pure_transformer.configs import TransformerConfig


class LightningTransformer(L.LightningModule):
    """
    Lightning module for distributed transformer training.
    
    Features:
    - Automatic distributed training (DDP)
    - Learning rate scheduling with warmup
    - Gradient clipping
    - Loss tracking and logging
    - Checkpoint management
    """
    
    def __init__(
        self,
        config: TransformerConfig,
        learning_rate: float = 3e-4,
        min_learning_rate: float = 3e-5,
        weight_decay: float = 0.1,
        warmup_steps: int = 2000,
        max_steps: int = 100000,
        max_grad_norm: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['config'])
        
        # Model
        self.model = TransformerLM(config)
        self.config = config
        
        # Training hyperparameters
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.max_grad_norm = max_grad_norm
        
        # Metrics
        self.train_loss = []
        
    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        """Forward pass through the model."""
        return self.model(input_ids, labels=labels)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step for one batch."""
        input_ids = batch['input_ids']
        labels = batch['labels']
        
        # Forward pass
        loss, logits = self(input_ids, labels=labels)
        
        # Log metrics (only if trainer is attached)
        try:
            self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log('train/ppl', torch.exp(loss), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log('train/lr', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, prog_bar=True)
            
            # Log throughput (tokens/sec)
            if batch_idx % 100 == 0:
                batch_size = input_ids.size(0)
                seq_len = input_ids.size(1)
                tokens = batch_size * seq_len * self.trainer.world_size
                self.log('train/tokens', float(tokens), on_step=True, sync_dist=True)
        except (RuntimeError, AttributeError):
            # Trainer not attached (testing mode)
            pass
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step for one batch."""
        input_ids = batch['input_ids']
        labels = batch['labels']
        
        # Forward pass
        loss, logits = self(input_ids, labels=labels)
        
        # Log metrics (only if trainer attached)
        try:
            self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log('val/ppl', torch.exp(loss), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        except (RuntimeError, AttributeError):
            pass
        
        return loss
    
    def configure_optimizers(self):
        """
        Configure optimizer with parameter groups and learning rate schedule.
        
        Separates parameters into:
        - Decayed: weights (with weight decay)
        - Non-decayed: biases, LayerNorm, embeddings (no weight decay)
        """
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            # No weight decay for biases, layernorm, and embeddings
            if any(nd in name for nd in ['bias', 'norm', 'embedding']):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        # Parameter groups
        param_groups = [
            {'params': decay_params, 'weight_decay': self.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]
        
        # AdamW optimizer (fused=False for compatibility with gradient clipping in bf16-mixed)
        # Note: fused=True causes issues with AMP precision plugin gradient clipping
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=False,  # Disabled for Lightning AMP compatibility
        )
        
        # Cosine learning rate schedule with warmup
        def lr_lambda(current_step: int):
            """Learning rate schedule: warmup + cosine decay."""
            if current_step < self.warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, self.warmup_steps))
            else:
                # Cosine decay
                progress = float(current_step - self.warmup_steps) / float(
                    max(1, self.max_steps - self.warmup_steps)
                )
                cosine_decay = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)))
                return max(
                    self.min_learning_rate / self.learning_rate,
                    cosine_decay.item()
                )
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        }
    
    def on_before_optimizer_step(self, optimizer):
        """Gradient clipping before optimizer step."""
        # Compute gradient norm (useful for monitoring)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.parameters(),
            self.max_grad_norm
        )
        try:
            self.log('train/grad_norm', grad_norm, on_step=True, prog_bar=False, sync_dist=True)
        except (RuntimeError, AttributeError):
            pass
    
    @rank_zero_only
    def on_train_epoch_end(self):
        """Called at the end of training epoch (rank 0 only)."""
        avg_loss = torch.stack(self.train_loss).mean() if self.train_loss else torch.tensor(0.0)
        print(f"\nEpoch {self.current_epoch} complete. Avg loss: {avg_loss:.4f}")
        self.train_loss = []
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return self.model.count_parameters()


def create_lightning_module(
    config: TransformerConfig,
    total_tokens: int,
    global_batch_size: int,
    learning_rate: float = 3e-4,
    min_learning_rate: float = 3e-5,
    weight_decay: float = 0.1,
    warmup_tokens: int = 100_000_000,
    max_grad_norm: float = 1.0,
) -> LightningTransformer:
    """
    Create a Lightning module with appropriate settings for training.
    
    Args:
        config: Model configuration
        total_tokens: Total training tokens
        global_batch_size: Global batch size in tokens
        learning_rate: Peak learning rate
        min_learning_rate: Minimum learning rate
        weight_decay: Weight decay coefficient
        warmup_tokens: Tokens for warmup
        max_grad_norm: Gradient clipping threshold
    
    Returns:
        LightningTransformer module ready for training
    """
    # Calculate steps
    max_steps = total_tokens // global_batch_size
    warmup_steps = warmup_tokens // global_batch_size
    
    return LightningTransformer(
        config=config,
        learning_rate=learning_rate,
        min_learning_rate=min_learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        max_grad_norm=max_grad_norm,
    )
