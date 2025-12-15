"""
PyTorch Lightning Module for DSA Transformer Training

Extends the base lightning module to support:
- DeepSeek Sparse Attention model
- Indexer alignment loss training
- Combined loss optimization
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import lightning as L
from lightning.pytorch.utilities import rank_zero_only

from pure_transformer.model.dsa_transformer import DSATransformer, DSAConfig, get_dsa_config
from pure_transformer.configs import TransformerConfig


class DSALightningModule(L.LightningModule):
    """
    Lightning module for DSA Transformer training.
    
    Features:
    - Combined LM loss + Indexer alignment loss
    - Automatic distributed training (DDP)
    - Learning rate scheduling with warmup
    - Gradient clipping
    """
    
    def __init__(
        self,
        config: DSAConfig,
        learning_rate: float = 3e-4,
        min_learning_rate: float = 3e-5,
        weight_decay: float = 0.1,
        warmup_steps: int = 2000,
        max_steps: int = 100000,
        max_grad_norm: float = 1.0,
        indexer_loss_weight: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['config'])
        
        # Model
        self.model = DSATransformer(config)
        self.config = config
        
        # Training hyperparameters
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.max_grad_norm = max_grad_norm
        self.indexer_loss_weight = indexer_loss_weight
    
    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        """Forward pass through DSA model."""
        return self.model(input_ids, labels=labels)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step with combined LM + indexer loss."""
        input_ids = batch['input_ids']
        labels = batch['labels']
        
        # Forward pass - returns (logits, lm_loss, indexer_loss)
        logits, lm_loss, indexer_loss = self(input_ids, labels=labels)
        
        # Combined loss
        total_loss = lm_loss
        if indexer_loss is not None and self.config.train_indexer:
            total_loss = lm_loss + self.indexer_loss_weight * indexer_loss
        
        # Log metrics
        try:
            self.log('train/loss', lm_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log('train/ppl', torch.exp(lm_loss), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log('train/total_loss', total_loss, on_step=True, sync_dist=True)
            
            if indexer_loss is not None:
                self.log('train/indexer_loss', indexer_loss, on_step=True, sync_dist=True)
            
            self.log('train/lr', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, prog_bar=True)
            
            # Log tokens/step
            if batch_idx % 100 == 0:
                batch_size = input_ids.size(0)
                seq_len = input_ids.size(1)
                tokens = batch_size * seq_len * self.trainer.world_size
                self.log('train/tokens_step', float(tokens), on_step=True, sync_dist=True)
        except (RuntimeError, AttributeError):
            pass
        
        return total_loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        input_ids = batch['input_ids']
        labels = batch['labels']
        
        logits, loss, _ = self(input_ids, labels=labels)
        
        try:
            self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log('val/ppl', torch.exp(loss), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        except (RuntimeError, AttributeError):
            pass
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer with separate groups for indexer."""
        # Separate parameters
        decay_params = []
        no_decay_params = []
        indexer_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Indexer parameters (may want different LR)
            if 'indexer' in name:
                indexer_params.append(param)
            # No weight decay for biases, norms, embeddings
            elif any(nd in name for nd in ['bias', 'norm', 'embedding']):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        param_groups = [
            {'params': decay_params, 'weight_decay': self.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
            {'params': indexer_params, 'weight_decay': 0.0, 'lr': self.learning_rate},
        ]
        
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=False,
        )
        
        # Cosine LR schedule with warmup
        def lr_lambda(current_step: int):
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            else:
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
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def count_indexer_parameters(self) -> int:
        """Count indexer parameters specifically."""
        return sum(
            p.numel() for name, p in self.model.named_parameters() 
            if p.requires_grad and 'indexer' in name
        )


def create_dsa_lightning_module(
    base_config: TransformerConfig,
    total_tokens: int,
    global_batch_size: int,
    learning_rate: float = 3e-4,
    min_learning_rate: float = 3e-5,
    weight_decay: float = 0.1,
    warmup_tokens: int = 100_000_000,
    index_topk: int = 2048,
    train_indexer: bool = True,
    indexer_loss_weight: float = 0.1,
) -> DSALightningModule:
    """
    Create a DSA Lightning module from a base config.
    
    Args:
        base_config: Base TransformerConfig
        total_tokens: Total training tokens
        global_batch_size: Global batch size in tokens
        learning_rate: Peak learning rate
        min_learning_rate: Minimum learning rate
        weight_decay: Weight decay
        warmup_tokens: Warmup tokens
        index_topk: Top-k for sparse attention
        train_indexer: Whether to train indexer
        indexer_loss_weight: Weight for indexer loss
    """
    # Create DSA config
    dsa_config = get_dsa_config(
        base_config,
        use_sparse_attention=True,
        index_topk=index_topk,
        train_indexer=train_indexer,
        indexer_loss_weight=indexer_loss_weight,
    )
    
    # Compute steps
    max_steps = total_tokens // global_batch_size
    warmup_steps = warmup_tokens // global_batch_size
    
    return DSALightningModule(
        config=dsa_config,
        learning_rate=learning_rate,
        min_learning_rate=min_learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        indexer_loss_weight=indexer_loss_weight,
    )
