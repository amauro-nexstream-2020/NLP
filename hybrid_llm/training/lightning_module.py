"""
PyTorch Lightning module for the Hybrid Mamba + Gated Attention LLM.
"""

from math import pi, cos
from typing import Any, Dict, List
import torch
import lightning as L
from torch import nn

from hybrid_llm.model.hybrid_llm import create_model
from hybrid_llm.configs import ModelConfig, TrainingConfig


class HybridLightningModule(L.LightningModule):
    """
    Lightning wrapper that wires the Hybrid LLM, optimizer, and schedulers.
    """
    
    def __init__(self, model_cfg: ModelConfig, train_cfg: TrainingConfig):
        super().__init__()
        self.save_hyperparameters(ignore=["model_cfg", "train_cfg"])
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.model = create_model(model_cfg)
        self.example_input_array = torch.zeros(1, train_cfg.max_seq_length, dtype=torch.long)
    
    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None):
        return self.model(input_ids, labels=labels)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        loss, _ = self.model(batch["input_ids"], labels=batch["labels"])
        self.log(
            "train/loss",
            loss,
            on_step=True,
            prog_bar=True,
            batch_size=batch["input_ids"].size(0),
            sync_dist=False,
        )
        return loss
    
    def configure_optimizers(self):
        # Parameter groups for weight decay
        decay_params, nodecay_params = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or "bias" in name or "norm" in name:
                nodecay_params.append(param)
            else:
                decay_params.append(param)
        
        optimizer_grouped_parameters = [
            {
                "params": decay_params,
                "weight_decay": self.train_cfg.weight_decay,
                "lr": self.train_cfg.learning_rate,
            },
            {
                "params": nodecay_params,
                "weight_decay": 0.0,
                "lr": self.train_cfg.learning_rate,
            },
        ]
        
        use_bnb = False
        optimizer = None
        if self.train_cfg.optimizer_type == "adamw_8bit":
            try:
                from bitsandbytes.optim import AdamW8bit
                optimizer = AdamW8bit(
                    optimizer_grouped_parameters,
                    lr=self.train_cfg.learning_rate,
                    betas=(self.train_cfg.adam_beta1, self.train_cfg.adam_beta2),
                    eps=self.train_cfg.adam_epsilon,
                )
                use_bnb = True
            except Exception:
                # Fallback to standard AdamW if bitsandbytes is unavailable
                pass
        
        if optimizer is None:
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.train_cfg.learning_rate,
                betas=(self.train_cfg.adam_beta1, self.train_cfg.adam_beta2),
                eps=self.train_cfg.adam_epsilon,
                weight_decay=self.train_cfg.weight_decay,
            )
        
        # Scheduler: warmup + cosine decay to min LR
        def lr_lambda(current_step: int):
            warmup = max(1, self.train_cfg.warmup_steps)
            total_steps = max(warmup + 1, self.train_cfg.max_steps)
            if current_step < warmup:
                return float(current_step) / float(warmup)
            progress = min(1.0, (current_step - warmup) / float(total_steps - warmup))
            min_lr_scale = self.train_cfg.min_learning_rate / self.train_cfg.learning_rate
            cosine_decay = 0.5 * (1.0 + cos(pi * progress))
            return min_lr_scale + (1.0 - min_lr_scale) * cosine_decay
        
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda),
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [scheduler]
    
    def on_train_start(self):
        # Helpful logging for ClearML and console
        total_params = sum(p.numel() for p in self.model.parameters())
        self.log("params/total", float(total_params), prog_bar=False)
        if not self.trainer.is_global_zero:
            return
        print(f"Total parameters: {total_params/1e9:.3f}B")
