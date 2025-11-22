"""
Training utilities and training loop implementation.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import Optional, Callable
import os

from .utils import AverageMeter, save_checkpoint


class Trainer:
    """
    Training class for the LLM.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[torch.device] = None,
        checkpoint_dir: str = "./checkpoints",
        log_interval: int = 10,
        eval_interval: int = 500,
        save_interval: int = 1000,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = checkpoint_dir
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        
        # Move model to device
        self.model.to(self.device)
        
        # Tracking
        self.global_step = 0
        self.epoch = 0
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def train_epoch(self, max_steps: Optional[int] = None) -> float:
        """
        Train for one epoch.
        
        Args:
            max_steps: Maximum number of steps to train for (useful for streaming datasets)
        """
        self.model.train()
        losses = AverageMeter()
        
        # Determine total steps for progress bar
        total_steps = len(self.train_loader) if hasattr(self.train_loader, '__len__') else max_steps
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}", total=total_steps)
        
        for batch_idx, batch in enumerate(pbar):
            if max_steps is not None and batch_idx >= max_steps:
                break
                
            # Move data to device
            input_ids = batch['input_ids'].to(self.device)
            targets = batch.get('labels', input_ids).to(self.device)
            
            # Forward pass
            logits, loss = self.model(input_ids, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Optimizer step
            self.optimizer.step()
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Update metrics
            losses.update(loss.item())
            self.global_step += 1
            
            # Logging
            if batch_idx % self.log_interval == 0:
                lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'loss': f'{losses.avg:.4f}',
                    'lr': f'{lr:.2e}'
                })
            
            # Evaluation
            if self.val_loader is not None and self.global_step % self.eval_interval == 0:
                val_loss = self.evaluate()
                print(f"Step {self.global_step}: Val Loss = {val_loss:.4f}")
                self.model.train()
            
            # Save checkpoint
            if self.global_step % self.save_interval == 0:
                self.save(f"checkpoint_step_{self.global_step}.pt")
        
        return losses.avg
    
    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate on validation set."""
        if self.val_loader is None:
            return 0.0
        
        self.model.eval()
        losses = AverageMeter()
        
        for batch in tqdm(self.val_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(self.device)
            targets = batch.get('labels', input_ids).to(self.device)
            
            logits, loss = self.model(input_ids, targets)
            losses.update(loss.item())
        
        return losses.avg
    
    def train(self, num_epochs: int, max_steps_per_epoch: Optional[int] = None):
        """
        Train for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
            max_steps_per_epoch: Maximum steps per epoch (for streaming)
        """
        for epoch in range(num_epochs):
            self.epoch = epoch
            train_loss = self.train_epoch(max_steps=max_steps_per_epoch)
            
            if self.val_loader is not None:
                val_loss = self.evaluate()
                print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            else:
                print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")
            
            # Save epoch checkpoint
            self.save(f"checkpoint_epoch_{epoch}.pt")
    
    def save(self, filename: str):
        """Save checkpoint."""
        path = os.path.join(self.checkpoint_dir, filename)
        save_checkpoint(
            self.model,
            self.optimizer,
            self.epoch,
            0.0,  # Placeholder for loss
            path,
            global_step=self.global_step
        )
        print(f"Checkpoint saved: {path}")
