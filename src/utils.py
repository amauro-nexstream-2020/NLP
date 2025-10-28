"""
Utility functions for the LLM project.
"""

import torch
import random
import numpy as np
from typing import Optional


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the total number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_number(num: int) -> str:
    """
    Format large numbers in a readable way.
    
    Args:
        num: Number to format
    
    Returns:
        Formatted string (e.g., "1.5M", "2.3B")
    """
    if num >= 1e9:
        return f"{num / 1e9:.2f}B"
    elif num >= 1e6:
        return f"{num / 1e6:.2f}M"
    elif num >= 1e3:
        return f"{num / 1e3:.2f}K"
    else:
        return str(num)


def get_device(prefer_mps: bool = True) -> torch.device:
    """
    Get the best available device (CUDA > MPS > CPU).
    
    Args:
        prefer_mps: Whether to prefer MPS (Apple Silicon) over CPU
    
    Returns:
        PyTorch device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def print_model_summary(model: torch.nn.Module, input_size: Optional[tuple] = None):
    """
    Print a summary of the model architecture.
    
    Args:
        model: PyTorch model
        input_size: Optional input size for detailed summary
    """
    print("=" * 80)
    print("MODEL SUMMARY")
    print("=" * 80)
    print(f"Total Parameters: {format_number(count_parameters(model))}")
    print("=" * 80)
    print(model)
    print("=" * 80)


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str,
    **kwargs
):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        path: Path to save checkpoint
        **kwargs: Additional items to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        **kwargs
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    model: torch.nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None
) -> dict:
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model
        path: Path to checkpoint
        optimizer: Optional optimizer to load state
        device: Device to load checkpoint to
    
    Returns:
        Checkpoint dictionary
    """
    if device is None:
        device = get_device()
    
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint
