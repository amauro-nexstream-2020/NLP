"""
Inference utilities for Pure Transformer model.

Provides checkpoint loading and text generation utilities.
"""

import torch
from pathlib import Path
from typing import Optional, Tuple
from transformers import GPT2Tokenizer

from pure_transformer.model import TransformerLM
from pure_transformer.configs import get_model_config


def load_model_from_checkpoint(
    checkpoint_path: str,
    model_size: str = "xlarge",
    device: str = "cuda"
) -> Tuple[TransformerLM, GPT2Tokenizer]:
    """
    Load a trained TransformerLM model from a Lightning checkpoint.
    
    Args:
        checkpoint_path: Path to the .ckpt file
        model_size: Model configuration name (tiny, small, medium, large, xlarge)
        device: Device to load model on
        
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model config
    config = get_model_config(model_size)
    
    # Create model
    model = TransformerLM(config)
    
    # Extract state dict from Lightning checkpoint
    state_dict = checkpoint.get("state_dict", checkpoint)
    
    # Remove 'model.' prefix if present (Lightning wraps model)
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            cleaned_state_dict[key[6:]] = value
        else:
            cleaned_state_dict[key] = value
    
    # Load weights
    model.load_state_dict(cleaned_state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully: {config.model_name}")
    print(f"Parameters: {model.count_parameters():,}")
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def generate_text(
    model: TransformerLM,
    tokenizer: GPT2Tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 50,
    device: str = "cuda"
) -> str:
    """
    Generate text from a prompt.
    
    Args:
        model: Loaded TransformerLM model
        tokenizer: GPT2 tokenizer
        prompt: Input prompt text
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (higher = more creative)
        top_k: Top-k sampling parameter
        device: Device to run on
        
    Returns:
        Generated text (including prompt)
    """
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )
    
    # Decode and return
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text


def get_latest_checkpoint(checkpoint_dir: str = "/checkpoints") -> Optional[str]:
    """
    Get the path to the latest checkpoint (last.ckpt preferred).
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        Path to latest checkpoint or None if not found
    """
    checkpoint_path = Path(checkpoint_dir)
    
    # Prefer last.ckpt
    last_ckpt = checkpoint_path / "last.ckpt"
    if last_ckpt.exists():
        return str(last_ckpt)
    
    # Otherwise find most recent
    ckpt_files = list(checkpoint_path.glob("*.ckpt"))
    if not ckpt_files:
        return None
    
    # Sort by modification time
    ckpt_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(ckpt_files[0])
