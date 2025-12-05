#!/usr/bin/env python3
"""
Pretraining entry point for Pure Transformer.

Usage:
    python -m pure_transformer.run_pretrain --config a100_3day --model medium
    
For debugging:
    python -m pure_transformer.run_pretrain --config debug --model tiny
"""

import argparse
import os
import sys

import torch

from pure_transformer.configs import get_model_config, get_training_config
from pure_transformer.model import TransformerLM
from pure_transformer.training import run_pretraining


def main():
    parser = argparse.ArgumentParser(description="Pretrain Pure Transformer")
    parser.add_argument("--config", type=str, default="a100_3day",
                       choices=["debug", "a100_3day", "a100_1day"],
                       help="Training config preset")
    parser.add_argument("--model", type=str, default="medium",
                       choices=["tiny", "small", "medium"],
                       help="Model size")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                       help="Override checkpoint directory")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from checkpoint")
    args = parser.parse_args()
    
    # Load configs
    model_config = get_model_config(args.model)
    train_config = get_training_config(args.config)
    
    if args.checkpoint_dir:
        train_config.checkpoint_dir = args.checkpoint_dir
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    print(f"Creating model: {model_config.model_name}")
    model = TransformerLM(model_config)
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Resume if specified
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    
    # Setup tokenizer (use GPT-2 tokenizer for simplicity)
    try:
        from transformers import GPT2TokenizerFast
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
    except ImportError:
        print("WARNING: transformers not installed. Using dummy tokenizer.")
        # Dummy tokenizer for testing
        class DummyTokenizer:
            def encode(self, text):
                return [ord(c) % 50304 for c in text]
            def decode(self, ids):
                return "".join(chr(i % 128) for i in ids)
        tokenizer = DummyTokenizer()
    
    # Optional: ClearML logging
    logger = None
    try:
        from clearml import Task
        task = Task.init(
            project_name="pure-transformer",
            task_name=f"pretrain-{args.model}-{args.config}",
        )
        clearml_logger = task.get_logger()

        # Adapter to expose a `log()` method for our training code
        class _ClearMLAdapter:
            def __init__(self, logger):
                self._logger = logger

            def log(self, metrics: dict):
                step = int(metrics.get("step", 0)) if metrics.get("step") is not None else 0
                for k, v in metrics.items():
                    if k == "step":
                        continue
                    try:
                        # Use report_scalar with namespace 'metrics'
                        self._logger.report_scalar("metrics", k, v, step)
                    except Exception:
                        # fallback: print
                        print(f"Could not report metric {k}: {v}")

        logger = _ClearMLAdapter(clearml_logger)
        print("ClearML logging enabled")
    except ImportError:
        # Fallback to a simple logger that prints to stdout and supports .log(dict)
        class _StdoutLogger:
            def log(self, metrics: dict):
                print("LOG:", metrics)
        logger = _StdoutLogger()
        print("ClearML not available. Logging to stdout only.")
    
    # Run training
    run_pretraining(
        model=model,
        tokenizer=tokenizer,
        config=train_config,
        device=device,
        logger=logger,
    )


if __name__ == "__main__":
    main()
