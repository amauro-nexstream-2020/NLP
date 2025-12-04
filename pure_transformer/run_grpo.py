#!/usr/bin/env python3
"""
GRPO/ProRL Training entry point for Pure Transformer.

Usage:
    python -m pure_transformer.run_grpo --config grpo_gsm8k --checkpoint ./checkpoints/pretrain/final.pt
    
For debugging:
    python -m pure_transformer.run_grpo --config debug --model tiny
"""

import argparse
import os
import sys

import torch

from pure_transformer.configs import get_model_config, get_rl_config
from pure_transformer.model import TransformerLM
from pure_transformer.training import run_grpo_training


class GSM8KDataset:
    """Simple GSM8K dataset wrapper."""
    
    def __init__(self, tokenizer, split="train", max_examples=None):
        try:
            from datasets import load_dataset
            ds = load_dataset("gsm8k", "main", split=split)
            if max_examples:
                ds = ds.select(range(min(max_examples, len(ds))))
            self.data = list(ds)
        except Exception as e:
            print(f"Warning: Could not load GSM8K: {e}")
            # Dummy data for testing
            self.data = [
                {"question": "What is 2 + 2?", "answer": "#### 4"},
                {"question": "What is 3 * 5?", "answer": "#### 15"},
            ] * 100
        
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = f"Question: {item['question']}\nAnswer: Let me solve this step by step.\n"
        input_ids = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.long)
        return {
            "input_ids": input_ids,
            "ground_truth": item["answer"],
        }


class MedQADataset:
    """Simple MedQA dataset wrapper."""
    
    def __init__(self, tokenizer, split="train", max_examples=None):
        try:
            from datasets import load_dataset
            ds = load_dataset("GBaker/MedQA-USMLE-4-options", split=split)
            if max_examples:
                ds = ds.select(range(min(max_examples, len(ds))))
            self.data = list(ds)
        except Exception as e:
            print(f"Warning: Could not load MedQA: {e}")
            # Dummy data
            self.data = [
                {
                    "question": "What is the function of the heart?",
                    "options": ["Pump blood", "Digest food", "Filter air", "Produce hormones"],
                    "answer": "A",
                },
            ] * 100
        
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        options = item.get("options", [])
        prompt = f"Question: {item['question']}\n"
        for i, opt in enumerate(options):
            prompt += f"{chr(65+i)}) {opt}\n"
        prompt += "Answer: "
        
        input_ids = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.long)
        return {
            "input_ids": input_ids,
            "ground_truth": f"Answer: {item.get('answer', 'A')}",
        }


def main():
    parser = argparse.ArgumentParser(description="GRPO Training for Pure Transformer")
    parser.add_argument("--config", type=str, default="grpo_gsm8k",
                       choices=["debug", "grpo_gsm8k", "grpo_medqa", "grpo_full"],
                       help="RL config preset")
    parser.add_argument("--model", type=str, default="medium",
                       choices=["tiny", "small", "medium"],
                       help="Model size")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Pretrained checkpoint to load")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                       help="Override checkpoint directory")
    args = parser.parse_args()
    
    # Load configs
    model_config = get_model_config(args.model)
    rl_config = get_rl_config(args.config)
    
    if args.checkpoint:
        rl_config.pretrained_checkpoint = args.checkpoint
    if args.checkpoint_dir:
        rl_config.checkpoint_dir = args.checkpoint_dir
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    print(f"Creating model: {model_config.model_name}")
    model = TransformerLM(model_config)
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Setup tokenizer
    try:
        from transformers import GPT2TokenizerFast
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
    except ImportError:
        print("WARNING: transformers not installed. Using dummy tokenizer.")
        class DummyTokenizer:
            def encode(self, text):
                return [ord(c) % 50304 for c in text]
            def decode(self, ids):
                return "".join(chr(i % 128) for i in ids)
        tokenizer = DummyTokenizer()
    
    # Create dataset based on task
    task = rl_config.tasks[0]
    print(f"Training on task: {task}")
    
    if task == "gsm8k":
        train_dataset = GSM8KDataset(tokenizer, split="train")
    elif task == "medqa":
        train_dataset = MedQADataset(tokenizer, split="train")
    else:
        raise ValueError(f"Unknown task: {task}")
    
    print(f"Dataset size: {len(train_dataset)}")
    
    # Optional: ClearML logging
    logger = None
    try:
        from clearml import Task
        task_obj = Task.init(
            project_name="pure-transformer",
            task_name=f"grpo-{args.model}-{task}",
        )
        logger = task_obj.get_logger()
        print("ClearML logging enabled")
    except ImportError:
        print("ClearML not available. Logging to stdout only.")
    
    # Run GRPO training
    run_grpo_training(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        config=rl_config,
        device=device,
        logger=logger,
    )


if __name__ == "__main__":
    main()
