"""
Training script for the Decoder-Only Transformer LLM.
"""

import argparse
import torch
from transformers import AutoTokenizer
from src.model import DecoderLM, ModelConfig
from src.training import Trainer
from src.data import create_dataloader
from src.utils import set_seed, get_device

def main():
    parser = argparse.ArgumentParser(description="Train LLM on FineWeb")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum number of training steps")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=1024, help="Sequence length")
    parser.add_argument("--n_embd", type=int, default=768, help="Embedding dimension")
    parser.add_argument("--n_head", type=int, default=12, help="Number of heads")
    parser.add_argument("--n_layer", type=int, default=12, help="Number of layers")
    parser.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb-edu", help="Dataset name")
    parser.add_argument("--subset", type=str, default="sample-10BT", help="Dataset subset")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Checkpoint directory")
    args = parser.parse_args()

    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    # 1. Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Data
    print(f"Creating dataloader for {args.dataset} ({args.subset})...")
    train_loader = create_dataloader(
        dataset_name=args.dataset,
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        subset=args.subset,
        split="train"
    )

    # 3. Model
    print("Initializing model...")
    config = ModelConfig(
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        n_positions=args.seq_len,
        vocab_size=tokenizer.vocab_size
    )
    model = DecoderLM(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 4. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # 5. Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        save_interval=500
    )

    # 6. Train
    print("Starting training...")
    # We treat max_steps as one "epoch" for the purpose of this script
    trainer.train_epoch(max_steps=args.max_steps)
    print("Training complete.")

if __name__ == "__main__":
    main()
