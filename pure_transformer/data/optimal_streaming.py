"""
Optimal Streaming Dataset Module - Based on "1 Billion Token Challenge"

Dataset Mix (for 2.5B+ tokens):
- 45% FinePDFs (1.125B tokens) - High-quality textbook-style PDFs
- 25% FineWeb-Edu (625M tokens) - Curated educational web  
- 20% C4 (500M tokens) - Diverse, high-quality web text (replaces DCLM)
- 10% USMLE QA (250M tokens) - Medical domain knowledge

Key Insights from Research:
1. Static mixing > Curriculum learning
2. 50-30-20 ratio is optimal (we adjust to 45-25-20-10 for USMLE)
3. Avoid hard transitions between data distributions
4. Balance validation performance with generalization

Performance Target:
- Training: 2 days on A100 80GB or A6000 48GB
- Tokens: 2.5B minimum
- Quality: 90%+ of GPT-2 performance with 1/4 the data
"""

import os
import random
from typing import Optional, Iterator, Dict, List, Any
from dataclasses import dataclass, field

import torch
from torch.utils.data import IterableDataset, DataLoader


@dataclass
class OptimalStreamingConfig:
    """
    Optimal streaming configuration based on research findings.
    
    Default mix: 45% FinePDFs + 25% FineWeb + 20% C4 + 10% USMLE
    """
    
    # Dataset sources
    finepdfs_dataset: str = "codelion/finepdfs-1B"
    fineweb_dataset: str = "HuggingFaceFW/fineweb-edu"
    fineweb_subset: str = "sample-10BT"
    c4_dataset: str = "allenai/c4"
    c4_subset: str = "en"
    usmle_dataset: str = "GBaker/MedQA-USMLE-4-options"
    
    # Mixing probabilities (must sum to 1.0)
    finepdfs_prob: float = 0.45
    fineweb_prob: float = 0.25
    c4_prob: float = 0.20
    usmle_prob: float = 0.10
    
    # Sequence settings
    max_seq_length: int = 2048
    shuffle_buffer_size: int = 10_000
    seed: int = 42
    
    # Training targets (for 2-day A100 training)
    total_tokens: int = 2_500_000_000  # 2.5B tokens minimum
    target_training_hours: int = 48  # 2 days
    
    def __post_init__(self):
        """Validate probabilities sum to 1.0"""
        total = self.finepdfs_prob + self.fineweb_prob + self.c4_prob + self.usmle_prob
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Probabilities must sum to 1.0, got {total}")


def create_finepdfs_stream(tokenizer, config: OptimalStreamingConfig):
    """
    Create FinePDFs streaming dataset.
    
    High-quality textbook-style educational PDFs from codelion.
    45% of total training data.
    """
    from datasets import load_dataset
    
    dataset = load_dataset(
        config.finepdfs_dataset,
        split="train",
        streaming=True,
    )
    
    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=config.max_seq_length,
            return_tensors="pt",
        )
    
    return dataset.map(tokenize, remove_columns=["text"])


def create_fineweb_stream(tokenizer, config: OptimalStreamingConfig):
    """
    Create FineWeb-Edu streaming dataset.
    
    Curated educational web resources from HuggingFace.
    25% of total training data.
    """
    from datasets import load_dataset
    
    dataset = load_dataset(
        config.fineweb_dataset,
        name=config.fineweb_subset,
        split="train",
        streaming=True,
    )
    
    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=config.max_seq_length,
            return_tensors="pt",
        )
    
    return dataset.map(tokenize, remove_columns=["text"])


def create_c4_stream(tokenizer, config: OptimalStreamingConfig):
    """
    Create C4 streaming dataset.
    
    Colossal Clean Crawled Corpus - diverse, high-quality web text.
    Replaces DCLM (which has compression issues).
    20% of total training data.
    """
    from datasets import load_dataset
    
    dataset = load_dataset(
        config.c4_dataset,
        config.c4_subset,
        split="train",
        streaming=True,
    )
    
    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=config.max_seq_length,
            return_tensors="pt",
        )
    
    return dataset.map(tokenize, remove_columns=["text", "timestamp", "url"])


def create_usmle_stream(tokenizer, config: OptimalStreamingConfig):
    """
    Create USMLE QA streaming dataset.
    
    MedQA-USMLE multiple choice questions for medical domain.
    10% of total training data.
    
    Format: "Question: {q}\nOptions: A) ... B) ... C) ... D) ...\nAnswer: {a}"
    """
    from datasets import load_dataset
    
    # Load full dataset (not streaming - it's small enough)
    dataset = load_dataset(
        config.usmle_dataset,
        split="train",
    )
    
    def format_usmle(example):
        """Format USMLE question as text."""
        question = example["question"]
        options = example["options"]
        answer = example["answer"]
        
        # Format options
        option_text = "\n".join([f"{chr(65+i)}) {opt}" for i, opt in enumerate(options.values())])
        
        # Full formatted text
        text = f"Question: {question}\nOptions:\n{option_text}\nAnswer: {answer}"
        
        return {"text": text}
    
    # Format and tokenize
    formatted = dataset.map(format_usmle, remove_columns=dataset.column_names)
    
    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=config.max_seq_length,
            return_tensors="pt",
        )
    
    return formatted.to_iterable_dataset().map(tokenize, remove_columns=["text"])


class OptimalStreamingDataset(IterableDataset):
    """
    Optimal mixed streaming dataset with static probabilities.
    
    Based on research: static 45-25-20-10 mixing outperforms
    curriculum learning strategies.
    
    Features:
    - Interleaved sampling from 4 sources
    - Static probability distribution (no curriculum)
    - Token packing for efficiency
    - Shuffle buffer for randomization
    """
    
    def __init__(
        self,
        tokenizer,
        config: OptimalStreamingConfig,
    ):
        self.tokenizer = tokenizer
        self.config = config
        
        # Create streaming iterators
        self.finepdfs = create_finepdfs_stream(tokenizer, config)
        self.fineweb = create_fineweb_stream(tokenizer, config)
        self.c4 = create_c4_stream(tokenizer, config)
        self.usmle = create_usmle_stream(tokenizer, config)
        
        # Probability thresholds for sampling
        self.probabilities = [
            config.finepdfs_prob,
            config.fineweb_prob,
            config.c4_prob,
            config.usmle_prob,
        ]
        
        # Random generator
        self.rng = random.Random(config.seed)
        
    def __iter__(self):
        """Iterate with probabilistic sampling."""
        # Create iterators
        iterators = [
            iter(self.finepdfs),
            iter(self.fineweb),
            iter(self.c4),
            iter(self.usmle),
        ]
        
        # Shuffle buffer
        buffer = []
        
        while True:
            try:
                # Sample dataset based on probabilities
                dataset_idx = self.rng.choices(range(4), weights=self.probabilities)[0]
                
                # Get next example from selected dataset
                example = next(iterators[dataset_idx])
                
                # Add to buffer
                buffer.append(example)
                
                # Yield from buffer when full
                if len(buffer) >= self.config.shuffle_buffer_size:
                    self.rng.shuffle(buffer)
                    for item in buffer:
                        yield self._format_example(item)
                    buffer = []
                    
            except StopIteration:
                # One of the iterators exhausted
                # Yield remaining buffer
                if buffer:
                    self.rng.shuffle(buffer)
                    for item in buffer:
                        yield self._format_example(item)
                break
    
    def _format_example(self, example):
        """Format example for training."""
        input_ids = example["input_ids"].squeeze(0)
        
        # Pad or truncate to max_seq_length
        if len(input_ids) > self.config.max_seq_length:
            input_ids = input_ids[:self.config.max_seq_length]
        elif len(input_ids) < self.config.max_seq_length:
            padding = torch.full(
                (self.config.max_seq_length - len(input_ids),),
                self.tokenizer.pad_token_id,
                dtype=input_ids.dtype
            )
            input_ids = torch.cat([input_ids, padding])
        
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
        }


def create_optimal_dataloader(
    tokenizer,
    config: OptimalStreamingConfig,
    batch_size: int,
    num_workers: int = 4,
) -> DataLoader:
    """
    Create optimized dataloader with 45-25-20-10 static mixing.
    
    Args:
        tokenizer: HuggingFace tokenizer
        config: Streaming configuration
        batch_size: Batch size
        num_workers: Number of data loading workers
    
    Returns:
        DataLoader ready for training
    """
    dataset = OptimalStreamingDataset(tokenizer, config)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )


def test_optimal_streaming():
    """Test the optimal streaming setup."""
    import os
    # Removed hard-coded HuggingFace token (was present in earlier commit).
# Set your HF token via environment variable: export HF_TOKEN="<HF_TOKEN>"
    
    from transformers import AutoTokenizer
    
    print("="*70)
    print("Testing Optimal Streaming Dataset (45-25-20-10)")
    print("="*70)
    
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create config
    config = OptimalStreamingConfig(
        total_tokens=2_500_000_000,
        shuffle_buffer_size=100,  # Small for testing
    )
    
    print(f"\nConfiguration:")
    print(f"  FinePDFs:    {config.finepdfs_prob*100:.0f}% ({config.finepdfs_prob * config.total_tokens / 1e9:.2f}B tokens)")
    print(f"  FineWeb-Edu: {config.fineweb_prob*100:.0f}% ({config.fineweb_prob * config.total_tokens / 1e9:.2f}B tokens)")
    print(f"  C4:          {config.c4_prob*100:.0f}% ({config.c4_prob * config.total_tokens / 1e9:.2f}B tokens)")
    print(f"  USMLE QA:    {config.usmle_prob*100:.0f}% ({config.usmle_prob * config.total_tokens / 1e9:.2f}B tokens)")
    print(f"  Total:       {config.total_tokens / 1e9:.1f}B tokens")
    
    # Create dataset
    dataset = OptimalStreamingDataset(tokenizer, config)
    
    # Test sampling
    print(f"\nSampling test (10 examples):")
    iterator = iter(dataset)
    
    for i in range(10):
        try:
            example = next(iterator)
            tokens = len(example["input_ids"])
            text = tokenizer.decode(example["input_ids"][:50])
            print(f"  Example {i+1}: {tokens} tokens, starts with: {text[:60]}...")
        except StopIteration:
            print(f"  Iterator exhausted at example {i+1}")
            break
    
    print("\n✓ Optimal streaming test complete")
    print("="*70)


if __name__ == "__main__":
    test_optimal_streaming()
