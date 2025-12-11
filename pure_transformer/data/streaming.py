"""
Streaming Dataset Module for Pure Transformer

Supports:
- FineWeb-Edu (general web text)
- USMLE/MedQA (medical QA)
- Flexible interleaving with configurable ratios
- Memory-efficient streaming

Optimized for:
- Large-scale pretraining
- Multi-domain training
- Reinforcement learning prompts
"""

import os
import random
from typing import Optional, Iterator, Dict, List, Any, Callable
from dataclasses import dataclass

import torch
from torch.utils.data import IterableDataset, DataLoader


@dataclass
class StreamingConfig:
    """Configuration for streaming datasets."""
    
    # FineWeb-Edu settings (15T tokens available)
    # Options: "sample-10BT" (~10B tokens), "sample-100BT" (~100B), "CC-MAIN-2024-10" (full crawl)
    fineweb_subset: str = "sample-10BT"  # Default to sample for testing
    fineweb_probability: float = 0.65  # 65% of data (high-quality web)
    
    # FinePDF settings - Official HuggingFace FinePDFs dataset
    # 1.19 TRILLION tokens available in English (eng_Latn)
    finepdf_dataset: str = "HuggingFaceFW/finepdfs"
    finepdf_language: str = "eng_Latn"  # English language subset
    finepdf_probability: float = 0.34  # 34% of data (long-context PDFs)
    
    # USMLE/Medical QA settings (~10M tokens only, will fine-tune with GRPO later)
    usmle_dataset: str = "GBaker/MedQA-USMLE-4-options"
    usmle_probability: float = 0.01  # 1% of data (tiny dataset, GRPO fine-tuning later)
    
    # Streaming settings
    shuffle_buffer_size: int = 10_000
    max_seq_length: int = 2048
    seed: int = 42
    
    # Tokenization
    truncation: bool = True
    padding: bool = False


def create_fineweb_stream(
    tokenizer,
    config: StreamingConfig,
    split: str = "train",
):
    """
    Create FineWeb-Edu streaming dataset.
    
    FineWeb-Edu is a high-quality educational web text dataset
    curated by HuggingFace for LLM pretraining.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")
    
    # Load in streaming mode
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name=config.fineweb_subset,
        split=split,
        streaming=True,
    )
    
    return dataset


def create_finepdf_stream(
    tokenizer,
    config: StreamingConfig,
    split: str = "train",
):
    """
    Create FinePDF streaming dataset.
    
    Uses the official HuggingFace FinePDFs dataset:
    - 1.19 trillion tokens in English (eng_Latn)
    - 207M documents from CommonCrawl PDFs (2013-2025)
    - Carefully deduplicated and filtered
    - Longer documents than typical web data (avg 2x length)
    - Excellent for long-context pretraining
    
    Returns None if dataset is not available.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")
    
    try:
        # Load official FinePDFs dataset (English subset)
        print(f"Loading FinePDFs ({config.finepdf_language})...")
        dataset = load_dataset(
            config.finepdf_dataset,
            name=config.finepdf_language,
            split=split,
            streaming=True,
        )
        print(f"✓ Using FinePDFs: {config.finepdf_dataset} ({config.finepdf_language})")
        print(f"  Available: 1.19T tokens, 207M documents")
        return dataset
    except Exception as e:
        print(f"⚠ FinePDFs not available: {e}")
        print(f"  Will use FineWeb-Edu only")
        return None


def create_usmle_stream(
    tokenizer,
    config: StreamingConfig,
    split: str = "train",
):
    """
    Create USMLE/MedQA streaming dataset.
    
    MedQA contains USMLE-style multiple choice questions
    for medical domain knowledge.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")
    
    # Load USMLE/MedQA dataset
    dataset = load_dataset(
        config.usmle_dataset,
        split=split,
        streaming=True,
    )
    
    def format_medqa_example(example):
        """Format MedQA example into text."""
        question = example.get("question", "")
        options = example.get("options", {})
        answer_idx = example.get("answer_idx", example.get("answer", ""))
        
        # Format options
        text = f"Question: {question}\n\nOptions:\n"
        
        if isinstance(options, dict):
            for key, value in sorted(options.items()):
                text += f"{key}) {value}\n"
        elif isinstance(options, list):
            for i, opt in enumerate(options):
                text += f"{chr(65+i)}) {opt}\n"
        
        # Add answer
        if isinstance(answer_idx, int) and answer_idx < 4:
            answer_letter = chr(65 + answer_idx)
            text += f"\nAnswer: {answer_letter}"
        elif isinstance(answer_idx, str):
            text += f"\nAnswer: {answer_idx}"
        
        return {"text": text, "is_medical": True}
    
    return dataset.map(format_medqa_example)


def create_usmle_rl_prompts(
    tokenizer,
    config: StreamingConfig,
    split: str = "train",
) -> Iterator[Dict[str, Any]]:
    """
    Create USMLE prompts for RL training.
    
    Returns prompts with ground truth for reward computation.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")
    
    # Load USMLE
    dataset = load_dataset(
        config.usmle_dataset,
        split=split,
        streaming=True,
    )
    
    for example in dataset:
        question = example.get("question", "")
        options = example.get("options", {})
        answer_idx = example.get("answer_idx", example.get("answer", ""))
        
        # Format prompt
        prompt_text = f"Question: {question}\n\nOptions:\n"
        if isinstance(options, dict):
            for key, value in sorted(options.items()):
                prompt_text += f"{key}) {value}\n"
        elif isinstance(options, list):
            for i, opt in enumerate(options):
                prompt_text += f"{chr(65+i)}) {opt}\n"
        
        prompt_text += "\nPlease select the correct answer (A, B, C, or D) and explain your reasoning.\n\nAnswer:"
        
        # Get ground truth answer
        if isinstance(answer_idx, int):
            ground_truth = chr(65 + answer_idx)
        else:
            ground_truth = str(answer_idx)
        
        # Tokenize prompt
        prompt_ids = tokenizer.encode(prompt_text)
        
        yield {
            "input_ids": torch.tensor(prompt_ids, dtype=torch.long),
            "prompt_text": prompt_text,
            "ground_truth": ground_truth,
            "question": question,
            "task": "usmle",
        }


class StreamingLMDataset(IterableDataset):
    """
    Streaming language modeling dataset.
    
    Features:
    - Interleaves multiple data sources
    - Efficient token packing
    - Memory-efficient streaming
    - DDP-compatible with proper worker splitting
    """
    
    def __init__(
        self,
        tokenizer,
        config: StreamingConfig,
        datasets: List[Any] = None,
        probabilities: List[float] = None,
    ):
        self.tokenizer = tokenizer
        self.config = config
        self.datasets = datasets
        self.probabilities = probabilities or [1.0 / len(datasets)] * len(datasets)
        self.base_seed = config.seed
    
    def _get_worker_info(self):
        """Get worker and distributed info for proper data splitting."""
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        
        # Get distributed rank if available
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                rank = dist.get_rank()
                world_size = dist.get_world_size()
            else:
                rank = 0
                world_size = 1
        except:
            rank = 0
            world_size = 1
        
        # Create unique seed for this worker+rank combo
        unique_seed = self.base_seed + rank * 1000 + worker_id
        return worker_id, num_workers, rank, world_size, unique_seed
    
    def _get_mixed_stream(self, rng: random.Random) -> Iterator[Dict]:
        """Create interleaved stream from multiple datasets."""
        # Manual interleaving is more reliable for streaming datasets
        iterators = [iter(d) for d in self.datasets]
        probs = list(self.probabilities)  # Copy to avoid modifying original
        
        while iterators:
            # Sample dataset based on probabilities
            idx = rng.choices(range(len(iterators)), weights=probs[:len(iterators)])[0]
            try:
                example = next(iterators[idx])
                yield example
            except StopIteration:
                # Remove exhausted iterator
                iterators.pop(idx)
                probs.pop(idx)
                if not iterators:
                    break
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over packed sequences with proper DDP handling."""
        # Get worker-specific info for proper sharding
        worker_id, num_workers, rank, world_size, unique_seed = self._get_worker_info()
        
        # Create worker-local state to avoid shared state issues
        rng = random.Random(unique_seed)
        buffer = []
        
        # Skip examples based on global worker index for proper sharding
        global_worker_id = rank * num_workers + worker_id
        total_workers = world_size * num_workers
        
        example_idx = 0
        for example in self._get_mixed_stream(rng):
            # Simple round-robin sharding across all workers
            if example_idx % total_workers != global_worker_id:
                example_idx += 1
                continue
            example_idx += 1
            
            text = example.get("text", "")
            if not text:
                continue

            # Tokenize (fast) without special tokens
            tokens = self.tokenizer.encode(text, add_special_tokens=False)

            # If document is longer than a single chunk, chunk it to avoid oversized sequences
            max_chunk = self.config.max_seq_length + 1
            if len(tokens) > max_chunk:
                # iterate through longer tokens in chunks
                for start in range(0, len(tokens), self.config.max_seq_length):
                    chunk = tokens[start:start + max_chunk]
                    if len(chunk) < 2:
                        continue
                    # If chunk is longer than allowed, trim it
                    if len(chunk) > max_chunk:
                        chunk = chunk[:max_chunk]
                    # create a sequence pair
                    input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                    labels = torch.tensor(chunk[1:], dtype=torch.long)
                    yield {"input_ids": input_ids, "labels": labels}
                # Continue to next example after yielding chunks
                continue

            # Otherwise extend buffer and stream normally
            buffer.extend(tokens)
            
            # Yield complete sequences
            while len(buffer) >= self.config.max_seq_length + 1:
                seq = buffer[:self.config.max_seq_length + 1]
                buffer = buffer[self.config.max_seq_length:]
                
                input_ids = torch.tensor(seq[:-1], dtype=torch.long)
                labels = torch.tensor(seq[1:], dtype=torch.long)
                
                yield {
                    "input_ids": input_ids,
                    "labels": labels,
                }


class RLPromptDataset(IterableDataset):
    """
    Dataset for RL training prompts.
    
    Supports:
    - USMLE/MedQA prompts
    - GSM8K math prompts
    - Custom task prompts
    """
    
    def __init__(
        self,
        tokenizer,
        config: StreamingConfig,
        task: str = "medqa",
    ):
        self.tokenizer = tokenizer
        self.config = config
        self.task = task
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over prompts."""
        if self.task == "medqa":
            yield from create_usmle_rl_prompts(
                self.tokenizer, self.config
            )
        elif self.task == "gsm8k":
            yield from self._create_gsm8k_prompts()
        else:
            raise ValueError(f"Unknown task: {self.task}")
    
    def _create_gsm8k_prompts(self) -> Iterator[Dict[str, Any]]:
        """Create GSM8K math prompts."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")
        
        dataset = load_dataset("gsm8k", "main", split="train", streaming=True)
        
        for example in dataset:
            question = example.get("question", "")
            answer = example.get("answer", "")
            
            prompt_text = f"Question: {question}\n\nSolve this step by step.\n\nAnswer:"
            prompt_ids = self.tokenizer.encode(prompt_text)
            
            yield {
                "input_ids": torch.tensor(prompt_ids, dtype=torch.long),
                "prompt_text": prompt_text,
                "ground_truth": answer,
                "question": question,
                "task": "gsm8k",
            }


def create_pretraining_dataloader(
    tokenizer,
    config: StreamingConfig,
    batch_size: int,
    num_workers: int = 2,
) -> DataLoader:
    """
    Create streaming dataloader for pretraining.
    
    Interleaves FineWeb-Edu (60%), FinePDF (30%), and USMLE QA (10%).
    Total available: 35B+ tokens
    
    If FinePDF is not available, adjusts to 90% FineWeb, 10% USMLE.
    """
    # Create streaming datasets
    fineweb = create_fineweb_stream(tokenizer, config)
    finepdf = create_finepdf_stream(tokenizer, config)
    usmle = create_usmle_stream(tokenizer, config)
    
    # Adjust dataset mix based on availability
    if finepdf is not None:
        datasets = [fineweb, finepdf, usmle]
        probabilities = [
            config.fineweb_probability,
            config.finepdf_probability,
            config.usmle_probability,
        ]
        print(f"Dataset mix: FineWeb {config.fineweb_probability*100:.0f}%, FinePDFs {config.finepdf_probability*100:.0f}%, USMLE {config.usmle_probability*100:.0f}%")
        print(f"  Total available: >35B tokens (sufficient for training)")
    else:
        datasets = [fineweb, usmle]
        probabilities = [0.9, 0.1]  # Fallback without PDF
        print(f"Dataset mix: FineWeb 90%, USMLE 10% (FinePDFs not available)")
    
    # Create combined dataset
    dataset = StreamingLMDataset(
        tokenizer=tokenizer,
        config=config,
        datasets=datasets,
        probabilities=probabilities,
    )
    
    def collate_fn(batch):
        # Batch is a list of dicts with tensors of shape (seq_len,) that may vary
        # Pad to the longest sequence length in the batch
        import torch
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        max_len = max([b["input_ids"].size(0) for b in batch])
        input_padded = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        labels_padded = torch.full((len(batch), max_len), -100, dtype=torch.long)
        for i, b in enumerate(batch):
            l = b["input_ids"].size(0)
            input_padded[i, :l] = b["input_ids"]
            labels_padded[i, :l] = b["labels"]
        return {"input_ids": input_padded, "labels": labels_padded}

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
        collate_fn=collate_fn,
    )


def create_rl_dataloader(
    tokenizer,
    config: StreamingConfig,
    task: str = "medqa",
    batch_size: int = 8,
) -> DataLoader:
    """
    Create dataloader for RL training prompts.
    """
    dataset = RLPromptDataset(
        tokenizer=tokenizer,
        config=config,
        task=task,
    )
    
    def collate_fn(batch):
        """Collate function for variable-length prompts."""
        return batch  # Return list, handle in trainer
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
    )


# =============================================================================
# Testing utilities
# =============================================================================

def test_fineweb_streaming():
    """Test FineWeb-Edu streaming."""
    print("Testing FineWeb-Edu streaming...")
    
    try:
        from transformers import GPT2TokenizerFast
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    except ImportError:
        print("transformers not installed, skipping test")
        return False
    
    config = StreamingConfig(fineweb_subset="sample-10BT")
    
    try:
        stream = create_fineweb_stream(tokenizer, config)
        
        # Test first 5 examples
        for i, example in enumerate(stream):
            if i >= 5:
                break
            text = example.get("text", "")
            print(f"  Example {i+1}: {len(text)} chars, {len(tokenizer.encode(text))} tokens")
        
        print("✓ FineWeb-Edu streaming works!")
        return True
        
    except Exception as e:
        print(f"✗ FineWeb-Edu streaming failed: {e}")
        return False


def test_medqa_streaming():
    """Test MedQA streaming."""
    print("Testing MedQA streaming...")
    
    try:
        from transformers import GPT2TokenizerFast
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    except ImportError:
        print("transformers not installed, skipping test")
        return False
    
    config = StreamingConfig()
    
    try:
        stream = create_medqa_stream(tokenizer, config)
        
        # Test first 5 examples
        for i, example in enumerate(stream):
            if i >= 5:
                break
            text = example.get("text", "")
            is_medical = example.get("is_medical", False)
            print(f"  Example {i+1}: {len(text)} chars, medical={is_medical}")
        
        print("✓ MedQA streaming works!")
        return True
        
    except Exception as e:
        print(f"✗ MedQA streaming failed: {e}")
        return False


def test_interleaved_streaming():
    """Test interleaved streaming."""
    print("Testing interleaved streaming...")
    
    try:
        from transformers import GPT2TokenizerFast
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    except ImportError:
        print("transformers not installed, skipping test")
        return False
    
    config = StreamingConfig(
        fineweb_probability=0.7,
        medqa_probability=0.3,
        max_seq_length=512,
    )
    
    try:
        dataloader = create_pretraining_dataloader(
            tokenizer, config, batch_size=2
        )
        
        # Test first batch
        for i, batch in enumerate(dataloader):
            if i >= 1:
                break
            print(f"  Batch input_ids shape: {batch['input_ids'].shape}")
            print(f"  Batch labels shape: {batch['labels'].shape}")
        
        print("✓ Interleaved streaming works!")
        return True
        
    except Exception as e:
        print(f"✗ Interleaved streaming failed: {e}")
        return False


def test_rl_prompts():
    """Test RL prompt generation."""
    print("Testing RL prompt generation...")
    
    try:
        from transformers import GPT2TokenizerFast
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    except ImportError:
        print("transformers not installed, skipping test")
        return False
    
    config = StreamingConfig()
    
    try:
        prompt_iter = create_usmle_rl_prompts(tokenizer, config)
        
        # Test first 3 prompts
        for i, prompt in enumerate(prompt_iter):
            if i >= 3:
                break
            print(f"  Prompt {i+1}:")
            print(f"    - Input IDs: {len(prompt['input_ids'])} tokens")
            print(f"    - Ground truth: {prompt['ground_truth']}")
            print(f"    - Task: {prompt['task']}")
        
        print("✓ RL prompt generation works!")
        return True
        
    except Exception as e:
        print(f"✗ RL prompt generation failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Streaming Dataset Tests")
    print("=" * 60)
    
    results = []
    results.append(("FineWeb-Edu", test_fineweb_streaming()))
    results.append(("MedQA", test_medqa_streaming()))
    results.append(("Interleaved", test_interleaved_streaming()))
    results.append(("RL Prompts", test_rl_prompts()))
    
    print("\n" + "=" * 60)
    print("Summary:")
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
