"""
Streaming dataloaders for the Hybrid LLM.

Key features:
- Streams FineWeb-Edu and MedQA directly from Hugging Face Hub (no local disk blow-up)
- Interleaves datasets with configurable probabilities
- Shuffle buffer to break locality in streaming mode
- Packs tokens into fixed-length causal LM examples (seq_len + 1)
"""

from typing import Iterable, Iterator, List, Optional, Dict
import random
import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset, interleave_datasets
from transformers import PreTrainedTokenizerBase
import numpy as np

from hybrid_llm.configs.training_config import DataConfig, TrainingConfig


ANSWER_LETTERS = ["A", "B", "C", "D", "E"]


def _safe_get(options, idx, default: str = "") -> str:
    """Safely fetch an option string."""
    if options is None:
        return default
    if isinstance(options, dict):
        options = list(options.values())
    try:
        return options[idx] if options[idx] else default
    except Exception:
        return default


def format_medqa_example(example: Dict) -> Dict[str, str]:
    """
    Convert MedQA fields into a single text block suitable for causal LM training.
    """
    question = example.get("question", "").strip()
    options = example.get("options") or [
        example.get("option_0"),
        example.get("option_1"),
        example.get("option_2"),
        example.get("option_3"),
    ]
    options = [opt for opt in options if opt]
    answer_idx = example.get("answer_idx")
    answer_label = example.get("answer")
    if answer_label is None and answer_idx is not None and answer_idx < len(ANSWER_LETTERS):
        answer_label = ANSWER_LETTERS[answer_idx]
    explanation = example.get("explanation", "") or example.get("rationale", "")
    
    option_lines = "\n".join(f"{ANSWER_LETTERS[i]}) {opt}" for i, opt in enumerate(options))
    answer_line = f"Answer: {answer_label}" if answer_label else "Answer:"
    if explanation:
        answer_line = f"{answer_line} because {explanation}"
    
    text = f"Question: {question}\nOptions:\n{option_lines}\n{answer_line}"
    return {"text": text}


def load_fineweb_stream(
    subset: str,
    min_quality: Optional[float],
) -> Iterable:
    """Load FineWeb-Edu in streaming mode with optional quality filter."""
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name=subset,
        split="train",
        streaming=True,
    )
    if min_quality is not None:
        ds = ds.filter(lambda ex: ex.get("language_score", 1.0) >= min_quality)
    ds = ds.map(lambda ex: {"text": ex.get("text", "")})
    return ds


def load_medqa_stream(split: str) -> Iterable:
    """Load MedQA-USMLE in streaming mode and format to text."""
    ds = load_dataset(
        "GBaker/MedQA-USMLE-4-options",
        split=split,
        streaming=True,
    )
    ds = ds.map(format_medqa_example)
    return ds


def build_streaming_text_mixture(cfg: DataConfig) -> Iterable:
    """
    Interleave FineWeb-Edu and MedQA streams according to configured probabilities.
    """
    fineweb_ds = load_fineweb_stream(cfg.fineweb_subset, cfg.fineweb_min_quality)
    medqa_ds = load_medqa_stream(cfg.medqa_split)
    
    mixed = interleave_datasets(
        [fineweb_ds, medqa_ds],
        probabilities=[cfg.fineweb_probability, cfg.medqa_probability],
        seed=cfg.seed,
    )
    mixed = mixed.shuffle(buffer_size=cfg.shuffle_buffer_size, seed=cfg.seed)
    return mixed


class StreamingCausalDataset(IterableDataset):
    """
    Packs a streaming text dataset into fixed-length token chunks for causal LM.
    """
    
    def __init__(
        self,
        dataset: Iterable,
        tokenizer: PreTrainedTokenizerBase,
        seq_len: int,
    ):
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.eos_id = tokenizer.eos_token_id
        self.bos_id = tokenizer.bos_token_id
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        buffer: List[int] = []
        for sample in self.dataset:
            text = sample.get("text", "")
            if not text:
                continue
            
            tokens = self.tokenizer.encode(
                text,
                add_special_tokens=False,
            )
            if self.bos_id is not None:
                buffer.append(self.bos_id)
            buffer.extend(tokens)
            if self.eos_id is not None:
                buffer.append(self.eos_id)
            
            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[: self.seq_len + 1]
                buffer = buffer[self.seq_len + 1 :]
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)
                yield {"input_ids": input_ids, "labels": labels}


def create_streaming_dataloader(
    tokenizer: PreTrainedTokenizerBase,
    train_cfg: TrainingConfig,
    num_workers: int = 0,
) -> DataLoader:
    """
    Build the PyTorch DataLoader that streams, shuffles, and tokenizes on the fly.
    """
    dataset = build_streaming_text_mixture(train_cfg.data)
    tokenized = StreamingCausalDataset(dataset, tokenizer, seq_len=train_cfg.max_seq_length)
    
    # Iterable datasets are not safe with multiple workers that clone the iterator,
    # so default to single-worker unless the user explicitly overrides.
    loader = DataLoader(
        tokenized,
        batch_size=train_cfg.micro_batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_seed_worker,
    )
    return loader


def _seed_worker(worker_id: int):
    """Ensure each worker has a different seed for shuffling buffers."""
    seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(seed)
    random.seed(seed)


__all__ = [
    "build_streaming_text_mixture",
    "StreamingCausalDataset",
    "create_streaming_dataloader",
]
