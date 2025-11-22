"""
Data loading utilities, including streaming from Hugging Face Hub.
"""

import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from typing import Optional, Iterator, List
import transformers

class StreamedTextDataset(IterableDataset):
    """
    IterableDataset for streaming text data from Hugging Face Hub.
    """
    def __init__(
        self, 
        dataset_name: str, 
        tokenizer: transformers.PreTrainedTokenizer, 
        seq_len: int,
        subset: Optional[str] = None,
        split: str = "train",
        buffer_size: int = 10000
    ):
        self.dataset_name = dataset_name
        self.subset = subset
        self.split = split
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.buffer_size = buffer_size
        
        # Load dataset in streaming mode
        self.dataset = load_dataset(
            dataset_name, 
            name=subset, 
            split=split, 
            streaming=True
        )

    def _tokenization_generator(self) -> Iterator[torch.Tensor]:
        buffer = []
        for sample in self.dataset:
            text = sample.get('text', '')
            if not text:
                continue
                
            tokens = self.tokenizer.encode(text)
            buffer.extend(tokens)
            
            # Yield chunks of seq_len + 1 (input + target)
            while len(buffer) >= self.seq_len + 1:
                yield torch.tensor(buffer[:self.seq_len + 1], dtype=torch.long)
                buffer = buffer[self.seq_len + 1:]

    def __iter__(self):
        return self._tokenization_generator()

def create_dataloader(
    dataset_name: str,
    tokenizer: transformers.PreTrainedTokenizer,
    seq_len: int,
    batch_size: int,
    subset: Optional[str] = None,
    split: str = "train",
    num_workers: int = 0
) -> DataLoader:
    """
    Create a DataLoader for streaming text data.
    """
    dataset = StreamedTextDataset(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        seq_len=seq_len,
        subset=subset,
        split=split
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
