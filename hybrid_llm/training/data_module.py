"""
LightningDataModule that wires streaming datasets into PyTorch Lightning.
"""

from typing import Optional
import lightning as L
from transformers import AutoTokenizer

from hybrid_llm.data import create_streaming_dataloader
from hybrid_llm.configs import TrainingConfig


class StreamingDataModule(L.LightningDataModule):
    """
    Streams FineWeb-Edu + MedQA and tokenizes on the fly.
    """
    
    def __init__(self, train_cfg: TrainingConfig, num_workers: int = 0):
        super().__init__()
        self.train_cfg = train_cfg
        self.num_workers = num_workers
        self.tokenizer = AutoTokenizer.from_pretrained(train_cfg.tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
    
    def train_dataloader(self):
        return create_streaming_dataloader(
            tokenizer=self.tokenizer,
            train_cfg=self.train_cfg,
            num_workers=self.num_workers,
        )
    
    def val_dataloader(self):
        # Optional: hook in eval streams later. For now rely on train-only streaming.
        return None
