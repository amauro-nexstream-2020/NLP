"""
Tokenizer utilities for the LLM project.
Wrapper around Hugging Face tokenizers with convenience functions.
"""

import os
from typing import List, Union, Optional
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


class BPETokenizerWrapper:
    """
    Wrapper for Byte Pair Encoding tokenizer using Hugging Face tokenizers.
    """
    
    def __init__(
        self,
        vocab_size: int = 50257,
        special_tokens: Optional[List[str]] = None
    ):
        """
        Initialize tokenizer.
        
        Args:
            vocab_size: Target vocabulary size
            special_tokens: List of special tokens
        """
        self.vocab_size = vocab_size
        
        if special_tokens is None:
            self.special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        else:
            self.special_tokens = special_tokens
        
        # Initialize tokenizer
        self.tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        self.tokenizer.decoder = decoders.BPEDecoder()
        
        self.is_trained = False
    
    def train(self, texts: Union[List[str], str], verbose: bool = True):
        """
        Train the tokenizer on texts.
        
        Args:
            texts: Training texts (list of strings or path to file)
            verbose: Show training progress
        """
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens,
            show_progress=verbose
        )
        
        if isinstance(texts, str):
            # Path to file
            self.tokenizer.train([texts], trainer=trainer)
        else:
            # List of texts
            self.tokenizer.train_from_iterator(texts, trainer=trainer)
        
        self.is_trained = True
        
        if verbose:
            print(f"✅ Tokenizer trained! Vocabulary size: {self.tokenizer.get_vocab_size()}")
    
    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_special_tokens: Whether to add BOS/EOS tokens
        
        Returns:
            List of token IDs
        """
        if not self.is_trained:
            raise ValueError("Tokenizer must be trained before encoding!")
        
        encoding = self.tokenizer.encode(text)
        ids = encoding.ids
        
        if add_special_tokens:
            bos_id = self.token_to_id("<BOS>")
            eos_id = self.token_to_id("<EOS>")
            ids = [bos_id] + ids + [eos_id]
        
        return ids
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.
        
        Args:
            ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens in output
        
        Returns:
            Decoded text
        """
        if not self.is_trained:
            raise ValueError("Tokenizer must be trained before decoding!")
        
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
    
    def token_to_id(self, token: str) -> Optional[int]:
        """Get ID for a token."""
        return self.tokenizer.token_to_id(token)
    
    def id_to_token(self, id: int) -> Optional[str]:
        """Get token for an ID."""
        return self.tokenizer.id_to_token(id)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.tokenizer.get_vocab_size()
    
    def save(self, path: str):
        """
        Save tokenizer to file.
        
        Args:
            path: Path to save tokenizer
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.tokenizer.save(path)
    
    def load(self, path: str):
        """
        Load tokenizer from file.
        
        Args:
            path: Path to tokenizer file
        """
        self.tokenizer = Tokenizer.from_file(path)
        self.is_trained = True
    
    @property
    def pad_token_id(self) -> int:
        """Get PAD token ID."""
        return self.token_to_id("<PAD>")
    
    @property
    def unk_token_id(self) -> int:
        """Get UNK token ID."""
        return self.token_to_id("<UNK>")
    
    @property
    def bos_token_id(self) -> int:
        """Get BOS token ID."""
        return self.token_to_id("<BOS>")
    
    @property
    def eos_token_id(self) -> int:
        """Get EOS token ID."""
        return self.token_to_id("<EOS>")


def load_tokenizer(path: str) -> BPETokenizerWrapper:
    """
    Load a trained tokenizer from file.
    
    Args:
        path: Path to tokenizer file
    
    Returns:
        Loaded tokenizer
    """
    tokenizer = BPETokenizerWrapper()
    tokenizer.load(path)
    return tokenizer
