"""
Tokenizer utilities for the LLM project.
Wrapper around Hugging Face tokenizers with convenience functions.
"""

import os
from typing import List, Union, Optional
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
class BPETokenizerWrapper:
    """
    Trains a subword tokenizer on text.
    Provides a convenient interface for encoding text into token IDs and decoding token IDs back into text.
    Handles special tokens like <PAD>, <BOS>, <EOS>, <UNK>.
    Makes saving and loading the tokenizer easier.
    """
#vocab_size - max vocabulary size for BPE training
#min_frequency - minimum number of occurences for a subword to be included
#special_tokens - list of custom tokens
    def __init__(self, vocab_size: int = 50257, min_frequency: int = 2,
                 special_tokens: Optional[List[str]] = None):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency

        if special_tokens is None:
            self.special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        else:
            self.special_tokens = special_tokens
        #tokenizer is not initialized here!
        self.tokenizer: Optional[ByteLevelBPETokenizer] = None
        self.is_trained = False

    def train(self, texts: Union[List[str], str], verbose: bool = True):
        #creates new byte-level BPE tokenizer
        self.tokenizer = ByteLevelBPETokenizer()

        if isinstance(texts, str):
          #directly trains using that file
            self.tokenizer.train(
                files=[texts],
                vocab_size=self.vocab_size,
                min_frequency=self.min_frequency,
                special_tokens=self.special_tokens,
                show_progress=verbose
            )
        else:
            #temporary file created to contain all text samples
            #helps simulate file training
            from tempfile import NamedTemporaryFile
            with NamedTemporaryFile("w+", encoding="utf-8", delete=False) as tmp:
                tmp.write("\n".join(texts))
                tmp.flush()
                self.tokenizer.train(
                    files=[tmp.name],
                    vocab_size=self.vocab_size,
                    min_frequency=self.min_frequency,
                    special_tokens=self.special_tokens,
                    show_progress=verbose
                )
        #tokenizer is trained!!
        self.is_trained = True
        if verbose:
            print(f"Tokenizer trained! Vocab size: {self.tokenizer.get_vocab_size()}")
    #converts text to token IDs
    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        if not self.is_trained:
            raise ValueError("Tokenizer must be trained before encoding!")
        #encodes using default tokenizer behavior
        ids = self.tokenizer.encode(text).ids
        if add_special_tokens:
          #gets special token IDs
            bos_id = self.token_to_id("<BOS>")
            eos_id = self.token_to_id("<EOS>")
            ids = [bos_id] + ids + [eos_id]
        return ids
    #tokens back to text
    #sometimes removes special tokens during decoding
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        if not self.is_trained:
            raise ValueError("Tokenizer must be trained before decoding!")
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
    #gets the integer ID for a specific token
    def token_to_id(self, token: str) -> Optional[int]:
        return self.tokenizer.token_to_id(token) if self.tokenizer else None
    #gets the token string for a specific ID
    def id_to_token(self, id: int) -> Optional[str]:
        return self.tokenizer.id_to_token(id) if self.tokenizer else None
    #returns vocab size of the trained tokenizer
    def get_vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size() if self.tokenizer else 0
    #vocab file, merges file, saves tokenizer configuration
    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.tokenizer.save_model(path)
        json_path = os.path.join(path, "tokenizer.json")
        self.tokenizer.save(json_path)
        print(f"Tokenizer saved to directory: {path}")
    #load from directory or from json file
    def load(self, path: str):
        if os.path.isdir(path):
            vocab_file = os.path.join(path, "vocab.json")
            merges_file = os.path.join(path, "merges.txt")
            self.tokenizer = ByteLevelBPETokenizer(vocab_file, merges_file)
        elif os.path.isfile(path):
            from tokenizers import Tokenizer
            from tokenizers.implementations import BaseTokenizer
            tokenizer = Tokenizer.from_file(path)
            self.tokenizer = BaseTokenizer(tokenizer)
        else:
            raise FileNotFoundError(f"Tokenizer not found at {path}")
        self.is_trained = True
        print(f"Tokenizer loaded from {path}")
    #provides a convenient attribute style access!
    @property
    def pad_token_id(self) -> int:
        return self.token_to_id("<PAD>")

    @property
    def unk_token_id(self) -> int:
        return self.token_to_id("<UNK>")

    @property
    def bos_token_id(self) -> int:
        return self.token_to_id("<BOS>")

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id("<EOS>")

def load_tokenizer(path: str) -> BPETokenizerWrapper:
    tok = BPETokenizerWrapper()
    tok.load(path)
    return tok
