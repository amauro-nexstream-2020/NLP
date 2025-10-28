"""
NLP - Decoder-Only Transformer LLM

A modular, educational implementation of a decoder-only transformer-based
Large Language Model for both general-purpose and domain-specific tasks.
"""

__version__ = "0.1.0"

from .model import DecoderLM, MultiHeadAttention, TransformerBlock
from .tokenizer import BPETokenizerWrapper, load_tokenizer
from .utils import (
    set_seed,
    count_parameters,
    get_device,
    print_model_summary,
    save_checkpoint,
    load_checkpoint
)

__all__ = [
    # Model
    "DecoderLM",
    "MultiHeadAttention",
    "TransformerBlock",
    # Tokenizer
    "BPETokenizerWrapper",
    "load_tokenizer",
    # Utils
    "set_seed",
    "count_parameters",
    "get_device",
    "print_model_summary",
    "save_checkpoint",
    "load_checkpoint",
]
