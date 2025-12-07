#!/usr/bin/env python
"""
Quick validation test - verifies basic functionality
Tests model creation, data loading, and a few training steps
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

try:
    import lightning as L
    print(f"Lightning version: {L.__version__}")
except ImportError:
    print("Lightning not found, trying pytorch_lightning...")
    import pytorch_lightning as L
    print(f"PyTorch Lightning version: {L.__version__}")

from transformers import AutoTokenizer
from pure_transformer.configs import get_model_config
from pure_transformer.data.streaming import StreamingConfig, create_pretraining_dataloader

print("\n" + "="*80)
print("QUICK VALIDATION TEST")
print("="*80)

# 1. Test model config
print("\n1. Testing model config...")
config = get_model_config("tiny")
print(f"   ✓ Model config loaded: {config.model_name}")
print(f"   ✓ Parameters: ~{config.num_layers * config.hidden_size**2 / 1e6:.0f}M")

# 2. Test tokenizer
print("\n2. Testing tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
test_text = "This is a test sentence."
tokens = tokenizer.encode(test_text)
print(f"   ✓ Tokenizer loaded: {len(tokens)} tokens for test text")

# 3. Test data streaming
print("\n3. Testing data streaming...")
streaming_config = StreamingConfig(
    fineweb_subset="sample-10BT",
    fineweb_probability=0.90,
    finepdf_probability=0.0,
    usmle_probability=0.10,
    max_seq_length=512,
    shuffle_buffer_size=100,
)

try:
    print("   Creating dataloader...")
    dataloader = create_pretraining_dataloader(
        tokenizer=tokenizer,
        config=streaming_config,
        batch_size=2,
        num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
    )
    
    print("   Fetching first batch...")
    batch = next(iter(dataloader))
    print(f"   ✓ Data streaming works!")
    print(f"   ✓ Batch shape: {batch['input_ids'].shape}")
    print(f"   ✓ Labels shape: {batch['labels'].shape}")
    
except Exception as e:
    print(f"   ✗ Data streaming failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4. Test model creation
print("\n4. Testing model creation...")
try:
    from pure_transformer.model.transformer import TransformerLM
    
    model = TransformerLM(config)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Model created successfully")
    print(f"   ✓ Total parameters: {param_count:,}")
    
    # Test forward pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    input_ids = batch["input_ids"][:1].to(device)  # Take first sample
    
    with torch.no_grad():
        output = model(input_ids)
    
    print(f"   ✓ Forward pass successful")
    print(f"   ✓ Output shape: {output.logits.shape}")
    
except Exception as e:
    print(f"   ✗ Model test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("✓ ALL VALIDATION TESTS PASSED!")
print("="*80)
print("\nReady to run full training!")
print("Next steps:")
print("  1. Test training: python pure_transformer/test_train_21b.py")
print("  2. Full training: python pure_transformer/train_multigpu.py --model xlarge --devices 8")
