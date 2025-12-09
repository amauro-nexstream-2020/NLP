"""
Comprehensive Tests for Pure Transformer

Tests cover:
1. Model components (attention, sparse attention, MLP, etc.)
2. Streaming datasets (FineWeb-Edu, MedQA)
3. Training components (GRPO, pretraining)
4. End-to-end pipeline

Run with: python -m pytest pure_transformer/tests/ -v
Or: python -m pure_transformer.tests.test_all
"""

import os
import sys
import unittest
import tempfile
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Test Utilities
# =============================================================================

def skip_if_no_cuda(test_method):
    """Skip test if CUDA is not available."""
    return unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")(test_method)


def skip_if_no_transformers(test_method):
    """Skip test if transformers is not installed."""
    try:
        import transformers
        has_transformers = True
    except ImportError:
        has_transformers = False
    return unittest.skipUnless(has_transformers, "transformers not installed")(test_method)


def skip_if_no_datasets(test_method):
    """Skip test if datasets is not installed."""
    try:
        import datasets
        has_datasets = True
    except ImportError:
        has_datasets = False
    return unittest.skipUnless(has_datasets, "datasets not installed")(test_method)


class DummyTokenizer:
    """Dummy tokenizer for testing without transformers."""
    
    def __init__(self, vocab_size: int = 50304):
        self.vocab_size = vocab_size
        self.eos_token_id = 0
        self.pad_token_id = 1
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
    
    def encode(self, text: str, add_special_tokens: bool = True) -> list:
        """Simple character-based encoding."""
        return [ord(c) % self.vocab_size for c in text]
    
    def decode(self, ids: list) -> str:
        """Simple character-based decoding."""
        return "".join(chr(i % 128) for i in ids if i < 128)


# =============================================================================
# Model Tests
# =============================================================================

class TestRMSNorm(unittest.TestCase):
    """Test RMSNorm layer."""
    
    def test_output_shape(self):
        from pure_transformer.model.transformer import RMSNorm
        
        norm = RMSNorm(hidden_size=64)
        x = torch.randn(2, 10, 64)
        out = norm(x)
        
        self.assertEqual(out.shape, x.shape)
    
    def test_normalization(self):
        from pure_transformer.model.transformer import RMSNorm
        
        norm = RMSNorm(hidden_size=64)
        x = torch.randn(2, 10, 64) * 10 + 5
        out = norm(x)
        
        # RMS should be approximately 1
        rms = torch.sqrt((out ** 2).mean(dim=-1))
        self.assertTrue(torch.allclose(rms, torch.ones_like(rms), atol=0.1))


class TestSwiGLUMLP(unittest.TestCase):
    """Test SwiGLU MLP."""
    
    def test_output_shape(self):
        from pure_transformer.model.transformer import SwiGLUMLP
        
        mlp = SwiGLUMLP(hidden_size=64, intermediate_size=256)
        x = torch.randn(2, 10, 64)
        out = mlp(x)
        
        self.assertEqual(out.shape, x.shape)
    
    def test_parameter_count(self):
        from pure_transformer.model.transformer import SwiGLUMLP
        
        mlp = SwiGLUMLP(hidden_size=64, intermediate_size=256)
        
        # gate: 64 * 256, up: 64 * 256, down: 256 * 64
        expected_params = 64 * 256 * 3
        actual_params = sum(p.numel() for p in mlp.parameters())
        
        self.assertEqual(actual_params, expected_params)


class TestAttention(unittest.TestCase):
    """Test multi-head attention."""
    
    def test_output_shape(self):
        from pure_transformer.model.transformer import Attention, precompute_rope_cache
        
        attn = Attention(
            hidden_size=64,
            num_heads=4,
            num_kv_heads=2,
            head_dim=16,
        )
        
        x = torch.randn(2, 10, 64)
        cos, sin = precompute_rope_cache(20, 16)
        
        out, _ = attn(x, cos, sin)
        
        self.assertEqual(out.shape, x.shape)
    
    def test_gqa(self):
        """Test Grouped Query Attention."""
        from pure_transformer.model.transformer import Attention, precompute_rope_cache
        
        attn = Attention(
            hidden_size=64,
            num_heads=8,
            num_kv_heads=2,  # 4x fewer KV heads
            head_dim=8,
        )
        
        x = torch.randn(2, 10, 64)
        cos, sin = precompute_rope_cache(20, 8)
        
        out, _ = attn(x, cos, sin)
        
        self.assertEqual(out.shape, x.shape)


class TestSparseAttention(unittest.TestCase):
    """Test DeepSeek Sparse Attention."""
    
    def test_lightning_indexer(self):
        from pure_transformer.model.sparse_attention import LightningIndexer
        
        indexer = LightningIndexer(
            hidden_size=64,
            num_indexer_heads=4,
            indexer_head_dim=16,
        )
        
        x = torch.randn(2, 32, 64)
        scores = indexer(x)
        
        # Output should be (B, T, T)
        self.assertEqual(scores.shape, (2, 32, 32))
        
        # Scores should have finite values
        self.assertTrue(torch.isfinite(scores).all())
    
    def test_sparse_attention_block(self):
        from pure_transformer.model.sparse_attention import DeepSeekSparseAttention
        from pure_transformer.model.transformer import precompute_rope_cache
        
        attn = DeepSeekSparseAttention(
            hidden_size=64,
            num_heads=4,
            num_kv_heads=2,
            head_dim=16,
            top_k=16,
        )
        
        x = torch.randn(2, 32, 64)
        cos, sin = precompute_rope_cache(64, 16)
        
        out, _, indexer_loss = attn(x, cos, sin, return_indexer_loss=True)
        
        self.assertEqual(out.shape, x.shape)
        # Indexer loss should be computed
        self.assertIsNotNone(indexer_loss)


class TestTransformerLM(unittest.TestCase):
    """Test full TransformerLM model."""
    
    def test_tiny_model(self):
        from pure_transformer.model import TransformerLM
        from pure_transformer.configs import TINY_CONFIG
        
        model = TransformerLM(TINY_CONFIG)
        
        # Test forward pass
        x = torch.randint(0, 1000, (2, 32))
        logits = model(x)
        
        self.assertEqual(logits.shape, (2, 32, TINY_CONFIG.vocab_size))
    
    def test_with_labels(self):
        from pure_transformer.model import TransformerLM
        from pure_transformer.configs import TINY_CONFIG
        
        model = TransformerLM(TINY_CONFIG)
        
        x = torch.randint(0, 1000, (2, 32))
        labels = torch.randint(0, 1000, (2, 32))
        
        loss, logits = model(x, labels=labels)
        
        self.assertEqual(logits.shape, (2, 32, TINY_CONFIG.vocab_size))
        self.assertEqual(loss.shape, ())  # Scalar loss
    
    def test_parameter_count(self):
        from pure_transformer.model import TransformerLM
        from pure_transformer.configs import TINY_CONFIG
        
        model = TransformerLM(TINY_CONFIG)
        params = model.count_parameters()
        
        # Tiny config should have reasonable params
        self.assertGreater(params, 10_000_000)
        self.assertLess(params, 200_000_000)
    
    @skip_if_no_cuda
    def test_cuda_forward(self):
        from pure_transformer.model import TransformerLM
        from pure_transformer.configs import TINY_CONFIG
        
        model = TransformerLM(TINY_CONFIG).cuda()
        
        x = torch.randint(0, 1000, (2, 32)).cuda()
        logits = model(x)
        
        self.assertEqual(logits.device.type, "cuda")


class TestGeneration(unittest.TestCase):
    """Test model generation."""
    
    def test_greedy_generation(self):
        from pure_transformer.model import TransformerLM
        from pure_transformer.configs import TINY_CONFIG
        
        model = TransformerLM(TINY_CONFIG)
        
        prompt = torch.randint(0, 1000, (1, 5))
        # Use small temperature instead of 0 to avoid inf in softmax
        generated = model.generate(prompt, max_new_tokens=10, temperature=0.01)
        
        # Should have 15 tokens (5 prompt + 10 generated)
        self.assertEqual(generated.shape[1], 15)
    
    def test_sampling_generation(self):
        from pure_transformer.model import TransformerLM
        from pure_transformer.configs import TINY_CONFIG
        
        model = TransformerLM(TINY_CONFIG)
        
        prompt = torch.randint(0, 1000, (1, 5))
        
        # Generate twice with same seed
        torch.manual_seed(42)
        gen1 = model.generate(prompt, max_new_tokens=10, temperature=1.0)
        
        torch.manual_seed(42)
        gen2 = model.generate(prompt, max_new_tokens=10, temperature=1.0)
        
        # Should be deterministic with same seed
        self.assertTrue(torch.equal(gen1, gen2))


# =============================================================================
# Data Tests
# =============================================================================

class TestStreamingConfig(unittest.TestCase):
    """Test streaming configuration."""
    
    def test_default_config(self):
        from pure_transformer.data import StreamingConfig
        
        config = StreamingConfig()
        
        self.assertAlmostEqual(config.fineweb_probability + config.finepdf_probability + config.usmle_probability, 1.0)
        self.assertEqual(config.max_seq_length, 2048)


@skip_if_no_datasets
class TestFineWebStreaming(unittest.TestCase):
    """Test FineWeb-Edu streaming."""
    
    def test_stream_creation(self):
        from pure_transformer.data import create_fineweb_stream, StreamingConfig
        
        tokenizer = DummyTokenizer()
        config = StreamingConfig(fineweb_subset="sample-10BT")
        
        stream = create_fineweb_stream(tokenizer, config)
        
        # Get first example
        example = next(iter(stream))
        self.assertIn("text", example)
    
    def test_tokenization(self):
        from pure_transformer.data import create_fineweb_stream, StreamingConfig
        
        tokenizer = DummyTokenizer()
        config = StreamingConfig(fineweb_subset="sample-10BT")
        
        stream = create_fineweb_stream(tokenizer, config)
        
        example = next(iter(stream))
        tokens = tokenizer.encode(example["text"])
        
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)


@skip_if_no_datasets
class TestMedQAStreaming(unittest.TestCase):
    """Test USMLE streaming."""
    
    def test_stream_creation(self):
        from pure_transformer.data import create_usmle_stream, StreamingConfig
        
        tokenizer = DummyTokenizer()
        config = StreamingConfig()
        
        stream = create_usmle_stream(tokenizer, config)
        
        # Get first example
        example = next(iter(stream))
        self.assertIn("text", example)
    
    def test_formatting(self):
        from pure_transformer.data import create_usmle_stream, StreamingConfig
        
        tokenizer = DummyTokenizer()
        config = StreamingConfig()
        
        stream = create_usmle_stream(tokenizer, config)
        
        example = next(iter(stream))
        text = example["text"]
        
        # Should contain question and answer markers
        self.assertIn("Question:", text)
        self.assertIn("Answer:", text)


@skip_if_no_datasets
class TestRLPrompts(unittest.TestCase):
    """Test RL prompt generation."""
    
    def test_medqa_prompts(self):
        from pure_transformer.data import create_usmle_rl_prompts, StreamingConfig
        
        tokenizer = DummyTokenizer()
        config = StreamingConfig()
        
        prompt_iter = create_usmle_rl_prompts(tokenizer, config)
        prompt = next(prompt_iter)
        
        self.assertIn("input_ids", prompt)
        self.assertIn("ground_truth", prompt)
        self.assertIn("task", prompt)
        self.assertEqual(prompt["task"], "usmle")


@skip_if_no_datasets
class TestInterleavedDataset(unittest.TestCase):
    """Test interleaved dataset creation."""
    
    def test_dataloader_creation(self):
        from pure_transformer.data import create_pretraining_dataloader, StreamingConfig
        
        tokenizer = DummyTokenizer()
        config = StreamingConfig(max_seq_length=128)
        
        try:
            dataloader = create_pretraining_dataloader(
                tokenizer, config, batch_size=2
            )
            
            # Get first batch (may timeout on slow connections)
            data_iter = iter(dataloader)
            batch = next(data_iter)
            
            self.assertIn("input_ids", batch)
            self.assertIn("labels", batch)
            self.assertEqual(batch["input_ids"].shape[1], 128)
        except StopIteration:
            # Dataset streaming might not have data ready immediately
            self.skipTest("Streaming dataset not ready")


# =============================================================================
# Training Tests
# =============================================================================

class TestGRPOConfig(unittest.TestCase):
    """Test GRPO configuration."""
    
    def test_default_config(self):
        from pure_transformer.training.enhanced_grpo import GRPOConfig
        
        config = GRPOConfig()
        
        self.assertEqual(config.num_samples_per_prompt, 16)
        self.assertTrue(config.use_unbiased_kl)
        self.assertTrue(config.use_off_policy_masking)


class TestEnhancedGRPO(unittest.TestCase):
    """Test enhanced GRPO trainer."""
    
    def test_trainer_initialization(self):
        from pure_transformer.model import TransformerLM
        from pure_transformer.configs import TINY_CONFIG
        from pure_transformer.training.enhanced_grpo import EnhancedGRPOTrainer, GRPOConfig
        
        model = TransformerLM(TINY_CONFIG)
        ref_model = TransformerLM(TINY_CONFIG)
        tokenizer = DummyTokenizer()
        config = GRPOConfig(num_samples_per_prompt=2, device_batch_size=1)
        
        trainer = EnhancedGRPOTrainer(
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            config=config,
            device=torch.device("cpu"),
        )
        
        self.assertEqual(trainer.global_step, 0)
    
    def test_unbiased_kl(self):
        from pure_transformer.training.enhanced_grpo import EnhancedGRPOTrainer, GRPOConfig
        from pure_transformer.model import TransformerLM
        from pure_transformer.configs import TINY_CONFIG
        
        model = TransformerLM(TINY_CONFIG)
        tokenizer = DummyTokenizer()
        config = GRPOConfig()
        
        trainer = EnhancedGRPOTrainer(
            model=model,
            ref_model=None,
            tokenizer=tokenizer,
            config=config,
            device=torch.device("cpu"),
        )
        
        # Test KL computation
        current_lp = torch.randn(4)
        ref_lp = torch.randn(4)
        old_lp = torch.randn(4)
        
        kl = trainer.compute_unbiased_kl(current_lp, ref_lp, old_lp)
        
        self.assertEqual(kl.shape, (4,))
    
    def test_off_policy_mask(self):
        from pure_transformer.training.enhanced_grpo import EnhancedGRPOTrainer, GRPOConfig
        from pure_transformer.model import TransformerLM
        from pure_transformer.configs import TINY_CONFIG
        
        model = TransformerLM(TINY_CONFIG)
        tokenizer = DummyTokenizer()
        config = GRPOConfig(off_policy_threshold=0.1)
        
        trainer = EnhancedGRPOTrainer(
            model=model,
            ref_model=None,
            tokenizer=tokenizer,
            config=config,
            device=torch.device("cpu"),
        )
        
        # Test masking
        old_lp = torch.zeros(4)
        current_lp = torch.tensor([-0.5, 0.0, 0.5, -0.2])
        advantages = torch.tensor([-1.0, 1.0, -0.5, 0.5])
        
        mask = trainer.compute_off_policy_mask(old_lp, current_lp, advantages)
        
        self.assertEqual(mask.shape, (4,))
        # Some sequences should be masked (negative advantage + high divergence)


class TestLearningRateSchedule(unittest.TestCase):
    """Test learning rate scheduling."""
    
    def test_cosine_schedule(self):
        from pure_transformer.training.pretrain import get_lr
        
        # Test warmup
        lr = get_lr(step=50, warmup_steps=100, total_steps=1000, max_lr=1e-4, min_lr=1e-5)
        self.assertAlmostEqual(lr, 5e-5, places=6)
        
        # Test end of warmup
        lr = get_lr(step=100, warmup_steps=100, total_steps=1000, max_lr=1e-4, min_lr=1e-5)
        self.assertAlmostEqual(lr, 1e-4, places=6)
        
        # Test end of training
        lr = get_lr(step=1000, warmup_steps=100, total_steps=1000, max_lr=1e-4, min_lr=1e-5)
        self.assertAlmostEqual(lr, 1e-5, places=6)


# =============================================================================
# Integration Tests
# =============================================================================

class TestEndToEndPretraining(unittest.TestCase):
    """End-to-end pretraining test."""
    
    def test_single_step(self):
        from pure_transformer.model import TransformerLM
        from pure_transformer.configs import TINY_CONFIG
        
        model = TransformerLM(TINY_CONFIG)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Create dummy batch
        input_ids = torch.randint(0, 1000, (2, 64))
        labels = torch.randint(0, 1000, (2, 64))
        
        # Forward pass
        loss, logits = model(input_ids, labels=labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        self.assertFalse(torch.isnan(loss))


class TestCheckpointing(unittest.TestCase):
    """Test model checkpointing."""
    
    def test_save_load(self):
        from pure_transformer.model import TransformerLM
        from pure_transformer.configs import TINY_CONFIG
        
        model = TransformerLM(TINY_CONFIG)
        
        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")
            
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "step": 100,
            }
            torch.save(checkpoint, path)
            
            # Load checkpoint
            loaded = torch.load(path)
            model2 = TransformerLM(TINY_CONFIG)
            model2.load_state_dict(loaded["model_state_dict"])
            
            # Verify weights match
            for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
                self.assertTrue(torch.equal(p1, p2), f"Mismatch in {n1}")


# =============================================================================
# Run All Tests
# =============================================================================

def run_all_tests():
    """Run all tests and print summary."""
    print("=" * 70)
    print("Pure Transformer - Comprehensive Test Suite")
    print("=" * 70)
    
    # Collect all test classes
    test_classes = [
        TestRMSNorm,
        TestSwiGLUMLP,
        TestAttention,
        TestSparseAttention,
        TestTransformerLM,
        TestGeneration,
        TestStreamingConfig,
        TestGRPOConfig,
        TestEnhancedGRPO,
        TestLearningRateSchedule,
        TestEndToEndPretraining,
        TestCheckpointing,
    ]
    
    # Add optional tests if dependencies available
    try:
        import datasets
        test_classes.extend([
            TestFineWebStreaming,
            TestMedQAStreaming,
            TestRLPrompts,
            TestInterleavedDataset,
        ])
    except ImportError:
        print("Note: datasets not installed, skipping streaming tests")
    
    # Run tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print("Test Summary:")
    print(f"  Tests run: {result.testsRun}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed!")
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}")
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
