"""
Test suite for 0.76B parameter model.

Tests:
- Model creation and parameter count
- Forward pass and memory usage
- Training with mixed precision
- Domain-specific prompt handling
- Batch size scaling on A100
"""

import unittest
import torch
from pure_transformer.configs import get_model_config
from pure_transformer.model import TransformerLM


class TestLargeModel(unittest.TestCase):
    """Tests for 0.76B parameter model."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cls.config = get_model_config('large')
        cls.model = TransformerLM(cls.config).to(cls.device)
    
    def test_parameter_count(self):
        """Test that large model has ~0.76B parameters."""
        params = self.model.count_parameters()
        # Target: 763M ± 5M
        self.assertGreater(params, 758_000_000, "Model too small")
        self.assertLess(params, 768_000_000, "Model too large")
        print(f"✓ Large model: {params:,} parameters ({params/1e9:.2f}B)")
    
    def test_model_dimensions(self):
        """Test model architecture dimensions."""
        self.assertEqual(self.config.hidden_size, 1536)
        self.assertEqual(self.config.num_layers, 24)
        self.assertEqual(self.config.num_heads, 16)
        self.assertEqual(self.config.head_dim, 96)
        self.assertEqual(self.config.intermediate_size, 4224)
        print("✓ Model dimensions correct for 0.76B scale")
    
    def test_forward_pass(self):
        """Test forward pass produces correct output shapes."""
        batch_size, seq_len = 2, 128
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(self.device)
        labels = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(self.device)
        
        with torch.no_grad():
            loss, logits = self.model(input_ids, labels=labels)
        
        self.assertEqual(logits.shape, (batch_size, seq_len, self.config.vocab_size))
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0)
        print(f"✓ Forward pass works, loss={loss.item():.4f}")
    
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_memory_usage(self):
        """Test memory usage is reasonable for A100."""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        batch_size, seq_len = 8, 2048
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(self.device)
        labels = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(self.device)
        
        optimizer = self.model.setup_optimizers(learning_rate=1e-4)
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            loss, _ = self.model(input_ids, labels=labels)
        
        loss.backward()
        optimizer.step()
        
        peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9
        
        # Should fit in A100 80GB with headroom (batch 8 uses ~45GB)
        self.assertLess(peak_memory_gb, 60, f"Memory usage too high: {peak_memory_gb:.2f}GB")
        print(f"✓ Memory usage: {peak_memory_gb:.2f}GB (fits in A100 80GB)")
    
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_mixed_precision_training(self):
        """Test training with mixed precision."""
        batch_size, seq_len = 4, 512
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(self.device)
        labels = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(self.device)
        
        optimizer = self.model.setup_optimizers(learning_rate=1e-4)
        scaler = torch.cuda.amp.GradScaler()
        
        # Training step
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            loss, _ = self.model(input_ids, labels=labels)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        self.assertTrue(torch.isfinite(loss))
        print(f"✓ Mixed precision training works, loss={loss.item():.4f}")
    
    def test_generation(self):
        """Test text generation."""
        prompt_len = 10
        max_new_tokens = 20
        
        prompt = torch.randint(0, self.config.vocab_size, (1, prompt_len)).to(self.device)
        
        with torch.no_grad():
            generated = self.model.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.8,
                top_k=50,
            )
        
        self.assertEqual(generated.shape[0], 1)
        self.assertEqual(generated.shape[1], prompt_len + max_new_tokens)
        print(f"✓ Generation works: {prompt_len} -> {generated.shape[1]} tokens")
    
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_batch_size_scaling(self):
        """Test different batch sizes to find optimal setting."""
        seq_len = 2048
        max_batch_size = 0
        
        for batch_size in [2, 4, 6, 8, 10, 12, 14, 16]:
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(self.device)
                labels = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(self.device)
                
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    loss, _ = self.model(input_ids, labels=labels)
                
                loss.backward()
                
                peak_gb = torch.cuda.max_memory_allocated() / 1e9
                
                # Check if it fits in A100 80GB with headroom (60GB limit)
                if peak_gb < 60:
                    max_batch_size = batch_size
                else:
                    break
                    
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    break
                raise
        
        self.assertGreaterEqual(max_batch_size, 8, "Should support at least batch size 8")
        print(f"✓ Max batch size on A100: {max_batch_size} (seq_len={seq_len})")
    
    def test_gradient_checkpointing(self):
        """Test that gradient checkpointing is enabled."""
        self.assertTrue(
            self.config.use_gradient_checkpointing,
            "Gradient checkpointing should be enabled for 0.76B model"
        )
        print("✓ Gradient checkpointing enabled")
    
    def test_optimizer_setup(self):
        """Test optimizer configuration."""
        optimizer = self.model.setup_optimizers(learning_rate=3e-4, weight_decay=0.1)
        
        # Check optimizer type
        self.assertTrue(isinstance(optimizer, torch.optim.AdamW))
        
        # Check parameter groups
        param_groups = optimizer.param_groups
        self.assertGreater(len(param_groups), 0)
        
        print(f"✓ Optimizer setup: {len(param_groups)} param groups")


class TestDomainSpecific(unittest.TestCase):
    """Tests for domain-specific capabilities."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cls.config = get_model_config('large')
        cls.model = TransformerLM(cls.config).to(cls.device)
        
        try:
            from transformers import AutoTokenizer
            cls.tokenizer = AutoTokenizer.from_pretrained('gpt2')
            cls.tokenizer.pad_token = cls.tokenizer.eos_token
            cls.has_tokenizer = True
        except:
            cls.has_tokenizer = False
    
    @unittest.skipUnless(hasattr(setUpClass, 'has_tokenizer'), "Tokenizer not available")
    def test_medical_prompt_encoding(self):
        """Test encoding of medical prompts."""
        if not self.has_tokenizer:
            self.skipTest("Tokenizer not available")
        
        medical_prompts = [
            "What are the symptoms of diabetes?",
            "Explain the mechanism of action of aspirin.",
            "A patient presents with chest pain.",
        ]
        
        for prompt in medical_prompts:
            encoded = self.tokenizer.encode(prompt, return_tensors='pt')
            self.assertGreater(encoded.shape[1], 0)
        
        print("✓ Medical prompts encode correctly")
    
    def test_long_context_support(self):
        """Test model handles full 2048 context."""
        seq_len = 2048
        input_ids = torch.randint(0, self.config.vocab_size, (1, seq_len)).to(self.device)
        
        with torch.no_grad():
            loss, logits = self.model(input_ids, labels=input_ids)
        
        self.assertEqual(logits.shape[1], seq_len)
        self.assertTrue(torch.isfinite(loss))
        print("✓ Full 2048 context supported")


class TestTrainingEfficiency(unittest.TestCase):
    """Tests for training efficiency and throughput."""
    
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_training_throughput(self):
        """Test training throughput is reasonable."""
        import time
        
        device = torch.device('cuda')
        config = get_model_config('large')
        model = TransformerLM(config).to(device)
        optimizer = model.setup_optimizers(learning_rate=3e-4)
        
        batch_size = 8
        seq_len = 2048
        num_steps = 5
        
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(num_steps):
            optimizer.zero_grad()
            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)
            labels = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)
            
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                loss, _ = model(input_ids, labels=labels)
            
            loss.backward()
            optimizer.step()
        
        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        
        tokens_per_sec = (batch_size * seq_len * num_steps) / elapsed
        
        # Should achieve at least 3K tokens/sec with bf16 on A100
        self.assertGreater(tokens_per_sec, 3000, f"Throughput too low: {tokens_per_sec:.0f} tokens/sec")
        
        print(f"✓ Training throughput: {tokens_per_sec:,.0f} tokens/sec")
        print(f"  Estimated time for 20B tokens: {20e9/tokens_per_sec/86400:.1f} days")


if __name__ == '__main__':
    unittest.main()
