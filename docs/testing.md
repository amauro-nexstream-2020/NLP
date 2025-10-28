# Testing Plan

## 1. Unit Tests

### 1.1 Tokenizer Tests
**File**: `tests/test_tokenizer.py`

**Tests**:
- `test_bpe_training`: Verify tokenizer trains on sample corpus
- `test_encode_decode`: Verify encoding/decoding roundtrip
- `test_special_tokens`: Verify special token handling
- `test_save_load`: Verify tokenizer persistence
- `test_vocab_size`: Verify vocabulary size constraints
- `test_unknown_tokens`: Verify unknown token handling

### 1.2 Model Tests
**File**: `tests/test_model.py`

**Tests**:
- `test_model_initialization`: Verify model initializes correctly
- `test_forward_pass`: Verify forward pass with dummy input
- `test_output_shape`: Verify output shapes are correct
- `test_causal_masking`: Verify attention is causal
- `test_parameter_count`: Verify parameter count matches config
- `test_gradient_flow`: Verify gradients flow through all layers

### 1.3 Training Tests
**File**: `tests/test_training.py`

**Tests**:
- `test_training_step`: Verify single training step
- `test_loss_computation`: Verify loss calculation
- `test_gradient_clipping`: Verify gradient clipping
- `test_checkpoint_save_load`: Verify checkpoint saving/loading
- `test_optimizer_state`: Verify optimizer state persistence

### 1.4 Generation Tests
**File**: `tests/test_generation.py`

**Tests**:
- `test_greedy_generation`: Verify greedy decoding
- `test_top_k_sampling`: Verify top-k sampling
- `test_top_p_sampling`: Verify nucleus sampling
- `test_temperature_scaling`: Verify temperature effects
- `test_max_length`: Verify max length constraints

## 2. Integration Tests

### 2.1 End-to-End Training
**File**: `tests/test_e2e_training.py`

**Test**: Train small model on tiny dataset
**Steps**:
1. Initialize tiny model (10M params)
2. Create small training dataset (1000 samples)
3. Train for 100 steps
4. Verify loss decreases
5. Save checkpoint
6. Load checkpoint and continue training

**Acceptance Criteria**:
- Training completes without errors
- Loss decreases by at least 10%
- Checkpoint loads successfully

### 2.2 End-to-End Generation
**File**: `tests/test_e2e_generation.py`

**Test**: Generate text from trained model
**Steps**:
1. Load pre-trained model
2. Generate text from various prompts
3. Verify outputs are coherent
4. Verify no infinite loops

**Acceptance Criteria**:
- Generation completes in < 10 seconds
- Output length matches requested length
- No token repetition loops

### 2.3 Fine-Tuning Pipeline
**File**: `tests/test_fine_tuning.py`

**Test**: Fine-tune model on domain-specific data
**Steps**:
1. Load pre-trained model
2. Fine-tune on small domain dataset
3. Verify performance improves on domain task
4. Compare with baseline model

**Acceptance Criteria**:
- Fine-tuning completes successfully
- Domain-specific metrics improve
- General capabilities retained

## 3. Performance Tests

### 3.1 Speed Benchmarks
**File**: `tests/test_performance.py`

**Tests**:
- `test_training_throughput`: Measure tokens/second during training
- `test_inference_speed`: Measure tokens/second during generation
- `test_batch_processing`: Measure throughput vs batch size
- `test_memory_usage`: Measure peak memory usage

**Targets**:
- Training: > 1000 tokens/sec on GPU
- Inference: > 10 tokens/sec on CPU
- Memory: < 16GB for 1B param model

### 3.2 Accuracy Benchmarks
**File**: `tests/test_accuracy.py`

**Tests**:
- `test_perplexity`: Calculate perplexity on validation set
- `test_qa_accuracy`: Measure accuracy on Q&A tasks
- `test_generation_quality`: Human evaluation of generations

**Targets**:
- Perplexity: < 30 on validation set
- Q&A accuracy: > 60% on SQuAD
- Generation quality: > 3.5/5 human rating

## 4. Acceptance Tests

### Test 1: Sanity Check Training
**Objective**: Verify model can train on small dataset

**Procedure**:
1. Create tiny Q&A dataset (100 samples)
2. Initialize small model (50M params)
3. Train for 1 epoch
4. Verify loss decreases

**Success Criteria**:
- ✅ Training completes without errors
- ✅ Final loss < 0.8 × initial loss
- ✅ Checkpoint saves successfully

### Test 2: Generation Quality
**Objective**: Verify model generates coherent text

**Procedure**:
1. Load trained model
2. Generate responses to 10 validation prompts
3. Evaluate coherence and relevance

**Success Criteria**:
- ✅ All generations complete successfully
- ✅ No infinite loops or repetitions
- ✅ At least 7/10 generations are coherent

### Test 3: Baseline Performance
**Objective**: Compare with NanoGPT baseline

**Procedure**:
1. Train model on same dataset as NanoGPT
2. Evaluate on same validation set
3. Compare perplexity and metrics

**Success Criteria**:
- ✅ Perplexity within 10% of NanoGPT
- ✅ Training time comparable
- ✅ Model size within constraints

### Test 4: Domain Fine-Tuning
**Objective**: Verify domain adaptation works

**Procedure**:
1. Train general model
2. Fine-tune on biomedical dataset (1000 samples)
3. Evaluate on biomedical Q&A
4. Compare with non-fine-tuned model

**Success Criteria**:
- ✅ Fine-tuning improves domain accuracy by > 10%
- ✅ General capabilities retained (< 5% degradation)
- ✅ Domain-specific vocabulary learned

### Test 5: Documentation Completeness
**Objective**: Verify all documentation is complete

**Procedure**:
1. Review README
2. Execute all notebooks
3. Check API documentation
4. Verify architecture diagrams

**Success Criteria**:
- ✅ README has clear setup instructions
- ✅ All notebooks execute without errors
- ✅ All functions have docstrings
- ✅ Architecture diagrams are accurate

## 5. Test Execution

### 5.1 Running Tests
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_model.py

# Run with coverage
pytest --cov=src tests/
```

### 5.2 Continuous Integration
```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - run: pip install -r requirements.txt
      - run: pytest tests/
```

## 6. Test Data

### 6.1 Sample Datasets
- **Tiny corpus**: 1000 sentences for quick testing
- **Small Q&A**: 100 question-answer pairs
- **Validation set**: 500 held-out samples
- **Domain dataset**: 1000 biomedical abstracts

### 6.2 Expected Outputs
Store expected outputs for regression testing:
- `tests/fixtures/expected_tokens.json`
- `tests/fixtures/expected_embeddings.pt`
- `tests/fixtures/expected_generation.txt`

## 7. Success Metrics

### Overall Project Success
- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] All acceptance tests pass
- [ ] Performance targets met
- [ ] Documentation complete
- [ ] Code coverage > 80%

### Release Criteria
- [ ] All tests passing
- [ ] No critical bugs
- [ ] Documentation reviewed
- [ ] Performance benchmarks met
- [ ] Example notebooks working

---

**Version**: 1.0  
**Last Updated**: October 28, 2025
