# Training Pipeline Documentation

## Overview

Complete 2-day training pipeline for 0.4B Pure Transformer model optimized for USMLE Question Answering.

**Total Duration:** 48 hours (2 days) on single A100 80GB  
**Phase 1:** Pretraining (1.5 days, 2B tokens)  
**Phase 2:** GRPO Fine-tuning (0.5 days, 0.6B tokens)  
**Final Model:** USMLE QA specialist with 35-45% accuracy

---

## Table of Contents

1. [Training Overview](#training-overview)
2. [Phase 1: Pretraining](#phase-1-pretraining)
3. [Phase 2: GRPO Fine-tuning](#phase-2-grpo-fine-tuning)
4. [Hardware Setup](#hardware-setup)
5. [Running the Pipeline](#running-the-pipeline)
6. [Monitoring](#monitoring)
7. [Troubleshooting](#troubleshooting)

---

## Training Overview

### Timeline

```
Day 1 (0-24h):  Pretraining Phase
├── Hour 0-1:   Warmup (100M tokens)
├── Hour 1-24:  Main training (1.4B tokens)
└── Hour 24:    Checkpoint & validation

Day 2 (24-38h): Pretraining Phase (continued)
├── Hour 24-29: Final pretraining (600M tokens)
├── Hour 29:    Save pretrain checkpoint
│
└── Hour 29-38: GRPO Fine-tuning Phase
    ├── Hour 29-30: Warmup on USMLE
    ├── Hour 30-37: GRPO training (600M tokens)
    └── Hour 37-38: Final evaluation

Day 2 (38-48h): Buffer (10h)
└── Evaluation, testing, checkpointing
```

### Token Budget

| Phase | Tokens | Duration | Purpose |
|-------|--------|----------|---------|
| Pretrain Warmup | 100M | 1h | LR warmup, stability |
| Pretrain Main | 1,900M | 28h | General language + domain |
| GRPO Training | 600M | 9h | USMLE QA optimization |
| **Total** | **2,600M** | **38h** | **With 10h buffer** |

### Data Mix

**Pretraining (2B tokens):**
- FineWeb-Edu: 85% (1.7B tokens) - General knowledge
- MedQA-USMLE: 15% (300M tokens) - Medical domain exposure

**GRPO (600M tokens):**
- MedQA-USMLE: 100% - Pure question answering

---

## Phase 1: Pretraining

### Objective

Build a strong foundation model with:
- General language understanding (FineWeb-Edu)
- Medical domain knowledge (MedQA)
- Coherent text generation capability

### Configuration

```yaml
Model: medium-large (0.4B parameters)
  - Layers: 20
  - Hidden: 1152
  - Context: 2048 tokens

Training Config: a100_2day
  - Total Tokens: 2,000,000,000 (2B)
  - Global Batch: 524,288 tokens (512K)
  - Micro Batch: 16 sequences
  - Gradient Accumulation: 16 steps
  - Sequence Length: 2048 tokens

Optimizer: AdamW (fused)
  - Learning Rate: 3e-4 (peak)
  - Min LR: 3e-5
  - Betas: (0.9, 0.95)
  - Weight Decay: 0.1
  - Gradient Clip: 1.0

LR Schedule: Cosine with warmup
  - Warmup: 100M tokens (1 hour)
  - Cosine Decay: 1,900M tokens (28 hours)
  - Final LR: 3e-5

Mixed Precision: bfloat16
  - Forward/Backward: bf16
  - Optimizer States: fp32
  - No loss scaling needed
```

### Streaming Data Pipeline

```python
# Dataset interleaving
fineweb = load_dataset("HuggingFaceFW/fineweb-edu", 
                       name="sample-10BT", 
                       streaming=True)
medqa = load_dataset("medmcqa", streaming=True)

# Interleave with probabilities
dataset = interleave_datasets(
    [fineweb, medqa],
    probabilities=[0.85, 0.15],
    stopping_strategy="all_exhausted"
)

# Token packing for efficiency
dataloader = PackedDataLoader(
    dataset,
    tokenizer=tokenizer,
    max_length=2048,
    batch_size=16,
    num_workers=4
)
```

**Key Features:**
- **Streaming:** No disk storage needed
- **Token Packing:** Multiple docs per sequence
- **Shuffling:** 10K buffer for randomness
- **Interleaving:** Smooth mix of datasets

### Training Loop

```python
model = TransformerLM(config).cuda()
optimizer = model.setup_optimizers(learning_rate=3e-4)
scaler = torch.cuda.amp.GradScaler()

step = 0
tokens_seen = 0

for epoch in range(num_epochs):
    for batch in dataloader:
        # Forward pass
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            loss, logits = model(
                batch['input_ids'],
                labels=batch['labels']
            )
            loss = loss / gradient_accumulation_steps
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Optimizer step (every 16 micro-batches)
        if (step + 1) % gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Update LR
            lr = get_lr(tokens_seen, warmup_tokens, total_tokens, 
                       max_lr, min_lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # Logging
            if step % 10 == 0:
                log_metrics(step, loss, lr, tokens_seen)
        
        step += 1
        tokens_seen += batch['input_ids'].numel()
        
        # Checkpointing
        if step % 500 == 0:
            save_checkpoint(model, optimizer, step)
```

### Expected Metrics

| Metric | Initial | After 100M | After 1B | After 2B |
|--------|---------|------------|----------|----------|
| Loss | 10.8 | 7.5 | 5.2 | 4.1 |
| Perplexity | 49,000 | 1,808 | 181 | 60 |
| Tokens/sec | 19,000 | 19,000 | 19,000 | 19,000 |
| Memory (GB) | 24.8 | 24.8 | 24.8 | 24.8 |

### Checkpointing Strategy

**Frequency:**
- Every 500 steps (~15M tokens)
- ~133 checkpoints total
- Keep last 3 + best validation

**Checkpoint Contents:**
```python
{
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scaler_state_dict': scaler.state_dict(),
    'step': step,
    'tokens_seen': tokens_seen,
    'config': config,
    'loss': current_loss,
}
```

### Validation

**Every 250 steps:**
- Evaluate on 1M tokens
- Track perplexity on FineWeb-Edu
- Track perplexity on MedQA separately
- Save best checkpoint

---

## Phase 2: GRPO Fine-tuning

### Objective

Optimize model specifically for USMLE question answering using Group Relative Policy Optimization (GRPO).

### Why GRPO for USMLE?

**Traditional fine-tuning limitations:**
- Supervised learning: Overfits to answer patterns
- Doesn't optimize for reasoning quality
- No exploration of solution paths

**GRPO advantages:**
- Samples multiple answers per question
- Rewards correct reasoning, penalizes incorrect
- Learns from relative quality (no absolute reward needed)
- Off-policy learning: Uses failed attempts as negative examples

### Configuration

```yaml
Model: Load from pretrain checkpoint (0.4B)

RL Config: grpo_usmle_2day
  - Algorithm: Enhanced GRPO
  - Samples per Prompt: 16
  - Prompts per Step: 16
  - Device Batch: 8
  - Epochs: 2 (~600M tokens)

Optimizer: AdamW
  - Base LR: 1e-5 (low for stability)
  - Embedding LR: 2e-4 (higher for adaptation)
  - Weight Decay: 0.0
  - Gradient Clip: 1.0

Sampling:
  - Temperature: 1.0
  - Top-K: 40
  - Max Tokens: 512 per answer

GRPO Settings:
  - Unbiased KL: True
  - Off-Policy Masking: True
  - Use Baseline: True (subtract mean reward)
  - KL Coefficient: 0.01
```

### Enhanced GRPO Algorithm

```python
# Step 1: Sample multiple answers for each question
prompts = load_usmle_questions(batch_size=16)  # 16 questions
samples = []

for prompt in prompts:
    # Generate 16 different answers
    answers = model.generate(
        prompt,
        num_return_sequences=16,
        temperature=1.0,
        top_k=40,
        max_new_tokens=512,
        do_sample=True
    )
    samples.append(answers)

# Step 2: Evaluate rewards (correctness)
rewards = []
for prompt, answers in zip(prompts, samples):
    for answer in answers:
        # Check if answer is correct (A/B/C/D)
        predicted = extract_answer(answer)
        correct = prompt.ground_truth
        reward = 1.0 if predicted == correct else 0.0
        rewards.append(reward)

# Step 3: Compute advantages (relative quality)
# Unbiased KL estimate
log_probs = model.get_log_probs(prompts, samples)
ref_log_probs = reference_model.get_log_probs(prompts, samples)
kl_divergence = log_probs - ref_log_probs

# Group-relative advantages
for i in range(num_prompts):
    group_rewards = rewards[i*16:(i+1)*16]
    baseline = np.mean(group_rewards)
    
    for j in range(16):
        advantages[i*16 + j] = group_rewards[j] - baseline

# Step 4: Off-policy masking
# Identify samples from old policy (KL > threshold)
off_policy_mask = (kl_divergence > 0.5).float()

# Step 5: Compute GRPO loss
policy_loss = -torch.mean(
    log_probs * advantages * off_policy_mask
)

kl_penalty = kl_coef * torch.mean(kl_divergence)

loss = policy_loss + kl_penalty

# Step 6: Optimize
loss.backward()
optimizer.step()
```

### GRPO vs Standard RL

| Feature | Standard REINFORCE | Enhanced GRPO |
|---------|-------------------|---------------|
| Samples per prompt | 1 | 16 |
| Baseline | Global | Group-relative |
| KL estimation | Biased | Unbiased |
| Off-policy data | Discarded | Used with masking |
| Exploration | Limited | Extensive |
| Sample efficiency | Low | High |

### USMLE Dataset Format

```python
{
    "question": "A 45-year-old man with hypertension...",
    "options": {
        "A": "Start ACE inhibitor",
        "B": "Start beta blocker", 
        "C": "Start calcium channel blocker",
        "D": "Observation only"
    },
    "answer": "A",
    "explanation": "ACE inhibitors are first-line..."
}
```

**Processing:**
```python
def format_usmle_prompt(example):
    prompt = f"Question: {example['question']}\n\n"
    for key, value in example['options'].items():
        prompt += f"{key}. {value}\n"
    prompt += "\nAnswer: "
    return prompt

def extract_answer(text):
    # Look for "A", "B", "C", or "D" in response
    matches = re.findall(r'\b[A-D]\b', text)
    return matches[0] if matches else None
```

### Training Loop

```python
reference_model = copy.deepcopy(model).eval()

for epoch in range(num_epochs):
    for batch_prompts in usmle_dataloader:
        # Generate samples
        all_samples = []
        all_log_probs = []
        
        for prompt in batch_prompts:
            samples = model.generate(
                prompt,
                num_return_sequences=16,
                **generation_config
            )
            log_probs = model.get_log_probs(prompt, samples)
            all_samples.extend(samples)
            all_log_probs.extend(log_probs)
        
        # Compute rewards
        rewards = evaluate_samples(batch_prompts, all_samples)
        
        # Compute advantages
        advantages = compute_group_advantages(
            rewards,
            num_samples_per_prompt=16
        )
        
        # Compute KL divergence
        with torch.no_grad():
            ref_log_probs = reference_model.get_log_probs(
                batch_prompts, all_samples
            )
        kl = all_log_probs - ref_log_probs
        
        # Off-policy masking
        mask = (kl < 0.5).float()
        
        # GRPO loss
        policy_loss = -(all_log_probs * advantages * mask).mean()
        kl_loss = 0.01 * kl.mean()
        loss = policy_loss + kl_loss
        
        # Optimize
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Update reference model periodically
        if step % 100 == 0:
            reference_model = copy.deepcopy(model).eval()
```

### Expected Metrics

| Metric | Initial | After 100M | After 300M | After 600M |
|--------|---------|------------|------------|------------|
| USMLE Accuracy | 25% | 32% | 38% | 42% |
| Avg Reward | 0.25 | 0.32 | 0.38 | 0.42 |
| KL Divergence | 0.0 | 0.15 | 0.25 | 0.35 |
| Policy Loss | - | -0.05 | -0.08 | -0.10 |

**Target:** 35-45% accuracy (comparable to GPT-3.5 on USMLE)

---

## Hardware Setup

### Requirements

**GPU:** NVIDIA A100 80GB
- Compute Capability: 8.0+
- CUDA: 12.1+
- Memory: 80GB HBM2e

**CPU:** 8+ cores recommended
- For data loading and preprocessing
- 4 workers for streaming

**RAM:** 64GB+
- Streaming buffers
- Checkpoint staging

**Storage:**
- Model checkpoints: ~5GB each × 3 = 15GB
- Logs and metrics: ~1GB
- Total: ~20GB

### Software Stack

```bash
# CUDA & Drivers
nvidia-driver: 535+
cuda-toolkit: 12.1+

# Python Environment
python: 3.10+
pytorch: 2.2.0+ (with CUDA 12.1)

# Core Dependencies
transformers: 4.35+
datasets: 2.15+
accelerate: 0.25+
tokenizers: 0.15+

# Optimization
flash-attn: 2.4+ (CRITICAL for 2x speedup)
```

### Installation

```bash
# Create environment
conda create -n pure_transformer python=3.10
conda activate pure_transformer

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install transformers datasets accelerate tokenizers

# Install flash-attention (2x speedup)
pip install flash-attn --no-build-isolation

# Verify installation
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import flash_attn; print('Flash Attention: OK')"
```

---

## Running the Pipeline

### Quick Start (K8s)

```bash
cd pure_transformer/k8s

# Deploy pretraining (Phase 1)
./deploy.sh pretrain

# Monitor logs
kubectl logs -f job/pure-transformer-pretrain -n ucsdfutures

# After 29 hours, deploy GRPO (Phase 2)
./deploy.sh rl

# Monitor GRPO logs
kubectl logs -f job/pure-transformer-grpo -n ucsdfutures
```

### Local Training

```bash
# Phase 1: Pretraining (29 hours)
python -m pure_transformer.run_train pretrain \
    --config a100_2day \
    --model medium-large \
    --checkpoint-dir ./checkpoints/pretrain \
    --num-workers 4 \
    --log-every 10

# Phase 2: GRPO (9 hours)
python -m pure_transformer.run_train rl \
    --checkpoint ./checkpoints/pretrain/final.pt \
    --config grpo_usmle_2day \
    --checkpoint-dir ./checkpoints/grpo \
    --task medqa
```

### Debug Mode (Fast Testing)

```bash
# Test pretraining (10M tokens, ~30 seconds)
python -m pure_transformer.run_train pretrain \
    --config debug \
    --model tiny

# Test GRPO (100 steps, ~5 minutes)
python -m pure_transformer.run_train rl \
    --config debug \
    --checkpoint ./checkpoints/debug.pt
```

---

## Monitoring

### Key Metrics

**Pretraining:**
- Loss: Should decrease from 10.8 → 4.1
- Learning Rate: Warmup then cosine decay
- Throughput: Should stay ~19K tokens/sec
- Memory: Should stay ~25GB

**GRPO:**
- Accuracy: Should increase 25% → 42%
- Average Reward: Should track accuracy
- KL Divergence: Should stay < 0.5
- Policy Loss: Should become more negative

### Logging

```python
# Every 10 steps
log = {
    'step': step,
    'tokens_seen': tokens_seen,
    'loss': loss.item(),
    'learning_rate': current_lr,
    'throughput': tokens_per_sec,
    'memory_gb': torch.cuda.max_memory_allocated() / 1e9,
}

# Every 250 steps (validation)
val_log = {
    'val_loss': val_loss,
    'val_perplexity': val_ppl,
    'fineweb_ppl': fineweb_ppl,
    'medqa_ppl': medqa_ppl,
}

# GRPO specific
grpo_log = {
    'accuracy': correct / total,
    'avg_reward': rewards.mean(),
    'kl_divergence': kl.mean(),
    'policy_loss': policy_loss.item(),
}
```

### TensorBoard

```bash
tensorboard --logdir ./checkpoints/logs --port 6006
```

### Monitoring Dashboard

```bash
# GPU utilization
nvidia-smi -l 1

# Training progress
watch -n 1 "tail -20 logs/train.log"

# Checkpoint sizes
du -sh checkpoints/*
```

---

## Troubleshooting

### Common Issues

**1. Out of Memory (OOM)**

```
Symptoms: RuntimeError: CUDA out of memory

Solutions:
- Reduce micro_batch_size: 16 → 8
- Verify gradient checkpointing is enabled
- Check for memory leaks (clear cache periodically)
- Reduce max_seq_length: 2048 → 1024
```

**2. Slow Training (<10K tokens/sec)**

```
Symptoms: Throughput below expected

Solutions:
- Install flash-attention: pip install flash-attn --no-build-isolation
- Verify bf16 is enabled: Check autocast
- Increase num_workers: 2 → 4
- Check GPU utilization: nvidia-smi
```

**3. NaN Loss**

```
Symptoms: Loss becomes NaN during training

Solutions:
- Check learning rate (should be ≤3e-4)
- Verify gradient clipping is enabled
- Use bf16 instead of fp16
- Check for corrupted data in streaming
```

**4. Streaming Dataset Timeout**

```
Symptoms: Dataset loading hangs

Solutions:
- Check internet connection
- Increase timeout in dataset config
- Use local cache if available
- Reduce shuffle_buffer_size temporarily
```

**5. GRPO Not Improving**

```
Symptoms: Accuracy stuck at ~25%

Solutions:
- Verify reference model is frozen
- Check reward computation (should be 0/1)
- Increase num_samples_per_prompt
- Lower learning rate (1e-5 → 5e-6)
- Check answer extraction regex
```

### Performance Optimization

**If training slower than 38 hours:**

1. **Install flash-attention** (most important)
   ```bash
   pip install flash-attn --no-build-isolation
   ```

2. **Use torch.compile** (optional, 1.3x speedup)
   ```python
   model = torch.compile(model, mode='max-autotune')
   ```

3. **Optimize data loading**
   ```python
   dataloader = DataLoader(
       dataset,
       batch_size=16,
       num_workers=4,  # Increase if CPU allows
       pin_memory=True,
       prefetch_factor=2
   )
   ```

4. **Profile to find bottlenecks**
   ```python
   with torch.profiler.profile() as prof:
       # Training code
       pass
   prof.export_chrome_trace("trace.json")
   ```

---

## Next Steps

After training completes:

1. **Evaluate on USMLE test set**
   ```bash
   python -m pure_transformer.evaluate \
       --checkpoint ./checkpoints/grpo/final.pt \
       --task medqa \
       --split test
   ```

2. **Generate sample answers**
   ```bash
   python -m pure_transformer.generate \
       --checkpoint ./checkpoints/grpo/final.pt \
       --prompt "A 45-year-old man presents with..."
   ```

3. **Deploy for inference**
   ```bash
   python -m pure_transformer.serve \
       --checkpoint ./checkpoints/grpo/final.pt \
       --port 8000
   ```

See:
- [EVALUATION.md](EVALUATION.md) - Testing and metrics
- [INFERENCE.md](INFERENCE.md) - Deployment guide
- [ARCHITECTURE.md](ARCHITECTURE.md) - Model details

---

**Last Updated:** December 5, 2025  
**Pipeline Version:** 2-day-optimized  
**Status:** Production Ready
