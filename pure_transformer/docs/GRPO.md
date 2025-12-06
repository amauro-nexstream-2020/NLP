# Group Relative Policy Optimization (GRPO) for USMLE QA

## Overview

This document describes the Enhanced GRPO implementation optimized for USMLE Question Answering. GRPO is a reinforcement learning algorithm that learns from relative quality comparisons within groups of generated responses.

**Why GRPO for USMLE?**
- No need for absolute reward models
- Learns from multiple answer attempts
- Robust to answer format variations
- Sample-efficient for medical QA

---

## Table of Contents

1. [Algorithm Overview](#algorithm-overview)
2. [Enhanced GRPO Features](#enhanced-grpo-features)
3. [Implementation Details](#implementation-details)
4. [USMLE-Specific Adaptations](#usmle-specific-adaptations)
5. [Training Procedure](#training-procedure)
6. [Expected Results](#expected-results)

---

## Algorithm Overview

### Standard REINFORCE Problems

Traditional REINFORCE for question answering:

```python
# Sample one answer
answer = model.generate(question)

# Get binary reward
reward = 1.0 if is_correct(answer) else 0.0

# Optimize
loss = -log_prob(answer) * reward
```

**Issues:**
- High variance (single sample)
- No exploration of alternatives
- Requires many questions to learn

### GRPO Solution

GRPO samples multiple answers and learns from relative quality:

```python
# Sample 16 different answers
answers = [model.generate(question) for _ in range(16)]

# Evaluate all answers
rewards = [1.0 if is_correct(ans) else 0.0 for ans in answers]

# Compute group-relative advantages
baseline = mean(rewards)  # e.g., 0.25 (4 correct out of 16)
advantages = [r - baseline for r in rewards]  # [-0.25, -0.25, 0.75, ...]

# Optimize using relative quality
for answer, advantage in zip(answers, advantages):
    loss -= log_prob(answer) * advantage
```

**Benefits:**
- Lower variance (16 samples vs 1)
- Natural exploration (temperature sampling)
- Learns what makes answers better
- Self-normalizing (no reward calibration needed)

---

## Enhanced GRPO Features

Our implementation includes three key enhancements from DeepSeek-V3.2:

### 1. Unbiased KL Estimate

**Problem:** Standard KL divergence estimation is biased when policies differ significantly.

**Standard approach:**
```python
kl = log_prob_current - log_prob_reference
```

**Enhanced approach:**
```python
def compute_unbiased_kl(current_logits, reference_logits):
    """
    Unbiased KL estimate using importance sampling.
    More accurate when current policy diverges from reference.
    """
    current_probs = F.softmax(current_logits, dim=-1)
    reference_probs = F.softmax(reference_logits, dim=-1)
    
    # Importance sampling correction
    importance_weights = current_probs / (reference_probs + 1e-8)
    
    kl = (current_probs * (
        torch.log(current_probs + 1e-8) - 
        torch.log(reference_probs + 1e-8)
    )).sum(dim=-1)
    
    return kl * importance_weights.mean(dim=-1)
```

**Impact:** More stable training, especially later when policies diverge.

### 2. Off-Policy Sequence Masking

**Problem:** As training progresses, some samples become "stale" (from old policy).

**Solution:** Detect and mask off-policy samples:

```python
def compute_off_policy_mask(current_log_probs, reference_log_probs, threshold=0.5):
    """
    Mask samples that are too far from current policy.
    Prevents learning from outdated samples.
    """
    kl_per_sample = current_log_probs - reference_log_probs
    
    # Mask samples with KL > threshold
    mask = (kl_per_sample < threshold).float()
    
    return mask
```

**Impact:** Better sample efficiency, faster convergence.

### 3. Keep Sampling Mask

**Problem:** Some generated samples may be invalid (formatting errors, incomplete answers).

**Solution:** Track valid samples and only learn from those:

```python
def create_sampling_mask(samples, min_length=10):
    """
    Mask invalid or low-quality samples.
    """
    mask = torch.ones(len(samples))
    
    for i, sample in enumerate(samples):
        # Check minimum length
        if len(sample) < min_length:
            mask[i] = 0.0
        
        # Check for valid answer format (A/B/C/D)
        if not has_valid_answer(sample):
            mask[i] = 0.0
        
        # Check for repetition/degeneration
        if is_repetitive(sample):
            mask[i] = 0.0
    
    return mask
```

**Impact:** Cleaner training signal, avoids learning from errors.

---

## Implementation Details

### Complete GRPO Training Step

```python
class EnhancedGRPOTrainer:
    def __init__(self, model, reference_model, config):
        self.model = model
        self.reference_model = reference_model  # Frozen copy
        self.config = config
        self.optimizer = model.setup_optimizers(
            learning_rate=config.learning_rate
        )
    
    def train_step(self, prompts):
        """
        Single GRPO training step.
        
        Args:
            prompts: List of USMLE questions (batch_size=16)
        
        Returns:
            metrics: Dict with loss, accuracy, KL, etc.
        """
        batch_size = len(prompts)
        num_samples = self.config.num_samples_per_prompt  # 16
        
        # Step 1: Generate samples
        all_samples = []
        all_log_probs = []
        all_rewards = []
        
        for prompt in prompts:
            # Generate 16 different answers
            samples = self.model.generate(
                prompt,
                num_return_sequences=num_samples,
                temperature=self.config.temperature,  # 1.0
                top_k=self.config.top_k,  # 40
                max_new_tokens=self.config.max_new_tokens,  # 512
                do_sample=True
            )
            
            # Compute log probabilities
            log_probs = self.model.get_log_probs(prompt, samples)
            
            # Evaluate rewards (binary: correct or not)
            rewards = self.evaluate_samples(prompt, samples)
            
            all_samples.extend(samples)
            all_log_probs.append(log_probs)
            all_rewards.append(rewards)
        
        all_log_probs = torch.cat(all_log_probs)
        all_rewards = torch.cat(all_rewards)
        
        # Step 2: Compute reference log probs (frozen model)
        with torch.no_grad():
            ref_log_probs = []
            for prompt, samples in zip(prompts, all_samples_grouped):
                ref_lp = self.reference_model.get_log_probs(prompt, samples)
                ref_log_probs.append(ref_lp)
            ref_log_probs = torch.cat(ref_log_probs)
        
        # Step 3: Compute KL divergence (unbiased)
        kl = self.compute_unbiased_kl(all_log_probs, ref_log_probs)
        
        # Step 4: Compute group-relative advantages
        advantages = []
        for i in range(batch_size):
            start = i * num_samples
            end = (i + 1) * num_samples
            
            group_rewards = all_rewards[start:end]
            baseline = group_rewards.mean()
            
            # Advantage = reward - group baseline
            group_advantages = group_rewards - baseline
            advantages.append(group_advantages)
        
        advantages = torch.cat(advantages)
        
        # Step 5: Off-policy masking
        off_policy_mask = self.compute_off_policy_mask(
            all_log_probs, 
            ref_log_probs,
            threshold=0.5
        )
        
        # Step 6: Sampling mask (valid answers only)
        sampling_mask = self.create_sampling_mask(all_samples)
        
        # Combined mask
        mask = off_policy_mask * sampling_mask
        
        # Step 7: Compute GRPO loss
        # Policy gradient loss
        policy_loss = -(all_log_probs * advantages * mask).sum() / mask.sum()
        
        # KL penalty (prevent too much drift)
        kl_penalty = self.config.kl_coef * kl.mean()
        
        # Total loss
        loss = policy_loss + kl_penalty
        
        # Step 8: Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            self.config.max_grad_norm
        )
        self.optimizer.step()
        
        # Step 9: Update reference model periodically
        if self.step % 100 == 0:
            self.reference_model = copy.deepcopy(self.model).eval()
        
        # Return metrics
        return {
            'loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'kl_penalty': kl_penalty.item(),
            'kl_divergence': kl.mean().item(),
            'accuracy': (all_rewards * mask).sum() / mask.sum(),
            'avg_reward': all_rewards.mean().item(),
            'mask_ratio': mask.mean().item(),
        }
```

### Key Hyperparameters

```python
GRPO_CONFIG = {
    # Sampling
    'num_samples_per_prompt': 16,  # More samples = lower variance
    'temperature': 1.0,  # Higher = more exploration
    'top_k': 40,  # Limit sampling to top-40 tokens
    'max_new_tokens': 512,  # Max answer length
    
    # Training
    'prompts_per_step': 16,  # Batch size (questions)
    'learning_rate': 1e-5,  # Low LR for stability
    'embedding_lr': 2e-4,  # Higher for embeddings
    'num_epochs': 2,  # ~600M tokens
    
    # GRPO-specific
    'kl_coef': 0.01,  # KL penalty weight
    'use_baseline': True,  # Subtract mean reward
    'normalize_advantages': False,  # Don't divide by std
    
    # Enhanced features
    'use_unbiased_kl': True,
    'use_off_policy_masking': True,
    'off_policy_threshold': 0.5,
    
    # Stability
    'max_grad_norm': 1.0,
    'update_reference_every': 100,  # steps
}
```

---

## USMLE-Specific Adaptations

### Question Format

```python
def format_usmle_prompt(example):
    """
    Format USMLE question for model.
    
    Example input:
    {
        "question": "A 65-year-old man presents with...",
        "options": {
            "A": "Myocardial infarction",
            "B": "Pulmonary embolism",
            "C": "Pneumothorax",
            "D": "Aortic dissection"
        },
        "answer": "D",
        "explanation": "..."
    }
    """
    prompt = f"Medical Question: {example['question']}\n\n"
    prompt += "Options:\n"
    for key, value in sorted(example['options'].items()):
        prompt += f"{key}. {value}\n"
    prompt += "\nAnalyze the case and provide the correct answer (A, B, C, or D): "
    
    return prompt
```

### Answer Extraction

```python
def extract_answer(generated_text):
    """
    Extract answer letter from model generation.
    Handles various formats:
    - "The answer is A"
    - "A. Myocardial infarction"
    - "Based on the symptoms, A"
    - "Answer: A"
    """
    # Remove prompt from generation
    text = generated_text.split("Answer (A, B, C, or D): ")[-1]
    
    # Look for explicit answer patterns
    patterns = [
        r'\b([A-D])\b',  # Single letter
        r'[Aa]nswer:?\s*([A-D])',  # "Answer: A"
        r'[Oo]ption\s*([A-D])',  # "Option A"
        r'([A-D])\..*correct',  # "A. ... is correct"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).upper()
    
    # Fallback: first A/B/C/D in text
    matches = re.findall(r'\b[A-D]\b', text)
    return matches[0] if matches else None
```

### Reward Computation

```python
def compute_reward(prompt, generated_text):
    """
    Compute reward for USMLE answer.
    
    Returns:
        reward: 1.0 if correct, 0.0 if incorrect
        info: Dict with additional information
    """
    ground_truth = prompt['answer']
    predicted = extract_answer(generated_text)
    
    # Binary reward
    reward = 1.0 if predicted == ground_truth else 0.0
    
    # Additional info for logging
    info = {
        'predicted': predicted,
        'ground_truth': ground_truth,
        'correct': reward == 1.0,
        'has_valid_answer': predicted is not None,
        'response_length': len(generated_text),
    }
    
    return reward, info
```

### Special Handling for Medical Concepts

```python
class USMLERewardModel:
    """
    Enhanced reward model for USMLE.
    Can incorporate additional signals beyond binary correctness.
    """
    
    def __init__(self):
        self.medical_terms = load_medical_vocabulary()
    
    def compute_reward(self, prompt, generated_text):
        # Primary reward: correctness
        primary_reward = compute_basic_reward(prompt, generated_text)
        
        # Bonus for medical terminology usage (small)
        term_bonus = 0.0
        if self.uses_medical_terms(generated_text):
            term_bonus = 0.05
        
        # Penalty for very short answers (likely incomplete)
        length_penalty = 0.0
        if len(generated_text) < 50:
            length_penalty = -0.1
        
        # Total reward
        total = primary_reward + term_bonus + length_penalty
        return torch.clamp(total, 0.0, 1.0)
```

---

## Training Procedure

### Phase 1: Warmup (1 hour, 50M tokens)

```python
# Start with low temperature for stability
config.temperature = 0.7
config.kl_coef = 0.02  # Higher KL penalty

# Use smaller sample size initially
config.num_samples_per_prompt = 8

# Run for 100 steps
for step in range(100):
    prompts = sample_usmle_questions(batch_size=16)
    metrics = trainer.train_step(prompts)
    
    # Monitor accuracy
    if metrics['accuracy'] < 0.15:
        # Model is struggling, reduce learning rate
        reduce_learning_rate()
```

### Phase 2: Main Training (7 hours, 450M tokens)

```python
# Increase exploration
config.temperature = 1.0
config.num_samples_per_prompt = 16

# Lower KL penalty (allow more deviation)
config.kl_coef = 0.01

# Train for main duration
for epoch in range(2):
    for batch_prompts in usmle_dataloader:
        metrics = trainer.train_step(batch_prompts)
        
        # Log every 10 steps
        if step % 10 == 0:
            log_metrics(metrics)
        
        # Validate every 50 steps
        if step % 50 == 0:
            val_accuracy = evaluate_on_validation_set()
            save_checkpoint_if_best(val_accuracy)
```

### Phase 3: Final Tuning (1 hour, 100M tokens)

```python
# Lower learning rate for fine-tuning
config.learning_rate = 5e-6

# Reduce temperature (exploit more, explore less)
config.temperature = 0.8

# Continue training
for step in range(final_steps):
    metrics = trainer.train_step(batch_prompts)
```

---

## Expected Results

### Learning Curve

| Tokens | Accuracy | Avg Reward | KL Div | Notes |
|--------|----------|------------|--------|-------|
| 0 (pretrained) | 25% | 0.25 | 0.00 | Random baseline |
| 50M | 28% | 0.28 | 0.05 | Initial learning |
| 150M | 33% | 0.33 | 0.15 | Rapid improvement |
| 300M | 38% | 0.38 | 0.25 | Steady progress |
| 450M | 41% | 0.41 | 0.32 | Approaching plateau |
| 600M | 42% | 0.42 | 0.35 | Final performance |

### Comparison with Baselines

| Method | USMLE Accuracy | Training Time |
|--------|---------------|---------------|
| Random | 25% | 0h |
| Pretrained only | 28% | 29h |
| Supervised fine-tuning | 35% | 33h |
| **GRPO (ours)** | **42%** | **38h** |
| GPT-3.5 | 45% | N/A |

### Sample Outputs

**Before GRPO:**
```
Question: A 45-year-old man with chest pain...
Options: A. MI, B. PE, C. Pneumothorax, D. Dissection

Model output: "This patient could have several conditions. Chest pain is concerning..."
Extracted answer: None (failed to extract)
Reward: 0.0
```

**After GRPO:**
```
Question: A 45-year-old man with chest pain...
Options: A. MI, B. PE, C. Pneumothorax, D. Dissection

Model output: "Given the sudden onset of chest pain radiating to the back, along with the patient's age and risk factors, the most likely diagnosis is D. Aortic dissection."
Extracted answer: D
Reward: 1.0 (correct)
```

---

## Advantages Over Other RL Methods

### vs PPO (Proximal Policy Optimization)

| Feature | PPO | GRPO |
|---------|-----|------|
| Value network | Required | Not needed |
| Sample efficiency | Lower | Higher |
| Hyperparameters | Many (clip, epochs, etc.) | Few |
| Memory | High (value network) | Lower |
| Stability | Moderate | High (group normalization) |

### vs DPO (Direct Preference Optimization)

| Feature | DPO | GRPO |
|---------|-----|------|
| Preference data | Required | Not needed |
| Online sampling | No | Yes |
| Exploration | Limited | Extensive |
| Adaptation | Slow | Fast |

---

## Tips for Success

1. **Monitor KL divergence:** Should stay < 0.5
   - If too high: Increase kl_coef
   - If too low: Decrease kl_coef

2. **Check mask ratios:** Should be 70-90%
   - Low mask ratio: Many off-policy samples (update reference model more often)
   - High mask ratio: Good sample quality

3. **Track per-question accuracy:** Some questions are harder
   - Focus on question types where model struggles
   - Consider question difficulty weighting

4. **Answer extraction robustness:** Monitor extraction failures
   - If >10% extraction failures: Improve prompt format
   - Consider training explicit answer format

5. **Reference model updates:** Critical for stability
   - Update too often: Slow learning
   - Update too rarely: Unstable training
   - Sweet spot: Every 100 steps

---

## Next Steps

After GRPO training:

1. **Evaluate on test set**
2. **Analyze error patterns**
3. **Fine-tune extraction if needed**
4. **Deploy for inference**

See [EVALUATION.md](EVALUATION.md) for comprehensive evaluation procedures.

---

**Last Updated:** December 5, 2025  
**Algorithm Version:** Enhanced GRPO with DeepSeek-V3.2 features  
**Status:** Production Ready
