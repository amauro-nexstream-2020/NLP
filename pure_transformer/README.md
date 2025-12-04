# Pure Transformer with GRPO/ProRL

A clean, efficient pure transformer architecture optimized for 3-day A100 training with SOTA RL fine-tuning.

## Architecture

**Target Model: ~350M parameters** (trainable in <72 hours on A100-80GB)

| Component | Details |
|-----------|---------|
| Hidden size | 1024 |
| Layers | 24 |
| Attention heads | 16 (GQA: 4 KV heads) |
| Head dimension | 64 |
| Intermediate size | 2816 (SwiGLU) |
| Max sequence length | 2048 |
| Vocab size | 50304 |

### Key Features

- **RoPE** positional embeddings
- **QK Normalization** for training stability
- **GQA** (Grouped Query Attention) for memory efficiency
- **SwiGLU MLP** activation
- **Pre-norm** architecture with RMSNorm
- **Flash Attention** (when available)
- **Gradient checkpointing** for memory efficiency

## Training Pipeline

### Phase 1: Pretraining (SFT)

Streaming pretraining on FineWeb-Edu + MedQA mixture:

```bash
# Local
python -m pure_transformer.run_pretrain --config a100_3day --model medium

# Debug mode
python -m pure_transformer.run_pretrain --config debug --model tiny
```

**3-Day A100 Target:**
- 50B tokens
- 512K global batch size
- 3e-4 learning rate with cosine decay
- ~500K tokens/sec throughput

### Phase 2: GRPO (Reinforcement Learning)

Group Relative Policy Optimization for task-specific fine-tuning:

```bash
# GSM8K (math reasoning)
python -m pure_transformer.run_grpo --config grpo_gsm8k --checkpoint ./checkpoints/pretrain/final.pt

# MedQA (medical QA)
python -m pure_transformer.run_grpo --config grpo_medqa --checkpoint ./checkpoints/pretrain/final.pt
```

**GRPO Features:**
- No KL regularization (simpler than PPO)
- On-policy sampling
- Group advantage normalization
- 16 samples per prompt for stable gradients

## K8s Deployment

### Quick Start

```bash
cd pure_transformer/k8s

# Configure
export REGISTRY="your-registry.io"
export NAMESPACE="ucsdfutures"

# Deploy
chmod +x deploy.sh
./deploy.sh
```

### Manual Deployment

```bash
# Build image
docker build -t $REGISTRY/pure-transformer:latest -f k8s/Dockerfile ..
docker push $REGISTRY/pure-transformer:latest

# Create secrets
cp k8s/secrets.yaml.template k8s/secrets.yaml
# Edit secrets.yaml with your credentials
kubectl apply -f k8s/secrets.yaml -n $NAMESPACE

# Start pretraining
kubectl apply -f k8s/pretrain-job.yaml -n $NAMESPACE

# Monitor
kubectl logs -f job/pure-transformer-pretrain -n $NAMESPACE

# After pretraining, run GRPO
kubectl apply -f k8s/grpo-job.yaml -n $NAMESPACE
```

## Configuration

### Model Configs

| Name | Params | Hidden | Layers | Use Case |
|------|--------|--------|--------|----------|
| tiny | ~45M | 512 | 12 | Debugging |
| small | ~125M | 768 | 16 | Quick experiments |
| **medium** | ~350M | 1024 | 24 | **3-day A100 target** |

### Training Configs

| Name | Tokens | Batch | Description |
|------|--------|-------|-------------|
| debug | 10M | 32K | Quick testing |
| a100_1day | 15B | 512K | Single day run |
| **a100_3day** | 50B | 512K | **Full training** |

### RL Configs

| Name | Task | Samples | Description |
|------|------|---------|-------------|
| debug | - | 4 | Quick testing |
| grpo_gsm8k | GSM8K | 16 | Math reasoning |
| grpo_medqa | MedQA | 16 | Medical QA |
| grpo_full | Both | 16 | Multi-task |

## Requirements

```bash
pip install -r requirements.txt

# Optional: Flash Attention for faster training
pip install flash-attn --no-build-isolation
```

## Project Structure

```
pure_transformer/
├── __init__.py
├── run_pretrain.py      # Pretraining entry point
├── run_grpo.py          # GRPO training entry point
├── requirements.txt
├── configs/
│   ├── __init__.py
│   ├── model_config.py   # Model architectures
│   └── training_config.py # Training & RL configs
├── model/
│   ├── __init__.py
│   └── transformer.py    # Pure Transformer implementation
├── training/
│   ├── __init__.py
│   ├── pretrain.py       # Pretraining loop
│   └── grpo.py           # GRPO implementation
└── k8s/
    ├── Dockerfile
    ├── deploy.sh
    ├── pretrain-job.yaml
    ├── grpo-job.yaml
    └── secrets.yaml.template
```

## References

- GRPO: [DeepSeekMath](https://arxiv.org/abs/2402.03300)
- SwiGLU: [Shazeer 2020](https://arxiv.org/abs/2002.05202)
- GQA: [Ainslie et al.](https://arxiv.org/abs/2305.13245)
- QK Norm: [Henry et al.](https://arxiv.org/abs/2302.05442)
- RoPE: [Su et al.](https://arxiv.org/abs/2104.09864)
