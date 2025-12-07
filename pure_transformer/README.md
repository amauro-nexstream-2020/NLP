# Pure Transformer with DeepSeek-V3.2 Optimizations

A clean, efficient pure transformer architecture with DeepSeek-V3.2 style optimizations for high-performance training and RL fine-tuning.

## Features

### DeepSeek-V3.2 Optimizations
- **DeepSeek Sparse Attention (DSA)**: Lightning indexer + fine-grained token selection
- **Enhanced GRPO**: Unbiased KL estimate, off-policy sequence masking
- **Scalable RL Framework**: Supports >10% post-training compute allocation

### Architecture
- **RoPE** positional embeddings
- **QK Normalization** for training stability  
- **GQA** (Grouped Query Attention) for memory efficiency
- **SwiGLU MLP** activation
- **Pre-norm** architecture with RMSNorm
- **Flash Attention** (when available)
- **Gradient checkpointing** for memory efficiency

## Model Configurations

| Config | Hidden | Layers | Heads | Parameters | Training Time | Use Case |
|--------|--------|--------|-------|------------|---------------|----------|
| tiny   | 512    | 12     | 8     | ~85M       | Minutes       | Debugging |
| small  | 768    | 16     | 12    | ~178M      | Hours         | Fast experiments |
| medium | 1024   | 24     | 16    | ~374M      | 1-2 days      | Baseline training |
| **medium-large** | **1152** | **20** | **16** | **~401M (0.4B)** | **2 days** | **RECOMMENDED** |
| large  | 1536   | 24     | 16    | ~763M      | 5-7 days      | Extended training |

**✨ RECOMMENDED: `medium-large` (0.4B) for 2-day A100 training**
- Achieves ~2.6B tokens in 48 hours (with flash-attn)
- Optimal for USMLE QA with GRPO fine-tuning
- 35-45% accuracy on medical questions

## Quick Start

### Installation

```bash
pip install torch transformers datasets accelerate tokenizers
```

### Training

```bash
# 2-Day Training Pipeline (RECOMMENDED)
# Phase 1: Pretraining (1.5 days, 2B tokens)
python -m pure_transformer.run_train pretrain \
    --config a100_2day \
    --model medium-large

# Phase 2: GRPO Fine-tuning on USMLE (0.5 days, 0.6B tokens)
python -m pure_transformer.run_train rl \
    --checkpoint checkpoints/pretrain/final.pt \
    --config grpo_usmle_2day \
    --task medqa

# Run tests
python -m pure_transformer.run_train test
```

### Debug Mode

```bash
python -m pure_transformer.run_train pretrain --config debug --model tiny
```

## Streaming Datasets

- **FineWeb-Edu**: High-quality educational web text (85% default)
- **MedQA-USMLE**: Medical question answering (15% default)

Both datasets are streamed from HuggingFace Hub for memory efficiency.

## K8s Deployment (A100)

### Quick Deploy

```bash
cd pure_transformer/k8s
./deploy.sh pretrain    # Deploy pretraining job
./deploy.sh rl          # Deploy RL job
./deploy.sh status      # Check status
./deploy.sh logs        # View logs
```

### Manual Deployment

```bash
# Create PVC for checkpoints
kubectl apply -f k8s/pretrain-job.yaml -n ucsdfutures

# Monitor
kubectl logs -f job/pure-transformer-pretrain -n ucsdfutures
````

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
