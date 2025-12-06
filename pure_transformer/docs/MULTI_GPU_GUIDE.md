# Pure Transformer - 8x A100 Multi-GPU Training Guide

## Executive Summary

**Configuration:** 1.46B parameter model trained on 11B tokens in 48 hours using 8x A100 (80GB) GPUs

**Key Specs:**
- Model: 1.46B parameters (32L x 1920H, Chinchilla-optimal)
- Training data: 11B tokens (8.5 tokens/param ratio)
- Hardware: 8x NVIDIA A100 80GB GPUs  
- Training time: ~40-42 hours 
- Strategy: PyTorch Lightning DDP with flash-attention
- Dataset: 45% FineWeb-Edu + 25% Cosmopedia + 20% FinePDFs + 10% USMLE QA

## Quick Start

### Prerequisites
```bash
# Install dependencies
cd pure_transformer
pip install -r requirements.txt

# Install flash-attention (2x speedup)
pip install flash-attn --no-build-isolation
```

### Local Testing (Single GPU)
```bash
# Test with smaller model
python train_multigpu.py \
    --model medium-large \
    --devices 1 \
    --total-tokens 1000000000 \
    --global-batch-size 65536 \
    --micro-batch-size 8
```

### Production Training (8x A100)
```bash
# Full 1.46B model training
python train_multigpu.py \
    --model xlarge \
    --devices 8 \
    --nodes 1 \
    --strategy ddp \
    --precision bf16-mixed \
    --total-tokens 11000000000 \
    --global-batch-size 131072 \
    --micro-batch-size 16 \
    --seq-length 2048 \
    --learning-rate 3e-4 \
    --min-lr 3e-5 \
    --warmup-tokens 100000000 \
    --checkpoint-dir ./checkpoints \
    --save-every-n-steps 1000 \
    --use-wandb
```

### Kubernetes Deployment
```bash
# Deploy to K8s cluster
kubectl apply -f k8s/multigpu-job.yaml -n ucsdfutures

# Monitor training
kubectl logs -f job/pure-transformer-multigpu -n ucsdfutures

# Check GPU utilization
kubectl exec -it <pod-name> -n ucsdfutures -- nvidia-smi
```

## Model Configurations

| Config | Params | Layers | Hidden | Use Case | Training Time (8 GPUs) |
|--------|--------|--------|--------|----------|------------------------|
| tiny | 85M | 12 | 512 | Debugging | ~1 hour |
| small | 178M | 16 | 768 | Quick tests | ~3 hours |
| medium | 374M | 24 | 1024 | Baseline | ~8 hours |
| medium-large | 401M | 20 | 1152 | Single GPU | ~10 hours |
| large | 763M | 24 | 1536 | Extended | ~18 hours |
| **xlarge** | **1.46B** | **32** | **1920** | **Production (8 GPUs)** | **~40 hours** |

**Recommended:** `xlarge` for 8x A100 training - Chinchilla-optimal ratio

## Performance Optimization

### 1. Flash Attention (2x Speedup)
```bash
pip install flash-attn --no-build-isolation
```
Expected throughput: 11,000 tokens/sec per GPU

### 2. Mixed Precision (bf16)
Automatically enabled with `--precision bf16-mixed`
- Reduces memory by ~50%
- Minimal quality loss
- Required for large batch sizes

### 3. Gradient Accumulation
Automatically calculated to achieve target global batch size:
```
grad_accum_steps = global_batch_size / (micro_batch_size * seq_length * num_gpus)
```

### 4. Optimized Data Loading
- `num_workers=4` per GPU (32 total workers)
- Streaming datasets (no disk bottleneck)
- Shuffled buffer size: 10,000

### 5. NCCL Optimization
Set environment variables:
```bash
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export CUDA_DEVICE_MAX_CONNECTIONS=1
```

## Dataset Composition (50-30-20-10 Mix)

Based on the "1 Billion Token Challenge" paper, optimized for USMLE QA:

### Breakdown
- **45% FineWeb-Edu** (4.95B tokens)
  - High-quality educational web content
  - Filtered for clarity and educational value
  
- **25% Cosmopedia** (2.75B tokens)
  - Synthetic textbook-style content
  - Generated educational materials
  
- **20% FinePDFs** (2.2B tokens)
  - Academic papers and textbooks
  - High-quality PDF extractions
  
- **10% USMLE QA** (1.1B tokens)
  - Medical licensing exam questions
  - Domain-specific fine-tuning data

### Why This Mix?
1. **Educational Foundation (70%):** FineWeb-Edu + Cosmopedia provide broad educational grounding
2. **Academic Rigor (20%):** FinePDFs add depth and technical accuracy
3. **Domain Specialization (10%):** USMLE QA tunes for medical domain
4. **Avoids Overfitting:** Diverse sources prevent memorization

## Training Pipeline

### Phase 1: Pre-training (40 hours)
```bash
python train_multigpu.py \
    --model xlarge \
    --devices 8 \
    --total-tokens 11000000000 \
    --fineweb-prob 0.45 \
    --cosmopedia-prob 0.25 \
    --finepdf-prob 0.20 \
    --usmle-prob 0.10
```

**Checkpoints saved every 1000 steps** (~131M tokens)

### Phase 2: GRPO Fine-tuning (Optional, 8 hours)
After pre-training, fine-tune with GRPO on USMLE:
```bash
python -m pure_transformer.run_train rl \
    --checkpoint ./checkpoints/pure-transformer-last.ckpt \
    --task usmle \
    --num-samples 16 \
    --learning-rate 1e-5
```

## Memory Requirements

### Per-GPU Memory Usage (A100 80GB)
```
Model weights:        ~6 GB  (1.46B params × 4 bytes)
Gradients:           ~6 GB
Optimizer states:    ~12 GB (AdamW: 2x params)
Activations:         ~15 GB (batch_size=16, seq_len=2048)
Working memory:      ~5 GB  (buffers, CUDA)
-------------------------------------------
Total:               ~44 GB per GPU
Headroom:            ~36 GB
```

**Safe margin:** Can handle micro_batch_size up to 24 per GPU

## Throughput Estimates

### Single A100
- Base (SDPA): 5,500 tokens/sec
- Flash-attn: 11,000 tokens/sec (2x)

### 8x A100 (DDP)
- Expected: 79,200 tokens/sec (90% scaling efficiency)
- Per GPU overhead: ~10% (communication, synchronization)

### Training Time Calculation
```
Total tokens: 11,000,000,000
Throughput: 79,200 tokens/sec
Time: 11B / 79.2K = 138,889 seconds = 38.6 hours
With overhead: ~40-42 hours
```

## Monitoring & Logging

### TensorBoard
```bash
tensorboard --logdir ./checkpoints/lightning_logs
```

Metrics tracked:
- `train/loss` - Training loss
- `train/ppl` - Perplexity
- `train/lr` - Learning rate
- `train/grad_norm` - Gradient norm
- `train/tokens` - Tokens processed

### Weights & Biases
Enable with `--use-wandb`:
```bash
export WANDB_API_KEY=<your-key>
python train_multigpu.py --use-wandb --wandb-project pure-transformer
```

### GPU Monitoring
```bash
# Watch GPU utilization
watch -n 1 nvidia-smi

# Detailed GPU stats
nvidia-smi dmon -s pucvmet
```

## Checkpointing

### Automatic Checkpoints
- **Every 1000 steps** (~131M tokens, ~20 minutes)
- **Top 3 checkpoints** saved (by val loss)
- **Last checkpoint** always saved

### Checkpoint Structure
```
checkpoints/
├── pure-transformer-epoch=0-step=1000.ckpt
├── pure-transformer-epoch=0-step=2000.ckpt
├── pure-transformer-last.ckpt
└── lightning_logs/
    └── version_0/
        ├── events.out.tfevents...
        └── hparams.yaml
```

### Resume Training
```bash
python train_multigpu.py \
    --resume-from ./checkpoints/pure-transformer-last.ckpt \
    --model xlarge \
    --devices 8
```

## Troubleshooting

### Out of Memory (OOM)
1. Reduce `--micro-batch-size` from 16 to 12 or 8
2. Reduce `--seq-length` from 2048 to 1024
3. Enable gradient checkpointing (already on by default)

### Slow Training
1. Verify flash-attention is installed: `python -c "import flash_attn; print('OK')"`
2. Check GPU utilization: `nvidia-smi` (should be >90%)
3. Increase `--num-workers` from 4 to 8
4. Check NCCL settings for multi-GPU communication

### NaN Loss
1. Reduce learning rate: `--learning-rate 1e-4`
2. Increase warmup: `--warmup-tokens 200000000`
3. Check gradient clipping (enabled at 1.0 by default)

### Dataset Streaming Timeout
1. Increase timeout in streaming config
2. Use cached datasets if available
3. Check network connectivity to HuggingFace Hub

### Poor GPU Scaling
1. Verify NCCL is using InfiniBand: `NCCL_DEBUG=INFO`
2. Check for stragglers: imbalanced data loading
3. Increase global batch size for better GPU utilization

## Production Checklist

Before starting 48-hour training run:

- [ ] Flash-attention installed (`pip install flash-attn --no-build-isolation`)
- [ ] All 8 GPUs available (`nvidia-smi` shows 8x A100)
- [ ] Checkpoint directory has 500GB+ space
- [ ] Monitoring setup (TensorBoard or W&B)
- [ ] HuggingFace token set (`HF_TOKEN` env var)
- [ ] Resume capability tested
- [ ] Expected throughput validated (>75K tokens/sec)
- [ ] Alerting configured for OOM/NaN errors
- [ ] Backup strategy for checkpoints

## Expected Results

### After 11B Tokens
- **Validation Loss:** ~3.5-4.0
- **Perplexity:** ~33-55
- **USMLE QA Accuracy:** ~35-45% (after GRPO fine-tuning)
- **Generation Quality:** Coherent multi-sentence medical text
- **Checkpoint Size:** ~5.8GB per checkpoint

### Comparison to Baselines
- **GPT-2 (124M):** Lower quality, trained on 10B tokens
- **GPT-3.5 (175B):** Comparable size, our model specialized for medical/educational
- **LLaMA-2 (7B):** Higher quality but 5x larger

## Next Steps After Training

1. **Evaluate on benchmarks:**
   ```bash
   python -m pure_transformer.evaluate \
       --checkpoint ./checkpoints/pure-transformer-last.ckpt \
       --tasks mmlu,medqa,arc
   ```

2. **GRPO fine-tuning:**
   ```bash
   python -m pure_transformer.run_train rl \
       --checkpoint ./checkpoints/pure-transformer-last.ckpt \
       --task usmle
   ```

3. **Deploy for inference:**
   ```bash
   python -m pure_transformer.serve \
       --checkpoint ./checkpoints/pure-transformer-last.ckpt \
       --port 8000
   ```

## Support & Resources

- **Documentation:** `docs/`
- **Examples:** `notebooks/`
- **Issues:** GitHub Issues
- **Monitoring:** TensorBoard / W&B dashboards

## Citations

```bibtex
@article{sharma2025billion,
  title={The 1 Billion Token Challenge: Finding the Perfect Pre-training Mix},
  author={Sharma, Asankhaya},
  year={2025}
}
```
