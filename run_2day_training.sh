#!/bin/bash
# 2-Day Training Run for Pure Transformer
# Started: $(date -Iseconds)
# Target: 2 days on 2x RTX 6000 Ada (48GB each)
# Model: xlarge (1.5B parameters)

set -e

cd /home/achalamlasetty/mscproj/NLP
source /home/achalamlasetty/mscproj/NLP/NLP/bin/activate

export PYTHONPATH=$(pwd)
export WANDB_API_KEY="1651cbec695294a8a1b99aebdbf4f11f66e5789e"
export WANDB_PROJECT="pure-transformer"
export WANDB_NAME="xlarge-2xRTX6000-2day-v2"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True"

# NCCL settings for stability (prevent watchdog timeouts)
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800  # 30 min timeout instead of 8 min default
export TORCH_NCCL_ENABLE_MONITORING=1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1  # Disable InfiniBand for local training

echo "========================================"
echo "2-DAY LOCAL TRAINING SESSION v2"
echo "========================================"
echo "Start time: $(date -Iseconds)"
echo "Model: xlarge (1.5B params)"
echo "GPUs: 2x RTX 6000 Ada"
echo "Target: 3.5B tokens (realistic for 2 days)"
echo "NCCL timeout: 30 min"
echo "========================================"

nvidia-smi

LOGFILE=/home/achalamlasetty/mscproj/NLP/checkpoints/2day_local/training_$(date +%Y%m%d_%H%M%S).log

python -m pure_transformer.train_multigpu \
  --model xlarge \
  --devices 2 \
  --nodes 1 \
  --strategy ddp \
  --precision bf16-mixed \
  --total-tokens 3500000000 \
  --global-batch-size 524288 \
  --micro-batch-size 8 \
  --seq-length 2048 \
  --learning-rate 3e-4 \
  --min-lr 3e-5 \
  --weight-decay 0.1 \
  --warmup-tokens 100000000 \
  --num-workers 2 \
  --checkpoint-dir /home/achalamlasetty/mscproj/NLP/checkpoints/2day_local \
  --save-every-n-steps 250 \
  --log-every-n-steps 10 \
  --use-wandb \
  --wandb-project pure-transformer \
  --fineweb-subset sample-100BT \
  --fineweb-prob 0.65 \
  --finepdf-prob 0.34 \
  --usmle-prob 0.01 \
  --seed 42 \
  2>&1 | tee ${LOGFILE}

echo ""
echo "========================================"
echo "TRAINING COMPLETED"
echo "End time: $(date -Iseconds)"
echo "========================================"
