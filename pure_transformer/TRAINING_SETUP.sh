#!/bin/bash
# Training Setup Script for 35B Token Pretraining
# 
# Dataset Configuration:
# - FineWeb-Edu: 60% (sample-100BT for 35B training, or full for >35B)
# - FinePDFs (English): 30% (1.19T tokens available from HuggingFace)
# - USMLE QA: 10% (medical knowledge)
#
# Hardware: Optimized for 8x A100 80GB GPUs
# Training time: ~6-8 hours for 35B tokens at 1.2M tokens/sec
#
# Model: XLarge (1.3B parameters) - Chinchilla optimal for 35B tokens

set -e

echo "========================================="
echo "35B Token Training Setup"
echo "========================================="
echo ""

# Activate environment
echo "Activating NLP environment..."
PYTHON_BIN="/home/achalamlasetty/mscproj/NLP/NLP/bin/python"

# Verify environment
echo "Checking Python and dependencies..."
$PYTHON_BIN -c "import torch; import lightning; import transformers; import datasets; print('✓ All dependencies installed')"

echo ""
echo "========================================="
echo "Dataset Information"
echo "========================================="
echo ""
echo "1. FineWeb-Edu (60%)"
echo "   - sample-10BT: ~10B tokens (for testing)"
echo "   - sample-100BT: ~100B tokens (recommended for 35B training)"
echo "   - Full dataset: 15T tokens"
echo ""
echo "2. FinePDFs English (30%)"
echo "   - Official HuggingFace dataset: HuggingFaceFW/finepdfs"
echo "   - 1.19 TRILLION tokens available"
echo "   - 207M documents from CommonCrawl PDFs"
echo "   - Longer documents (2x avg length vs web)"
echo "   - Excellent for long-context capabilities"
echo ""
echo "3. USMLE QA (10%)"
echo "   - Medical knowledge questions"
echo "   - ~10M tokens"
echo ""
echo "Total available: >35B tokens ✓"
echo ""

echo "========================================="
echo "Training Commands"
echo "========================================="
echo ""

echo "=== 1. Test Run (10 steps, verify setup) ==="
echo "cd /home/achalamlasetty/mscproj/NLP"
echo "CUDA_VISIBLE_DEVICES=0 $PYTHON_BIN pure_transformer/simple_train_test.py"
echo ""

echo "=== 2. Single GPU Test (A6000, 10B tokens) ==="
echo "cd /home/achalamlasetty/mscproj/NLP"
echo "$PYTHON_BIN pure_transformer/train_multigpu.py \\"
echo "  --model medium \\"
echo "  --total-tokens 10000000000 \\"
echo "  --fineweb-subset sample-10BT \\"
echo "  --devices 1 \\"
echo "  --micro-batch-size 8 \\"
echo "  --checkpoint-dir ./checkpoints/test_10b"
echo ""

echo "=== 3. Production Training (8x A100, 35B tokens) ==="
echo "cd /home/achalamlasetty/mscproj/NLP"
echo "$PYTHON_BIN pure_transformer/train_multigpu.py \\"
echo "  --model xlarge \\"
echo "  --total-tokens 35000000000 \\"
echo "  --fineweb-subset sample-100BT \\"
echo "  --devices 8 \\"
echo "  --micro-batch-size 8 \\"
echo "  --global-batch-size 262144 \\"
echo "  --fineweb-prob 0.60 \\"
echo "  --finepdf-prob 0.30 \\"
echo "  --usmle-prob 0.10 \\"
echo "  --num-workers 8 \\"
echo "  --checkpoint-dir ./checkpoints/xlarge_35b \\"
echo "  --use-wandb"
echo ""

echo "=== 4. Extended Training (8x A100, 50B tokens with full FinePDFs) ==="
echo "cd /home/achalamlasetty/mscproj/NLP"
echo "$PYTHON_BIN pure_transformer/train_multigpu.py \\"
echo "  --model xlarge \\"
echo "  --total-tokens 50000000000 \\"
echo "  --fineweb-subset sample-100BT \\"
echo "  --devices 8 \\"
echo "  --micro-batch-size 8 \\"
echo "  --global-batch-size 262144 \\"
echo "  --fineweb-prob 0.50 \\"
echo "  --finepdf-prob 0.40 \\"
echo "  --usmle-prob 0.10 \\"
echo "  --checkpoint-dir ./checkpoints/xlarge_50b \\"
echo "  --use-wandb"
echo ""

echo "========================================="
echo "Performance Estimates (8x A100 80GB)"
echo "========================================="
echo ""
echo "Model: XLarge (1.3B params)"
echo "Expected throughput: 1.0-1.5M tokens/sec"
echo ""
echo "Training Duration:"
echo "  • 35B tokens: ~6-8 hours"
echo "  • 50B tokens: ~9-11 hours"
echo ""
echo "Well within 2-day target! ✓"
echo ""

echo "========================================="
echo "Monitoring Training"
echo "========================================="
echo ""
echo "1. Watch GPU utilization:"
echo "   watch -n 1 nvidia-smi"
echo ""
echo "2. TensorBoard (if not using wandb):"
echo "   tensorboard --logdir ./checkpoints/xlarge_35b/lightning_logs"
echo ""
echo "3. Check training logs:"
echo "   tail -f ./checkpoints/xlarge_35b/train.log"
echo ""

echo "========================================="
echo "Dataset Verification"
echo "========================================="
echo ""
echo "Run this to verify all datasets are accessible:"
echo "$PYTHON_BIN pure_transformer/test_datasets.py"
echo ""

echo "Ready to train! 🚀"
