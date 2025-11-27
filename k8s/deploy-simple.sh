#!/bin/bash
# Simple deploy script - No Docker required
# Uses pre-built PyTorch image and clones repo at runtime
set -e

NAMESPACE="ucsdfutures"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=========================================="
echo "  Hybrid LLM Deploy (No Docker Required) "
echo "=========================================="
echo ""
echo "This uses a pre-built PyTorch image and installs dependencies at runtime."
echo ""

# Check kubectl
if ! kubectl auth can-i create jobs -n $NAMESPACE &>/dev/null; then
    echo "ERROR: Cannot create jobs in namespace $NAMESPACE"
    exit 1
fi
echo "✓ Kubernetes access verified"

# Get API keys
echo ""
echo "Enter your API credentials:"
echo ""

echo "--- ClearML ---"
echo "(Get credentials from: app.clear.ml -> Settings -> Workspace -> Create credentials)"
read -p "ClearML Access Key: " CLEARML_KEY
read -sp "ClearML Secret Key: " CLEARML_SECRET
echo ""

echo ""
echo "--- Hugging Face ---"
echo "(Get token from: huggingface.co -> Settings -> Access Tokens)"
read -sp "HF Token: " HF_TOKEN
echo ""

echo ""
echo "--- WandB (optional, press Enter to skip) ---"
read -sp "WandB API Key: " WANDB_KEY
echo ""
WANDB_KEY=${WANDB_KEY:-"not-configured"}

# Create secrets
echo ""
echo "Creating Kubernetes secret..."

kubectl create secret generic hybrid-llm-secrets \
    --namespace=$NAMESPACE \
    --from-literal=clearml-web-host="https://app.clear.ml" \
    --from-literal=clearml-api-host="https://api.clear.ml" \
    --from-literal=clearml-files-host="https://files.clear.ml" \
    --from-literal=clearml-api-access-key="$CLEARML_KEY" \
    --from-literal=clearml-api-secret-key="$CLEARML_SECRET" \
    --from-literal=hf-token="$HF_TOKEN" \
    --from-literal=wandb-api-key="$WANDB_KEY" \
    --dry-run=client -o yaml | kubectl apply -f -

echo "✓ Secret created/updated"

# Deploy job
echo ""
echo "Deploying training job..."
kubectl apply -f "$SCRIPT_DIR/pretrain-job-simple.yaml"
echo "✓ Job submitted"

# Show status
echo ""
echo "=========================================="
echo "  Deployment Complete!"
echo "=========================================="
echo ""
echo "The pod will:"
echo "  1. Clone your repo from GitHub"
echo "  2. Install dependencies (flash-attn, mamba-ssm, etc.)"
echo "  3. Start pretraining"
echo ""
echo "This takes ~10-15 minutes for initial setup."
echo ""
echo "Monitor commands:"
echo "  kubectl get pods -n $NAMESPACE -l app=hybrid-llm -w"
echo "  kubectl logs -n $NAMESPACE -l app=hybrid-llm -f --all-containers"
echo ""
echo "Delete job (to restart):"
echo "  kubectl delete job hybrid-llm-pretrain -n $NAMESPACE"
echo ""

# Show initial status
sleep 2
echo "Current status:"
kubectl get pods -n $NAMESPACE -l app=hybrid-llm 2>/dev/null || echo "(Pod scheduling...)"
