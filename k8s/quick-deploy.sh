#!/bin/bash
# Quick deployment script for Hybrid LLM on Nautilus
# Prompts for API keys and deploys immediately
set -e

NAMESPACE="ucsdfutures"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "  Hybrid LLM Quick Deploy on Nautilus    "
echo "=========================================="
echo ""
echo "Namespace: $NAMESPACE"
echo "Project: $PROJECT_DIR"
echo ""

# Step 1: Get API Keys
echo "Enter your API credentials (press Enter to use defaults where shown):"
echo ""

echo "--- ClearML (get from app.clear.ml -> Settings -> Workspace -> Create credentials) ---"
read -p "ClearML Web Host [https://app.clear.ml]: " CLEARML_WEB
CLEARML_WEB=${CLEARML_WEB:-"https://app.clear.ml"}

read -p "ClearML API Host [https://api.clear.ml]: " CLEARML_API
CLEARML_API=${CLEARML_API:-"https://api.clear.ml"}

read -p "ClearML Files Host [https://files.clear.ml]: " CLEARML_FILES
CLEARML_FILES=${CLEARML_FILES:-"https://files.clear.ml"}

read -p "ClearML Access Key: " CLEARML_KEY
read -sp "ClearML Secret Key: " CLEARML_SECRET
echo ""

echo ""
echo "--- Hugging Face (get from huggingface.co -> Settings -> Access Tokens) ---"
read -sp "HF Token: " HF_TOKEN
echo ""

echo ""
echo "--- WandB (optional, press Enter to skip) ---"
read -sp "WandB API Key: " WANDB_KEY
echo ""
WANDB_KEY=${WANDB_KEY:-"not-configured"}

echo ""
echo "--- Container Registry ---"
read -p "Registry (e.g., docker.io/username, ghcr.io/username): " REGISTRY

# Step 2: Create secrets YAML
cat > "$SCRIPT_DIR/secrets.yaml" << EOF
apiVersion: v1
kind: Secret
metadata:
  name: hybrid-llm-secrets
  namespace: $NAMESPACE
type: Opaque
stringData:
  clearml-web-host: "$CLEARML_WEB"
  clearml-api-host: "$CLEARML_API"
  clearml-files-host: "$CLEARML_FILES"
  clearml-api-access-key: "$CLEARML_KEY"
  clearml-api-secret-key: "$CLEARML_SECRET"
  hf-token: "$HF_TOKEN"
  wandb-api-key: "$WANDB_KEY"
EOF

echo ""
echo "✓ Secrets file created"

# Step 3: Build and push Docker image
FULL_IMAGE="${REGISTRY}/hybrid-llm:latest"
echo ""
echo "Building Docker image: $FULL_IMAGE"
cd "$PROJECT_DIR"
docker build -t "$FULL_IMAGE" -f k8s/Dockerfile .

echo "Pushing to registry..."
docker push "$FULL_IMAGE"
echo "✓ Image pushed"

# Step 4: Update job manifest with image
sed -i.bak "s|image: <YOUR_REGISTRY>/hybrid-llm:latest|image: ${FULL_IMAGE}|g" "$SCRIPT_DIR/pretrain-job.yaml"

# Step 5: Deploy
echo ""
echo "Deploying to Kubernetes..."

kubectl apply -f "$SCRIPT_DIR/secrets.yaml"
echo "✓ Secrets applied"

kubectl apply -f "$SCRIPT_DIR/pretrain-job.yaml"
echo "✓ Job created"

# Step 6: Show status
echo ""
echo "=========================================="
echo "  Deployment Complete!"
echo "=========================================="
echo ""
echo "Monitor with:"
echo "  kubectl get pods -n $NAMESPACE -l app=hybrid-llm -w"
echo ""
echo "View logs:"
echo "  kubectl logs -n $NAMESPACE -l app=hybrid-llm -f"
echo ""

# Wait and show initial status
sleep 3
echo "Current status:"
kubectl get pods -n $NAMESPACE -l app=hybrid-llm 2>/dev/null || echo "Pod starting..."
