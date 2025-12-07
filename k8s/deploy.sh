#!/bin/bash
# Deployment script for Hybrid LLM Pretraining on Kubernetes
# UCSD Futures Namespace with A100 GPU
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

NAMESPACE="ucsdfutures"
IMAGE_NAME="hybrid-llm"
IMAGE_TAG="latest"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Hybrid LLM Pretraining Deployment${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"

if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}Error: kubectl not found. Please install kubectl first.${NC}"
    exit 1
fi

if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: docker not found. Please install docker first.${NC}"
    exit 1
fi

# Check kubernetes context
CURRENT_CONTEXT=$(kubectl config current-context 2>/dev/null || echo "none")
echo -e "${GREEN}Current kubectl context: ${CURRENT_CONTEXT}${NC}"
read -p "Is this the correct context for UCSD Futures? (y/n): " confirm
if [[ $confirm != "y" ]]; then
    echo "Please set the correct context with: kubectl config use-context <context-name>"
    exit 1
fi

# Check namespace access
if ! kubectl get namespace $NAMESPACE &> /dev/null; then
    echo -e "${RED}Error: Cannot access namespace '$NAMESPACE'${NC}"
    echo "Please ensure you have access to the ucsd-futures namespace"
    exit 1
fi
echo -e "${GREEN}✓ Namespace '$NAMESPACE' accessible${NC}"

# Step 1: Configure secrets
echo ""
echo -e "${YELLOW}Step 1: Configure API Secrets${NC}"
echo "==============================="

if [[ ! -f "k8s/secrets.yaml" ]]; then
    echo -e "${YELLOW}Creating secrets.yaml from template...${NC}"
    cp k8s/secrets.yaml.template k8s/secrets.yaml
    
    echo ""
    echo "Please enter your API credentials:"
    echo ""
    
    # ClearML
    echo -e "${BLUE}ClearML Configuration (get from app.clear.ml -> Settings -> Workspace)${NC}"
    read -p "ClearML Web Host [https://app.clear.ml]: " CLEARML_WEB_HOST
    CLEARML_WEB_HOST=${CLEARML_WEB_HOST:-"https://app.clear.ml"}
    
    read -p "ClearML API Host [https://api.clear.ml]: " CLEARML_API_HOST
    CLEARML_API_HOST=${CLEARML_API_HOST:-"https://api.clear.ml"}
    
    read -p "ClearML Files Host [https://files.clear.ml]: " CLEARML_FILES_HOST
    CLEARML_FILES_HOST=${CLEARML_FILES_HOST:-"https://files.clear.ml"}
    
    read -p "ClearML API Access Key: " CLEARML_ACCESS_KEY
    read -sp "ClearML API Secret Key: " CLEARML_SECRET_KEY
    echo ""
    
    # Hugging Face
    echo ""
    echo -e "${BLUE}Hugging Face Configuration (get from huggingface.co -> Settings -> Access Tokens)${NC}"
    read -sp "Hugging Face Token: " HF_TOKEN
    echo ""
    
    # WandB (optional)
    echo ""
    echo -e "${BLUE}WandB Configuration (optional - press Enter to skip)${NC}"
    read -sp "WandB API Key (optional): " WANDB_KEY
    echo ""
    WANDB_KEY=${WANDB_KEY:-"placeholder"}
    
    # Encode and update secrets.yaml
    echo -e "${YELLOW}Encoding secrets...${NC}"
    
    # Use sed to replace placeholders
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS sed
        sed -i '' "s|clearml-web-host: <BASE64_ENCODED_VALUE>|clearml-web-host: $(echo -n "$CLEARML_WEB_HOST" | base64)|" k8s/secrets.yaml
        sed -i '' "s|clearml-api-host: <BASE64_ENCODED_VALUE>|clearml-api-host: $(echo -n "$CLEARML_API_HOST" | base64)|" k8s/secrets.yaml
        sed -i '' "s|clearml-files-host: <BASE64_ENCODED_VALUE>|clearml-files-host: $(echo -n "$CLEARML_FILES_HOST" | base64)|" k8s/secrets.yaml
        sed -i '' "s|clearml-api-access-key: <BASE64_ENCODED_VALUE>|clearml-api-access-key: $(echo -n "$CLEARML_ACCESS_KEY" | base64)|" k8s/secrets.yaml
        sed -i '' "s|clearml-api-secret-key: <BASE64_ENCODED_VALUE>|clearml-api-secret-key: $(echo -n "$CLEARML_SECRET_KEY" | base64)|" k8s/secrets.yaml
        sed -i '' "s|hf-token: <BASE64_ENCODED_VALUE>|hf-token: $(echo -n "$HF_TOKEN" | base64)|" k8s/secrets.yaml
        sed -i '' "s|wandb-api-key: <BASE64_ENCODED_VALUE>|wandb-api-key: $(echo -n "$WANDB_KEY" | base64)|" k8s/secrets.yaml
    else
        # Linux sed
        sed -i "s|clearml-web-host: <BASE64_ENCODED_VALUE>|clearml-web-host: $(echo -n "$CLEARML_WEB_HOST" | base64)|" k8s/secrets.yaml
        sed -i "s|clearml-api-host: <BASE64_ENCODED_VALUE>|clearml-api-host: $(echo -n "$CLEARML_API_HOST" | base64)|" k8s/secrets.yaml
        sed -i "s|clearml-files-host: <BASE64_ENCODED_VALUE>|clearml-files-host: $(echo -n "$CLEARML_FILES_HOST" | base64)|" k8s/secrets.yaml
        sed -i "s|clearml-api-access-key: <BASE64_ENCODED_VALUE>|clearml-api-access-key: $(echo -n "$CLEARML_ACCESS_KEY" | base64)|" k8s/secrets.yaml
        sed -i "s|clearml-api-secret-key: <BASE64_ENCODED_VALUE>|clearml-api-secret-key: $(echo -n "$CLEARML_SECRET_KEY" | base64)|" k8s/secrets.yaml
        sed -i "s|hf-token: <BASE64_ENCODED_VALUE>|hf-token: $(echo -n "$HF_TOKEN" | base64)|" k8s/secrets.yaml
        sed -i "s|wandb-api-key: <BASE64_ENCODED_VALUE>|wandb-api-key: $(echo -n "$WANDB_KEY" | base64)|" k8s/secrets.yaml
    fi
    
    echo -e "${GREEN}✓ Secrets configured${NC}"
else
    echo -e "${GREEN}✓ secrets.yaml already exists${NC}"
    read -p "Do you want to reconfigure secrets? (y/n): " reconfig
    if [[ $reconfig == "y" ]]; then
        rm k8s/secrets.yaml
        exec "$0"  # Re-run script
    fi
fi

# Step 2: Build and push Docker image
echo ""
echo -e "${YELLOW}Step 2: Build and Push Docker Image${NC}"
echo "====================================="

read -p "Enter your container registry (e.g., docker.io/username or gcr.io/project): " REGISTRY
FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"

echo -e "${YELLOW}Building Docker image: ${FULL_IMAGE}${NC}"
docker build -t ${FULL_IMAGE} -f k8s/Dockerfile .

echo -e "${YELLOW}Pushing image to registry...${NC}"
docker push ${FULL_IMAGE}

echo -e "${GREEN}✓ Image pushed: ${FULL_IMAGE}${NC}"

# Update the job manifest with the correct image
if [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i '' "s|image: <YOUR_REGISTRY>/hybrid-llm:latest|image: ${FULL_IMAGE}|" k8s/pretrain-job.yaml
else
    sed -i "s|image: <YOUR_REGISTRY>/hybrid-llm:latest|image: ${FULL_IMAGE}|" k8s/pretrain-job.yaml
fi

# Step 3: Deploy to Kubernetes
echo ""
echo -e "${YELLOW}Step 3: Deploy to Kubernetes${NC}"
echo "=============================="

# Apply secrets
echo -e "${YELLOW}Applying secrets...${NC}"
kubectl apply -f k8s/secrets.yaml
echo -e "${GREEN}✓ Secrets applied${NC}"

# Apply PVCs and Job
echo -e "${YELLOW}Applying Kubernetes resources...${NC}"
kubectl apply -f k8s/pretrain-job.yaml
echo -e "${GREEN}✓ Resources applied${NC}"

# Step 4: Monitor deployment
echo ""
echo -e "${YELLOW}Step 4: Deployment Status${NC}"
echo "=========================="

echo "Waiting for pod to start..."
sleep 5

POD_NAME=$(kubectl get pods -n $NAMESPACE -l app=hybrid-llm,task=pretraining -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")

if [[ -z "$POD_NAME" ]]; then
    echo -e "${YELLOW}Pod is still being created. Check status with:${NC}"
    echo "  kubectl get pods -n $NAMESPACE -l app=hybrid-llm"
else
    echo -e "${GREEN}Pod created: ${POD_NAME}${NC}"
    kubectl get pod $POD_NAME -n $NAMESPACE
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Deployment Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Useful commands:"
echo -e "  ${BLUE}Watch pods:${NC}"
echo "    kubectl get pods -n $NAMESPACE -l app=hybrid-llm -w"
echo ""
echo -e "  ${BLUE}View logs:${NC}"
echo "    kubectl logs -n $NAMESPACE -l app=hybrid-llm -f"
echo ""
echo -e "  ${BLUE}Describe job:${NC}"
echo "    kubectl describe job hybrid-llm-pretrain -n $NAMESPACE"
echo ""
echo -e "  ${BLUE}Delete job (to restart):${NC}"
echo "    kubectl delete job hybrid-llm-pretrain -n $NAMESPACE"
echo ""
echo -e "  ${BLUE}Check GPU allocation:${NC}"
echo "    kubectl describe node | grep -A 5 'Allocated resources'"
