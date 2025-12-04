#!/bin/bash
# Quick deployment script for Pure Transformer on K8s

set -e

# Configuration
NAMESPACE="${NAMESPACE:-ucsdfutures}"
REGISTRY="${REGISTRY:-<YOUR_REGISTRY>}"
TAG="${TAG:-latest}"

echo "=== Pure Transformer K8s Deployment ==="
echo "Namespace: $NAMESPACE"
echo "Registry: $REGISTRY"
echo "Tag: $TAG"

# Build and push Docker image
echo ""
echo "Building Docker image..."
docker build -t $REGISTRY/pure-transformer:$TAG -f k8s/Dockerfile ..

echo "Pushing Docker image..."
docker push $REGISTRY/pure-transformer:$TAG

# Update image in manifests
echo ""
echo "Updating manifests..."
sed -i "s|<YOUR_REGISTRY>|$REGISTRY|g" k8s/pretrain-job.yaml
sed -i "s|<YOUR_REGISTRY>|$REGISTRY|g" k8s/grpo-job.yaml

# Create namespace if needed
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Apply secrets (if template exists)
if [ -f "k8s/secrets.yaml" ]; then
    echo "Applying secrets..."
    kubectl apply -f k8s/secrets.yaml -n $NAMESPACE
fi

# Apply PVC
echo ""
echo "Creating PVC..."
kubectl apply -f k8s/pretrain-job.yaml -n $NAMESPACE --dry-run=client -o yaml | \
    grep -A 20 "kind: PersistentVolumeClaim" | kubectl apply -f - -n $NAMESPACE || true

# Start pretraining job
echo ""
echo "Starting pretraining job..."
kubectl apply -f k8s/pretrain-job.yaml -n $NAMESPACE

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "Monitor with:"
echo "  kubectl logs -f job/pure-transformer-pretrain -n $NAMESPACE"
echo ""
echo "Check status:"
echo "  kubectl get pods -n $NAMESPACE -l app=pure-transformer"
echo ""
echo "After pretraining, run GRPO with:"
echo "  kubectl apply -f k8s/grpo-job.yaml -n $NAMESPACE"
