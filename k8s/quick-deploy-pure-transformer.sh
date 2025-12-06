#!/bin/bash
# Quick Deploy Script for Pure Transformer on Kubernetes

set -e

NAMESPACE="ucsdfutures"
JOB_NAME="pure-transformer-a100-8gpu"

echo "🚀 Pure Transformer Kubernetes Deployment"
echo "=========================================="
echo ""

# Check cluster connection
echo "📡 Checking cluster connection..."
if ! kubectl cluster-info &>/dev/null; then
    echo "❌ Not connected to Kubernetes cluster"
    exit 1
fi
echo "✅ Connected to: $(kubectl config current-context)"
echo ""

# Check namespace
echo "📦 Checking namespace..."
if ! kubectl get namespace $NAMESPACE &>/dev/null; then
    echo "❌ Namespace $NAMESPACE not found"
    exit 1
fi
echo "✅ Namespace $NAMESPACE exists"
echo ""

# Check secrets
echo "🔐 Checking secrets..."
if ! kubectl get secret -n $NAMESPACE hybrid-llm-secrets &>/dev/null; then
    echo "❌ Secret hybrid-llm-secrets not found"
    exit 1
fi
echo "✅ Secrets configured"
echo ""

# Apply manifest
echo "📄 Applying Kubernetes manifest..."
kubectl apply -f k8s/pure-transformer-a100-job.yaml
echo ""

# Check job status
echo "✅ Deployment complete!"
echo ""
echo "📊 Current status:"
kubectl get jobs -n $NAMESPACE | grep pure-transformer || true
echo ""
kubectl get pods -n $NAMESPACE -l app=pure-transformer,task=pretraining
echo ""

# Get pod name
POD_NAME=$(kubectl get pods -n $NAMESPACE -l app=pure-transformer,task=pretraining -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")

if [ -n "$POD_NAME" ]; then
    POD_STATUS=$(kubectl get pod -n $NAMESPACE $POD_NAME -o jsonpath='{.status.phase}')
    echo "Pod: $POD_NAME"
    echo "Status: $POD_STATUS"
    echo ""
    
    if [ "$POD_STATUS" == "Pending" ]; then
        echo "⏳ Pod is pending - waiting for cluster resources"
        echo ""
        echo "💡 Monitor with:"
        echo "   kubectl get pods -n $NAMESPACE -l app=pure-transformer -w"
    elif [ "$POD_STATUS" == "Running" ]; then
        echo "✅ Pod is running!"
        echo ""
        echo "💡 Follow logs with:"
        echo "   kubectl logs -f -n $NAMESPACE $POD_NAME"
    fi
else
    echo "⚠️  Pod not found yet - still initializing"
fi

echo ""
echo "=========================================="
echo "📚 Useful Commands:"
echo "=========================================="
echo ""
echo "Monitor status:"
echo "  bash k8s/monitor-pure-transformer.sh"
echo ""
echo "Follow logs (once running):"
echo "  kubectl logs -f -n $NAMESPACE \$POD_NAME"
echo ""
echo "Check job status:"
echo "  kubectl get jobs -n $NAMESPACE"
echo ""
echo "Delete job (cleanup):"
echo "  kubectl delete job -n $NAMESPACE $JOB_NAME"
echo ""
