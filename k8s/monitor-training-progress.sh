#!/bin/bash
# Monitor training progress across all pods

echo "==================================="
echo "DDP Training Monitor - 8x A100"
echo "==================================="
echo ""

# Check job status
echo "📊 Job Status:"
kubectl get jobs -n ucsdfutures -l job-name=pure-transformer-8xa100 -o wide
echo ""

# Check pod status
echo "🔧 Pod Status:"
kubectl get pods -n ucsdfutures -l job-name=pure-transformer-8xa100 -o wide
echo ""

# Get logs from each pod for training progress
echo "📝 Training Progress (recent logs):"
for pod in $(kubectl get pods -n ucsdfutures -l job-name=pure-transformer-8xa100 -o name | cut -d'/' -f2); do
    echo ""
    echo ">>> Pod: $pod <<<"
    kubectl logs -n ucsdfutures $pod -c trainer --tail=20 2>/dev/null | grep -E "STARTING TRAINING|Initializing distributed|MEMBER|tokens/sec|Step [0-9]+|loss|throughput|GPU" | tail -10 || echo "  (Still initializing...)"
done

echo ""
echo "==================================="
echo "🎯 GPU Utilization Check"
echo "==================================="

# Check GPU utilization on each pod
for pod in $(kubectl get pods -n ucsdfutures -l job-name=pure-transformer-8xa100 -o name | cut -d'/' -f2); do
    echo ""
    echo ">>> Pod: $pod <<<"
    kubectl exec -n ucsdfutures $pod -c trainer -- nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "  (Cannot query GPU yet)"
done

echo ""
echo "==================================="
echo "Run this script again with: bash k8s/monitor-training-progress.sh"
echo "Or watch continuously: watch -n 30 bash k8s/monitor-training-progress.sh"
echo "==================================="
