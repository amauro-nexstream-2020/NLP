#!/bin/bash
# Monitor script for 8xA100 Pure Transformer training

echo "================================================"
echo "Pure Transformer 8xA100 Multi-Node Monitor"
echo "================================================"
echo ""

# Job status
echo "=== Job Status ==="
kubectl get job pure-transformer-8xa100 -n ucsdfutures
echo ""

# Pod status
echo "=== Pod Status ==="
kubectl get pods -n ucsdfutures -l job-name=pure-transformer-8xa100 -o wide
echo ""

# Check which nodes and GPUs
echo "=== Node Distribution ==="
kubectl get pods -n ucsdfutures -l job-name=pure-transformer-8xa100 -o json | \
  jq -r '.items[] | "\(.metadata.name): \(.spec.nodeName)"' | sort
echo ""

# Get a quick log sample from master node (rank 0)
echo "=== Master Node (Rank 0) Latest Logs ==="
MASTER_POD=$(kubectl get pods -n ucsdfutures -l job-name=pure-transformer-8xa100,batch.kubernetes.io/job-completion-index=0 -o jsonpath='{.items[0].metadata.name}')
if [ ! -z "$MASTER_POD" ]; then
  echo "Pod: $MASTER_POD"
  kubectl logs -n ucsdfutures $MASTER_POD --tail=30 2>/dev/null || echo "Logs not ready yet"
else
  echo "Master pod not found or not ready"
fi

echo ""
echo "================================================"
echo "Live Monitoring Commands:"
echo "================================================"
echo ""
echo "# Watch all pods:"
echo "  kubectl get pods -n ucsdfutures -l job-name=pure-transformer-8xa100 -w"
echo ""
echo "# Master node logs (rank 0):"
echo "  kubectl logs -n ucsdfutures -l job-name=pure-transformer-8xa100,batch.kubernetes.io/job-completion-index=0 -f"
echo ""
echo "# Worker node logs (rank 1, 2, 3):"
echo "  kubectl logs -n ucsdfutures -l job-name=pure-transformer-8xa100,batch.kubernetes.io/job-completion-index=1 -f"
echo ""
echo "# Check GPU usage on a pod:"
echo "  kubectl exec -n ucsdfutures <pod-name> -- nvidia-smi"
echo ""
echo "# W&B dashboard:"
echo "  https://wandb.ai/your-entity/pure-transformer/runs"
echo ""
echo "# Get training metrics:"
echo "  kubectl logs -n ucsdfutures -l job-name=pure-transformer-8xa100,batch.kubernetes.io/job-completion-index=0 -f | grep -E '(loss|tokens/sec|step)'"
echo ""
