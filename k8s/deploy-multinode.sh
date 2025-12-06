#!/bin/bash
set -e

echo "=========================================="
echo "Pure Transformer Multi-Node Deployment"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  - 2 nodes × 4 A100 GPUs = 8 total GPUs"
echo "  - Model: xlarge (1.3B parameters)"
echo "  - Tokens: 100B"
echo "  - Strategy: Multi-node DDP"
echo "  - Precision: bf16-mixed"
echo ""

# Check cluster connection
echo "Checking cluster connection..."
kubectl cluster-info | head -n 1

# Check namespace
echo "Checking namespace..."
kubectl get namespace ucsdfutures

# Check PVCs
echo ""
echo "Checking PVCs..."
kubectl get pvc -n ucsdfutures | grep pure-transformer || echo "No PVCs found, will be created"

# Apply PVCs if not exist
if ! kubectl get pvc pure-transformer-checkpoints -n ucsdfutures &>/dev/null; then
  echo ""
  echo "Creating checkpoints PVC..."
  cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: pure-transformer-checkpoints
  namespace: ucsdfutures
spec:
  accessModes:
  - ReadWriteMany
  resources:
    requests:
      storage: 100Gi
  storageClassName: rook-cephfs
EOF
fi

if ! kubectl get pvc pure-transformer-hf-cache -n ucsdfutures &>/dev/null; then
  echo ""
  echo "Creating HuggingFace cache PVC..."
  cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: pure-transformer-hf-cache
  namespace: ucsdfutures
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 200Gi
  storageClassName: rook-ceph-block
EOF
fi

# Wait for PVCs
echo ""
echo "Waiting for PVCs to be bound..."
kubectl wait --for=jsonpath='{.status.phase}'=Bound \
  pvc/pure-transformer-checkpoints \
  -n ucsdfutures \
  --timeout=120s || echo "Checkpoints PVC may take longer to bind"

kubectl wait --for=jsonpath='{.status.phase}'=Bound \
  pvc/pure-transformer-hf-cache \
  -n ucsdfutures \
  --timeout=120s || echo "HF cache PVC may take longer to bind"

# Delete old job if exists
if kubectl get job pure-transformer-multinode -n ucsdfutures &>/dev/null; then
  echo ""
  echo "Deleting old job..."
  kubectl delete job pure-transformer-multinode -n ucsdfutures
  sleep 5
fi

# Delete old service if exists
if kubectl get service pure-transformer-multinode-master -n ucsdfutures &>/dev/null; then
  echo ""
  echo "Deleting old service..."
  kubectl delete service pure-transformer-multinode-master -n ucsdfutures
  sleep 2
fi

# Apply job
echo ""
echo "Deploying multi-node job..."
kubectl apply -f k8s/pure-transformer-multinode-job.yaml

echo ""
echo "=========================================="
echo "Deployment initiated!"
echo "=========================================="
echo ""
echo "Monitor with:"
echo "  kubectl get pods -n ucsdfutures -l job-name=pure-transformer-multinode -w"
echo ""
echo "Check logs:"
echo "  # Master node (rank 0):"
echo "  kubectl logs -n ucsdfutures -l job-name=pure-transformer-multinode,batch.kubernetes.io/job-completion-index=0 -f"
echo ""
echo "  # Worker node (rank 1):"
echo "  kubectl logs -n ucsdfutures -l job-name=pure-transformer-multinode,batch.kubernetes.io/job-completion-index=1 -f"
echo ""
echo "Check job status:"
echo "  kubectl describe job pure-transformer-multinode -n ucsdfutures"
echo ""
echo "Live training metrics:"
echo "  - W&B: https://wandb.ai/your-entity/pure-transformer"
echo "  - TensorBoard: kubectl port-forward -n ucsdfutures <pod-name> 6006:6006"
echo "  - GPU stats: kubectl exec -n ucsdfutures <pod-name> -- nvidia-smi"
echo ""
