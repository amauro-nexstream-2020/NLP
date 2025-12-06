# Pure Transformer Deployment Status

## ✅ Successfully Deployed

Your Pure Transformer training jobs have been successfully deployed to the Nautilus Kubernetes cluster!

### Deployment Details

**Cluster**: Nautilus (nautilus context)  
**Namespace**: ucsdfutures  
**Jobs Created**:
- `pure-transformer-a100-8gpu` (8×A100 GPUs)
- `pure-transformer-a100-9gpu` (9×A100 GPUs)

**Pod Status**: Currently **Pending** - waiting for node with 8 A100 GPUs to become available

### What's Happening Now

The scheduler is looking for a node with:
- 8 A100 GPUs available
- 24 CPU cores
- 120Gi memory
- Access to the PVCs

**Available 8-GPU nodes**:
- node-1-1.sdsc.optiputer.net
- node-1-3.sdsc.optiputer.net
- node-1-4.sdsc.optiputer.net
- node-2-1.sdsc.optiputer.net
- node-2-2.sdsc.optiputer.net
- node-2-3.sdsc.optiputer.net

Your pod will automatically start once resources are available. This is normal in shared clusters.

### Storage Configuration

**Checkpoints PVC**: `pure-transformer-checkpoints` (100Gi, rook-cephfs) - *already existed*  
**HF Cache PVC**: `pure-transformer-hf-cache` (200Gi, rook-ceph-block) - *newly created*

Checkpoints will be saved to `/checkpoints/pure_transformer/` every 500 steps.

### Training Configuration

- **Model**: xlarge (1.3B parameters)
- **Total Tokens**: 100B tokens
- **Global Batch Size**: 524,288 tokens
- **Micro Batch Size**: 16
- **Sequence Length**: 2048
- **Precision**: bf16-mixed
- **Strategy**: DDP (DistributedDataParallel)
- **Checkpoint Frequency**: Every 500 steps
- **Logging**: Every 10 steps to W&B and TensorBoard

### Expected Performance

With 8×A100 GPUs @ ~200k tokens/sec/GPU:
- Throughput: ~1.6M tokens/sec
- Time for 100B tokens: ~17 hours

### Monitoring Commands

```bash
# Check pod status
kubectl get pods -n ucsdfutures -l app=pure-transformer,task=pretraining

# Watch pod until it starts
kubectl get pods -n ucsdfutures -l app=pure-transformer,task=pretraining -w

# Once running, follow logs
kubectl logs -f -n ucsdfutures pure-transformer-a100-8gpu-65gwl

# Check detailed pod info
kubectl describe pod -n ucsdfutures pure-transformer-a100-8gpu-65gwl

# Quick monitoring script
bash k8s/monitor-pure-transformer.sh
```

### Access Training Logs

Once the pod is running:

**W&B Dashboard**: https://wandb.ai (project: `pure-transformer`)  
**TensorBoard Logs**: `/checkpoints/pure_transformer/lightning_logs/` on PVC

### Troubleshooting

If the pod stays pending for >1 hour:
```bash
# Check what's blocking scheduling
kubectl describe pod -n ucsdfutures pure-transformer-a100-8gpu-65gwl | grep -A 10 Events

# Check if node resources are available
kubectl get nodes -o json | jq -r '.items[] | select(.status.allocatable."nvidia.com/a100" == "8") | .metadata.name'
```

### Cleanup (when finished)

```bash
# Stop the training job
kubectl delete job -n ucsdfutures pure-transformer-a100-8gpu

# Optional: Delete PVCs to free storage
kubectl delete pvc -n ucsdfutures pure-transformer-hf-cache
# Note: pure-transformer-checkpoints was pre-existing, consider keeping it
```

### Next Steps

1. **Monitor**: Keep checking pod status until it moves to Running
2. **Watch logs**: Once running, follow logs to verify training starts
3. **Check W&B**: Monitor training metrics in real-time
4. **Verify checkpoints**: After ~500 steps, check if checkpoints are being saved

The job is configured to:
- Resume from `/checkpoints/pure_transformer/last.ckpt` if it exists
- Save checkpoints every 500 steps
- Run for up to 48 hours (activeDeadlineSeconds)
- Retry up to 2 times if it fails (backoffLimit)

---

**Status**: ⏳ Waiting for cluster resources  
**Action Required**: None - the job will start automatically when resources are available
