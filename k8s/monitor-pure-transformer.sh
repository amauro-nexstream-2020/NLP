#!/bin/bash
# Monitor Pure Transformer Training on Kubernetes

NAMESPACE="ucsdfutures"
APP_LABEL="app=pure-transformer,task=pretraining"

echo "======================================"
echo "Pure Transformer Training Monitor"
echo "======================================"
echo ""

# Check job status
echo "📊 Job Status:"
kubectl get jobs -n $NAMESPACE | grep pure-transformer
echo ""

# Check pod status
echo "🔷 Pod Status:"
kubectl get pods -n $NAMESPACE -l $APP_LABEL
echo ""

# Get pod name
POD_NAME=$(kubectl get pods -n $NAMESPACE -l $APP_LABEL -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

if [ -n "$POD_NAME" ]; then
    POD_STATUS=$(kubectl get pod -n $NAMESPACE $POD_NAME -o jsonpath='{.status.phase}')
    echo "📦 Pod: $POD_NAME (Status: $POD_STATUS)"
    echo ""
    
    if [ "$POD_STATUS" == "Pending" ]; then
        echo "⏳ Pod is pending. Checking events..."
        kubectl describe pod -n $NAMESPACE $POD_NAME | grep -A 5 "Events:"
        echo ""
    elif [ "$POD_STATUS" == "Running" ]; then
        echo "✅ Pod is running!"
        echo ""
        echo "📝 Recent logs (last 20 lines):"
        kubectl logs -n $NAMESPACE $POD_NAME --tail=20
        echo ""
        echo "💡 To follow logs in real-time:"
        echo "   kubectl logs -f -n $NAMESPACE $POD_NAME"
    elif [ "$POD_STATUS" == "Failed" ] || [ "$POD_STATUS" == "Error" ]; then
        echo "❌ Pod failed. Checking logs..."
        kubectl logs -n $NAMESPACE $POD_NAME --tail=50
        echo ""
    fi
else
    echo "⚠️  No pod found yet. Job may still be scheduling..."
fi

echo ""
echo "======================================"
echo "Useful Commands:"
echo "======================================"
echo "Follow logs:       kubectl logs -f -n $NAMESPACE \$POD_NAME"
echo "Get pod details:   kubectl describe pod -n $NAMESPACE \$POD_NAME"
echo "Check PVCs:        kubectl get pvc -n $NAMESPACE"
echo "Delete job:        kubectl delete job -n $NAMESPACE pure-transformer-a100-8gpu"
echo "Check checkpoints: kubectl exec -n $NAMESPACE \$POD_NAME -- ls -lh /checkpoints/pure_transformer/"
echo ""
