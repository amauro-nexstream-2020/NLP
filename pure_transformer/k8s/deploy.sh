#!/bin/bash
# Quick deployment script for Pure Transformer on K8s

set -e

# Location helpers — make this script invocable from anywhere
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PURE_K8S_DIR="$REPO_ROOT/pure_transformer/k8s"

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

# For local environments with docker installed, use docker
if command -v docker >/dev/null 2>&1; then
    docker build -t $REGISTRY/pure-transformer:$TAG -f k8s/Dockerfile ..
    echo "Pushing Docker image..."
    docker push $REGISTRY/pure-transformer:$TAG

# Podman is an alternative that provides a docker CLI replacement
elif command -v podman >/dev/null 2>&1; then
    podman build -t $REGISTRY/pure-transformer:$TAG -f k8s/Dockerfile ..
    echo "Pushing Docker image with podman..."
    podman push $REGISTRY/pure-transformer:$TAG

# If neither Docker nor Podman are present, run an in-cluster build using Kaniko
else
    echo "Docker/Podman not found. Falling back to in-cluster Kaniko build."
    # Kaniko requires the registry secret to be present. Assume it's called 'regcred'.
    # Create a simple Kaniko Pod to build and push the image from the repo context.
    cat > "$PURE_K8S_DIR/kaniko-pod.yaml" <<EOF
apiVersion: v1
kind: Pod
metadata:
    name: kaniko-build
    namespace: ${NAMESPACE}
spec:
    restartPolicy: Never
    containers:
    - name: kaniko
        image: gcr.io/kaniko-project/executor:debug
        args:
            - "--context=git://github.com/amauro-nexstream-2020/NLP.git#refs/heads/mamba-hybrid"
            - "--dockerfile=k8s/Dockerfile"
            - "--destination=${REGISTRY}/pure-transformer:${TAG}"
            - "--insecure"
        env:
            - name: DOCKER_CONFIG
                value: /kaniko/.docker
        volumeMounts:
            - name: kaniko-secret
                mountPath: /kaniko/.docker
    volumes:
        - name: kaniko-secret
            secret:
                secretName: regcred
EOF

    # Ensure the regcred secret exists; otherwise provide an example
    if ! kubectl get secret regcred -n $NAMESPACE >/dev/null 2>&1; then
        echo "Warning: Kubernetes image registry secret 'regcred' not found in namespace $NAMESPACE."
        echo "Create it using:"
        echo "  kubectl create secret docker-registry regcred --docker-server=<REGISTRY> --docker-username=<USER> --docker-password=<PASSWORD> --docker-email=<EMAIL> -n $NAMESPACE"
        echo "Exiting."
        exit 1
    fi

    echo "Applying Kaniko Pod (this will run in the cluster and push to your registry)."
    kubectl apply -f "$PURE_K8S_DIR/kaniko-pod.yaml" -n $NAMESPACE
    echo "Waiting for Kaniko Pod to complete..."
    kubectl wait --for=condition=Succeeded pod/kaniko-build -n $NAMESPACE --timeout=900s || {
        echo "Kaniko build failed, check logs: kubectl logs -n ${NAMESPACE} pod/kaniko-build"; exit 1
    }
    echo "Kaniko build finished. Pushed ${REGISTRY}/pure-transformer:${TAG}"
    kubectl delete pod -n $NAMESPACE kaniko-build --ignore-not-found=true || true
fi

# Update image in manifests
echo ""
echo "Updating manifests..."
# Use sed -i.bak for compatibility (works on macOS & GNU sed)
sed -i.bak "s|<YOUR_REGISTRY>|$REGISTRY|g" "$PURE_K8S_DIR/pretrain-job.yaml"
sed -i.bak "s|<YOUR_REGISTRY>|$REGISTRY|g" "$PURE_K8S_DIR/grpo-job.yaml"
rm -f "$PURE_K8S_DIR/pretrain-job.yaml.bak" "$PURE_K8S_DIR/grpo-job.yaml.bak" || true
sed -i.bak "s|:latest|:${TAG}|g" "$PURE_K8S_DIR/pretrain-job.yaml"
sed -i.bak "s|:latest|:${TAG}|g" "$PURE_K8S_DIR/grpo-job.yaml"
rm -f "$PURE_K8S_DIR/pretrain-job.yaml.bak" "$PURE_K8S_DIR/grpo-job.yaml.bak" || true

# Create namespace if needed
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Apply secrets (if template exists)
if [ -f "$PURE_K8S_DIR/secrets.yaml" ]; then
    echo "Applying secrets..."
    kubectl apply -f "$PURE_K8S_DIR/secrets.yaml" -n $NAMESPACE
fi

# Optionally create docker registry secrets in the namespace when provided
if [ -n "$DOCKER_USERNAME" ] && [ -n "$DOCKER_PASSWORD" ] && [ -n "$REGISTRY" ]; then
    echo "Creating registry secrets in $NAMESPACE (regcred & registry-credentials)"
    kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
    kubectl create secret docker-registry regcred \
        --docker-server=${REGISTRY} \
        --docker-username=${DOCKER_USERNAME} \
        --docker-password=${DOCKER_PASSWORD} \
        --docker-email=${DOCKER_EMAIL:-none} \
        -n $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

    kubectl create secret docker-registry registry-credentials \
        --docker-server=${REGISTRY} \
        --docker-username=${DOCKER_USERNAME} \
        --docker-password=${DOCKER_PASSWORD} \
        --docker-email=${DOCKER_EMAIL:-none} \
        -n $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
fi

# Apply PVC
echo ""
echo "Creating PVC..."
kubectl apply -f "$PURE_K8S_DIR/pretrain-job.yaml" -n $NAMESPACE --dry-run=client -o yaml | \
    grep -A 20 "kind: PersistentVolumeClaim" | kubectl apply -f - -n $NAMESPACE || true

# Start pretraining job
echo ""
echo "Starting pretraining job..."
kubectl apply -f "$PURE_K8S_DIR/pretrain-job.yaml" -n $NAMESPACE

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
echo "  kubectl apply -f $PURE_K8S_DIR/grpo-job.yaml -n $NAMESPACE"
