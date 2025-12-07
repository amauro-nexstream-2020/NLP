#!/usr/bin/env bash

# Auto-deploy single-node 8xA100 training job when a node with 8 free A100 GPUs and sufficient memory/CPU is available.
# Usage: ./auto_deploy_single_node.sh
# Note: Requires kubectl and jq installed and running in configured kubeconfig context.

set -euo pipefail

NAMESPACE=ucsdfutures
JOB_YAML=k8s/pure-transformer-8xa100-single-node.yaml
TMP_YAML=/tmp/pt-fast-single-node.yaml
REQUIRED_GPUS=8
REQUIRED_CPUS=64
REQUIRED_MEM_Gi=350

echo "Auto-deploy: looking for node with >=${REQUIRED_GPUS} free A100 GPUs and >=${REQUIRED_CPUS} CPUs and >=${REQUIRED_MEM_Gi}Gi memory free"

while true; do
  # Get list of nodes with A100 capacity
  nodes=$(kubectl get nodes -o json | jq -r '.items[] | select(.status.allocatable["nvidia.com/a100"] != null) | .metadata.name')

  for node in $nodes; do
    # capacity and allocatable
    cap_a100=$(kubectl get node $node -o json | jq -r '.status.capacity["nvidia.com/a100"] // "0"')
    alloc_a100=$(kubectl get node $node -o json | jq -r '.status.allocatable["nvidia.com/a100"] // "0"')

    # count used a100 on this node
    used=$(kubectl get pods --all-namespaces -o json | jq -r --arg node "$node" '.items[] | select(.spec.nodeName==$node) | (.spec.containers[].resources.requests["nvidia.com/a100"] // 0) | tonumber' | awk '{s+=$1} END{print s+0}')

    # convert to ints
    cap=${cap_a100:-0}
    used=${used:-0}
    free=$((cap - used))

    if [ "$free" -ge "$REQUIRED_GPUS" ]; then
      echo "Node $node has $free free A100 GPUs (cap $cap, used $used)"

      # check CPU and memory free
      # get allocatable vs allocated CPU and memory
      alloc_cpu=$(kubectl get node $node -o json | jq -r '.status.allocatable.cpu')
      alloc_mem=$(kubectl get node $node -o json | jq -r '.status.allocatable.memory')

      # allocated resources
      requested_cpu=$(kubectl describe node $node | awk '/Requests:/,/Limits:/' | grep cpu | head -1 | awk '{print $2}') || requested_cpu=0
      # Convert values (cpu in m -> to cores)
      if [[ "$alloc_cpu" == *m ]]; then
        alloc_cpu_cores=$(echo $alloc_cpu | sed 's/m$//')
        # m to cores later
      else
        alloc_cpu_cores=$(echo $alloc_cpu | sed 's/\..*$//')
      fi

      # Parse alloc_mem like 430075635200 (Ki?) - Kubernetes uses Ki
      # We'll use describe node allocated to compute available memory (Ki)
      allocated_mem_kib=$(kubectl describe node $node | awk '/Allocated resources:/,/Events:/' | grep memory | head -1 | awk '{print $2}' | sed 's/\(.*\)i//') || allocated_mem_kib=0

      # if allocated mem is 430075635200 -> this is in Ki not Gi. convert
      if [[ $allocated_mem_kib =~ [0-9]+ ]]; then
        # get allocatable memory from json in Ki
        alloc_mem_kib=$(kubectl get node $node -o json | jq -r '.status.allocatable.memory')
        alloc_mem_kib=$(echo "$alloc_mem_kib" | sed 's/Ki$//')
        free_mem_kib=$((alloc_mem_kib - allocated_mem_kib))
        free_mem_gi=$((free_mem_kib/1024/1024))
      else
        free_mem_gi=0
      fi

      # convert CPU allocatable and used
      # We'll check CPU cores roughly by parsing Allocatable and Requests
      # (finer-grained calculation would require parsing the exact units)
      # Get allocatable cpu as cores or m
      alloc_cpu_val=$(kubectl get node $node -o json | jq -r '.status.allocatable.cpu')
      alloc_cpu_cores=0
      if [[ $alloc_cpu_val == *m ]]; then
        alloc_cpu_cores=$(( $(echo $alloc_cpu_val | sed 's/m$//') / 1000 ))
      else
        alloc_cpu_cores=$(echo $alloc_cpu_val | sed 's/\..*$//')
      fi

      # compute used CPU from describe node
      cpu_used_m=$(kubectl describe node $node | awk '/Allocated resources:/,/Events:/' | grep -A2 cpu | head -1 | awk '{print $2}')
      cpu_used_cores=0
      if [[ $cpu_used_m == *m ]]; then
        cpu_used_cores=$(( $(echo $cpu_used_m | sed 's/m$//') / 1000 ))
      else
        cpu_used_cores=$(echo $cpu_used_m | sed 's/\..*$//')
      fi

      cpu_free=$((alloc_cpu_cores - cpu_used_cores))

      echo "  CPU free cores (approx): $cpu_free | Mem free Gi: $free_mem_gi"

      if [[ $cpu_free -ge $REQUIRED_CPUS && $free_mem_gi -ge $REQUIRED_MEM_Gi ]]; then
        echo "Found eligible node: $node"
        echo "Patching job yaml and applying"
        # If job already exists, patch nodeSelector; otherwise apply the template with new node
        if kubectl get job pure-transformer-8xa100-fast -n $NAMESPACE >/dev/null 2>&1; then
          echo "Patching existing job to schedule on node $node"
          kubectl patch job pure-transformer-8xa100-fast -n $NAMESPACE --type='json' -p "[{\"op\":\"replace\",\"path\":\"/spec/template/spec/nodeSelector/kubernetes.io~1hostname\",\"value\":\"$node\"}]" || true
        else
          cp $JOB_YAML $TMP_YAML
          # Replace node selector line - assumes is set as kubernetes.io/hostname: node-2-2...
          sed -i "s/kubernetes.io\/hostname:.*/kubernetes.io\/hostname: $node/" $TMP_YAML
          kubectl apply -f $TMP_YAML
        fi
        echo "Applied job pointing to $node"
        exit 0
      else
        echo "Node $node has insufficient CPU/memory for our request; continuing"
      fi
    fi
  done

  echo "No eligible node found. Sleeping 30s and will retry..."
  sleep 30
done
