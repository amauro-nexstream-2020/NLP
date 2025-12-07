
import subprocess
import sys
import re

def get_node_availability(nodes_file):
    with open(nodes_file, 'r') as f:
        nodes = [line.strip() for line in f if line.strip()]

    print(f"Checking {len(nodes)} nodes for A100 availability...")
    
    available_nodes = []

    for node in nodes:
        try:
            # Run kubectl describe
            result = subprocess.run(
                ['kubectl', 'describe', 'node', node], 
                capture_output=True, text=True, timeout=10
            )
            output = result.stdout
            
            # Find Capacity
            cap_match = re.search(r'nvidia\.com/a100:\s+(\d+)', output) # Capacity usually in Capacity section but also Requests/Limits
            # Wait, describe output has "Capacity:" section and "Allocated resources:" section.
            # Capacity section format: 
            # Capacity:
            #   nvidia.com/a100: 4
            
            capacity = 0
            if cap_match:
                # This regex might match multiple places. 
                # Let's search specifically in Capacity block is hard with regex on full text.
                # simpler: find all occurrences, usually first is Capacity, second Allocatable?
                # Actually, in "Allocated resources" section, it shows "Allocated resources: ... nvidia.com/a100 Request Limit"
                pass

            # Parsing "Allocated resources" table
            # pattern: nvidia.com/a100  4  4
            alloc_match = re.search(r'nvidia\.com/a100\s+(\d+)\s+(\d+)', output)
            
            allocated = 0
            limit = 0
            if alloc_match:
                allocated = int(alloc_match.group(1))
                # limit = int(alloc_match.group(2))
            
            # Find actual Capacity from "Capacity" or "Allocatable"
            # Allocatable is better
            # Allocatable:
            #   nvidia.com/a100: 4
            allocatable_match = re.search(r'Allocatable:\s+.*\n.*nvidia\.com/a100:\s+(\d+)', output, re.MULTILINE)
            # The above regex is risky. 
            
            # Use `kubectl get node <node> -o json` is safer.
            json_proc = subprocess.run(
                ['kubectl', 'get', 'node', node, '-o', 'json'],
                capture_output=True, text=True
            )
            import json
            node_data = json.loads(json_proc.stdout)
            
            allocatable = int(node_data['status']['allocatable'].get('nvidia.com/a100', 0))
            
            # But "Allocated" is not in JSON. It's only in describe (calculated by metrics server or controller manager).
            # So I need describe for Allocated.
            
            # Re-parse describe for Allocated
            # Look for "Allocated resources:" then "nvidia.com/a100"
            lines = output.split('\n')
            in_allocated = False
            allocated_req = 0
            for line in lines:
                if 'Allocated resources:' in line:
                    in_allocated = True
                if in_allocated and 'nvidia.com/a100' in line:
                    # Line looks like: "  nvidia.com/a100  4  4" or "  nvidia.com/a100  4 (100%)  4 (100%)"
                    parts = line.split()
                    # parts[0] is nvidia.com/a100
                    # parts[1] is Requests count (might have % e.g. "4")
                    try:
                        allocated_req = int(parts[1])
                    except:
                        # try stripping (
                        val = parts[1].split('(')[0]
                        allocated_req = int(val)
                    break
            
            free = allocatable - allocated_req
            
            if free > 0:
                print(f"FOUND: {node} - Capacity: {allocatable}, Allocated: {allocated_req}, Free: {free}")
                available_nodes.append((node, free))
            else:
                # print(f"FULL: {node} ({allocated_req}/{allocatable})")
                pass

        except Exception as e:
            print(f"Error checking {node}: {e}")

    print("\nSummary of Available A100 Nodes:")
    total_free = sum(n[1] for n in available_nodes)
    print(f"Total Free A100s: {total_free}")
    for node, count in available_nodes:
        print(f"- {node}: {count}")

get_node_availability('a100_nodes.txt')
