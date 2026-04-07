#!/bin/bash
# GPU environment checks before running benchmarks.
# Verifies NVIDIA drivers, Docker GPU support, and prints H200 specs.
set -euo pipefail

echo "=== GPU Environment Check ==="
echo ""

# 1. Driver version
echo "Driver:"
nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | xargs echo "  Version:"

# 2. GPU details
echo ""
echo "GPU(s):"
nvidia-smi --query-gpu=index,name,memory.total,compute_cap \
  --format=csv,noheader | while IFS=',' read idx name mem cap; do
  echo "  [$idx] $name — VRAM: $mem — Compute: $cap"
done

# 3. CUDA version
echo ""
echo "CUDA:"
nvidia-smi | grep "CUDA Version" | awk '{print "  " $NF}'

# 4. Docker GPU access
echo ""
echo "Docker GPU access:"
if docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi -L 2>/dev/null; then
  echo "  ✓ Docker can access GPU"
else
  echo "  ✗ Docker GPU access failed — check nvidia-container-toolkit" >&2
  echo "    Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html" >&2
  exit 1
fi

echo ""
echo "=== All checks passed. Ready to benchmark. ==="
