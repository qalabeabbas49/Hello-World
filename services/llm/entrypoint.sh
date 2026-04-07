#!/bin/bash
# vLLM server entrypoint for locally-hosted 20B LLM.
#
# Key tuning flags:
#   --enable-prefix-caching  Reuses KV cache for the shared medical system prompt
#                            (~120 tokens identical across all sessions → big throughput win)
#   --max-num-seqs 256       H200 with 20B FP16 (~40 GB weights) leaves ~100 GB for KV
#                            cache. At 8192 ctx, each sequence ≈ 400 MB → fits ~250 seqs.
#   --gpu-memory-utilization Pre-allocates KV cache blocks at startup. Set lower in
#                            mixed-mode (see docker-compose.yml LLM_GPU_UTIL env var).
#
# Override any flag via environment variables defined in .env or docker-compose.
set -euo pipefail

MODEL="${LLM_MODEL_PATH:-/models/gpt-oss-20b}"
MAX_MODEL_LEN="${LLM_MAX_MODEL_LEN:-8192}"
GPU_UTIL="${LLM_GPU_UTIL:-0.92}"
TENSOR_PARALLEL="${LLM_TENSOR_PARALLEL:-1}"
MAX_NUM_SEQS="${LLM_MAX_NUM_SEQS:-256}"
PORT="${LLM_PORT:-8002}"
DTYPE="${LLM_DTYPE:-float16}"

echo "Starting vLLM: model=${MODEL}, gpu_util=${GPU_UTIL}, max_seqs=${MAX_NUM_SEQS}"

exec python -m vllm.entrypoints.openai.api_server \
    --model                   "${MODEL}" \
    --max-model-len           "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization  "${GPU_UTIL}" \
    --tensor-parallel-size    "${TENSOR_PARALLEL}" \
    --dtype                   "${DTYPE}" \
    --enable-prefix-caching \
    --max-num-seqs            "${MAX_NUM_SEQS}" \
    --port                    "${PORT}" \
    --host                    "0.0.0.0" \
    --served-model-name       "gpt-oss-20b"
