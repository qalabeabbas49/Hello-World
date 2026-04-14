#!/bin/bash
# vLLM server entrypoint for Whisper transcription models.
#
# Exposes the OpenAI-compatible endpoint:
#   POST /v1/audio/transcriptions
set -euo pipefail

MODEL="${VLLM_WHISPER_MODEL:-openai/whisper-large-v3}"
PORT="${VLLM_WHISPER_PORT:-8003}"
GPU_UTIL="${VLLM_WHISPER_GPU_UTIL:-0.95}"
MAX_NUM_SEQS="${VLLM_WHISPER_MAX_NUM_SEQS:-2048}"
DTYPE="${VLLM_WHISPER_DTYPE:-auto}"
EXTRA_ARGS="${VLLM_WHISPER_EXTRA_ARGS:-}"

echo "Starting vLLM Whisper: model=${MODEL}, gpu_util=${GPU_UTIL}, max_seqs=${MAX_NUM_SEQS}"

cmd=(
  vllm serve "${MODEL}"
  --host "0.0.0.0"
  --port "${PORT}"
  --served-model-name "${MODEL}"
  --gpu-memory-utilization "${GPU_UTIL}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --dtype "${DTYPE}"
)

if [[ -n "${EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  extra_parts=(${EXTRA_ARGS})
  cmd+=("${extra_parts[@]}")
fi

# Compose injects these for this script only; vLLM warns on unknown VLLM_* env vars.
unset VLLM_WHISPER_MODEL VLLM_WHISPER_PORT VLLM_WHISPER_GPU_UTIL \
  VLLM_WHISPER_MAX_NUM_SEQS VLLM_WHISPER_DTYPE VLLM_WHISPER_EXTRA_ARGS

exec "${cmd[@]}"
