#!/bin/bash
# Wait for services to become healthy before starting benchmarks.
# Usage: bash scripts/verify_services.sh whisper|llm|all
set -euo pipefail

MODE="${1:-all}"
WHISPER_URL="${WHISPER_URL:-http://localhost:8001}"
LLM_URL="${LLM_URL:-http://localhost:8002}"
MAX_RETRIES=24    # 24 × 5s = 120s max wait
SLEEP_S=5

wait_for() {
  local name="$1"
  local url="$2"
  echo "Waiting for $name at $url..."
  for i in $(seq 1 $MAX_RETRIES); do
    if curl -sf "$url/health" -o /dev/null 2>/dev/null; then
      echo "  ✓ $name is ready"
      return 0
    fi
    echo "  attempt $i/$MAX_RETRIES — retrying in ${SLEEP_S}s..."
    sleep $SLEEP_S
  done
  echo "  ✗ $name did not become healthy within $((MAX_RETRIES * SLEEP_S))s" >&2
  return 1
}

case "$MODE" in
  whisper) wait_for "Whisper" "$WHISPER_URL" ;;
  llm)     wait_for "LLM"     "$LLM_URL"     ;;
  all)
    wait_for "Whisper" "$WHISPER_URL"
    wait_for "LLM"     "$LLM_URL"
    ;;
  *)
    echo "Usage: $0 whisper|llm|all" >&2
    exit 1
    ;;
esac
