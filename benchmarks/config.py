"""
Central benchmark configuration.
All values can be overridden via environment variables.
"""
import os

# ── Service URLs ──────────────────────────────────────────────────────────────
WHISPER_URL = os.getenv("WHISPER_URL", "http://localhost:8001")
LLM_URL     = os.getenv("LLM_URL",     "http://localhost:8002")

# ── Concurrency sweep levels ──────────────────────────────────────────────────
# Whisper: how many 47-second audio chunks to send simultaneously
WHISPER_CONCURRENCY = [1, 5, 10, 20, 50, 100]

# LLM: how many chat completion requests to send simultaneously
LLM_CONCURRENCY = [1, 5, 10, 20, 50]

# E2E: how many full "sessions" (Whisper chunks + LLM note) to run simultaneously
E2E_SESSION_COUNTS = [20, 50, 100, 200]

# ── Per-run parameters ────────────────────────────────────────────────────────
# Minimum samples per concurrency level (actual count = max(MIN_SAMPLES, concurrency))
MIN_SAMPLES    = 30
WARMUP_REQUESTS = 3

# Seconds to wait between concurrency levels (GPU thermal + scheduler cooldown)
COOLDOWN_S = 10.0

# Audio chunk duration (seconds) — matches your production chunk size
AUDIO_DURATION_S = 47.0

# Number of unique audio files in the pool (cycles to avoid cache artifacts)
AUDIO_POOL_SIZE = 10

# Number of audio chunks per E2E session (47 s × 13 ≈ 10-minute appointment)
CHUNKS_PER_SESSION = 13

# LLM model name as served by vLLM (must match --served-model-name in entrypoint.sh)
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-oss-20b")

# Whisper model sizes to sweep in the whisper-only benchmark
# Override via env: WHISPER_MODELS=medium,large-v2
_raw_models = os.getenv("WHISPER_MODELS", "tiny,base,small,medium,large-v2,large-v3")
WHISPER_MODELS = [m.strip() for m in _raw_models.split(",") if m.strip()]

# ── Output ────────────────────────────────────────────────────────────────────
RESULTS_DIR      = os.getenv("RESULTS_DIR",      "./results")
GPU_METRICS_FILE = os.getenv("GPU_METRICS_FILE",  "./results/gpu_metrics.jsonl")

# aiohttp request timeouts (seconds)
WHISPER_TIMEOUT = 300   # 47s audio × worst-case 5 min queue
LLM_TIMEOUT     = 600   # generous for cold KV cache fill
