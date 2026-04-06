"""
Central benchmark configuration.
All values can be overridden via environment variables.
"""
import os

# ── Service URLs ──────────────────────────────────────────────────────────────
WHISPER_URL = os.getenv("WHISPER_URL", "http://localhost:8001")
LLM_URL     = os.getenv("LLM_URL",     "http://localhost:8002")

# ── Whisper model lists (per backend) ─────────────────────────────────────────
# faster-whisper (CTranslate2) model names
FASTER_WHISPER_MODELS = [
    "tiny", "base", "small", "medium",
    "large-v2", "large-v3", "large-v3-turbo",
]

# openai-whisper (PyTorch) model names
# "turbo" == large-v3-turbo; "large" == large-v1 (included for completeness)
OPENAI_WHISPER_MODELS = [
    "tiny", "base", "small", "medium",
    "large", "large-v2", "large-v3", "turbo",
]

# Canonical display order for the comparison matrix
# Maps display_name → {backend: actual_model_name}
WHISPER_MODEL_MAP = {
    "tiny":            {"faster_whisper": "tiny",            "openai_whisper": "tiny"},
    "base":            {"faster_whisper": "base",            "openai_whisper": "base"},
    "small":           {"faster_whisper": "small",           "openai_whisper": "small"},
    "medium":          {"faster_whisper": "medium",          "openai_whisper": "medium"},
    "large (v1)":      {"faster_whisper": None,              "openai_whisper": "large"},
    "large-v2":        {"faster_whisper": "large-v2",        "openai_whisper": "large-v2"},
    "large-v3":        {"faster_whisper": "large-v3",        "openai_whisper": "large-v3"},
    "turbo (lv3-t)":   {"faster_whisper": "large-v3-turbo",  "openai_whisper": "turbo"},
}

# Both backends to test in the comparison sweep
WHISPER_BACKENDS = ["faster_whisper", "openai_whisper"]

# ── Concurrency sweep levels ──────────────────────────────────────────────────
# Whisper: concurrent 47-second audio chunks
WHISPER_CONCURRENCY = [1, 5, 10, 20, 50, 100]

# Concurrency levels to show in the comparison matrix (subset for readability)
WHISPER_COMPARE_CONCURRENCY = [1, 10, 50]

# LLM: concurrent chat completions
LLM_CONCURRENCY = [1, 5, 10, 20, 50]

# E2E: simultaneous full sessions (Whisper chunks → LLM note)
E2E_SESSION_COUNTS = [20, 50, 100, 200]

# ── Per-run parameters ────────────────────────────────────────────────────────
MIN_SAMPLES      = 30       # minimum data points per concurrency level
WARMUP_REQUESTS  = 3        # sequential warmup before each level
COOLDOWN_S       = 10.0     # seconds between concurrency levels
AUDIO_DURATION_S = 47.0     # audio chunk length in seconds
AUDIO_POOL_SIZE  = 10       # unique audio files to cycle through
CHUNKS_PER_SESSION = 13     # 13 × 47s ≈ 10-minute appointment

LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-oss-20b")

# ── Output ────────────────────────────────────────────────────────────────────
RESULTS_DIR      = os.getenv("RESULTS_DIR",     "./results")
GPU_METRICS_FILE = os.getenv("GPU_METRICS_FILE", "./results/gpu_metrics.jsonl")

# aiohttp request timeouts (seconds)
WHISPER_TIMEOUT = 300
LLM_TIMEOUT     = 600
