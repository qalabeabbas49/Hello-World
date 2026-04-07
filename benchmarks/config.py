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

# Compute types for faster-whisper (openai-whisper always uses fp16 via PyTorch)
# float16        — full precision CTranslate2 (default, best accuracy)
# int8_float16   — INT8 weights, FP16 activations: ~10-15% faster, <1% WER penalty
# Set via --compute-types flag; default is float16 only.
WHISPER_COMPUTE_TYPES = ["float16"]   # override with e.g. ["float16", "int8_float16"]

# Recommended WHISPER_WORKERS per model size for a single H200 (141 GB VRAM).
# Rule of thumb: workers = floor(available_vram / model_vram_per_instance)
# Values below assume faster-whisper float16 with 70–80 GB available to Whisper
# (remaining VRAM reserved for LLM in mixed mode or system headroom).
# For Whisper-only mode (up to ~130 GB), multiply by ~1.6.
WHISPER_WORKERS_RECOMMENDATION: dict[str, int] = {
    "tiny":           48,   # ~0.15 GB/instance
    "base":           32,   # ~0.29 GB/instance
    "small":          16,   # ~0.93 GB/instance
    "medium":          8,   # ~3.06 GB/instance
    "large-v2":        4,   # ~6.17 GB/instance
    "large-v3":        4,   # ~6.17 GB/instance
    "large-v3-turbo": 8,   # ~3.10 GB/instance
    # openai-whisper (PyTorch) uses slightly more VRAM per instance
    "large":           4,   # ~6.4  GB/instance (large-v1 PyTorch)
    "turbo":           8,   # ~3.3  GB/instance (large-v3-turbo PyTorch)
}

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

# ── Audio format + sample rate sweep ─────────────────────────────────────────
# Formats to test in the format benchmark
_raw_fmts = os.getenv("AUDIO_FORMATS", "wav,flac,opus")
AUDIO_FORMATS = [f.strip() for f in _raw_fmts.split(",") if f.strip()]

# Sample rates to test (Hz)
_raw_rates = os.getenv("AUDIO_SAMPLE_RATES", "16000,48000")
AUDIO_SAMPLE_RATES = [int(r.strip()) for r in _raw_rates.split(",") if r.strip()]

# Concurrency used in the format sweep (fixed — isolates format overhead from scaling)
FORMAT_BENCH_CONCURRENCY = int(os.getenv("FORMAT_BENCH_CONCURRENCY", "10"))

# Directory for real audio files provided by the user
# Sub-folders named  <fmt>_<rate>/  are auto-detected (e.g. wav_16000/)
# Flat folder also works — format inferred from extension, rate from ffprobe
REAL_AUDIO_DIR = os.getenv("REAL_AUDIO_DIR", "./fixtures/real_audio")

# ── Output ────────────────────────────────────────────────────────────────────
RESULTS_DIR      = os.getenv("RESULTS_DIR",     "./results")
GPU_METRICS_FILE = os.getenv("GPU_METRICS_FILE", "./results/gpu_metrics.jsonl")

# aiohttp request timeouts (seconds)
WHISPER_TIMEOUT = 300
LLM_TIMEOUT     = 600
