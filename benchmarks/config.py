"""Minimal vLLM Whisper benchmark configuration."""
import os


def _int_list_from_env(name: str, default: list[int]) -> list[int]:
    raw = os.getenv(name, "")
    if not raw.strip():
        return default
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _float_from_env(name: str, default: float) -> float:
    raw = os.getenv(name, "")
    if not raw.strip():
        return default
    return float(raw.strip())


VLLM_WHISPER_BASE_URL = os.getenv("VLLM_WHISPER_BASE_URL", "http://localhost:8003").rstrip("/")
VLLM_WHISPER_URL = os.getenv("VLLM_WHISPER_URL", f"{VLLM_WHISPER_BASE_URL}/v1").rstrip("/")
VLLM_WHISPER_MODEL = os.getenv("VLLM_WHISPER_MODEL", "openai/whisper-large-v3")

# Core workload knobs for cross-GPU comparability.
WHISPER_CONCURRENCY = _int_list_from_env("WHISPER_CONCURRENCY", [8, 16, 24, 32, 40, 48, 64, 80, 100])
WHISPER_LOAD_MODE = os.getenv("WHISPER_LOAD_MODE", "steady").strip().lower()
WHISPER_STEADY_STATE_DURATION_S = float(os.getenv("WHISPER_STEADY_STATE_DURATION_S", "90"))
WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "1"))
WHISPER_INPUT_MODE = os.getenv("WHISPER_INPUT_MODE", "pcm_f32le").strip()

# Sampling and pooling.
MIN_SAMPLES = int(os.getenv("MIN_SAMPLES", "30"))
WARMUP_REQUESTS = int(os.getenv("WARMUP_REQUESTS", "3"))
COOLDOWN_S = float(os.getenv("COOLDOWN_S", "10"))
AUDIO_DURATION_S = float(os.getenv("AUDIO_DURATION_S", "47"))
AUDIO_POOL_SIZE = int(os.getenv("AUDIO_POOL_SIZE", "10"))

# SLA + planning.
WHISPER_MAX_P95_MS = _float_from_env("WHISPER_MAX_P95_MS", 12000.0)
WHISPER_PROJECTION_HEADROOM = _float_from_env("WHISPER_PROJECTION_HEADROOM", 1.0)
TARGET_CONCURRENCY_TIERS = _int_list_from_env(
    "TARGET_CONCURRENCY_TIERS",
    [200, 400, 600, 1000, 1500, 2000, 25000],
)
WHISPER_COMPARE_CONCURRENCY = WHISPER_CONCURRENCY
WHISPER_MODEL_MAP = {
    "large-v3": {"vllm_whisper": "openai/whisper-large-v3"},
    "medium": {"vllm_whisper": "openai/whisper-medium"},
}

# Output.
RESULTS_DIR = os.getenv("RESULTS_DIR", "./results")
GPU_METRICS_FILE = os.getenv("GPU_METRICS_FILE", "./results/gpu_metrics_vllm_whisper_only.jsonl")

# HTTP timeout for long audio requests.
WHISPER_TIMEOUT = int(os.getenv("WHISPER_TIMEOUT", "300"))
