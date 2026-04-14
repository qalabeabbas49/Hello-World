import os


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


WHISPER_BACKEND  = os.getenv("WHISPER_BACKEND", "faster_whisper")   # faster_whisper | openai_whisper
MODEL_SIZE       = os.getenv("WHISPER_MODEL", "large-v3")
DEVICE           = os.getenv("WHISPER_DEVICE", "cuda")
# CTranslate2 precision (faster_whisper only): float16 | int8_float16 | int8
COMPUTE_TYPE     = os.getenv("WHISPER_COMPUTE_TYPE", "float16")
# OpenAI Whisper backend precision on GPU: fp16 (True) or fp32 (False)
OPENAI_FP16      = _env_bool("WHISPER_OPENAI_FP16", True)
# Number of concurrent model instances sharing the GPU
NUM_WORKERS      = int(os.getenv("WHISPER_WORKERS", "4"))
MAX_QUEUE        = int(os.getenv("WHISPER_MAX_QUEUE", "200"))
HOST             = os.getenv("WHISPER_HOST", "0.0.0.0")
PORT             = int(os.getenv("WHISPER_PORT", "8001"))
