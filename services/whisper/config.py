import os

MODEL_SIZE   = os.getenv("WHISPER_MODEL", "large-v2")
DEVICE       = os.getenv("WHISPER_DEVICE", "cuda")
# float16 for large models; int8_float16 saves VRAM for small/medium with minimal quality loss
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")
# Number of concurrent WhisperModel instances sharing the GPU
NUM_WORKERS  = int(os.getenv("WHISPER_WORKERS", "4"))
MAX_QUEUE    = int(os.getenv("WHISPER_MAX_QUEUE", "200"))
HOST         = os.getenv("WHISPER_HOST", "0.0.0.0")
PORT         = int(os.getenv("WHISPER_PORT", "8001"))
