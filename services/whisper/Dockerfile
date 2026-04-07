# Base: PyTorch + CUDA 12.4 (needed by openai-whisper backend).
# faster-whisper uses CTranslate2 (its own CUDA runtime, no PyTorch dep),
# but both can coexist in the same image.
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

# ffmpeg required by both backends for audio decoding
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install requirements (torch is already in base image; pip won't reinstall it)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download model weights at build time for the default model so that the
# first request has zero cold-start latency.
# Both backends share the same HuggingFace/whisper cache directories.
ARG WHISPER_MODEL=large-v2
ARG WHISPER_BACKEND=faster_whisper
ARG WHISPER_COMPUTE_TYPE=float16

RUN python - <<'EOF'
import os
model   = os.getenv("WHISPER_MODEL",        "large-v2")
backend = os.getenv("WHISPER_BACKEND",      "faster_whisper")
ctype   = os.getenv("WHISPER_COMPUTE_TYPE", "float16")

if backend == "faster_whisper":
    from faster_whisper import WhisperModel
    # device="cpu" here so the build worker doesn't need a GPU
    WhisperModel(model, device="cpu", compute_type="int8")
    print(f"faster-whisper model '{model}' cached.")
elif backend == "openai_whisper":
    import whisper
    whisper.load_model(model, device="cpu")
    print(f"openai-whisper model '{model}' cached.")
EOF

HEALTHCHECK --interval=10s --timeout=5s --start-period=90s --retries=6 \
    CMD curl -sf http://localhost:8001/health || exit 1

CMD ["uvicorn", "app:app", \
     "--host", "0.0.0.0", \
     "--port", "8001", \
     "--workers", "1", \
     "--loop", "uvloop", \
     "--http", "httptools"]
