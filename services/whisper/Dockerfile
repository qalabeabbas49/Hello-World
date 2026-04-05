FROM nvidia/cuda:12.4.1-cudnn9-runtime-ubuntu22.04

# ffmpeg is required by faster-whisper for audio decoding
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg python3-pip curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Download model weights at build time so first-request latency is zero
ARG WHISPER_MODEL=large-v2
ARG WHISPER_COMPUTE_TYPE=float16
RUN python - <<'EOF'
import os
from faster_whisper import WhisperModel
model = os.environ.get("WHISPER_MODEL", "large-v2")
ctype = os.environ.get("WHISPER_COMPUTE_TYPE", "float16")
# Download only — don't keep in memory
WhisperModel(model, device="cpu", compute_type="int8")
print(f"Model {model} cached.")
EOF

CMD ["uvicorn", "app:app", \
     "--host", "0.0.0.0", \
     "--port", "8001", \
     "--workers", "1", \
     "--loop", "uvloop", \
     "--http", "httptools"]
