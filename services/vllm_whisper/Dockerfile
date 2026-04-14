ARG VLLM_WHISPER_IMAGE=vllm/vllm-openai:latest
FROM ${VLLM_WHISPER_IMAGE}

# Keep model downloads on the mounted /models volume when Hugging Face is used.
ENV HF_HOME=/models/.cache

# vLLM speech-to-text loads uploads via soundfile; base vllm-openai image omits audio extras.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --no-cache-dir soundfile

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

HEALTHCHECK --interval=15s --timeout=10s --start-period=180s --retries=12 \
    CMD curl -f http://localhost:${VLLM_WHISPER_PORT:-8003}/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]
