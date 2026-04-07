FROM vllm/vllm-openai:v0.6.3

# Model weights are volume-mounted at /models at runtime.
# Set HF_HOME so that any HuggingFace downloads go to the mounted volume too.
ENV HF_HOME=/models/.cache

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Health check hits vLLM's built-in /health endpoint
HEALTHCHECK --interval=15s --timeout=10s --start-period=120s --retries=6 \
    CMD curl -f http://localhost:${LLM_PORT:-8002}/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]
