FROM nvidia/cuda:12.4.1-base-ubuntu22.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

CMD ["python3", "/monitor/gpu_monitor.py"]
