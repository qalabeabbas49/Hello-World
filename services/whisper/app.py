"""
Whisper transcription service.

Endpoints:
  POST /transcribe   multipart/form-data, field "file" = audio bytes
  GET  /health       returns model name, worker count, active workers
  GET  /metrics      returns request counters for monitoring

Usage (env vars):
  WHISPER_MODEL        tiny | base | small | medium | large-v2 | large-v3
  WHISPER_WORKERS      number of concurrent model instances (default 4)
  WHISPER_COMPUTE_TYPE float16 | int8_float16 | int8
"""
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

import config
from batch_manager import WhisperPool

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

pool: WhisperPool | None = None

# Simple in-process counters (good enough for benchmarking)
_stats = {
    "total_requests": 0,
    "successful": 0,
    "failed": 0,
    "total_audio_seconds": 0.0,
    "total_latency_ms": 0.0,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pool
    pool = WhisperPool(
        config.MODEL_SIZE,
        config.NUM_WORKERS,
        config.DEVICE,
        config.COMPUTE_TYPE,
    )
    yield
    # Cleanup — executor shuts down on GC, nothing explicit needed


app = FastAPI(title="Whisper Transcription Service", lifespan=lifespan)


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Query(default="en", description="BCP-47 language code"),
    beam_size: int = Query(default=5, ge=1, le=10),
):
    if pool is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    t0 = time.perf_counter()
    _stats["total_requests"] += 1

    try:
        audio_bytes = await file.read()
        result = await pool.transcribe(audio_bytes, language=language, beam_size=beam_size)
    except Exception as exc:
        _stats["failed"] += 1
        logger.exception("Transcription error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    latency_ms = (time.perf_counter() - t0) * 1000
    _stats["successful"] += 1
    _stats["total_audio_seconds"] += result.get("duration", 0.0)
    _stats["total_latency_ms"] += latency_ms

    return {
        **result,
        "latency_ms": round(latency_ms, 2),
        "model": config.MODEL_SIZE,
        "workers": config.NUM_WORKERS,
        "active_workers": pool.active_workers,
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": config.MODEL_SIZE,
        "workers": config.NUM_WORKERS,
        "active_workers": pool.active_workers if pool else 0,
        "compute_type": config.COMPUTE_TYPE,
        "device": config.DEVICE,
    }


@app.get("/metrics")
async def metrics():
    total = _stats["total_requests"] or 1
    return {
        **_stats,
        "avg_latency_ms": round(_stats["total_latency_ms"] / max(_stats["successful"], 1), 2),
        "error_rate_pct": round(_stats["failed"] / total * 100, 2),
    }
