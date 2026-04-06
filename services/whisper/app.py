"""
Whisper transcription service — dual-backend (faster-whisper / openai-whisper).

Env vars:
  WHISPER_BACKEND      faster_whisper (default) | openai_whisper
  WHISPER_MODEL        Model size (see batch_manager.py for valid names per backend)
  WHISPER_WORKERS      Concurrent model instances (default 4)
  WHISPER_COMPUTE_TYPE CTranslate2 precision for faster_whisper (float16 default)

Endpoints:
  POST /transcribe     multipart/form-data, field "file" = audio bytes
  GET  /health         model, backend, workers, active_workers
  GET  /metrics        request counters
"""
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Query, UploadFile

import config
from batch_manager import BaseWhisperPool, create_pool

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

pool: BaseWhisperPool | None = None

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
    pool = create_pool(
        backend=config.WHISPER_BACKEND,
        model_size=config.MODEL_SIZE,
        n_workers=config.NUM_WORKERS,
        device=config.DEVICE,
        compute_type=config.COMPUTE_TYPE,
    )
    yield


app = FastAPI(title="Whisper Transcription Service", lifespan=lifespan)


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Query(default="en"),
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
        "latency_ms":     round(latency_ms, 2),
        "model":          config.MODEL_SIZE,
        "backend":        config.WHISPER_BACKEND,
        "workers":        config.NUM_WORKERS,
        "active_workers": pool.active_workers,
    }


@app.get("/health")
async def health():
    return {
        "status":         "ok",
        "model":          config.MODEL_SIZE,
        "backend":        config.WHISPER_BACKEND,
        "workers":        config.NUM_WORKERS,
        "active_workers": pool.active_workers if pool else 0,
        "compute_type":   config.COMPUTE_TYPE,
        "device":         config.DEVICE,
    }


@app.get("/metrics")
async def metrics():
    total = _stats["total_requests"] or 1
    return {
        **_stats,
        "avg_latency_ms": round(_stats["total_latency_ms"] / max(_stats["successful"], 1), 2),
        "error_rate_pct": round(_stats["failed"] / total * 100, 2),
    }
