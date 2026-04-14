"""
Whisper transcription service — dual-backend (faster-whisper / openai-whisper).

Env vars:
  WHISPER_BACKEND      faster_whisper (default) | openai_whisper
  WHISPER_MODEL        Model size (see batch_manager.py for valid names per backend)
  WHISPER_WORKERS      Concurrent model instances (default 4)
  WHISPER_COMPUTE_TYPE CTranslate2 precision for faster_whisper (float16 default)
  WHISPER_OPENAI_FP16  1/0 precision toggle for openai_whisper backend

Endpoints:
  POST /transcribe     multipart/form-data, field "file" = audio bytes
  POST /transcribe_pcm_f32le raw float32 LE mono PCM at 16 kHz
  GET  /health         model, backend, workers, active_workers
  GET  /metrics        request counters
"""
import asyncio
import logging
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile

import config
from batch_manager import BaseWhisperPool, create_pool

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

pool: BaseWhisperPool | None = None

_stats_lock = asyncio.Lock()
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
        openai_fp16=config.OPENAI_FP16,
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
    async with _stats_lock:
        _stats["total_requests"] += 1

    try:
        audio_bytes = await file.read()
        result = await pool.transcribe(audio_bytes, language=language, beam_size=beam_size)
    except Exception as exc:
        async with _stats_lock:
            _stats["failed"] += 1
        logger.exception("Transcription error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    latency_ms = (time.perf_counter() - t0) * 1000
    async with _stats_lock:
        _stats["successful"] += 1
        _stats["total_audio_seconds"] += result.get("duration", 0.0)
        _stats["total_latency_ms"] += latency_ms

    return {
        **result,
        "latency_ms":     round(latency_ms, 2),
        "model":          config.MODEL_SIZE,
        "backend":        config.WHISPER_BACKEND,
        "input_mode":     "file",
        "workers":        config.NUM_WORKERS,
        "active_workers": pool.active_workers,
    }


@app.post("/transcribe_pcm_f32le")
async def transcribe_pcm_f32le(
    request: Request,
    language: str = Query(default="en"),
    beam_size: int = Query(default=5, ge=1, le=10),
    sample_rate: int = Query(default=16_000),
):
    if pool is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    if sample_rate != 16_000:
        raise HTTPException(status_code=400, detail="Only sample_rate=16000 is supported")

    t0 = time.perf_counter()
    async with _stats_lock:
        _stats["total_requests"] += 1

    try:
        raw = await request.body()
        if not raw:
            raise HTTPException(status_code=400, detail="Empty request body")
        if len(raw) % 4 != 0:
            raise HTTPException(status_code=400, detail="Body size must be a multiple of 4 bytes for float32 PCM")
        audio = np.frombuffer(raw, dtype=np.float32).copy()
        if audio.size == 0:
            raise HTTPException(status_code=400, detail="No audio samples found")
        result = await pool.transcribe(audio, language=language, beam_size=beam_size)
    except HTTPException:
        async with _stats_lock:
            _stats["failed"] += 1
        raise
    except Exception as exc:
        async with _stats_lock:
            _stats["failed"] += 1
        logger.exception("PCM transcription error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    latency_ms = (time.perf_counter() - t0) * 1000
    async with _stats_lock:
        _stats["successful"] += 1
        _stats["total_audio_seconds"] += result.get("duration", 0.0)
        _stats["total_latency_ms"] += latency_ms

    return {
        **result,
        "latency_ms":     round(latency_ms, 2),
        "model":          config.MODEL_SIZE,
        "backend":        config.WHISPER_BACKEND,
        "input_mode":     "pcm_f32le_16khz",
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
        "openai_fp16":    config.OPENAI_FP16,
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
