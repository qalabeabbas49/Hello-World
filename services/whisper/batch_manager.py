"""
Dual-backend Whisper pool: faster-whisper (CTranslate2) and openai-whisper (PyTorch).

Both backends use the same interface:
  pool = create_pool(backend="faster_whisper", ...)
  result = await pool.transcribe(audio_bytes)

Design:
- Pool of N model instances sharing one CUDA device.
- ThreadPoolExecutor: faster-whisper releases the GIL via CTranslate2 native CUDA calls;
  openai-whisper releases it via PyTorch CUDA kernels. Both give true parallelism.
- asyncio.Queue acts as a semaphore: back-pressure is automatic.
- Multiple model instances on H200 (141 GB) → concurrent CUDA stream submission.
  Large-v3 FW: ~6 GB each → 20+ instances fit. Large-v3 OW (PyTorch): ~6 GB → same.
"""
import asyncio
import io
import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


# ── Common audio loading ──────────────────────────────────────────────────────

def _load_via_ffmpeg(audio_bytes: bytes) -> np.ndarray:
    """
    Decode audio bytes using ffmpeg — handles formats soundfile can't:
    Opus (.opus / .ogg), WebM, MP3, M4A, AAC, etc.
    ffmpeg outputs raw float32 LE PCM at 16 kHz mono directly.
    """
    import subprocess
    result = subprocess.run(
        [
            "ffmpeg", "-v", "quiet",
            "-i", "pipe:0",
            "-f", "f32le",      # float32 little-endian raw PCM
            "-ac", "1",         # mono
            "-ar", "16000",     # 16 kHz (Whisper-native; avoids double-resample)
            "pipe:1",
        ],
        input=audio_bytes,
        capture_output=True,
        check=True,
    )
    return np.frombuffer(result.stdout, dtype=np.float32).copy()


def _load_audio(audio_bytes: bytes) -> np.ndarray:
    """
    Decode any audio format → float32 mono numpy array at 16 kHz.

    Strategy:
    1. Try soundfile first (fast; supports WAV, FLAC, OGG Vorbis, AIFF…).
    2. Fall back to ffmpeg for formats soundfile can't handle
       (Opus, WebM, MP3, M4A, AAC…).
    3. Resample to 16 kHz if soundfile decoded at a different rate
       (e.g. 48 kHz browser capture).
    """
    try:
        arr, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        if sr != 16_000:
            import scipy.signal as sig_resamp
            n_out = int(len(arr) * 16_000 / sr)
            arr   = sig_resamp.resample(arr, n_out).astype(np.float32)
        return arr
    except Exception:
        # soundfile failed (Opus, WebM, etc.) — delegate to ffmpeg
        return _load_via_ffmpeg(audio_bytes)


# ── Base class ────────────────────────────────────────────────────────────────

class BaseWhisperPool(ABC):
    def __init__(self, model_size: str, n_workers: int, device: str):
        self._model_size = model_size
        self._n_workers  = n_workers
        self._device     = device
        self._executor   = ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="whisper")
        self._available: asyncio.Queue = asyncio.Queue(maxsize=n_workers)

    async def transcribe(self, audio_bytes: bytes, language: str = "en", beam_size: int = 5) -> dict:
        model = await self._available.get()
        loop  = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(
                self._executor, self._infer, model, audio_bytes, language, beam_size
            )
        finally:
            await self._available.put(model)

    @abstractmethod
    def _infer(self, model, audio_bytes: bytes, language: str, beam_size: int) -> dict: ...

    @property
    def active_workers(self) -> int:
        return self._n_workers - self._available.qsize()


# ── faster-whisper backend ────────────────────────────────────────────────────

class FasterWhisperPool(BaseWhisperPool):
    """
    Uses CTranslate2-based faster-whisper.
    Models: tiny, base, small, medium, large-v2, large-v3, large-v3-turbo
    """
    def __init__(self, model_size: str, n_workers: int, device: str, compute_type: str = "float16"):
        super().__init__(model_size, n_workers, device)
        from faster_whisper import WhisperModel

        logger.info(
            "FasterWhisper: loading %d × %s (device=%s, compute_type=%s)",
            n_workers, model_size, device, compute_type,
        )
        self._models = [
            WhisperModel(model_size, device=device, compute_type=compute_type,
                         num_workers=1, cpu_threads=2)
            for _ in range(n_workers)
        ]
        for m in self._models:
            self._available.put_nowait(m)
        logger.info("FasterWhisperPool ready (%d workers)", n_workers)

    def _infer(self, model, audio_bytes: bytes, language: str, beam_size: int) -> dict:
        audio = _load_audio(audio_bytes)
        segments, info = model.transcribe(
            audio, language=language, beam_size=beam_size,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
        )
        text = " ".join(s.text.strip() for s in segments)
        return {"text": text, "language": info.language, "duration": info.duration}


# ── openai-whisper backend ────────────────────────────────────────────────────

class OpenAIWhisperPool(BaseWhisperPool):
    """
    Uses the original OpenAI Whisper (PyTorch).
    Models: tiny, base, small, medium, large, large-v1, large-v2, large-v3, turbo
    Note: 'turbo' is large-v3-turbo (809 M params); 'large' = large-v1.
    """
    def __init__(self, model_size: str, n_workers: int, device: str):
        super().__init__(model_size, n_workers, device)
        import whisper as oai_whisper

        # openai-whisper uses torch; each instance occupies its own VRAM slot
        # exactly as faster-whisper does.
        logger.info(
            "OpenAIWhisper: loading %d × %s (device=%s)",
            n_workers, model_size, device,
        )
        self._models = [
            oai_whisper.load_model(model_size, device=device)
            for _ in range(n_workers)
        ]
        for m in self._models:
            self._available.put_nowait(m)
        logger.info("OpenAIWhisperPool ready (%d workers)", n_workers)

    def _infer(self, model, audio_bytes: bytes, language: str, beam_size: int) -> dict:
        audio = _load_audio(audio_bytes)
        # openai-whisper expects a 30-second padded mel internally;
        # passing a numpy array directly is supported since v20231117.
        result = model.transcribe(
            audio,
            language=language,
            beam_size=beam_size,
            fp16=(self._device != "cpu"),
        )
        duration = float(len(audio) / 16_000)
        return {
            "text": result["text"].strip(),
            "language": result.get("language", language),
            "duration": duration,
        }


# ── Factory ───────────────────────────────────────────────────────────────────

def create_pool(
    backend: str,
    model_size: str,
    n_workers: int,
    device: str,
    compute_type: str = "float16",
) -> BaseWhisperPool:
    """
    Factory that returns the right pool based on backend name.

    Args:
        backend:      "faster_whisper" | "openai_whisper"
        model_size:   Model name valid for the chosen backend.
        n_workers:    Number of concurrent model instances.
        device:       "cuda" | "cpu"
        compute_type: CTranslate2 precision (faster_whisper only).
    """
    if backend == "faster_whisper":
        return FasterWhisperPool(model_size, n_workers, device, compute_type)
    elif backend == "openai_whisper":
        return OpenAIWhisperPool(model_size, n_workers, device)
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Choose 'faster_whisper' or 'openai_whisper'.")
