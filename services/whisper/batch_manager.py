"""
WhisperPool — pool of faster-whisper model instances sharing one CUDA device.

Design rationale:
- CTranslate2 (faster-whisper backend) releases the GIL during GPU inference,
  so ThreadPoolExecutor gives true parallelism across model instances.
- Multiple WhisperModel instances on the same device submit to different CUDA
  streams concurrently, fully utilizing the GPU without dynamic batching complexity.
- The asyncio.Queue acts as a semaphore: callers await a model, use it, return it.
  Backpressure is automatic — requests queue behind available workers.
"""
import asyncio
import io
import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class WhisperPool:
    def __init__(
        self,
        model_size: str,
        n_workers: int,
        device: str,
        compute_type: str,
    ):
        self._n_workers = n_workers
        self._executor = ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="whisper")

        logger.info(
            "Loading %d x WhisperModel(%s, device=%s, compute_type=%s)",
            n_workers, model_size, device, compute_type,
        )
        # Each instance loads its own weight copy into VRAM.
        # H200 (141 GB) easily holds 12+ large-v3 instances (6 GB each = 72 GB).
        self._models: list[WhisperModel] = [
            WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
                num_workers=1,   # internal CTranslate2 threads per model
                cpu_threads=2,
            )
            for _ in range(n_workers)
        ]

        # Queue pre-filled with all model instances
        self._available: asyncio.Queue = asyncio.Queue(maxsize=n_workers)
        for m in self._models:
            self._available.put_nowait(m)

        logger.info("WhisperPool ready (%d workers)", n_workers)

    async def transcribe(
        self,
        audio_bytes: bytes,
        language: str = "en",
        beam_size: int = 5,
    ) -> dict:
        """
        Transcribe raw audio bytes (any format soundfile supports: WAV, FLAC, MP3…).
        Returns: {"text": str, "language": str, "duration": float}
        """
        model = await self._available.get()
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                self._executor,
                self._transcribe_sync,
                model,
                audio_bytes,
                language,
                beam_size,
            )
            return result
        finally:
            await self._available.put(model)

    @staticmethod
    def _transcribe_sync(
        model: WhisperModel,
        audio_bytes: bytes,
        language: str,
        beam_size: int,
    ) -> dict:
        audio_array, _ = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        # Ensure mono
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)

        segments, info = model.transcribe(
            audio_array,
            language=language,
            beam_size=beam_size,
            vad_filter=True,        # skip silence, speeds up processing
            vad_parameters={"min_silence_duration_ms": 300},
        )
        text = " ".join(s.text.strip() for s in segments)
        return {
            "text": text,
            "language": info.language,
            "duration": info.duration,
        }

    @property
    def active_workers(self) -> int:
        return self._n_workers - self._available.qsize()
