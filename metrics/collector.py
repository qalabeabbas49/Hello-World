"""
Asyncio-safe in-memory metrics accumulator.

Design:
- Uses asyncio.Lock (not threading.Lock) because all benchmark coroutines run
  in the same event loop; threading.Lock would unnecessarily block the loop.
- No per-request disk I/O; flush to disk only after each concurrency level
  completes to avoid serializing the benchmark at high concurrency.
- Stores raw RequestRecord objects so the reporter can compute any percentile
  post-hoc without needing pre-aggregated bins.
"""
import asyncio
import os
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# ── GPU label helpers ─────────────────────────────────────────────────────────
# Benchmark scripts call set_gpu_label() to annotate GPU metrics with the
# current test phase. gpu_monitor reads the same file on every poll cycle.

_GPU_LABEL_FILE = Path(os.getenv("LABEL_FILE", "/results/gpu_label.txt"))


def set_gpu_label(label: str) -> None:
    """Write the current benchmark phase label to the shared label file."""
    try:
        _GPU_LABEL_FILE.parent.mkdir(parents=True, exist_ok=True)
        _GPU_LABEL_FILE.write_text(label)
    except Exception:
        pass   # non-critical; silently ignore FS errors


def clear_gpu_label() -> None:
    """Clear the GPU label (end of a benchmark phase)."""
    try:
        _GPU_LABEL_FILE.unlink(missing_ok=True)
    except Exception:
        pass


@dataclass
class RequestRecord:
    request_id:       str
    start_ts:         float           # perf_counter timestamp
    end_ts:           float = 0.0
    latency_ms:       float = 0.0
    tokens_in:        int   = 0
    tokens_out:       int   = 0
    audio_duration_s: float = 0.0    # for Whisper requests; 0 for LLM
    success:          bool  = True
    error:            Optional[str] = None


class MetricsCollector:
    """
    Collects request records for a single benchmark run.

    Usage:
        collector = MetricsCollector("whisper_concurrent_20")
        rec = RequestRecord(request_id="r0", start_ts=time.perf_counter(), audio_duration_s=47.0)
        # ... perform request ...
        rec.end_ts     = time.perf_counter()
        rec.latency_ms = (rec.end_ts - rec.start_ts) * 1000
        await collector.record(rec)
        summary = collector.summarize()
    """

    def __init__(self, test_name: str):
        self.test_name  = test_name
        self.test_start = time.perf_counter()
        self._records: list[RequestRecord] = []
        self._lock = asyncio.Lock()

    async def record(self, rec: RequestRecord) -> None:
        async with self._lock:
            self._records.append(rec)

    def summarize(self) -> dict:
        """Return a JSON-serialisable summary dict."""
        ok     = [r for r in self._records if r.success]
        failed = [r for r in self._records if not r.success]

        if not ok:
            return {
                "test_name": self.test_name,
                "total_requests": len(self._records),
                "successful": 0,
                "failed": len(failed),
                "error": "no successful requests",
            }

        latencies  = sorted(r.latency_ms for r in ok)
        test_dur   = time.perf_counter() - self.test_start
        tokens_out = sum(r.tokens_out for r in ok)
        audio_secs = sum(r.audio_duration_s for r in ok)

        return {
            "test_name":             self.test_name,
            "wall_time_s":           round(test_dur, 3),
            "total_requests":        len(self._records),
            "successful":            len(ok),
            "failed":                len(failed),
            "error_rate_pct":        round(len(failed) / len(self._records) * 100, 2),
            "throughput_rps":        round(len(ok) / test_dur, 4),
            "tokens_out_per_sec":    round(tokens_out / test_dur, 1),
            # RTF > 1 means we process audio faster than real-time (good)
            "audio_realtime_factor": round(audio_secs / test_dur, 3) if audio_secs else None,
            "latency_ms": {
                "mean": round(statistics.mean(latencies), 1),
                "p50":  round(float(np.percentile(latencies, 50)), 1),
                "p90":  round(float(np.percentile(latencies, 90)), 1),
                "p95":  round(float(np.percentile(latencies, 95)), 1),
                "p99":  round(float(np.percentile(latencies, 99)), 1),
                "max":  round(max(latencies), 1),
                "min":  round(min(latencies), 1),
            },
            "tokens_out_total":   tokens_out,
            "audio_seconds_total": round(audio_secs, 1),
        }

    def reset(self) -> None:
        """Clear records (e.g. after warmup phase)."""
        self._records.clear()
        self.test_start = time.perf_counter()
