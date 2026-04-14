"""
Queue-backed vLLM Whisper benchmark using Redis Streams.

Simulates production-style ingestion with decoupled producer/consumer workers:
- producer publishes transcription jobs at a target ingress rate
- consumers pull jobs from Redis Streams consumer group and call vLLM
- metrics capture queue wait, service latency, and end-to-end latency
"""
import argparse
import asyncio
import io
import json
import time
import wave
from pathlib import Path

import aiohttp
import numpy as np
from redis.asyncio import Redis

from benchmarks import config as cfg
from generators.audio_gen import generate_pool, generate_pool_pcm_f32le


def _vllm_transcription_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return f"{base}/audio/transcriptions"


def _pcm_f32le_to_wav_bytes(audio_bytes: bytes, sample_rate: int = 16000) -> bytes:
    pcm = np.frombuffer(audio_bytes, dtype=np.float32)
    pcm_i16 = (np.clip(pcm, -1.0, 1.0) * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_i16.tobytes())
    return buf.getvalue()


class QueueMetrics:
    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.queue_wait_ms: list[float] = []
        self.service_ms: list[float] = []
        self.e2e_ms: list[float] = []
        self.queue_depth_samples: list[int] = []
        self.success = 0
        self.failed = 0
        self.produced = 0

    async def record_result(self, queue_wait_ms: float, service_ms: float, ok: bool) -> None:
        async with self.lock:
            self.queue_wait_ms.append(queue_wait_ms)
            self.service_ms.append(service_ms)
            self.e2e_ms.append(queue_wait_ms + service_ms)
            if ok:
                self.success += 1
            else:
                self.failed += 1

    async def record_produced(self, n: int = 1) -> None:
        async with self.lock:
            self.produced += n

    async def record_queue_depth(self, depth: int) -> None:
        async with self.lock:
            self.queue_depth_samples.append(depth)

    def _pct(self, values: list[float], p: float) -> float:
        if not values:
            return 0.0
        return float(np.percentile(values, p))

    def summarize(self, wall_time_s: float) -> dict:
        processed = self.success + self.failed
        return {
            "produced": self.produced,
            "processed": processed,
            "successful": self.success,
            "failed": self.failed,
            "error_rate_pct": round((self.failed / processed) * 100.0, 2) if processed else 0.0,
            "throughput_rps": round(self.success / max(wall_time_s, 1e-6), 4),
            "queue_wait_ms": {
                "p50": round(self._pct(self.queue_wait_ms, 50), 1),
                "p95": round(self._pct(self.queue_wait_ms, 95), 1),
                "p99": round(self._pct(self.queue_wait_ms, 99), 1),
                "max": round(max(self.queue_wait_ms), 1) if self.queue_wait_ms else 0.0,
            },
            "service_latency_ms": {
                "p50": round(self._pct(self.service_ms, 50), 1),
                "p95": round(self._pct(self.service_ms, 95), 1),
                "p99": round(self._pct(self.service_ms, 99), 1),
                "max": round(max(self.service_ms), 1) if self.service_ms else 0.0,
            },
            "e2e_latency_ms": {
                "p50": round(self._pct(self.e2e_ms, 50), 1),
                "p95": round(self._pct(self.e2e_ms, 95), 1),
                "p99": round(self._pct(self.e2e_ms, 99), 1),
                "max": round(max(self.e2e_ms), 1) if self.e2e_ms else 0.0,
            },
            "queue_depth": {
                "mean": round(float(np.mean(self.queue_depth_samples)), 1) if self.queue_depth_samples else 0.0,
                "max": int(max(self.queue_depth_samples)) if self.queue_depth_samples else 0,
            },
            "wall_time_s": round(wall_time_s, 3),
        }


async def _transcribe(
    session: aiohttp.ClientSession,
    url: str,
    audio_bytes: bytes,
    model: str,
    beam_size: int,
    input_mode: str,
) -> tuple[bool, float]:
    start = time.perf_counter()
    try:
        payload = audio_bytes
        if input_mode == "pcm_f32le":
            payload = _pcm_f32le_to_wav_bytes(audio_bytes, sample_rate=16000)
        form = aiohttp.FormData()
        form.add_field("file", payload, filename="audio.wav", content_type="audio/wav")
        form.add_field("model", model)
        form.add_field("language", "en")
        form.add_field("response_format", "json")
        form.add_field("temperature", "0.0")
        timeout = aiohttp.ClientTimeout(total=cfg.WHISPER_TIMEOUT)
        async with session.post(url, data=form, params={"beam_size": str(beam_size)}, timeout=timeout) as resp:
            ok = resp.status == 200
            if ok:
                await resp.json()
            else:
                await resp.text()
            return ok, (time.perf_counter() - start) * 1000
    except Exception:
        return False, (time.perf_counter() - start) * 1000


async def _producer(
    redis: Redis,
    stream: str,
    duration_s: float,
    ingress_rps: float,
    metrics: QueueMetrics,
) -> None:
    deadline = time.perf_counter() + duration_s
    interval = 1.0 / ingress_rps
    next_at = time.perf_counter()
    while time.perf_counter() < deadline:
        now = time.time()
        await redis.xadd(stream, {"enq_ts": f"{now:.6f}"})
        await metrics.record_produced(1)
        next_at += interval
        sleep_s = max(0.0, next_at - time.perf_counter())
        if sleep_s:
            await asyncio.sleep(sleep_s)


async def _consumer(
    redis: Redis,
    stream: str,
    group: str,
    consumer_name: str,
    url: str,
    audio_pool: list[bytes],
    model: str,
    beam_size: int,
    input_mode: str,
    metrics: QueueMetrics,
    stop_event: asyncio.Event,
) -> None:
    idx = 0
    connector = aiohttp.TCPConnector(limit=32)
    async with aiohttp.ClientSession(connector=connector) as session:
        while not stop_event.is_set():
            batch = await redis.xreadgroup(
                groupname=group,
                consumername=consumer_name,
                streams={stream: ">"},
                count=1,
                block=1000,
            )
            if not batch:
                continue
            _stream_name, messages = batch[0]
            for msg_id, fields in messages:
                enq_ts = float(fields.get(b"enq_ts", b"0").decode())
                queue_wait_ms = max(0.0, (time.time() - enq_ts) * 1000)
                audio_bytes = audio_pool[idx % len(audio_pool)]
                idx += 1
                ok, service_ms = await _transcribe(
                    session=session,
                    url=url,
                    audio_bytes=audio_bytes,
                    model=model,
                    beam_size=beam_size,
                    input_mode=input_mode,
                )
                await metrics.record_result(queue_wait_ms=queue_wait_ms, service_ms=service_ms, ok=ok)
                await redis.xack(stream, group, msg_id)


async def _queue_depth_sampler(
    redis: Redis,
    stream: str,
    metrics: QueueMetrics,
    stop_event: asyncio.Event,
) -> None:
    while not stop_event.is_set():
        try:
            depth = await redis.xlen(stream)
            await metrics.record_queue_depth(int(depth))
        except Exception:
            pass
        await asyncio.sleep(1.0)


async def run_queue_bench(
    redis_url: str,
    stream: str,
    group: str,
    consumers: int,
    ingress_rps: float,
    duration_s: float,
    model: str,
    whisper_base_url: str,
    beam_size: int,
    input_mode: str,
    reset_stream: bool,
    output_dir: str,
) -> dict:
    redis = Redis.from_url(redis_url, decode_responses=False)
    await redis.ping()

    if reset_stream:
        await redis.delete(stream)

    try:
        await redis.xgroup_create(stream, group, id="$", mkstream=True)
    except Exception:
        # BUSYGROUP is expected if rerunning.
        pass

    if input_mode == "pcm_f32le":
        print(f"Generating {cfg.AUDIO_POOL_SIZE} × {cfg.AUDIO_DURATION_S}s pre-decoded PCM payloads...")
        audio_pool = generate_pool_pcm_f32le(cfg.AUDIO_POOL_SIZE, cfg.AUDIO_DURATION_S)
    else:
        print(f"Generating {cfg.AUDIO_POOL_SIZE} × {cfg.AUDIO_DURATION_S}s WAV payloads...")
        audio_pool = generate_pool(cfg.AUDIO_POOL_SIZE, cfg.AUDIO_DURATION_S)

    metrics = QueueMetrics()
    stop_event = asyncio.Event()
    start = time.perf_counter()
    url = _vllm_transcription_url(whisper_base_url)

    sampler = asyncio.create_task(_queue_depth_sampler(redis, stream, metrics, stop_event))
    worker_tasks = [
        asyncio.create_task(
            _consumer(
                redis=redis,
                stream=stream,
                group=group,
                consumer_name=f"c{i}",
                url=url,
                audio_pool=audio_pool,
                model=model,
                beam_size=beam_size,
                input_mode=input_mode,
                metrics=metrics,
                stop_event=stop_event,
            )
        )
        for i in range(consumers)
    ]

    await _producer(redis, stream, duration_s, ingress_rps, metrics)

    # Drain phase: allow workers to clear queued jobs for a bounded window.
    drain_deadline = time.perf_counter() + max(10.0, duration_s * 0.5)
    while time.perf_counter() < drain_deadline:
        depth = await redis.xlen(stream)
        if depth == 0:
            break
        await asyncio.sleep(1.0)

    stop_event.set()
    for task in worker_tasks:
        task.cancel()
    sampler.cancel()
    await asyncio.gather(*worker_tasks, return_exceptions=True)
    await asyncio.gather(sampler, return_exceptions=True)

    wall_time_s = time.perf_counter() - start
    summary = metrics.summarize(wall_time_s)
    summary.update(
        {
            "redis_url": redis_url,
            "stream": stream,
            "group": group,
            "consumers": consumers,
            "ingress_rps_target": ingress_rps,
            "duration_s": duration_s,
            "model": model,
            "input_mode": input_mode,
            "beam_size": beam_size,
            "whisper_base_url": whisper_base_url,
        }
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "queue_bench_results.json"
    result_path.write_text(json.dumps(summary, indent=2))
    print(f"Results -> {result_path}")
    return summary


async def main() -> None:
    parser = argparse.ArgumentParser(description="Redis Streams queue benchmark for vLLM Whisper")
    parser.add_argument("--redis-url", default="redis://localhost:6379/0")
    parser.add_argument("--stream", default="whisper_jobs")
    parser.add_argument("--group", default="whisper_workers")
    parser.add_argument("--consumers", type=int, default=8)
    parser.add_argument("--ingress-rps", type=float, default=8.0)
    parser.add_argument("--duration-s", type=float, default=90.0)
    parser.add_argument("--model", default=cfg.VLLM_WHISPER_MODEL)
    parser.add_argument("--whisper-base-url", default=cfg.VLLM_WHISPER_BASE_URL)
    parser.add_argument("--beam-size", type=int, default=cfg.WHISPER_BEAM_SIZE)
    parser.add_argument("--input-mode", choices=["file", "pcm_f32le"], default=cfg.WHISPER_INPUT_MODE)
    parser.add_argument("--reset-stream", action="store_true", default=False)
    parser.add_argument("--output", default=str(Path(cfg.RESULTS_DIR) / "queue_bench"))
    args = parser.parse_args()

    summary = await run_queue_bench(
        redis_url=args.redis_url,
        stream=args.stream,
        group=args.group,
        consumers=args.consumers,
        ingress_rps=args.ingress_rps,
        duration_s=args.duration_s,
        model=args.model,
        whisper_base_url=args.whisper_base_url,
        beam_size=args.beam_size,
        input_mode=args.input_mode,
        reset_stream=args.reset_stream,
        output_dir=args.output,
    )
    print(
        "Queue benchmark summary: "
        f"produced={summary['produced']} processed={summary['processed']} ok={summary['successful']} "
        f"rps={summary['throughput_rps']:.2f} "
        f"queue_p95={summary['queue_wait_ms']['p95']:.0f}ms "
        f"service_p95={summary['service_latency_ms']['p95']:.0f}ms "
        f"e2e_p95={summary['e2e_latency_ms']['p95']:.0f}ms"
    )


if __name__ == "__main__":
    asyncio.run(main())
