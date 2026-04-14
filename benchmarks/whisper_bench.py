"""vLLM Whisper throughput benchmark (OpenAI-compatible transcription API)."""
import asyncio
import argparse
import io
import itertools
import json
import math
import time
import wave
from pathlib import Path

import aiohttp
import numpy as np

from benchmarks import config as cfg
from generators.audio_gen import generate_pool, generate_pool_pcm_f32le
from metrics.collector import MetricsCollector, RequestRecord, set_gpu_label, clear_gpu_label


def _vllm_transcription_url() -> str:
    base = cfg.VLLM_WHISPER_URL.rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return f"{base}/audio/transcriptions"


def _pcm_f32le_to_wav_bytes(audio_bytes: bytes, sample_rate: int = 16000) -> bytes:
    """
    Convert raw float32 mono PCM bytes to WAV bytes in-memory.
    This keeps decoding client-side while sending a transport format the
    OpenAI-compatible vLLM transcription endpoint accepts.
    """
    pcm = np.frombuffer(audio_bytes, dtype=np.float32)
    # Clamp to [-1, 1] and quantize to int16 PCM for broad WAV compatibility.
    pcm_i16 = (np.clip(pcm, -1.0, 1.0) * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_i16.tobytes())
    return buf.getvalue()


async def _single_transcribe(
    session: aiohttp.ClientSession,
    audio_bytes: bytes,
    request_id: str,
    collector: MetricsCollector,
    backend: str,
    model_label: str,
    input_mode: str,
    beam_size: int,
) -> None:
    rec = RequestRecord(
        request_id=request_id,
        start_ts=time.perf_counter(),
        audio_duration_s=cfg.AUDIO_DURATION_S,
    )
    try:
        timeout = aiohttp.ClientTimeout(total=cfg.WHISPER_TIMEOUT)
        form = aiohttp.FormData()
        upload_bytes = audio_bytes
        if input_mode == "pcm_f32le":
            upload_bytes = _pcm_f32le_to_wav_bytes(audio_bytes, sample_rate=16000)
        form.add_field("file", upload_bytes, filename="audio.wav", content_type="audio/wav")
        form.add_field("model", model_label)
        form.add_field("language", "en")
        form.add_field("response_format", "json")
        form.add_field("temperature", "0.0")
        url = _vllm_transcription_url()
        async with session.post(url, data=form, timeout=timeout, params={"beam_size": str(beam_size)}) as resp:
            rec.end_ts     = time.perf_counter()
            rec.latency_ms = (rec.end_ts - rec.start_ts) * 1000
            if resp.status == 200:
                body = await resp.json()
                rec.tokens_out = len((body.get("text") or "").split())
            else:
                rec.success = False
                rec.error   = f"HTTP {resp.status}"
    except Exception as exc:
        rec.end_ts     = time.perf_counter()
        rec.latency_ms = (rec.end_ts - rec.start_ts) * 1000
        rec.success    = False
        rec.error      = str(exc)
    await collector.record(rec)


async def _run_concurrency_level_batch(
    concurrency: int,
    audio_pool: list[bytes],
    test_name: str,
    session: aiohttp.ClientSession,
    backend: str,
    model_label: str,
    input_mode: str,
    beam_size: int,
) -> MetricsCollector:
    audio_cycle = itertools.cycle(audio_pool)
    collector = MetricsCollector(test_name)
    n_rounds = max(1, math.ceil(cfg.MIN_SAMPLES / concurrency))
    for round_i in range(n_rounds):
        tasks = [
            _single_transcribe(
                session,
                next(audio_cycle),
                f"r{round_i}_req{j}",
                collector,
                backend,
                model_label,
                input_mode,
                beam_size,
            )
            for j in range(concurrency)
        ]
        await asyncio.gather(*tasks)
    return collector


async def _run_concurrency_level_steady(
    concurrency: int,
    audio_pool: list[bytes],
    test_name: str,
    session: aiohttp.ClientSession,
    backend: str,
    model_label: str,
    input_mode: str,
    beam_size: int,
) -> MetricsCollector:
    """
    Maintain `concurrency` concurrent request pipelines until duration elapses
    (each pipeline issues the next request as soon as the previous completes).
    """
    audio_cycle = itertools.cycle(audio_pool)
    collector = MetricsCollector(test_name)
    duration = max(1.0, cfg.WHISPER_STEADY_STATE_DURATION_S)

    async def worker(wid: int) -> None:
        n = 0
        while True:
            if time.perf_counter() >= deadline:
                break
            await _single_transcribe(
                session,
                next(audio_cycle),
                f"w{wid}_r{n}",
                collector,
                backend,
                model_label,
                input_mode,
                beam_size,
            )
            n += 1

    deadline = time.perf_counter() + duration
    await asyncio.gather(*[worker(j) for j in range(concurrency)])
    return collector


async def _run_concurrency_level(
    concurrency: int,
    audio_pool: list[bytes],
    label: str,          # e.g. "fw_large-v2" or "ow_turbo"
    backend: str,
    model_label: str,
    input_mode: str,
    beam_size: int,
) -> dict:
    load_mode = cfg.WHISPER_LOAD_MODE if cfg.WHISPER_LOAD_MODE in ("batch", "steady") else "batch"
    test_name = f"whisper_{label}_c{concurrency}"
    audio_cycle = itertools.cycle(audio_pool)

    connector = aiohttp.TCPConnector(limit=concurrency + 10)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Warmup
        warmup_col = MetricsCollector(f"{test_name}_warmup")
        for i in range(cfg.WARMUP_REQUESTS):
            await _single_transcribe(
                session, next(audio_cycle), f"w{i}", warmup_col, backend, model_label, input_mode, beam_size
            )
        w = warmup_col.summarize()
        print(f"    warmup  avg={w.get('latency_ms', {}).get('mean', 0):.0f}ms")

        if load_mode == "steady":
            print(
                f"    load=steady  duration={cfg.WHISPER_STEADY_STATE_DURATION_S:.0f}s  "
                f"pipelines={concurrency}"
            )
            collector = await _run_concurrency_level_steady(
                concurrency,
                audio_pool,
                test_name,
                session,
                backend,
                model_label,
                input_mode,
                beam_size,
            )
        else:
            collector = await _run_concurrency_level_batch(
                concurrency,
                audio_pool,
                test_name,
                session,
                backend,
                model_label,
                input_mode,
                beam_size,
            )

    summary = collector.summarize()
    summary["concurrency"] = concurrency
    summary["load_mode"] = load_mode
    if load_mode == "steady":
        summary["steady_state_duration_s"] = cfg.WHISPER_STEADY_STATE_DURATION_S
    return summary


async def run_whisper_bench(
    model_label: str = "unknown",
    backend: str = "vllm_whisper",
    whisper_url: str | None = None,
    label_prefix: str | None = None,
    input_mode: str = "file",
    beam_size: int = 5,
) -> dict:
    """
    Run the full Whisper concurrency sweep.

    Returns:
        dict keyed by concurrency level (str) → summary dict.
        Each summary includes a "backend" and "model" field for the matrix reporter.
    """
    if whisper_url:
        cfg.VLLM_WHISPER_URL = whisper_url.rstrip("/")

    prefix = "vw"
    safe_model = model_label.replace("/", "_").replace(":", "_")
    label  = label_prefix or f"{prefix}_{safe_model}"
    target_url = cfg.VLLM_WHISPER_URL

    effective_input_mode = input_mode
    load_mode_label = cfg.WHISPER_LOAD_MODE if cfg.WHISPER_LOAD_MODE in ("batch", "steady") else "batch"
    print(
        f"\n=== Whisper Bench  backend={backend}  model={model_label}  url={target_url}  "
        f"input_mode={effective_input_mode}  beam_size={beam_size}  load_mode={load_mode_label} ==="
    )
    if load_mode_label == "steady":
        print(f"  steady_state_duration_s={cfg.WHISPER_STEADY_STATE_DURATION_S:.0f}")
    if effective_input_mode == "pcm_f32le":
        print(f"Generating {cfg.AUDIO_POOL_SIZE} × {cfg.AUDIO_DURATION_S}s pre-decoded PCM payloads...")
        audio_pool = generate_pool_pcm_f32le(cfg.AUDIO_POOL_SIZE, cfg.AUDIO_DURATION_S)
        print(f"  {len(audio_pool)} payloads ready  ({len(audio_pool[0]) // 1024} KB each)")
    else:
        print(f"Generating {cfg.AUDIO_POOL_SIZE} × {cfg.AUDIO_DURATION_S}s audio files...")
        audio_pool = generate_pool(cfg.AUDIO_POOL_SIZE, cfg.AUDIO_DURATION_S)
        print(f"  {len(audio_pool)} files ready  ({len(audio_pool[0]) // 1024} KB each)")

    results: dict[str, dict] = {}
    for concurrency in cfg.WHISPER_CONCURRENCY:
        print(f"\n  concurrency={concurrency}")
        set_gpu_label(f"{label}_c{concurrency}")
        summary = await _run_concurrency_level(
            concurrency,
            audio_pool,
            label,
            backend,
            model_label,
            effective_input_mode,
            beam_size,
        )
        # Stamp backend + model onto every result for the reporter
        summary["backend"] = backend
        summary["model"]   = model_label
        summary["input_mode"] = effective_input_mode
        summary["beam_size"] = beam_size
        results[str(concurrency)] = summary

        lm  = summary.get("latency_ms", {})
        rtf = summary.get("audio_realtime_factor")
        rtf_str = f"RTF={rtf:.1f}x" if rtf else ""
        print(
            f"    rps={summary.get('throughput_rps', 0):.2f}  "
            f"p50={lm.get('p50', 0):.0f}ms  p99={lm.get('p99', 0):.0f}ms  "
            f"{rtf_str}  err={summary.get('error_rate_pct', 0):.1f}%"
        )

        if concurrency != cfg.WHISPER_CONCURRENCY[-1]:
            print(f"  cooling down {cfg.COOLDOWN_S}s...")
            await asyncio.sleep(cfg.COOLDOWN_S)

    clear_gpu_label()
    return results


def _save(results: dict, label: str) -> None:
    out = Path(cfg.RESULTS_DIR)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"whisper_{label}.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {path}")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=cfg.VLLM_WHISPER_MODEL, help="vLLM Whisper model label")
    parser.add_argument("--backend", default="vllm_whisper", help="Only vllm_whisper is supported")
    parser.add_argument("--url", default=None, help="Override VLLM_WHISPER_URL")
    parser.add_argument("--input-mode", default="file", choices=["file", "pcm_f32le"], help="Benchmark input mode")
    parser.add_argument("--beam-size", type=int, default=5, help="Whisper beam size")
    parser.add_argument("--load-mode", choices=["batch", "steady"], default=None,
        help="Override WHISPER_LOAD_MODE (batch or steady in-flight pipelines)")
    parser.add_argument("--steady-duration-s", type=float, default=None,
        help="Override WHISPER_STEADY_STATE_DURATION_S for steady mode")
    args = parser.parse_args()

    if args.load_mode:
        cfg.WHISPER_LOAD_MODE = args.load_mode
    if args.steady_duration_s is not None:
        cfg.WHISPER_STEADY_STATE_DURATION_S = float(args.steady_duration_s)

    prefix = "vw"
    label = f"{prefix}_{args.model.replace('/', '_').replace(':', '_')}"
    results = await run_whisper_bench(
        model_label=args.model,
        backend=args.backend,
        whisper_url=args.url,
        input_mode=args.input_mode,
        beam_size=args.beam_size,
    )
    _save(results, label)


if __name__ == "__main__":
    asyncio.run(main())
