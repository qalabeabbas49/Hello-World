"""
Whisper throughput benchmark — supports both faster-whisper and openai-whisper backends.

The running Whisper service is already configured for a specific backend+model via env vars.
This script hits the HTTP service, so it's backend-agnostic at the client level.
The `backend` and `model_label` arguments are used purely for result labelling.

Methodology per concurrency level:
  1. Warmup: WARMUP_REQUESTS sequential requests.
  2. Test:   asyncio.gather(concurrency tasks) repeated for ceil(MIN_SAMPLES/concurrency) rounds.
  3. Cooldown: COOLDOWN_S before next level.

Run directly:
  python -m benchmarks.whisper_bench --model large-v2 --backend faster_whisper
"""
import asyncio
import argparse
import itertools
import json
import time
from pathlib import Path

import aiohttp

from benchmarks import config as cfg
from generators.audio_gen import generate_pool
from metrics.collector import MetricsCollector, RequestRecord, set_gpu_label, clear_gpu_label


async def _single_transcribe(
    session: aiohttp.ClientSession,
    audio_bytes: bytes,
    request_id: str,
    collector: MetricsCollector,
) -> None:
    rec = RequestRecord(
        request_id=request_id,
        start_ts=time.perf_counter(),
        audio_duration_s=cfg.AUDIO_DURATION_S,
    )
    try:
        form = aiohttp.FormData()
        form.add_field("file", audio_bytes, filename="audio.wav", content_type="audio/wav")
        timeout = aiohttp.ClientTimeout(total=cfg.WHISPER_TIMEOUT)
        async with session.post(f"{cfg.WHISPER_URL}/transcribe", data=form, timeout=timeout) as resp:
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


async def _run_concurrency_level(
    concurrency: int,
    audio_pool: list[bytes],
    label: str,          # e.g. "fw_large-v2" or "ow_turbo"
) -> dict:
    test_name   = f"whisper_{label}_c{concurrency}"
    audio_cycle = itertools.cycle(audio_pool)

    connector = aiohttp.TCPConnector(limit=concurrency + 10)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Warmup
        warmup_col = MetricsCollector(f"{test_name}_warmup")
        for i in range(cfg.WARMUP_REQUESTS):
            await _single_transcribe(session, next(audio_cycle), f"w{i}", warmup_col)
        w = warmup_col.summarize()
        print(f"    warmup  avg={w.get('latency_ms', {}).get('mean', 0):.0f}ms")

        # Actual test
        collector = MetricsCollector(test_name)
        n_rounds  = max(1, cfg.MIN_SAMPLES // concurrency)
        for round_i in range(n_rounds):
            tasks = [
                _single_transcribe(session, next(audio_cycle), f"r{round_i}_req{j}", collector)
                for j in range(concurrency)
            ]
            await asyncio.gather(*tasks)

    summary = collector.summarize()
    summary["concurrency"] = concurrency
    return summary


async def run_whisper_bench(
    model_label: str = "unknown",
    backend: str = "faster_whisper",
    whisper_url: str | None = None,
) -> dict:
    """
    Run the full Whisper concurrency sweep.

    Returns:
        dict keyed by concurrency level (str) → summary dict.
        Each summary includes a "backend" and "model" field for the matrix reporter.
    """
    if whisper_url:
        cfg.WHISPER_URL = whisper_url

    # Short prefix for result keys: "fw" or "ow"
    prefix = "fw" if "faster" in backend else "ow"
    label  = f"{prefix}_{model_label}"

    print(f"\n=== Whisper Bench  backend={backend}  model={model_label}  url={cfg.WHISPER_URL} ===")
    print(f"Generating {cfg.AUDIO_POOL_SIZE} × {cfg.AUDIO_DURATION_S}s audio files...")
    audio_pool = generate_pool(cfg.AUDIO_POOL_SIZE, cfg.AUDIO_DURATION_S)
    print(f"  {len(audio_pool)} files ready  ({len(audio_pool[0]) // 1024} KB each)")

    results: dict[str, dict] = {}
    for concurrency in cfg.WHISPER_CONCURRENCY:
        print(f"\n  concurrency={concurrency}")
        set_gpu_label(f"{label}_c{concurrency}")
        summary = await _run_concurrency_level(concurrency, audio_pool, label)
        # Stamp backend + model onto every result for the reporter
        summary["backend"] = backend
        summary["model"]   = model_label
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
    parser.add_argument("--model",   default="large-v2",      help="Whisper model label")
    parser.add_argument("--backend", default="faster_whisper", help="faster_whisper | openai_whisper")
    parser.add_argument("--url",     default=None,             help="Override WHISPER_URL")
    args = parser.parse_args()

    prefix  = "fw" if "faster" in args.backend else "ow"
    label   = f"{prefix}_{args.model}"
    results = await run_whisper_bench(model_label=args.model, backend=args.backend, whisper_url=args.url)
    _save(results, label)


if __name__ == "__main__":
    asyncio.run(main())
