"""
Whisper throughput benchmark.

For each model size (controlled by the running service's WHISPER_MODEL env):
  1. Pre-generate AUDIO_POOL_SIZE unique 47-second WAV files.
  2. Warm up (WARMUP_REQUESTS sequential requests).
  3. For each concurrency level:
     a. Send `concurrency` requests simultaneously via asyncio.gather().
     b. Repeat until ≥ MIN_SAMPLES data points are collected.
     c. Sleep COOLDOWN_S before the next level.
  4. Return summary dict.

Run directly:
  python -m benchmarks.whisper_bench          # uses .env / environment
  python -m benchmarks.whisper_bench --url http://gpu-box:8001
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
from metrics.collector import MetricsCollector, RequestRecord


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
    model_label: str,
) -> dict:
    test_name = f"whisper_{model_label}_c{concurrency}"
    audio_cycle = itertools.cycle(audio_pool)

    connector = aiohttp.TCPConnector(limit=concurrency + 10)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Warmup
        warmup_col = MetricsCollector(f"{test_name}_warmup")
        for i in range(cfg.WARMUP_REQUESTS):
            await _single_transcribe(session, next(audio_cycle), f"w{i}", warmup_col)
        print(f"    warmup done  (avg {warmup_col.summarize().get('latency_ms', {}).get('mean', 0):.0f} ms)")

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


async def run_whisper_bench(model_label: str = "unknown", whisper_url: str | None = None) -> dict:
    """
    Run the full Whisper concurrency sweep and return results keyed by concurrency level.
    """
    if whisper_url:
        cfg.WHISPER_URL = whisper_url

    print(f"\n=== Whisper Benchmark  model={model_label}  url={cfg.WHISPER_URL} ===")
    print(f"Generating {cfg.AUDIO_POOL_SIZE} × {cfg.AUDIO_DURATION_S}s audio files...")
    audio_pool = generate_pool(cfg.AUDIO_POOL_SIZE, cfg.AUDIO_DURATION_S)
    print(f"  {len(audio_pool)} files ready  ({len(audio_pool[0]) / 1024:.0f} KB each)")

    results: dict[str, dict] = {}
    for concurrency in cfg.WHISPER_CONCURRENCY:
        print(f"\n  concurrency = {concurrency}")
        summary = await _run_concurrency_level(concurrency, audio_pool, model_label)
        results[str(concurrency)] = summary

        lm  = summary.get("latency_ms", {})
        rtf = summary.get("audio_realtime_factor")
        print(
            f"    rps={summary.get('throughput_rps', 0):.2f}  "
            f"p50={lm.get('p50', 0):.0f}ms  p99={lm.get('p99', 0):.0f}ms  "
            f"RTF={rtf:.2f}×  err={summary.get('error_rate_pct', 0):.1f}%"
            if rtf else
            f"    rps={summary.get('throughput_rps', 0):.2f}  "
            f"p50={lm.get('p50', 0):.0f}ms  p99={lm.get('p99', 0):.0f}ms  "
            f"err={summary.get('error_rate_pct', 0):.1f}%"
        )

        if concurrency != cfg.WHISPER_CONCURRENCY[-1]:
            print(f"  cooling down {cfg.COOLDOWN_S}s...")
            await asyncio.sleep(cfg.COOLDOWN_S)

    return results


def _save(results: dict, model_label: str) -> None:
    out = Path(cfg.RESULTS_DIR)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"whisper_{model_label}.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {path}")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="unknown", help="Whisper model label (for filename)")
    parser.add_argument("--url",   default=None,      help="Override WHISPER_URL")
    args = parser.parse_args()

    results = await run_whisper_bench(model_label=args.model, whisper_url=args.url)
    _save(results, args.model)


if __name__ == "__main__":
    asyncio.run(main())
