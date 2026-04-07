"""
Audio format + sample rate benchmark.

Tests two independent dimensions:
  Format:      WAV | FLAC | Opus
  Sample rate: 16000 Hz | 48000 Hz

For each (format, sample_rate) combination, sends audio to the running
Whisper service and measures latency + throughput at a fixed concurrency.
This isolates the cost of: (a) format decoding, (b) service-side resampling.

Also supports real audio files from a user-provided directory — same
measurements, same metrics, but with actual patient/doctor recordings.

Usage (synthetic audio):
  python -m benchmarks.audio_format_bench --mode synthetic
  python -m benchmarks.audio_format_bench --mode synthetic --formats wav,flac,opus --rates 16000,48000

Usage (real audio files):
  python -m benchmarks.audio_format_bench --mode real --real-audio-dir /path/to/audio
  python -m benchmarks.audio_format_bench --mode real --real-audio-dir /data/audio --fmt wav --rate 48000

Usage (both):
  python -m benchmarks.audio_format_bench --mode both --real-audio-dir /path/to/audio
"""
import argparse
import asyncio
import itertools
import json
import time
from pathlib import Path

import aiohttp

from benchmarks import config as cfg
from metrics.collector import MetricsCollector, RequestRecord


# ── Single request ─────────────────────────────────────────────────────────────

async def _send_audio(
    session: aiohttp.ClientSession,
    audio_bytes: bytes,
    content_type: str,
    filename: str,
    request_id: str,
    collector: MetricsCollector,
    audio_duration_s: float | None = None,   # None → use cfg default (synthetic)
) -> None:
    rec = RequestRecord(
        request_id=request_id,
        start_ts=time.perf_counter(),
        # Use provided duration (real files) or fall back to synthetic default.
        # Correct duration is required for RTF to be meaningful.
        audio_duration_s=audio_duration_s if audio_duration_s is not None else cfg.AUDIO_DURATION_S,
    )
    try:
        form = aiohttp.FormData()
        form.add_field("file", audio_bytes, filename=filename, content_type=content_type)
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


# ── Pool runner ────────────────────────────────────────────────────────────────

async def _run_pool(
    audio_pool: list[bytes],
    content_type: str,
    filename_ext: str,
    test_label: str,
    concurrency: int,
    durations: list[float] | None = None,   # per-file durations; None → use cfg default
) -> dict:
    """
    Run one format+rate combination at a fixed concurrency.
    Cycles through the pool, collecting MIN_SAMPLES data points.

    Args:
        durations: List of audio durations (seconds) parallel to audio_pool.
                   Used for real audio files so RTF is calculated correctly.
                   Pass None for synthetic audio (uses cfg.AUDIO_DURATION_S).
    """
    collector      = MetricsCollector(test_label)
    audio_cycle    = itertools.cycle(audio_pool)
    duration_cycle = itertools.cycle(durations) if durations else itertools.repeat(None)

    connector = aiohttp.TCPConnector(limit=concurrency + 4)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Warmup
        for i in range(cfg.WARMUP_REQUESTS):
            await _send_audio(
                session, next(audio_cycle), content_type,
                f"warmup_{i}.{filename_ext}", f"w{i}", collector,
                audio_duration_s=next(duration_cycle),
            )
        collector.reset()

        # Test
        n_rounds = max(1, cfg.MIN_SAMPLES // concurrency)
        for round_i in range(n_rounds):
            tasks = [
                _send_audio(
                    session, next(audio_cycle), content_type,
                    f"r{round_i}_req{j}.{filename_ext}", f"r{round_i}_req{j}", collector,
                    audio_duration_s=next(duration_cycle),
                )
                for j in range(concurrency)
            ]
            await asyncio.gather(*tasks)

    return collector.summarize()


# ── Synthetic sweep ────────────────────────────────────────────────────────────

async def run_synthetic_format_bench(
    formats: list[str] | None = None,
    sample_rates: list[int] | None = None,
    concurrency: int = 10,
) -> dict:
    """
    Sweep all (format, sample_rate) combinations using synthetic audio.

    Returns:
        Nested dict:  results[format][sample_rate] = summary
    """
    from generators.audio_gen import generate_pool, _content_type

    if formats is None:
        formats = cfg.AUDIO_FORMATS
    if sample_rates is None:
        sample_rates = cfg.AUDIO_SAMPLE_RATES

    print(f"\n=== Synthetic Format Benchmark  c={concurrency} ===")
    print(f"Formats: {formats}   Rates: {sample_rates}")

    results: dict[str, dict] = {}

    for fmt in formats:
        results[fmt] = {}
        for rate in sample_rates:
            label = f"synthetic_{fmt}_{rate}"
            print(f"\n  [{fmt} @ {rate//1000}kHz]  generating pool...")
            try:
                pool = generate_pool(cfg.AUDIO_POOL_SIZE, cfg.AUDIO_DURATION_S, rate, fmt)
            except Exception as exc:
                print(f"    SKIP — could not generate {fmt}: {exc}")
                results[fmt][str(rate)] = {"error": str(exc), "skipped": True}
                continue

            ext  = "ogg" if fmt == "opus" else fmt
            ct   = _content_type(fmt)
            print(f"  [{fmt} @ {rate//1000}kHz]  {len(pool)} files, ~{len(pool[0])//1024}KB each")

            summary = await _run_pool(pool, ct, ext, label, concurrency)
            summary["fmt"]    = fmt
            summary["rate"]   = rate
            summary["source"] = "synthetic"
            results[fmt][str(rate)] = summary

            lm = summary.get("latency_ms", {})
            print(
                f"    p50={lm.get('p50', 0):.0f}ms  p95={lm.get('p95', 0):.0f}ms  "
                f"rps={summary.get('throughput_rps', 0):.2f}  "
                f"err={summary.get('error_rate_pct', 0):.1f}%"
            )

    return results


# ── Real audio sweep ───────────────────────────────────────────────────────────

async def run_real_audio_bench(
    real_audio_dir: str,
    formats: list[str] | None = None,
    sample_rates: list[int] | None = None,
    concurrency: int = 10,
    max_files_per_combo: int = 20,
    chunk_duration_s: float | None = None,
) -> dict:
    """
    Benchmark using real audio files split into production-sized chunks.

    Each source file is decoded to PCM and split into fixed-length WAV chunks
    (default: cfg.AUDIO_DURATION_S = 47 s) before being sent to Whisper.
    This exactly replicates the production streaming pipeline where the client
    segments continuous audio every 47 seconds.

    Files are grouped by original format + sample rate so results remain
    comparable across recording types. Within each group all derived chunks
    are sent as standard WAV at the source sample rate.

    Returns:
        Nested dict:  results[format][sample_rate] = summary
    """
    from generators.real_audio import RealAudioPool

    chunk_s = chunk_duration_s if chunk_duration_s is not None else cfg.AUDIO_DURATION_S

    pool_loader = RealAudioPool.from_directory(real_audio_dir, probe=True)
    pool_loader.print_summary()

    if formats is None:
        formats = list(pool_loader.summary().keys())
    if sample_rates is None:
        all_rates: set[int] = set()
        for info in pool_loader.summary().values():
            all_rates.update(info.get("sample_rates", []))
        sample_rates = sorted(all_rates) or cfg.AUDIO_SAMPLE_RATES

    print(f"\n=== Real Audio Benchmark  chunk={chunk_s:.0f}s  c={concurrency} ===")
    print(f"Formats: {formats}   Rates: {sample_rates}")

    results: dict[str, dict] = {}

    for fmt in formats:
        results[fmt] = {}
        for rate in sample_rates:
            label = f"real_{fmt}_{rate}"
            print(f"\n  [{fmt} @ {rate//1000}kHz]  loading + chunking files...")
            try:
                chunks = pool_loader.get_chunks(
                    fmt=fmt,
                    sample_rate=rate,
                    max_files=max_files_per_combo,
                    chunk_duration_s=chunk_s,
                )
            except ValueError as e:
                print(f"    SKIP — {e}")
                results[fmt][str(rate)] = {"skipped": True, "reason": str(e)}
                continue

            pool      = [c.data for c in chunks]
            durations = [c.duration_s for c in chunks]
            n_src     = len({c.source_path for c in chunks})

            print(
                f"  [{fmt} @ {rate//1000}kHz]  "
                f"{n_src} source files → {len(pool)} × {chunk_s:.0f}s WAV chunks"
            )

            # WAV chunks at source sample rate — content_type is always audio/wav
            summary = await _run_pool(
                pool, "audio/wav", "wav", label, concurrency, durations=durations,
            )
            summary["fmt"]          = fmt
            summary["rate"]         = rate
            summary["source"]       = "real"
            summary["n_source_files"] = n_src
            summary["n_chunks"]     = len(pool)
            summary["chunk_s"]      = chunk_s
            results[fmt][str(rate)] = summary

            lm = summary.get("latency_ms", {})
            print(
                f"    p50={lm.get('p50', 0):.0f}ms  p95={lm.get('p95', 0):.0f}ms  "
                f"rps={summary.get('throughput_rps', 0):.2f}  "
                f"RTF={summary.get('audio_realtime_factor', 0):.1f}x  "
                f"err={summary.get('error_rate_pct', 0):.1f}%"
            )

    return results


# ── CLI entry point ────────────────────────────────────────────────────────────

async def main() -> None:
    parser = argparse.ArgumentParser(description="Audio format + sample rate benchmark")
    parser.add_argument("--mode",
        choices=["synthetic", "real", "both"], default="synthetic",
        help="synthetic: use generated audio; real: use files from --real-audio-dir; both: run both")
    parser.add_argument("--real-audio-dir", default=None,
        help="Path to folder with real audio files (required for --mode real/both)")
    parser.add_argument("--formats",  default=None,
        help="Comma-separated formats to test (default: all). e.g. wav,flac,opus")
    parser.add_argument("--rates",    default=None,
        help="Comma-separated sample rates (default: all). e.g. 16000,48000")
    parser.add_argument("--concurrency", type=int, default=10,
        help="Concurrent requests per format/rate combo (default 10)")
    parser.add_argument("--max-files", type=int, default=20,
        help="Max real audio files per format+rate combo (default 20)")
    parser.add_argument("--chunk-duration", type=float, default=None,
        help=f"Chunk duration in seconds for real audio (default {cfg.AUDIO_DURATION_S:.0f}s)")
    parser.add_argument("--output",   default=cfg.RESULTS_DIR,
        help="Results output directory")
    args = parser.parse_args()

    fmts  = [f.strip() for f in args.formats.split(",")] if args.formats else None
    rates = [int(r.strip()) for r in args.rates.split(",")]   if args.rates   else None

    Path(args.output).mkdir(parents=True, exist_ok=True)
    all_results: dict = {}

    if args.mode in ("synthetic", "both"):
        synth = await run_synthetic_format_bench(
            formats=fmts, sample_rates=rates, concurrency=args.concurrency,
        )
        all_results["synthetic"] = synth

    if args.mode in ("real", "both"):
        if not args.real_audio_dir:
            parser.error("--real-audio-dir is required for --mode real/both")
        real = await run_real_audio_bench(
            real_audio_dir=args.real_audio_dir,
            formats=fmts, sample_rates=rates,
            concurrency=args.concurrency,
            max_files_per_combo=args.max_files,
            chunk_duration_s=args.chunk_duration,
        )
        all_results["real"] = real

    # Save
    out_path = Path(args.output) / "format_bench.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
