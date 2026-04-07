"""
Master benchmark orchestrator.

Modes:
  whisper   — sweep one backend × one model (service must already be up)
  llm       — sweep LLM concurrency levels
  e2e       — sweep E2E session counts (both services must be up)
  compare   — full backend × model comparison sweep
              Starts/stops the Whisper service for each combination via docker compose.
  all       — compare + llm + e2e

Usage examples:
  python -m benchmarks.run_all --mode whisper --model large-v2 --backend faster_whisper
  python -m benchmarks.run_all --mode compare
  python -m benchmarks.run_all --mode all --output ./results
"""
import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path

from benchmarks import config as cfg
from metrics.reporter import generate_report


# ── Service helpers ───────────────────────────────────────────────────────────

async def _check_service(url: str, name: str, retries: int = 24, delay: float = 5.0) -> bool:
    import aiohttp
    for attempt in range(retries):
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(f"{url}/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        print(f"  + {name} is up ({url})")
                        return True
        except Exception:
            pass
        print(f"  waiting for {name}... ({attempt + 1}/{retries})")
        await asyncio.sleep(delay)
    print(f"  ! {name} not reachable at {url}", file=sys.stderr)
    return False


def _docker_up(backend: str, model: str, workers: int = 16, compute_type: str = "float16") -> None:
    """Start the Whisper-only compose with the given backend + model + compute type."""
    env = {
        "WHISPER_BACKEND":       backend,
        "WHISPER_MODEL":         model,
        "WHISPER_WORKERS":       str(workers),
        "WHISPER_COMPUTE_TYPE":  compute_type,
    }
    import os
    full_env = {**os.environ, **env}
    print(f"  Starting Whisper service: backend={backend} model={model} workers={workers}")
    subprocess.run(
        ["docker", "compose", "-f", "docker-compose.whisper-only.yml", "up", "-d", "--build"],
        env=full_env, check=True,
    )


def _docker_down() -> None:
    subprocess.run(
        ["docker", "compose", "-f", "docker-compose.whisper-only.yml", "down"],
        check=False,
    )


# ── Individual suites ─────────────────────────────────────────────────────────

async def run_whisper_suite(model: str, backend: str) -> dict:
    from benchmarks.whisper_bench import run_whisper_bench
    if not await _check_service(cfg.WHISPER_URL, f"Whisper ({backend}/{model})"):
        return {}
    results = await run_whisper_bench(model_label=model, backend=backend)
    return results


async def run_llm_suite() -> dict:
    from benchmarks.llm_bench import run_llm_bench
    if not await _check_service(cfg.LLM_URL, "LLM"):
        return {}
    return await run_llm_bench()


async def run_e2e_suite(session_counts: list[int] | None = None, n_chunks: int | None = None) -> dict:
    from benchmarks.e2e_bench import run_e2e_bench
    if not (
        await _check_service(cfg.WHISPER_URL, "Whisper") and
        await _check_service(cfg.LLM_URL, "LLM")
    ):
        return {}
    return await run_e2e_bench(session_counts=session_counts, n_chunks=n_chunks)


# ── Compare mode: both backends × all models ──────────────────────────────────
# Results structure:
#   all_results["whisper"][backend][model][concurrency] = summary_dict

async def run_compare_suite(
    output_dir: str,
    use_docker: bool = True,
    workers: int = 16,
    compute_types: list[str] | None = None,
) -> dict:
    """
    Sweep both backends across all their supported models (and optionally compute types).
    For each combination: optionally (re)start the Whisper container, run the bench, save partial.

    Args:
        use_docker:     If True, start/stop the Docker service per combination.
                        If False, assumes the service is already running.
        workers:        WHISPER_WORKERS for the Whisper-only compose.
        compute_types:  List of CTranslate2 compute types to test for faster-whisper.
                        e.g. ["float16", "int8_float16"].  openai-whisper always uses fp16.
                        Defaults to cfg.WHISPER_COMPUTE_TYPES (["float16"]).
    """
    if compute_types is None:
        compute_types = cfg.WHISPER_COMPUTE_TYPES

    # Results: {backend: {model: {compute_type: {concurrency: summary}}}}
    # For openai_whisper, compute_type is always "fp16" (keyed as "fp16").
    compare: dict = {"faster_whisper": {}, "openai_whisper": {}}

    # Build full sweep list: (backend, model, compute_type)
    combos: list[tuple[str, str, str]] = []
    for m in cfg.FASTER_WHISPER_MODELS:
        for ct in compute_types:
            combos.append(("faster_whisper", m, ct))
    for m in cfg.OPENAI_WHISPER_MODELS:
        combos.append(("openai_whisper", m, "fp16"))

    total = len(combos)
    for idx, (backend, model, compute_type) in enumerate(combos, 1):
        print(f"\n{'='*62}")
        print(f" [{idx}/{total}]  {backend}  /  {model}  /  {compute_type}")
        print(f"{'='*62}")

        if use_docker:
            _docker_down()
            _docker_up(backend, model, workers=workers, compute_type=compute_type)

        svc_label = f"{backend}/{model}/{compute_type}"
        if await _check_service(cfg.WHISPER_URL, svc_label, retries=24, delay=5):
            results = await run_whisper_suite(model, backend)
            compare[backend].setdefault(model, {})[compute_type] = results
        else:
            compare[backend].setdefault(model, {})[compute_type] = {"error": "service did not start"}

        # Save checkpoint after each combo
        _save_partial({"whisper_compare": compare}, output_dir, "whisper_compare_partial.json")

        if use_docker:
            _docker_down()
            await asyncio.sleep(5)   # brief pause to let VRAM fully release

    return compare


# ── Orchestrator ──────────────────────────────────────────────────────────────

async def main() -> None:
    parser = argparse.ArgumentParser(description="GPU Stress Test Runner")
    parser.add_argument("--mode",
        choices=["whisper", "llm", "e2e", "compare", "format", "all"],
        default="all",
        help="Benchmark mode: 'format' tests WAV/FLAC/Opus at 16k/48k")
    parser.add_argument("--model",   default="large-v2",       help="Model for --mode whisper")
    parser.add_argument("--backend", default="faster_whisper", help="Backend for --mode whisper")
    parser.add_argument("--sessions",default=None,             help="Comma-separated E2E session counts")
    parser.add_argument("--chunks",  type=int, default=None,   help="Chunks per E2E session")
    parser.add_argument("--workers", type=int, default=16,     help="WHISPER_WORKERS for compare mode")
    parser.add_argument("--compute-types", default=None,
        help="Comma-separated compute types for faster-whisper (default: float16). "
             "e.g. float16,int8_float16")
    parser.add_argument("--no-docker", action="store_true",
        help="Skip docker compose management (service already running)")
    parser.add_argument("--real-audio-dir", default=None,
        help="Path to real audio files folder (for --mode format)")
    parser.add_argument("--audio-formats", default=None,
        help="Comma-separated formats for format bench (default: wav,flac,opus)")
    parser.add_argument("--audio-rates", default=None,
        help="Comma-separated sample rates for format bench (default: 16000,48000)")
    parser.add_argument("--chunk-duration", type=float, default=None,
        help=f"Chunk duration in seconds for real audio (default {cfg.AUDIO_DURATION_S:.0f}s)")
    parser.add_argument("--output",  default=cfg.RESULTS_DIR,  help="Results output directory")
    args = parser.parse_args()

    cfg.RESULTS_DIR = args.output
    Path(cfg.RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    session_counts = (
        [int(x) for x in args.sessions.split(",")]
        if args.sessions else None
    )

    fmts         = [f.strip() for f in args.audio_formats.split(",")]  if args.audio_formats  else None
    rates        = [int(r.strip()) for r in args.audio_rates.split(",")]  if args.audio_rates    else None
    compute_types = [c.strip() for c in args.compute_types.split(",")]  if args.compute_types  else None
    if compute_types:
        cfg.WHISPER_COMPUTE_TYPES = compute_types

    all_results: dict = {
        "whisper":         {},   # single backend/model run
        "whisper_compare": {},   # compare mode: both backends × all models
        "llm":             {},
        "e2e":             {},
        "format":          {},   # format × sample_rate benchmark
    }

    # ── whisper (single backend/model) ────────────────────────────────────────
    if args.mode == "whisper":
        if await _check_service(cfg.WHISPER_URL, f"Whisper ({args.backend}/{args.model})"):
            all_results["whisper"] = {
                args.backend: {
                    args.model: await run_whisper_suite(args.model, args.backend)
                }
            }
        _save_partial(all_results, args.output)

    # ── compare (both backends × all models) ──────────────────────────────────
    elif args.mode in ("compare", "all"):
        all_results["whisper_compare"] = await run_compare_suite(
            output_dir=args.output,
            use_docker=not args.no_docker,
            workers=args.workers,
            compute_types=compute_types,
        )
        _save_partial(all_results, args.output)

    # ── format / real-audio ───────────────────────────────────────────────────
    if args.mode in ("format", "all"):
        from benchmarks.audio_format_bench import (
            run_synthetic_format_bench, run_real_audio_bench,
        )
        print(f"\n{'='*62}\n Audio format + sample rate sweep\n{'='*62}")

        # In --mode all, compare already tore down Docker — restart Whisper
        # using large-v2 as the reference model for the format sweep.
        use_docker = not args.no_docker
        if args.mode == "all" and use_docker:
            _docker_down()
            _docker_up("faster_whisper", args.model, workers=args.workers)

        if not await _check_service(cfg.WHISPER_URL, "Whisper"):
            print("  ! Whisper not reachable — skipping format bench")
        else:
            synth = await run_synthetic_format_bench(
                formats=fmts, sample_rates=rates,
                concurrency=cfg.FORMAT_BENCH_CONCURRENCY,
            )
            all_results["format"]["synthetic"] = synth

            real_dir = args.real_audio_dir or cfg.REAL_AUDIO_DIR
            if Path(real_dir).exists():
                print(f"\n  Real audio dir: {real_dir}")
                real = await run_real_audio_bench(
                    real_audio_dir=real_dir,
                    formats=fmts, sample_rates=rates,
                    concurrency=cfg.FORMAT_BENCH_CONCURRENCY,
                    chunk_duration_s=args.chunk_duration,
                )
                all_results["format"]["real"] = real
            else:
                print(f"  (no real audio dir at {real_dir} — skipping real audio bench)")

        if args.mode == "all" and use_docker:
            _docker_down()
            await asyncio.sleep(5)

        _save_partial(all_results, args.output)

    # ── llm ───────────────────────────────────────────────────────────────────
    if args.mode in ("llm", "all"):
        print(f"\n{'='*62}\n LLM sweep\n{'='*62}")
        all_results["llm"] = await run_llm_suite()
        _save_partial(all_results, args.output)

    # ── e2e ───────────────────────────────────────────────────────────────────
    if args.mode in ("e2e", "all"):
        print(f"\n{'='*62}\n E2E mixed workload sweep\n{'='*62}")
        all_results["e2e"] = await run_e2e_suite(
            session_counts=session_counts,
            n_chunks=args.chunks,
        )
        _save_partial(all_results, args.output)

    # ── Write consolidated JSON + generate report ──────────────────────────────
    consolidated = Path(args.output) / "all_results.json"
    consolidated.write_text(json.dumps(all_results, indent=2))
    print(f"\nConsolidated results -> {consolidated}")

    generate_report(
        all_results,
        gpu_metrics_path=Path(args.output) / "gpu_metrics.jsonl",
        output_path=Path(args.output) / "report.md",
    )
    print(f"Report           -> {Path(args.output) / 'report.md'}")
    print(f"\n{'='*62}\n DONE\n{'='*62}")


def _save_partial(results: dict, output_dir: str, filename: str = "all_results_partial.json") -> None:
    path = Path(output_dir) / filename
    path.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
