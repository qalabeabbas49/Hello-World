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


def _docker_up(backend: str, model: str, workers: int = 16) -> None:
    """Start the Whisper-only compose with the given backend + model."""
    env = {
        "WHISPER_BACKEND":       backend,
        "WHISPER_MODEL":         model,
        "WHISPER_WORKERS":       str(workers),
        "WHISPER_COMPUTE_TYPE":  "float16",
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
) -> dict:
    """
    Sweep both backends across all their supported models.
    For each combination: optionally (re)start the Whisper container, run the bench, save partial.

    Args:
        use_docker: If True, start/stop the Docker service per combination.
                    If False, assumes the service is already running (useful for manual testing).
        workers:    WHISPER_WORKERS for the Whisper-only compose.
    """
    # Flat results: {backend: {model: {concurrency: summary}}}
    compare: dict = {"faster_whisper": {}, "openai_whisper": {}}

    backend_model_pairs = [
        ("faster_whisper", m) for m in cfg.FASTER_WHISPER_MODELS
    ] + [
        ("openai_whisper",  m) for m in cfg.OPENAI_WHISPER_MODELS
    ]

    total = len(backend_model_pairs)
    for idx, (backend, model) in enumerate(backend_model_pairs, 1):
        print(f"\n{'='*62}")
        print(f" [{idx}/{total}]  {backend}  /  {model}")
        print(f"{'='*62}")

        if use_docker:
            _docker_down()
            _docker_up(backend, model, workers=workers)

        if await _check_service(cfg.WHISPER_URL, f"{backend}/{model}", retries=24, delay=5):
            results = await run_whisper_suite(model, backend)
            compare[backend][model] = results
        else:
            compare[backend][model] = {"error": "service did not start"}

        # Save checkpoint after each model
        _save_partial({"whisper_compare": compare}, output_dir, "whisper_compare_partial.json")

        if use_docker:
            _docker_down()
            await asyncio.sleep(5)   # brief pause to let VRAM fully release

    return compare


# ── Orchestrator ──────────────────────────────────────────────────────────────

async def main() -> None:
    parser = argparse.ArgumentParser(description="GPU Stress Test Runner")
    parser.add_argument("--mode",
        choices=["whisper", "llm", "e2e", "compare", "all"],
        default="all",
        help="Benchmark mode")
    parser.add_argument("--model",   default="large-v2",       help="Model for --mode whisper")
    parser.add_argument("--backend", default="faster_whisper", help="Backend for --mode whisper")
    parser.add_argument("--sessions",default=None,             help="Comma-separated E2E session counts")
    parser.add_argument("--chunks",  type=int, default=None,   help="Chunks per E2E session")
    parser.add_argument("--workers", type=int, default=16,     help="WHISPER_WORKERS for compare mode")
    parser.add_argument("--no-docker", action="store_true",
        help="Skip docker compose management (service already running)")
    parser.add_argument("--output",  default=cfg.RESULTS_DIR,  help="Results output directory")
    args = parser.parse_args()

    cfg.RESULTS_DIR = args.output
    Path(cfg.RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    session_counts = (
        [int(x) for x in args.sessions.split(",")]
        if args.sessions else None
    )

    all_results: dict = {
        "whisper":         {},   # single backend/model run
        "whisper_compare": {},   # compare mode: both backends × all models
        "llm":             {},
        "e2e":             {},
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
        )
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
