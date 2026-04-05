"""
Master benchmark orchestrator.

Runs the full benchmark suite and generates a consolidated markdown report.

Usage:
  # Full suite (Whisper sweep across all models → LLM → E2E):
  python -m benchmarks.run_all --mode all

  # Whisper-only (service must already be running):
  python -m benchmarks.run_all --mode whisper --model large-v2

  # LLM-only:
  python -m benchmarks.run_all --mode llm

  # E2E mixed:
  python -m benchmarks.run_all --mode e2e

  # Custom session counts and chunk count:
  python -m benchmarks.run_all --mode e2e --sessions 50,100,200 --chunks 13
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

from benchmarks import config as cfg
from metrics.reporter import generate_report


async def _check_service(url: str, name: str, retries: int = 6, delay: float = 5.0) -> bool:
    import aiohttp
    for attempt in range(retries):
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(f"{url}/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        print(f"  ✓ {name} is up ({url})")
                        return True
        except Exception:
            pass
        print(f"  waiting for {name}... ({attempt + 1}/{retries})")
        await asyncio.sleep(delay)
    print(f"  ✗ {name} not reachable at {url}", file=sys.stderr)
    return False


async def run_whisper_suite(model_label: str) -> dict:
    from benchmarks.whisper_bench import run_whisper_bench
    if not await _check_service(cfg.WHISPER_URL, "Whisper"):
        return {}
    return {model_label: await run_whisper_bench(model_label=model_label)}


async def run_llm_suite() -> dict:
    from benchmarks.llm_bench import run_llm_bench
    if not await _check_service(cfg.LLM_URL, "LLM"):
        return {}
    return await run_llm_bench()


async def run_e2e_suite(session_counts: list[int] | None = None, n_chunks: int | None = None) -> dict:
    from benchmarks.e2e_bench import run_e2e_bench
    whisper_ok = await _check_service(cfg.WHISPER_URL, "Whisper")
    llm_ok     = await _check_service(cfg.LLM_URL,     "LLM")
    if not (whisper_ok and llm_ok):
        return {}
    return await run_e2e_bench(session_counts=session_counts, n_chunks=n_chunks)


async def main() -> None:
    parser = argparse.ArgumentParser(description="GPU Stress Test Runner")
    parser.add_argument("--mode",     choices=["whisper", "llm", "e2e", "all"], default="all")
    parser.add_argument("--model",    default="large-v2",  help="Whisper model label for --mode whisper")
    parser.add_argument("--sessions", default=None,        help="Comma-separated E2E session counts")
    parser.add_argument("--chunks",   type=int, default=None, help="Chunks per E2E session")
    parser.add_argument("--output",   default=cfg.RESULTS_DIR, help="Results output directory")
    args = parser.parse_args()

    cfg.RESULTS_DIR = args.output
    Path(cfg.RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    session_counts = (
        [int(x) for x in args.sessions.split(",")]
        if args.sessions else None
    )

    all_results: dict = {"whisper": {}, "llm": {}, "e2e": {}}

    if args.mode in ("whisper", "all"):
        models = cfg.WHISPER_MODELS if args.mode == "all" else [args.model]
        for model in models:
            print(f"\n{'='*60}")
            print(f" Whisper sweep: {model}")
            print(f"{'='*60}")
            whisper_res = await run_whisper_suite(model)
            all_results["whisper"].update(whisper_res)
            _save_partial(all_results, args.output)

    if args.mode in ("llm", "all"):
        print(f"\n{'='*60}")
        print(" LLM sweep")
        print(f"{'='*60}")
        all_results["llm"] = await run_llm_suite()
        _save_partial(all_results, args.output)

    if args.mode in ("e2e", "all"):
        print(f"\n{'='*60}")
        print(" E2E mixed workload sweep")
        print(f"{'='*60}")
        all_results["e2e"] = await run_e2e_suite(
            session_counts=session_counts,
            n_chunks=args.chunks,
        )
        _save_partial(all_results, args.output)

    # Write consolidated JSON
    consolidated = Path(args.output) / "all_results.json"
    consolidated.write_text(json.dumps(all_results, indent=2))
    print(f"\nConsolidated results → {consolidated}")

    # Generate markdown report
    report = generate_report(
        all_results,
        gpu_metrics_path=Path(args.output) / "gpu_metrics.jsonl",
        output_path=Path(args.output) / "report.md",
    )
    print(f"Report → {Path(args.output) / 'report.md'}")
    print("\n" + "=" * 60)
    print(" DONE")
    print("=" * 60)


def _save_partial(results: dict, output_dir: str) -> None:
    path = Path(output_dir) / "all_results_partial.json"
    path.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
