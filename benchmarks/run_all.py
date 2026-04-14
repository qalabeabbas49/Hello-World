"""Slim vLLM Whisper orchestrator for cross-GPU comparisons."""
import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path

from benchmarks import config as cfg
from metrics.reporter import generate_whisper_gpu_report


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


def _parse_int_list(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _ensure_output_dir_writable(output_dir: str) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile("w", dir=path, delete=True) as fh:
            fh.write("")
    except OSError as exc:
        raise SystemExit(
            f"Output directory is not writable: {path}. "
            f"Use a new directory like '--output ./results_tiny' or fix ownership first."
        ) from exc


async def run_whisper_suite(
    model: str,
    input_mode: str,
    beam_size: int,
    label_prefix: str | None = None,
) -> dict:
    from benchmarks.whisper_bench import run_whisper_bench
    if not await _check_service(cfg.VLLM_WHISPER_BASE_URL, f"Whisper (vllm_whisper/{model})"):
        return {}
    return await run_whisper_bench(
        model_label=model,
        backend="vllm_whisper",
        label_prefix=label_prefix,
        input_mode=input_mode,
        beam_size=beam_size,
    )
async def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM Whisper benchmark runner")
    parser.add_argument("--mode", choices=["whisper", "sweep", "sla"], default="sweep")
    parser.add_argument("--model", default=cfg.VLLM_WHISPER_MODEL)
    parser.add_argument("--whisper-concurrency", default=None, help="Comma-separated concurrency list")
    parser.add_argument("--whisper-input-mode", default=cfg.WHISPER_INPUT_MODE, choices=["file", "pcm_f32le"])
    parser.add_argument("--whisper-beam-size", type=int, default=cfg.WHISPER_BEAM_SIZE)
    parser.add_argument("--whisper-load-mode", choices=["batch", "steady"], default=cfg.WHISPER_LOAD_MODE)
    parser.add_argument("--whisper-steady-duration-s", type=float, default=cfg.WHISPER_STEADY_STATE_DURATION_S)
    parser.add_argument("--whisper-max-p95-ms", type=float, default=cfg.WHISPER_MAX_P95_MS)
    parser.add_argument("--whisper-projection-headroom", type=float, default=cfg.WHISPER_PROJECTION_HEADROOM)
    parser.add_argument("--target-tiers", default=None)
    parser.add_argument("--output", default=cfg.RESULTS_DIR)
    args = parser.parse_args()

    cfg.RESULTS_DIR = args.output
    cfg.WHISPER_LOAD_MODE = args.whisper_load_mode
    cfg.WHISPER_STEADY_STATE_DURATION_S = float(args.whisper_steady_duration_s)
    cfg.WHISPER_MAX_P95_MS = float(args.whisper_max_p95_ms)
    cfg.WHISPER_PROJECTION_HEADROOM = float(args.whisper_projection_headroom)
    _ensure_output_dir_writable(cfg.RESULTS_DIR)

    parsed_concurrency = _parse_int_list(args.whisper_concurrency)
    if parsed_concurrency:
        cfg.WHISPER_CONCURRENCY = parsed_concurrency
    target_tiers = _parse_int_list(args.target_tiers) or cfg.TARGET_CONCURRENCY_TIERS

    all_results: dict = {"whisper": {"vllm_whisper": {args.model: {}}}}
    results = await run_whisper_suite(
        model=args.model,
        input_mode=args.whisper_input_mode,
        beam_size=args.whisper_beam_size,
    )
    all_results["whisper"]["vllm_whisper"][args.model] = results
    _save_partial(all_results, args.output)

    consolidated = Path(args.output) / "all_results.json"
    consolidated.write_text(json.dumps(all_results, indent=2))
    print(f"\nConsolidated results -> {consolidated}")

    report_path = Path(args.output) / "vllm_whisper_report.md"
    load_mode_note = (
        f"{cfg.WHISPER_LOAD_MODE} "
        f"({cfg.WHISPER_STEADY_STATE_DURATION_S:.0f}s per level)"
        if cfg.WHISPER_LOAD_MODE == "steady"
        else "batch"
    )
    generate_whisper_gpu_report(
        all_results,
        gpu_metrics_path=Path(args.output) / "gpu_metrics_vllm_whisper_only.jsonl",
        output_path=report_path,
        target_tiers=target_tiers,
        max_p95_ms=cfg.WHISPER_MAX_P95_MS,
        headroom_factor=cfg.WHISPER_PROJECTION_HEADROOM,
        report_title="vLLM Whisper Cross-GPU Capacity Report",
        load_mode_note=load_mode_note,
    )
    print(f"Report           -> {report_path}")
    print(f"\n{'='*62}\n DONE\n{'='*62}")


def _save_partial(results: dict, output_dir: str, filename: str = "all_results_partial.json") -> None:
    path = Path(output_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as fh:
        json.dump(results, fh, indent=2)
        fh.write("\n")
        tmp_path = Path(fh.name)
    tmp_path.replace(path)


if __name__ == "__main__":
    asyncio.run(main())
