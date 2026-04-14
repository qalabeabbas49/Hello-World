"""Run reproducible vLLM Whisper scheduler/prefill sweeps."""
import argparse
import itertools
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_boolish_list(raw: str) -> list[int]:
    out: list[int] = []
    for x in raw.split(","):
        v = x.strip().lower()
        if not v:
            continue
        out.append(1 if v in ("1", "true", "yes", "on") else 0)
    return out


def _run(cmd: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep vLLM scheduler flags for Whisper SLA.")
    parser.add_argument("--results-root", default="./results")
    parser.add_argument("--model", default="openai/whisper-large-v3")
    parser.add_argument("--concurrency", default="4,8,12,16,20,24,28,32")
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--input-mode", choices=["file", "pcm_f32le"], default="pcm_f32le")
    parser.add_argument("--load-mode", choices=["batch", "steady"], default="steady")
    parser.add_argument("--steady-duration-s", type=float, default=90.0)
    parser.add_argument("--max-p95-ms", type=float, default=12000.0)
    parser.add_argument("--max-num-seqs", default="128,256,384,512")
    parser.add_argument("--scheduler-delay-factors", default="0.0,0.1,0.2")
    parser.add_argument("--chunked-prefill", default="0,1")
    parser.add_argument("--max-num-batched-tokens", default="")
    args = parser.parse_args()

    seqs = _parse_int_list(args.max_num_seqs)
    delay_factors = _parse_float_list(args.scheduler_delay_factors)
    chunked = _parse_boolish_list(args.chunked_prefill)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = Path(args.results_root).resolve() / f"vllm_scheduler_sweep_{ts}"
    root.mkdir(parents=True, exist_ok=True)

    summary: list[dict] = []
    matrix = list(itertools.product(seqs, delay_factors, chunked))
    print(f"Running {len(matrix)} configs into {root}")

    for idx, (max_num_seqs, delay_factor, chunk_prefill) in enumerate(matrix, start=1):
        run_name = f"s{max_num_seqs}_d{delay_factor:.2f}_cp{chunk_prefill}"
        out_dir = root / run_name
        out_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env["VLLM_WHISPER_MODEL"] = args.model
        env["VLLM_WHISPER_MAX_NUM_SEQS"] = str(max_num_seqs)
        env["VLLM_WHISPER_SCHEDULER_DELAY_FACTOR"] = str(delay_factor)
        env["VLLM_WHISPER_ENABLE_CHUNKED_PREFILL"] = str(chunk_prefill)
        env["VLLM_WHISPER_MAX_BATCHED_TOKENS"] = args.max_num_batched_tokens
        env["RESULTS_DIR"] = str(out_dir)

        print(
            f"\n[{idx}/{len(matrix)}] {run_name} "
            f"(seqs={max_num_seqs}, delay={delay_factor}, chunked_prefill={chunk_prefill})"
        )

        _run(
            [
                "make",
                "up-vllm-whisper",
                f"VLLM_WHISPER_MODEL={args.model}",
                f"VLLM_WHISPER_MAX_NUM_SEQS={max_num_seqs}",
                f"VLLM_WHISPER_SCHEDULER_DELAY_FACTOR={delay_factor}",
                f"VLLM_WHISPER_ENABLE_CHUNKED_PREFILL={chunk_prefill}",
                f"VLLM_WHISPER_MAX_BATCHED_TOKENS={args.max_num_batched_tokens}",
                f"RESULTS={out_dir}",
            ],
            env,
        )

        _run(
            [
                "python",
                "-m",
                "benchmarks.run_all",
                "--mode",
                "sla",
                "--model",
                args.model,
                "--whisper-concurrency",
                args.concurrency,
                "--whisper-input-mode",
                args.input_mode,
                "--whisper-beam-size",
                str(args.beam_size),
                "--whisper-load-mode",
                args.load_mode,
                "--whisper-steady-duration-s",
                str(args.steady_duration_s),
                "--whisper-max-p95-ms",
                str(args.max_p95_ms),
                "--output",
                str(out_dir),
            ],
            env,
        )

        summary.append(
            {
                "run_name": run_name,
                "results_dir": str(out_dir),
                "max_num_seqs": max_num_seqs,
                "scheduler_delay_factor": delay_factor,
                "enable_chunked_prefill": bool(chunk_prefill),
                "max_num_batched_tokens": args.max_num_batched_tokens,
            }
        )

    summary_path = root / "sweep_runs.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSweep complete -> {summary_path}")


if __name__ == "__main__":
    main()
