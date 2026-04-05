"""
Benchmark results reporter and capacity planner.

Reads collected benchmark results and GPU metrics, then:
1. Renders markdown tables for Whisper and LLM throughput across concurrency levels.
2. Projects the number of H200 GPUs required for a given user count.
3. Writes a self-contained results/report.md.

Capacity projection formula:
  required_rps = users × (appointments_per_hour / 3600) × peak_factor
  h200s_needed = ceil(required_rps / measured_sessions_per_sec)

Assumptions (adjustable via function args):
  appointments_per_hour = 3.0   (average provider appointment rate)
  peak_factor           = 1.5   (morning rush / burst headroom)
  target_p99_latency_s  = 60.0  (clinical SLA: note delivered within 60s)
"""
import json
import math
from pathlib import Path
from typing import Any


# ── Capacity projections ──────────────────────────────────────────────────────

def project_capacity(
    measured_sessions_per_sec: float,
    target_users: list[int] | None = None,
    appointments_per_hour: float = 3.0,
    peak_factor: float = 1.5,
) -> list[dict]:
    """
    Project the number of H200s needed for each user tier.

    Args:
        measured_sessions_per_sec: Throughput from the E2E benchmark (sessions/s).
        target_users:              User tiers to project (default: 500, 1000, 1500, 2500).
        appointments_per_hour:     Average appointments per provider per hour.
        peak_factor:               Burst multiplier for peak load.

    Returns:
        List of dicts with projection details per user tier.
    """
    if target_users is None:
        target_users = [500, 1000, 1500, 2500]

    rows = []
    for users in target_users:
        required_rps = users * (appointments_per_hour / 3600.0) * peak_factor
        h200s        = math.ceil(required_rps / max(measured_sessions_per_sec, 0.001))
        utilization  = required_rps / (h200s * measured_sessions_per_sec) * 100
        rows.append({
            "concurrent_users":        users,
            "required_sessions_per_s": round(required_rps, 4),
            "h200s_needed":            h200s,
            "utilization_pct":         round(utilization, 1),
        })
    return rows


# ── GPU metrics analysis ──────────────────────────────────────────────────────

def load_gpu_metrics(path: str | Path) -> list[dict]:
    records = []
    p = Path(path)
    if not p.exists():
        return records
    with p.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def summarize_gpu_metrics(records: list[dict]) -> dict:
    if not records:
        return {}
    valid = [r for r in records if "mem_used_mib" in r and r["mem_used_mib"] >= 0]
    if not valid:
        return {}

    mem_used   = [r["mem_used_mib"]  for r in valid]
    gpu_util   = [r["gpu_util_pct"]  for r in valid]
    power      = [r["power_w"]       for r in valid if r.get("power_w", -1) >= 0]
    temp       = [r["temp_c"]        for r in valid if r.get("temp_c",  -1) >= 0]

    return {
        "mem_used_mib":  {"max": max(mem_used),  "mean": round(sum(mem_used)  / len(mem_used),  1)},
        "gpu_util_pct":  {"max": max(gpu_util),  "mean": round(sum(gpu_util)  / len(gpu_util),  1)},
        "power_w":       {"max": max(power, default=0), "mean": round(sum(power)  / max(len(power), 1), 1)},
        "temp_c":        {"max": max(temp,  default=0), "mean": round(sum(temp)   / max(len(temp),  1), 1)},
        "samples":       len(valid),
    }


# ── Markdown report builder ───────────────────────────────────────────────────

def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    col_widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
                  for i, h in enumerate(headers)]
    sep  = "| " + " | ".join("-" * w for w in col_widths) + " |"
    head = "| " + " | ".join(str(h).ljust(w) for h, w in zip(headers, col_widths)) + " |"
    body = "\n".join(
        "| " + " | ".join(str(c).ljust(w) for c, w in zip(row, col_widths)) + " |"
        for row in rows
    )
    return f"{head}\n{sep}\n{body}"


def generate_report(
    all_results: dict,
    gpu_metrics_path: str | Path = "./results/gpu_metrics.jsonl",
    output_path: str | Path = "./results/report.md",
    appointments_per_hour: float = 3.0,
    peak_factor: float = 1.5,
) -> str:
    """
    Build and write a full markdown benchmark report.

    Args:
        all_results:       Dict with keys "whisper", "llm", "e2e" from run_all.py.
        gpu_metrics_path:  Path to the JSONL file written by gpu_monitor.py.
        output_path:       Where to write the report.
        appointments_per_hour / peak_factor: capacity projection parameters.

    Returns:
        The markdown string.
    """
    lines = ["# H200 GPU Stress Test Report", ""]
    lines += [
        "> **System**: NVIDIA H200 141 GB HBM3e · **LLM**: ~20B parameter model via vLLM",
        "> **ASR**: faster-whisper · **Audio chunks**: 47 s · **Date**: see filename",
        "",
    ]

    # ── GPU summary ──────────────────────────────────────────────────────────
    gpu_records = load_gpu_metrics(gpu_metrics_path)
    gpu_summary = summarize_gpu_metrics(gpu_records)
    if gpu_summary:
        lines += [
            "## GPU Resource Usage (peak across all tests)",
            "",
            f"| Metric | Peak | Mean |",
            f"|--------|------|------|",
            f"| VRAM used (MiB) | {gpu_summary['mem_used_mib']['max']:.0f} | {gpu_summary['mem_used_mib']['mean']:.0f} |",
            f"| GPU utilization (%) | {gpu_summary['gpu_util_pct']['max']:.1f} | {gpu_summary['gpu_util_pct']['mean']:.1f} |",
            f"| Power draw (W) | {gpu_summary['power_w']['max']:.0f} | {gpu_summary['power_w']['mean']:.0f} |",
            f"| Temperature (°C) | {gpu_summary['temp_c']['max']:.0f} | {gpu_summary['temp_c']['mean']:.0f} |",
            "",
        ]

    # ── Whisper results ───────────────────────────────────────────────────────
    whisper_results = all_results.get("whisper", {})
    if whisper_results:
        lines += ["## Whisper Throughput (47-second audio chunks)", ""]
        for model_name, model_results in whisper_results.items():
            lines += [f"### Model: `{model_name}`", ""]
            rows = []
            for concurrency, res in sorted(model_results.items(), key=lambda x: int(x[0])):
                if isinstance(res, dict) and "latency_ms" in res:
                    lm = res["latency_ms"]
                    rtf = res.get("audio_realtime_factor") or "N/A"
                    rows.append([
                        concurrency,
                        f"{res.get('throughput_rps', 0):.2f}",
                        f"{lm.get('p50', 0):.0f}",
                        f"{lm.get('p95', 0):.0f}",
                        f"{lm.get('p99', 0):.0f}",
                        f"{rtf:.1f}×" if isinstance(rtf, float) else rtf,
                        f"{res.get('error_rate_pct', 0):.1f}%",
                    ])
            if rows:
                lines.append(_md_table(
                    ["Concurrency", "req/s", "p50 ms", "p95 ms", "p99 ms", "RTF", "Error %"],
                    rows,
                ))
                lines.append("")
        lines += [
            "> **RTF** = Real-Time Factor. RTF = 47 s audio duration ÷ latency.",
            "> RTF > 1× means the service processes audio faster than real-time.",
            "",
        ]

    # ── LLM results ──────────────────────────────────────────────────────────
    llm_results = all_results.get("llm", {})
    if llm_results:
        lines += ["## LLM Throughput (20B model, ~800-token input → 512-token output)", ""]
        rows = []
        for concurrency, res in sorted(llm_results.items(), key=lambda x: int(x[0])):
            if isinstance(res, dict) and "latency_ms" in res:
                lm = res["latency_ms"]
                rows.append([
                    concurrency,
                    f"{res.get('throughput_rps', 0):.2f}",
                    f"{res.get('tokens_out_per_sec', 0):.0f}",
                    f"{lm.get('p50', 0):.0f}",
                    f"{lm.get('p95', 0):.0f}",
                    f"{lm.get('p99', 0):.0f}",
                    f"{res.get('error_rate_pct', 0):.1f}%",
                ])
        if rows:
            lines.append(_md_table(
                ["Concurrency", "req/s", "tok/s", "p50 ms", "p95 ms", "p99 ms", "Error %"],
                rows,
            ))
            lines.append("")

    # ── E2E results ───────────────────────────────────────────────────────────
    e2e_results = all_results.get("e2e", {})
    measured_rps = 0.0
    if e2e_results:
        lines += ["## End-to-End Pipeline (Whisper chunks → LLM note, mixed workload)", ""]
        rows = []
        best_rps = 0.0
        for session_count, res in sorted(e2e_results.items(), key=lambda x: int(x[0])):
            if isinstance(res, dict):
                rps = res.get("throughput_rps", 0)
                lm  = res.get("latency_ms", {})
                rows.append([
                    session_count,
                    f"{rps:.3f}",
                    f"{lm.get('p50', 0):.0f}",
                    f"{lm.get('p95', 0):.0f}",
                    f"{lm.get('p99', 0):.0f}",
                    f"{res.get('error_rate_pct', 0):.1f}%",
                ])
                if rps > best_rps:
                    best_rps = rps
        measured_rps = best_rps
        if rows:
            lines.append(_md_table(
                ["Concurrent Sessions", "sessions/s", "p50 ms", "p95 ms", "p99 ms", "Error %"],
                rows,
            ))
            lines.append("")

    # ── Capacity projections ──────────────────────────────────────────────────
    if measured_rps > 0:
        projections = project_capacity(
            measured_rps,
            appointments_per_hour=appointments_per_hour,
            peak_factor=peak_factor,
        )
        lines += [
            "## Capacity Projections",
            "",
            f"> Based on **{measured_rps:.3f} sessions/s** (peak measured throughput on 1× H200).",
            f"> Assumptions: **{appointments_per_hour} appt/hr per provider**, **{peak_factor}× peak burst factor**.",
            "",
        ]
        rows = [
            [
                p["concurrent_users"],
                f"{p['required_sessions_per_s']:.3f}",
                p["h200s_needed"],
                f"{p['utilization_pct']:.1f}%",
                "✅ Comfortable" if p["utilization_pct"] < 50 else
                "✅ Healthy"    if p["utilization_pct"] < 75 else
                "⚠️ Near limit" if p["utilization_pct"] < 90 else
                "❌ Over limit",
            ]
            for p in projections
        ]
        lines.append(_md_table(
            ["Concurrent Users", "Required sessions/s", "H200s needed", "Utilization", "Status"],
            rows,
        ))
        lines += [
            "",
            "> Add 1 extra H200 per tier for redundancy / rolling restarts in production.",
            "",
        ]
    else:
        lines += [
            "## Capacity Projections",
            "",
            "_No E2E results available. Run `make bench-e2e` to generate projections._",
            "",
        ]

    lines += [
        "## Methodology Notes",
        "",
        "- Whisper: synthetic 47 s WAV (formant synthesis, -3 dBFS, 16 kHz mono).",
        "  RTF > 1 is the production viability threshold.",
        "- LLM: templated medical transcripts (~800 tokens input, 512 max output tokens).",
        "  vLLM prefix caching enabled for the shared system prompt.",
        "- E2E sessions: 13 Whisper chunks + 1 LLM completion per session.",
        "- GPU metrics: nvidia-smi polled every 500 ms via sidecar container.",
        "- Warmup: 3 requests before each concurrency sweep level.",
        "- Cooldown: 10 s between concurrency levels.",
        "",
    ]

    report = "\n".join(lines)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    return report
