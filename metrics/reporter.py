"""
Benchmark results reporter and capacity planner.

Reads collected benchmark results and GPU metrics, then renders:
1. Whisper comparison matrix: backend × model × concurrency → RTF / latency / req/s
2. LLM throughput table: concurrency → tok/s / latency
3. E2E pipeline table: sessions → sessions/s / latency
4. Capacity projections: users → H200s needed
5. Full markdown report written to results/report.md

Capacity formula:
  required_rps = users × (appt/hr / 3600) × peak_factor
  h200s_needed = ceil(required_rps / measured_sessions_per_sec)
"""
import json
import math
from pathlib import Path
from typing import Any

from benchmarks.config import WHISPER_MODEL_MAP, WHISPER_COMPARE_CONCURRENCY


# ── Capacity projections ──────────────────────────────────────────────────────

def project_capacity(
    measured_sessions_per_sec: float,
    target_users: list[int] | None = None,
    appointments_per_hour: float = 3.0,
    peak_factor: float = 1.5,
) -> list[dict]:
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


# ── GPU metrics ───────────────────────────────────────────────────────────────

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
    mem_used = [r["mem_used_mib"]  for r in valid]
    gpu_util = [r["gpu_util_pct"]  for r in valid]
    power    = [r["power_w"]       for r in valid if r.get("power_w",  -1) >= 0]
    temp     = [r["temp_c"]        for r in valid if r.get("temp_c",   -1) >= 0]
    return {
        "mem_used_mib": {"max": max(mem_used),             "mean": round(sum(mem_used) / len(mem_used), 1)},
        "gpu_util_pct": {"max": max(gpu_util),             "mean": round(sum(gpu_util) / len(gpu_util), 1)},
        "power_w":      {"max": max(power, default=0),     "mean": round(sum(power)    / max(len(power),  1), 1)},
        "temp_c":       {"max": max(temp,  default=0),     "mean": round(sum(temp)     / max(len(temp),   1), 1)},
        "samples":      len(valid),
    }


# ── Markdown helpers ──────────────────────────────────────────────────────────

def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_no data_"
    col_widths = [
        max(len(str(h)), max(len(str(r[i])) for r in rows))
        for i, h in enumerate(headers)
    ]
    sep  = "| " + " | ".join("-" * w for w in col_widths) + " |"
    head = "| " + " | ".join(str(h).ljust(w) for h, w in zip(headers, col_widths)) + " |"
    body = "\n".join(
        "| " + " | ".join(str(c).ljust(w) for c, w in zip(row, col_widths)) + " |"
        for row in rows
    )
    return f"{head}\n{sep}\n{body}"


def _get(d: dict, *keys, default="—"):
    """Safe nested dict access."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
        if cur == default:
            return default
    return cur


def _fmt_rtf(val) -> str:
    if val is None or val == "—":
        return "—"
    try:
        return f"{float(val):.1f}×"
    except (TypeError, ValueError):
        return "—"


def _fmt_ms(val) -> str:
    if val is None or val == "—":
        return "—"
    try:
        return f"{float(val):.0f}"
    except (TypeError, ValueError):
        return "—"


# ── Comparison matrix ─────────────────────────────────────────────────────────

def _build_comparison_matrix(
    compare: dict,
    metric: str = "rtf",
    concurrency_levels: list[int] | None = None,
) -> str:
    """
    Build a markdown table: rows=models, cols=(backend, concurrency).

    Args:
        compare:            {backend: {model: {str(concurrency): summary}}}
        metric:             "rtf" | "p50" | "p95" | "rps"
        concurrency_levels: Which concurrency values to show as columns.
    """
    if concurrency_levels is None:
        concurrency_levels = WHISPER_COMPARE_CONCURRENCY

    backends = ["faster_whisper", "openai_whisper"]
    b_label  = {"faster_whisper": "FW", "openai_whisper": "OW"}

    # Build column headers: model | FW c=1 | FW c=10 | FW c=50 | OW c=1 | …
    col_headers = ["Model"]
    for b in backends:
        for c in concurrency_levels:
            col_headers.append(f"{b_label[b]} c={c}")

    rows = []
    for display_name, model_map in WHISPER_MODEL_MAP.items():
        row = [display_name]
        for b in backends:
            bm = model_map.get(b)
            for c in concurrency_levels:
                if bm is None:
                    row.append("N/A")
                    continue
                summary = _get(compare, b, bm, str(c))
                if summary == "—" or not isinstance(summary, dict):
                    row.append("—")
                    continue
                if metric == "rtf":
                    row.append(_fmt_rtf(_get(summary, "audio_realtime_factor")))
                elif metric == "p50":
                    row.append(_fmt_ms(_get(summary, "latency_ms", "p50")))
                elif metric == "p95":
                    row.append(_fmt_ms(_get(summary, "latency_ms", "p95")))
                elif metric == "p99":
                    row.append(_fmt_ms(_get(summary, "latency_ms", "p99")))
                elif metric == "rps":
                    val = _get(summary, "throughput_rps")
                    row.append(f"{float(val):.2f}" if val != "—" else "—")
        rows.append(row)

    return _md_table(col_headers, rows)


def _build_vram_matrix(compare: dict) -> str:
    """
    Rough VRAM reference table from model sizes.
    Values are known estimates — reported alongside measured GPU metrics.
    """
    vram_ref = {
        # (display_name, faster_whisper_vram_gb, openai_whisper_vram_gb)
        "tiny":            (0.15, 0.15),
        "base":            (0.29, 0.29),
        "small":           (0.93, 0.93),
        "medium":          (3.06, 3.06),
        "large (v1)":      (None, 6.17),
        "large-v2":        (6.17, 6.17),
        "large-v3":        (6.17, 6.17),
        "turbo (lv3-t)":   (3.10, 3.10),   # large-v3-turbo ≈ 809 M params
    }
    rows = []
    for display, (fw_gb, ow_gb) in vram_ref.items():
        fw_str = f"~{fw_gb:.2f} GB" if fw_gb else "N/A"
        ow_str = f"~{ow_gb:.2f} GB" if ow_gb else "N/A"
        rows.append([display, fw_str, ow_str])
    return _md_table(["Model", "faster-whisper VRAM", "openai-whisper VRAM"], rows)


# ── Per-backend per-model detailed tables ─────────────────────────────────────

def _build_single_model_table(model_results: dict, backend: str, model: str) -> str:
    """Full concurrency sweep for one backend+model combo."""
    rows = []
    for concurrency in sorted(model_results.keys(), key=lambda x: int(x)):
        res = model_results[concurrency]
        if not isinstance(res, dict) or "latency_ms" not in res:
            continue
        lm  = res["latency_ms"]
        rtf = res.get("audio_realtime_factor")
        rows.append([
            concurrency,
            f"{res.get('throughput_rps', 0):.2f}",
            _fmt_ms(lm.get("p50")),
            _fmt_ms(lm.get("p90")),
            _fmt_ms(lm.get("p95")),
            _fmt_ms(lm.get("p99")),
            _fmt_rtf(rtf),
            f"{res.get('error_rate_pct', 0):.1f}%",
        ])
    return _md_table(
        ["Concurrency", "req/s", "p50 ms", "p90 ms", "p95 ms", "p99 ms", "RTF", "Error %"],
        rows,
    )


# ── Main report builder ───────────────────────────────────────────────────────

def generate_report(
    all_results: dict,
    gpu_metrics_path: str | Path = "./results/gpu_metrics.jsonl",
    output_path: str | Path = "./results/report.md",
    appointments_per_hour: float = 3.0,
    peak_factor: float = 1.5,
) -> str:
    lines = [
        "# H200 GPU Stress Test Report — Medical Ambient Scribing",
        "",
        "> **GPU**: NVIDIA H200 141 GB HBM3e · **LLM**: ~20B model via vLLM  ",
        "> **ASR backends**: faster-whisper (CTranslate2) vs openai-whisper (PyTorch)  ",
        "> **Audio chunks**: 47 s · **Concurrency sweep**: 1 / 5 / 10 / 20 / 50 / 100",
        "",
        "---",
        "",
    ]

    # ── Table of contents ─────────────────────────────────────────────────────
    lines += [
        "## Contents",
        "",
        "1. [GPU Resource Usage](#gpu-resource-usage)",
        "2. [Whisper Comparison Matrix](#whisper-comparison-matrix)",
        "3. [Whisper Detailed Results](#whisper-detailed-results)",
        "4. [VRAM Reference](#vram-reference)",
        "5. [Audio Format & Sample Rate Matrix](#audio-format--sample-rate-matrix)",
        "6. [LLM Throughput](#llm-throughput)",
        "7. [End-to-End Pipeline](#end-to-end-pipeline)",
        "8. [Capacity Projections](#capacity-projections)",
        "9. [Methodology](#methodology)",
        "",
        "---",
        "",
    ]

    # ── GPU summary ───────────────────────────────────────────────────────────
    gpu_records = load_gpu_metrics(gpu_metrics_path)
    gpu_summary = summarize_gpu_metrics(gpu_records)
    lines.append("## GPU Resource Usage\n")
    if gpu_summary:
        lines += [
            _md_table(
                ["Metric", "Peak", "Mean"],
                [
                    ["VRAM used (MiB)",     f"{gpu_summary['mem_used_mib']['max']:.0f}",  f"{gpu_summary['mem_used_mib']['mean']:.0f}"],
                    ["GPU utilization (%)", f"{gpu_summary['gpu_util_pct']['max']:.1f}",  f"{gpu_summary['gpu_util_pct']['mean']:.1f}"],
                    ["Power draw (W)",      f"{gpu_summary['power_w']['max']:.0f}",        f"{gpu_summary['power_w']['mean']:.0f}"],
                    ["Temperature (°C)",    f"{gpu_summary['temp_c']['max']:.0f}",         f"{gpu_summary['temp_c']['mean']:.0f}"],
                ],
            ),
            "",
        ]
    else:
        lines += ["_GPU metrics not available (run gpu_monitor sidecar)._\n"]
    lines.append("---\n")

    # ── Whisper comparison matrix ─────────────────────────────────────────────
    compare = all_results.get("whisper_compare", {})
    lines.append("## Whisper Comparison Matrix\n")
    lines += [
        "> **FW** = faster-whisper (CTranslate2)  |  **OW** = openai-whisper (PyTorch)  ",
        f"> Columns: concurrency = {WHISPER_COMPARE_CONCURRENCY} simultaneous 47-second audio chunks",
        "",
    ]

    if compare:
        for metric, title, note in [
            ("rtf",  "### RTF (Real-Time Factor) — Higher is Better",
             "> RTF = 47 s ÷ latency. **RTF > 1×** = faster than real-time. Production viability threshold."),
            ("rps",  "### Throughput (requests/second) — Higher is Better", ""),
            ("p50",  "### p50 Latency (ms) — Lower is Better", ""),
            ("p95",  "### p95 Latency (ms) — Lower is Better", ""),
        ]:
            lines += [title, ""]
            if note:
                lines += [note, ""]
            lines += [_build_comparison_matrix(compare, metric=metric), ""]
    else:
        lines += ["_No comparison data. Run `make bench-compare` or `python -m benchmarks.run_all --mode compare`._\n"]

    lines.append("---\n")

    # ── Whisper detailed results ──────────────────────────────────────────────
    lines.append("## Whisper Detailed Results\n")

    # Collect from both whisper_compare and whisper keys
    detailed_sources = []
    if compare:
        for backend, models in compare.items():
            for model, results in models.items():
                if isinstance(results, dict) and not results.get("error"):
                    detailed_sources.append((backend, model, results))
    elif all_results.get("whisper"):
        w = all_results["whisper"]
        for backend, models in w.items():
            for model, results in models.items():
                if isinstance(results, dict):
                    detailed_sources.append((backend, model, results))

    if detailed_sources:
        b_label = {"faster_whisper": "faster-whisper", "openai_whisper": "openai-whisper"}
        for backend, model, results in detailed_sources:
            lines += [
                f"### {b_label.get(backend, backend)} · `{model}`",
                "",
                _build_single_model_table(results, backend, model),
                "",
            ]
    else:
        lines += ["_No detailed Whisper results available._\n"]
    lines.append("---\n")

    # ── VRAM reference ────────────────────────────────────────────────────────
    lines += [
        "## VRAM Reference\n",
        "Per-instance VRAM footprint. H200 has 141 GB total.\n",
        _build_vram_matrix(compare),
        "",
        "> faster-whisper (int8_float16): approximately half the FP16 VRAM above.  ",
        "> H200 can hold ~22 large-v3 instances simultaneously (FP16) for maximum parallelism.",
        "",
        "---\n",
    ]

    # ── Audio format / sample rate matrix ────────────────────────────────────
    lines.append("## Audio Format & Sample Rate Matrix\n")
    lines += [
        "> Fixed concurrency = 10. Measures decoding overhead and resampling cost per format.",
        "> **16 kHz** = Whisper-native (no resampling). **48 kHz** = browser/mic capture path (requires service resampling).",
        "",
    ]
    fmt_results = all_results.get("format", {})
    if fmt_results:
        for source_label, source_data in [("Synthetic Audio", fmt_results.get("synthetic", {})),
                                          ("Real Audio",      fmt_results.get("real",      {}))]:
            if not source_data:
                continue
            lines.append(f"### {source_label}\n")
            for metric, col_hdr, fmt_fn in [
                ("p50",  "p50 latency (ms)", _fmt_ms),
                ("p95",  "p95 latency (ms)", _fmt_ms),
                ("rps",  "req/s",            lambda v: f"{float(v):.2f}" if v != "—" else "—"),
                ("err",  "error %",          lambda v: f"{float(v):.1f}%" if v != "—" else "—"),
            ]:
                # Collect all rates that appear in data
                all_rates: list[int] = sorted({
                    int(r)
                    for fmt_data in source_data.values()
                    if isinstance(fmt_data, dict)
                    for r in fmt_data.keys()
                    if r.isdigit()
                })
                if not all_rates:
                    continue
                rate_labels = [f"{r//1000}kHz" for r in all_rates]
                headers = ["Format"] + rate_labels
                rows = []
                for fmt_key in sorted(source_data.keys()):
                    fmt_data = source_data[fmt_key]
                    if not isinstance(fmt_data, dict):
                        continue
                    row = [f"`{fmt_key}`"]
                    for rate in all_rates:
                        res = fmt_data.get(str(rate), {})
                        if not isinstance(res, dict) or res.get("skipped"):
                            row.append("—")
                            continue
                        lm = res.get("latency_ms", {})
                        if metric == "p50":
                            row.append(fmt_fn(_get(lm, "p50")))
                        elif metric == "p95":
                            row.append(fmt_fn(_get(lm, "p95")))
                        elif metric == "rps":
                            row.append(fmt_fn(_get(res, "throughput_rps")))
                        elif metric == "err":
                            row.append(fmt_fn(_get(res, "error_rate_pct")))
                    rows.append(row)
                if rows:
                    lines += [
                        f"**{col_hdr}** — lower is better"
                        if metric in ("p50", "p95", "err")
                        else f"**{col_hdr}** — higher is better",
                        "",
                        _md_table(headers, rows),
                        "",
                    ]
        lines += [
            "> Latency delta (48kHz vs 16kHz) = resampling overhead added by the service.",
            "> Latency delta (flac/opus vs wav) = decoding overhead (FLAC ≈ 0–5 ms, Opus ≈ 5–15 ms via ffmpeg).",
            "",
        ]
    else:
        lines += [
            "_No format results. Run `make bench-formats` or add `--mode format` to run_all._",
            "",
        ]
    lines.append("---\n")

    # ── LLM results ───────────────────────────────────────────────────────────
    lines.append("## LLM Throughput\n")
    lines += [
        "> Model: ~20B parameters, FP16 via vLLM · Input: ~800 tokens · Output: 512 max tokens  ",
        "> Prefix caching enabled (shared system prompt reused across all requests)\n",
    ]
    llm_results = all_results.get("llm", {})
    if llm_results:
        rows = []
        for concurrency in sorted(llm_results.keys(), key=lambda x: int(x)):
            res = llm_results[concurrency]
            if not isinstance(res, dict) or "latency_ms" not in res:
                continue
            lm = res["latency_ms"]
            rows.append([
                concurrency,
                f"{res.get('throughput_rps', 0):.3f}",
                f"{res.get('tokens_out_per_sec', 0):.0f}",
                _fmt_ms(lm.get("p50")),
                _fmt_ms(lm.get("p95")),
                _fmt_ms(lm.get("p99")),
                f"{res.get('ttft_ms', 0):.0f}" if res.get("ttft_ms", -1) > 0 else "—",
                f"{res.get('error_rate_pct', 0):.1f}%",
            ])
        if rows:
            lines += [
                _md_table(
                    ["Concurrency", "req/s", "tok/s", "p50 ms", "p95 ms", "p99 ms", "TTFT ms", "Error %"],
                    rows,
                ),
                "",
            ]
    else:
        lines += ["_No LLM results. Run `make bench-llm`._\n"]
    lines.append("---\n")

    # ── E2E results ───────────────────────────────────────────────────────────
    lines.append("## End-to-End Pipeline\n")
    lines += [
        "> Mixed workload: 13 Whisper chunks (47 s each) → LLM SOAP note per session.  ",
        "> VRAM split: LLM_GPU_UTIL=0.45 (~63 GB for LLM) + Whisper workers (~78 GB available)\n",
    ]
    e2e_results = all_results.get("e2e", {})
    measured_rps = 0.0
    if e2e_results:
        rows = []
        for sessions in sorted(e2e_results.keys(), key=lambda x: int(x)):
            res = e2e_results[sessions]
            if not isinstance(res, dict):
                continue
            lm  = res.get("latency_ms", {})
            rps = res.get("throughput_rps", 0)
            rows.append([
                sessions,
                f"{rps:.3f}",
                _fmt_ms(lm.get("p50")),
                _fmt_ms(lm.get("p95")),
                _fmt_ms(lm.get("p99")),
                f"{res.get('error_rate_pct', 0):.1f}%",
            ])
            if rps > measured_rps:
                measured_rps = rps
        if rows:
            lines += [
                _md_table(
                    ["Concurrent Sessions", "sessions/s", "p50 ms", "p95 ms", "p99 ms", "Error %"],
                    rows,
                ),
                "",
            ]
    else:
        lines += ["_No E2E results. Run `make bench-e2e`._\n"]
    lines.append("---\n")

    # ── Capacity projections ──────────────────────────────────────────────────
    lines.append("## Capacity Projections\n")
    if measured_rps > 0:
        projections = project_capacity(
            measured_rps,
            appointments_per_hour=appointments_per_hour,
            peak_factor=peak_factor,
        )
        lines += [
            f"> Measured peak throughput: **{measured_rps:.3f} sessions/s** on 1× H200  ",
            f"> Assumptions: **{appointments_per_hour} appt/hr per provider** · **{peak_factor}× peak burst**",
            "",
            _md_table(
                ["Concurrent Users", "Required sessions/s", "H200s needed", "Utilization", "Status"],
                [
                    [
                        p["concurrent_users"],
                        f"{p['required_sessions_per_s']:.3f}",
                        p["h200s_needed"],
                        f"{p['utilization_pct']:.1f}%",
                        "✅ Comfortable" if p["utilization_pct"] < 50
                        else "✅ Healthy"    if p["utilization_pct"] < 75
                        else "⚠️ Near limit" if p["utilization_pct"] < 90
                        else "❌ Over limit",
                    ]
                    for p in projections
                ],
            ),
            "",
            "> Add 1 redundant H200 per tier for rolling restarts and zero-downtime deploys.",
            "",
        ]
    else:
        lines += [
            "_Capacity projections require E2E results. Run `make bench-e2e` first._",
            "",
        ]
    lines.append("---\n")

    # ── Methodology ───────────────────────────────────────────────────────────
    lines += [
        "## Methodology",
        "",
        "### Audio generation",
        "- Synthetic 47-second WAV (16 kHz mono, PCM-16, -3 dBFS peak).",
        "- Formant synthesis (F1/F2 resonance + fricative bursts + lognormal pauses + -40 dBFS noise).",
        "- No TTS dependency; generates in ~80 ms on CPU. Reproducible via integer seed.",
        "- Exercises Whisper's encoder fully — unlike pure silence or pure noise.",
        "",
        "### Benchmark procedure",
        "- **Warmup**: 3 sequential requests before each concurrency level.",
        "- **Test**: `asyncio.gather(N)` tasks repeated for ≥ 30 total samples.",
        "- **Cooldown**: 10 s between levels (GPU thermal + scheduler drain).",
        "- **GPU metrics**: nvidia-smi polled every 500 ms by sidecar container.",
        "",
        "### Whisper backends",
        "| Backend | Library | Precision | Batch | Notes |",
        "|---------|---------|-----------|-------|-------|",
        "| faster-whisper | CTranslate2 | FP16 / INT8 | Pool of N model instances | 1.4–2× faster than OW on GPU |",
        "| openai-whisper | PyTorch | FP16 | Pool of N model instances | Reference implementation |",
        "",
        "### Metrics glossary",
        "- **RTF** (Real-Time Factor): `audio_duration ÷ transcription_latency`. RTF > 1 = faster than real-time.",
        "- **TTFT**: Time to first token (LLM streaming latency).",
        "- **tok/s**: Aggregate output tokens per second across all concurrent requests.",
        "- **sessions/s**: Complete E2E sessions (all chunks + LLM note) per second.",
        "",
    ]

    report = "\n".join(lines)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    return report
