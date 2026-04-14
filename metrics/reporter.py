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
import os
from pathlib import Path
from typing import Any

from benchmarks.config import TARGET_CONCURRENCY_TIERS, WHISPER_MODEL_MAP, WHISPER_COMPARE_CONCURRENCY


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


def filter_gpu_metrics(records: list[dict], label_prefixes: list[str] | None = None) -> list[dict]:
    if not label_prefixes:
        return records
    prefixes = [prefix for prefix in label_prefixes if prefix]
    if not prefixes:
        return records
    return [
        record
        for record in records
        if any(str(record.get("label", "")).startswith(prefix) for prefix in prefixes)
    ]


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
    names    = sorted({str(r.get("gpu_name", "unknown")) for r in valid if r.get("gpu_name")})
    indices  = sorted({str(r.get("gpu_index", "?")) for r in valid if r.get("gpu_index") is not None})
    totals   = [r["mem_total_mib"] for r in valid if r.get("mem_total_mib", -1) >= 0]
    return {
        "gpu_names":    names,
        "gpu_indices":  indices,
        "mem_total_mib": max(totals, default=0),
        "mem_used_mib": {"max": max(mem_used),             "mean": round(sum(mem_used) / len(mem_used), 1)},
        "gpu_util_pct": {"max": max(gpu_util),             "mean": round(sum(gpu_util) / len(gpu_util), 1)},
        "power_w":      {"max": max(power, default=0),     "mean": round(sum(power)    / max(len(power),  1), 1)},
        "temp_c":       {"max": max(temp,  default=0),     "mean": round(sum(temp)     / max(len(temp),   1), 1)},
        "samples":      len(valid),
    }


def _gpu_label(gpu_summary: dict) -> str:
    names = gpu_summary.get("gpu_names") or []
    if not names:
        return "GPU detected from benchmark host"
    if len(names) == 1:
        total_gb = gpu_summary.get("mem_total_mib", 0) / 1024
        suffix = f" ({total_gb:.1f} GiB VRAM)" if total_gb else ""
        return f"{names[0]}{suffix}"
    return ", ".join(names)


def _gpu_summary_table(gpu_summary: dict) -> str:
    if not gpu_summary:
        return "_GPU metrics not available. Start the matching gpu_monitor sidecar for this benchmark._"
    rows = [
        ["GPU(s)", _gpu_label(gpu_summary), ""],
        ["GPU index", ", ".join(gpu_summary.get("gpu_indices", [])) or "unknown", ""],
        ["VRAM total (MiB)", f"{gpu_summary.get('mem_total_mib', 0):.0f}", ""],
        ["VRAM used (MiB)", f"{gpu_summary['mem_used_mib']['max']:.0f}", f"{gpu_summary['mem_used_mib']['mean']:.0f}"],
        ["GPU utilization (%)", f"{gpu_summary['gpu_util_pct']['max']:.1f}", f"{gpu_summary['gpu_util_pct']['mean']:.1f}"],
        ["Power draw (W)", f"{gpu_summary['power_w']['max']:.0f}", f"{gpu_summary['power_w']['mean']:.0f}"],
        ["Temperature (C)", f"{gpu_summary['temp_c']['max']:.0f}", f"{gpu_summary['temp_c']['mean']:.0f}"],
    ]
    return _md_table(["Metric", "Peak / Value", "Mean"], rows)


def project_rps_capacity(
    throughput_rps_per_gpu: int | float,
    target_tiers: list[int] | None = None,
    headroom_factor: float = 1.0,
) -> list[dict]:
    """
    Fleet projection from measured req/s per GPU.

    headroom_factor: multiply effective capacity by this before dividing (e.g. 0.85
    to plan conservatively when measured load used batch mode or differs from prod).
    """
    if target_tiers is None:
        target_tiers = TARGET_CONCURRENCY_TIERS
    raw_cap = max(float(throughput_rps_per_gpu), 0.001)
    hf = max(float(headroom_factor), 0.001)
    capacity = raw_cap * hf
    rows = []
    for target in target_tiers:
        gpus_needed = math.ceil(target / capacity)
        utilization = target / (gpus_needed * capacity) * 100
        rows.append({
            "target_rps": target,
            "throughput_rps_per_gpu": round(capacity, 3),
            "gpus_needed": gpus_needed,
            "utilization_pct": round(utilization, 1),
        })
    return rows


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


def _looks_like_concurrency_results(results: dict) -> bool:
    return any(str(k).isdigit() and isinstance(v, dict) for k, v in results.items())


def _safe_slug(value: str) -> str:
    slug = value.strip().replace("/", "_").replace(":", "_").replace(" ", "_")
    return "".join(ch for ch in slug if ch.isalnum() or ch in ("-", "_", "."))


def _whisper_label_prefix(backend: str, model: str, workers: int | None = None) -> str:
    prefix = {
        "faster_whisper": "fw",
        "openai_whisper": "ow",
        "vllm_whisper": "vw",
    }.get(backend, backend[:2])
    base = f"{prefix}_{_safe_slug(model)}"
    if workers is None:
        return base
    return f"{base}_w{workers}"


def _unwrap_selected_results(results: dict) -> tuple[int | None, dict]:
    if isinstance(results, dict) and isinstance(results.get("selected_results"), dict):
        selected_workers = results.get("selected_workers")
        try:
            selected_workers = int(selected_workers) if selected_workers is not None else None
        except (TypeError, ValueError):
            selected_workers = None
        return selected_workers, results["selected_results"]
    return None, results


def _preferred_compute_results(model_results: dict) -> tuple[str | None, dict]:
    """Return one compute-type layer for legacy comparison tables."""
    if _looks_like_concurrency_results(model_results):
        return None, model_results
    for compute_type in ("float16", "fp16", "int8_float16", "int8"):
        results = model_results.get(compute_type)
        if isinstance(results, dict):
            _, selected = _unwrap_selected_results(results)
            return compute_type, selected
    for compute_type, results in model_results.items():
        if isinstance(results, dict):
            _, selected = _unwrap_selected_results(results)
            return str(compute_type), selected
    return None, {}


def _iter_whisper_result_sets(all_results: dict) -> list[dict]:
    """Flatten Whisper results across backend/model/compute_type shapes."""
    rows: list[dict] = []
    compare = all_results.get("whisper_compare") or {}
    for backend, models in compare.items():
        if not isinstance(models, dict):
            continue
        for model, model_results in models.items():
            if not isinstance(model_results, dict) or model_results.get("error"):
                continue
            if _looks_like_concurrency_results(model_results):
                rows.append({
                    "backend": backend,
                    "model": model,
                    "compute_type": None,
                    "workers": None,
                    "selected": True,
                    "label_prefix": _whisper_label_prefix(backend, model),
                    "results": model_results,
                })
                continue
            for compute_type, results in model_results.items():
                if isinstance(results, dict) and not results.get("error"):
                    if isinstance(results.get("workers"), dict):
                        selected_workers = results.get("selected_workers")
                        try:
                            selected_workers = int(selected_workers) if selected_workers is not None else None
                        except (TypeError, ValueError):
                            selected_workers = None
                        for worker, worker_results in results["workers"].items():
                            if not (str(worker).isdigit() and isinstance(worker_results, dict) and not worker_results.get("error")):
                                continue
                            worker_int = int(worker)
                            rows.append({
                                "backend": backend,
                                "model": model,
                                "compute_type": str(compute_type),
                                "workers": worker_int,
                                "selected": worker_int == selected_workers,
                                "label_prefix": _whisper_label_prefix(backend, model, worker_int),
                                "results": worker_results,
                            })
                        continue
                    rows.append({
                        "backend": backend,
                        "model": model,
                        "compute_type": str(compute_type),
                        "workers": None,
                        "selected": True,
                        "label_prefix": _whisper_label_prefix(backend, model),
                        "results": results,
                    })

    single = all_results.get("whisper") or {}
    for backend, models in single.items():
        if not isinstance(models, dict):
            continue
        for model, results in models.items():
            if isinstance(results, dict):
                rows.append({
                    "backend": backend,
                    "model": model,
                    "compute_type": None,
                    "workers": None,
                    "selected": True,
                    "label_prefix": _whisper_label_prefix(backend, model),
                    "results": results,
                })
    return rows


def _iter_concurrency_summaries(results: dict) -> list[tuple[int, dict]]:
    items: list[tuple[int, dict]] = []
    for concurrency, summary in results.items():
        if str(concurrency).isdigit() and isinstance(summary, dict):
            items.append((int(concurrency), summary))
    return sorted(items, key=lambda item: item[0])


def _stable_summary(results: dict, max_error_pct: float = 1.0) -> tuple[int, dict] | None:
    stable = [
        (concurrency, summary)
        for concurrency, summary in _iter_concurrency_summaries(results)
        if summary.get("successful", 0) > 0
        and float(summary.get("error_rate_pct", 100.0)) <= max_error_pct
        and "latency_ms" in summary
    ]
    if not stable:
        return None
    return max(stable, key=lambda item: item[0])


def _peak_throughput_summary(results: dict, max_error_pct: float | None = None) -> tuple[int, dict] | None:
    summaries = [
        (concurrency, summary)
        for concurrency, summary in _iter_concurrency_summaries(results)
        if summary.get("successful", 0) > 0
        and (
            max_error_pct is None
            or float(summary.get("error_rate_pct", 100.0)) <= max_error_pct
        )
    ]
    if not summaries:
        return None
    return max(summaries, key=lambda item: float(item[1].get("throughput_rps", 0.0)))


def max_concurrency_under_p95_sla(
    results: dict,
    max_p95_ms: float,
    max_error_pct: float = 1.0,
) -> tuple[int, dict] | None:
    """
    Highest concurrency level where p95 latency is <= max_p95_ms and errors are within threshold.
    """
    if max_p95_ms <= 0:
        return None
    candidates: list[tuple[int, dict]] = []
    for concurrency, summary in _iter_concurrency_summaries(results):
        if summary.get("successful", 0) <= 0:
            continue
        if float(summary.get("error_rate_pct", 100.0)) > max_error_pct:
            continue
        lm = summary.get("latency_ms")
        if not isinstance(lm, dict):
            continue
        p95 = float(lm.get("p95", float("inf")) or float("inf"))
        if p95 <= max_p95_ms:
            candidates.append((concurrency, summary))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])


def _fastest_latency_summary(results: dict, max_error_pct: float = 1.0) -> tuple[int, dict] | None:
    summaries = [
        (concurrency, summary)
        for concurrency, summary in _iter_concurrency_summaries(results)
        if summary.get("successful", 0) > 0
        and float(summary.get("error_rate_pct", 100.0)) <= max_error_pct
        and isinstance(summary.get("latency_ms"), dict)
    ]
    if not summaries:
        return None
    return min(
        summaries,
        key=lambda item: (
            float(item[1].get("latency_ms", {}).get("p95", float("inf")) or float("inf")),
            float(item[1].get("latency_ms", {}).get("p50", float("inf")) or float("inf")),
            -float(item[1].get("throughput_rps", 0.0) or 0.0),
            item[0],
        ),
    )


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

    backends = ["faster_whisper", "openai_whisper", "vllm_whisper"]
    b_label  = {"faster_whisper": "FW", "openai_whisper": "OW", "vllm_whisper": "VW"}

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
                model_results = _get(compare, b, bm)
                if model_results == "—" or not isinstance(model_results, dict):
                    row.append("—")
                    continue
                _, selected_results = _preferred_compute_results(model_results)
                summary = _get(selected_results, str(c))
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
        # (display_name, faster_whisper_vram_gb, openai_whisper_vram_gb, vllm_whisper_vram_gb)
        "medium":          (3.06, 3.06, 3.06),
        "large-v3":        (6.17, 6.17, 6.17),
    }
    rows = []
    for display, (fw_gb, ow_gb, vw_gb) in vram_ref.items():
        fw_str = f"~{fw_gb:.2f} GB" if fw_gb else "N/A"
        ow_str = f"~{ow_gb:.2f} GB" if ow_gb else "N/A"
        vw_str = f"~{vw_gb:.2f} GB+" if vw_gb else "N/A"
        rows.append([display, fw_str, ow_str, vw_str])
    return _md_table(
        ["Model", "faster-whisper VRAM", "openai-whisper VRAM", "vLLM Whisper VRAM"],
        rows,
    )


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
    gpu_records = load_gpu_metrics(gpu_metrics_path)
    gpu_summary = summarize_gpu_metrics(gpu_records)
    gpu_label = _gpu_label(gpu_summary) if gpu_summary else "detected GPU"

    lines = [
        "# GPU Stress Test Report — Medical Ambient Scribing",
        "",
        f"> **GPU**: {gpu_label} · **LLM**: ~20B model via vLLM  ",
        "> **ASR backends**: faster-whisper (CTranslate2) vs openai-whisper (PyTorch) vs vLLM Whisper  ",
        f"> **Audio chunks**: 47 s · **Concurrency sweep**: {' / '.join(str(x) for x in WHISPER_COMPARE_CONCURRENCY)}",
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
    lines.append("## GPU Resource Usage\n")
    if gpu_summary:
        lines += [_gpu_summary_table(gpu_summary), ""]
    else:
        lines += ["_GPU metrics not available (run gpu_monitor sidecar)._\n"]
    lines.append("---\n")

    # ── Whisper comparison matrix ─────────────────────────────────────────────
    compare = all_results.get("whisper_compare", {})
    lines.append("## Whisper Comparison Matrix\n")
    lines += [
        "> **FW** = faster-whisper (CTranslate2)  |  **OW** = openai-whisper (PyTorch)  |  **VW** = vLLM Whisper  ",
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
    for result_set in _iter_whisper_result_sets(all_results):
        model_label = result_set["model"]
        if result_set.get("compute_type"):
            model_label = f"{model_label} [{result_set['compute_type']}]"
        if result_set.get("workers") is not None:
            suffix = f"workers={result_set['workers']}"
            if result_set.get("selected"):
                suffix += ", selected"
            model_label = f"{model_label} ({suffix})"
        detailed_sources.append((result_set["backend"], model_label, result_set["results"]))

    if detailed_sources:
        b_label = {
            "faster_whisper": "faster-whisper",
            "openai_whisper": "openai-whisper",
            "vllm_whisper": "vLLM Whisper",
        }
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
        "> faster-whisper (int8_float16): approximately half the FP16 VRAM above. vLLM adds scheduler/KV-cache overhead above model weights.  ",
        "> Actual parallelism depends on the measured GPU VRAM, worker count, vLLM queue/cache settings, backend, and compute type.",
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
        "> VRAM split depends on LLM_GPU_UTIL plus the Whisper worker count on this GPU.\n",
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
            f"> Measured peak throughput: **{measured_rps:.3f} sessions/s** on 1 GPU  ",
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


def _projection_table(
    throughput_rps_per_gpu: int | float,
    target_tiers: list[int] | None = None,
    headroom_factor: float = 1.0,
) -> str:
    rows = [
        [
            p["target_rps"],
            f"{p['throughput_rps_per_gpu']:.3f}",
            p["gpus_needed"],
            f"{p['utilization_pct']:.1f}%",
        ]
        for p in project_rps_capacity(throughput_rps_per_gpu, target_tiers, headroom_factor=headroom_factor)
    ]
    col_measured = "Planning req/s/GPU" if headroom_factor < 0.999 else "Measured req/s/GPU"
    return _md_table(["Target requests/s", col_measured, "GPUs needed", "Utilization"], rows)


def generate_whisper_gpu_report(
    all_results: dict,
    gpu_metrics_path: str | Path,
    output_path: str | Path,
    target_tiers: list[int] | None = None,
    max_error_pct: float = 1.0,
    gpu_label_prefixes: list[str] | None = None,
    report_title: str = "Whisper-Only GPU Capacity Report",
    max_p95_ms: float | None = None,
    headroom_factor: float | None = None,
    load_mode_note: str | None = None,
) -> str:
    """Render a focused Whisper-only GPU capacity report."""
    if max_p95_ms is None:
        max_p95_ms = float(os.getenv("WHISPER_MAX_P95_MS") or 0.0)
    if headroom_factor is None:
        headroom_factor = float(os.getenv("WHISPER_PROJECTION_HEADROOM", "1.0"))
    hf = max(float(headroom_factor), 0.001)
    gpu_records = filter_gpu_metrics(load_gpu_metrics(gpu_metrics_path), gpu_label_prefixes)
    gpu_summary = summarize_gpu_metrics(gpu_records)
    result_sets = _iter_whisper_result_sets(all_results)

    rows = []
    throughput_candidates: list[dict] = []
    latency_candidates: list[dict] = []
    for result_set in result_sets:
        peak = _peak_throughput_summary(result_set["results"], max_error_pct=max_error_pct)
        fastest = _fastest_latency_summary(result_set["results"], max_error_pct=max_error_pct)
        row_gpu_summary = summarize_gpu_metrics(
            filter_gpu_metrics(gpu_records, [result_set.get("label_prefix", "")])
        ) if result_set.get("label_prefix") else {}
        if peak is None and fastest is None:
            rows.append([
                result_set["backend"],
                result_set["model"],
                result_set.get("compute_type") or "default",
                result_set.get("workers") if result_set.get("workers") is not None else "—",
                "yes" if result_set.get("selected", True) else "",
                "0.00",
                "—",
                f"{row_gpu_summary.get('gpu_util_pct', {}).get('max', 0):.1f}" if row_gpu_summary else "—",
                f"{row_gpu_summary.get('mem_used_mib', {}).get('max', 0):.0f}" if row_gpu_summary else "—",
                "—",
                "—",
                "—",
                "—",
            ])
            continue

        peak_c, peak_summary = peak if peak else fastest
        fastest_c, fastest_summary = fastest if fastest else peak
        peak_lm = peak_summary.get("latency_ms", {})
        fastest_lm = fastest_summary.get("latency_ms", {})
        row = {
            "backend": result_set["backend"],
            "model": result_set["model"],
            "compute_type": result_set.get("compute_type") or "default",
            "workers": result_set.get("workers"),
            "selected": result_set.get("selected", True),
            "peak_concurrency": peak_c,
            "peak_summary": peak_summary,
            "fastest_concurrency": fastest_c,
            "fastest_summary": fastest_summary,
            "gpu_summary": row_gpu_summary,
        }
        rows.append([
            row["backend"],
            row["model"],
            row["compute_type"],
            row["workers"] if row["workers"] is not None else "—",
            "yes" if row["selected"] else "",
            f"{peak_summary.get('throughput_rps', 0):.2f}",
            peak_c,
            f"{row_gpu_summary.get('gpu_util_pct', {}).get('max', 0):.1f}" if row_gpu_summary else "—",
            f"{row_gpu_summary.get('mem_used_mib', {}).get('max', 0):.0f}" if row_gpu_summary else "—",
            _fmt_ms(peak_lm.get("p95")),
            fastest_c,
            _fmt_ms(fastest_lm.get("p95")),
            f"{peak_summary.get('error_rate_pct', 0):.1f}%",
        ])
        if row["selected"] or row["workers"] is None:
            throughput_candidates.append(row)
            latency_candidates.append(row)

    best_throughput: dict | None = None
    for candidate in throughput_candidates:
        if best_throughput is None or (
            float(candidate["peak_summary"].get("throughput_rps", 0) or 0.0),
            -float(candidate["peak_summary"].get("latency_ms", {}).get("p95", float("inf")) or float("inf")),
        ) > (
            float(best_throughput["peak_summary"].get("throughput_rps", 0) or 0.0),
            -float(best_throughput["peak_summary"].get("latency_ms", {}).get("p95", float("inf")) or float("inf")),
        ):
            best_throughput = candidate

    best_latency: dict | None = None
    for candidate in latency_candidates:
        if best_latency is None or (
            float(candidate["fastest_summary"].get("latency_ms", {}).get("p95", float("inf")) or float("inf")),
            -float(candidate["fastest_summary"].get("throughput_rps", 0) or 0.0),
        ) < (
            float(best_latency["fastest_summary"].get("latency_ms", {}).get("p95", float("inf")) or float("inf")),
            -float(best_latency["fastest_summary"].get("throughput_rps", 0) or 0.0),
        ):
            best_latency = candidate

    sla_table_rows: list[list[Any]] = []
    if max_p95_ms > 0:
        for result_set in result_sets:
            res = result_set.get("results")
            if not isinstance(res, dict):
                continue
            sla = max_concurrency_under_p95_sla(res, max_p95_ms, max_error_pct)
            if not sla:
                continue
            sc, ss = sla
            lm = ss.get("latency_ms", {}) or {}
            sla_table_rows.append([
                result_set["backend"],
                result_set["model"],
                str(result_set.get("compute_type") or "default"),
                str(result_set.get("workers") if result_set.get("workers") is not None else "—"),
                str(sc),
                _fmt_ms(lm.get("p95")),
                f"{ss.get('throughput_rps', 0):.2f}",
                f"{ss.get('error_rate_pct', 0):.1f}%",
            ])

    methodology = [
        f"> GPU: {_gpu_label(gpu_summary) if gpu_summary else 'GPU metrics unavailable'}",
        "> Workload: concurrent 47-second audio chunk transcription requests.",
        "> Planning basis: highest measured requests/second with error rate within threshold.",
        "> Fastest response time is reported separately from max throughput so the latency/throughput tradeoff is visible.",
        "> `Selected=yes` marks the chosen worker count for that backend/model/compute variant when worker sweep was used.",
        "> External queues (e.g. Redis/RabbitMQ) are not simulated; results size GPU workers and safe in-flight concurrency.",
    ]
    if load_mode_note:
        methodology.append(f"> Load generator: {load_mode_note}")
    if hf < 0.999:
        methodology.append(
            f"> Projection uses planning headroom factor **{hf:g}** (effective req/s/GPU = measured × headroom)."
        )

    lines = [
        f"# {report_title}",
        "",
        *methodology,
        "",
        "## GPU Resource Usage",
        "",
        _gpu_summary_table(gpu_summary),
        "",
        "## Whisper Results",
        "",
        _md_table(
            [
                "Backend",
                "Model",
                "Compute",
                "Workers",
                "Selected",
                "Max req/s",
                "Max req/s @ c",
                "Peak GPU %",
                "Peak VRAM MiB",
                "Peak p95 ms",
                "Fastest @ c",
                "Fastest p95 ms",
                "Peak error",
            ],
            rows,
        ),
        "",
    ]

    best_sla: list[Any] | None = None
    if sla_table_rows:
        # Prefer highest req/s, then lowest p95 among SLA-valid rows.
        def _sla_sort_key(row: list[Any]) -> tuple[float, float]:
            req_s = float(row[6]) if str(row[6]).replace(".", "", 1).isdigit() else 0.0
            p95 = float(row[5]) if str(row[5]).replace(".", "", 1).isdigit() else float("inf")
            return (req_s, -p95)

        best_sla = max(sla_table_rows, key=_sla_sort_key)
        lines += [
            "## Latency SLA operating point",
            "",
            (
                f"Highest concurrency level where **p95 ≤ {max_p95_ms:g} ms** "
                f"and error rate ≤ {max_error_pct:g}% (per row)."
            ),
            "",
            _md_table(
                [
                    "Backend",
                    "Model",
                    "Compute",
                    "Workers",
                    "Max c @ SLA",
                    "p95 ms",
                    "req/s",
                    "Error",
                ],
                sla_table_rows,
            ),
            "",
        ]
        lines += [
            "### Recommended operating point",
            "",
            (
                f"`{best_sla[0]}` / `{best_sla[1]}` / `{best_sla[2]}` / workers `{best_sla[3]}` "
                f"at concurrency **{best_sla[4]}** (p95 **{best_sla[5]} ms**, "
                f"throughput **{best_sla[6]} req/s**, error **{best_sla[7]}**)."
            ),
            "",
        ]

    if best_throughput:
        peak_lm = best_throughput["peak_summary"].get("latency_ms", {})
        lines += [
            "## Projection",
            "",
            (
                f"Highest measured sustained throughput on this GPU: **{best_throughput['peak_summary'].get('throughput_rps', 0):.2f} req/s** "
                f"at concurrency **{best_throughput['peak_concurrency']}** using "
                f"`{best_throughput['backend']}` / `{best_throughput['model']}` / "
                f"`{best_throughput['compute_type']}`"
                + (
                    f" / `workers={best_throughput['workers']}`."
                    if best_throughput.get("workers") is not None else "."
                )
            ),
            (
                f"At that throughput point p95 latency was {_fmt_ms(peak_lm.get('p95'))} ms and error rate was "
                f"{best_throughput['peak_summary'].get('error_rate_pct', 0):.1f}%."
            ),
            (
                f"Observed GPU peak during that run: "
                f"{best_throughput.get('gpu_summary', {}).get('gpu_util_pct', {}).get('max', 0):.1f}% util, "
                f"{best_throughput.get('gpu_summary', {}).get('mem_used_mib', {}).get('max', 0):.0f} MiB VRAM."
                if best_throughput.get("gpu_summary") else "Observed GPU metrics were not available for that run."
            ),
            "",
        ]
        if best_latency:
            fastest_lm = best_latency["fastest_summary"].get("latency_ms", {})
            lines += [
                (
                    f"Fastest measured response time: p95 **{_fmt_ms(fastest_lm.get('p95'))} ms** "
                    f"at concurrency **{best_latency['fastest_concurrency']}** using "
                    f"`{best_latency['backend']}` / `{best_latency['model']}` / `{best_latency['compute_type']}`"
                    + (
                        f" / `workers={best_latency['workers']}`."
                        if best_latency.get("workers") is not None else "."
                    )
                ),
                (
                    f"At that latency point throughput was {best_latency['fastest_summary'].get('throughput_rps', 0):.2f} req/s "
                    f"and error rate was {best_latency['fastest_summary'].get('error_rate_pct', 0):.1f}%."
                ),
                "",
            ]
        raw_rps = float(best_throughput["peak_summary"].get("throughput_rps", 0) or 0.0)
        lines += [
            _projection_table(raw_rps, target_tiers, headroom_factor=hf),
            "",
        ]
        if hf < 0.999:
            eff = raw_rps * hf
            lines += [
                f"Raw measured peak req/s (above): **{raw_rps:.3f}**; effective planning req/s/GPU: **{eff:.3f}** (× headroom {hf:g}).",
                "",
            ]
        lines += [
            "Assumption: horizontal scaling is roughly linear across identical GPUs and the same request mix.",
            "Projection table uses effective req/s/GPU when headroom is below 1; otherwise measured req/s.",
            "Batch-mode benchmarks can over- or under-state sustained utilization versus steady load; see load generator notes above.",
            "",
        ]
    else:
        lines += [
            "## Projection",
            "",
            "_No valid Whisper throughput result was available, so GPU count projection was not generated._",
            "",
        ]

    report = "\n".join(lines)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    return report


def generate_vllm_gpu_report(
    all_results: dict,
    gpu_metrics_path: str | Path,
    output_path: str | Path,
    target_tiers: list[int] | None = None,
    max_error_pct: float = 1.0,
) -> str:
    """Render a focused vLLM-only GPU capacity report."""
    gpu_summary = summarize_gpu_metrics(load_gpu_metrics(gpu_metrics_path))
    llm_results = all_results.get("llm") or {}

    rows = []
    for concurrency, summary in _iter_concurrency_summaries(llm_results):
        lm = summary.get("latency_ms", {})
        rows.append([
            concurrency,
            f"{summary.get('throughput_rps', 0):.3f}",
            f"{summary.get('tokens_out_per_sec', 0):.0f}",
            _fmt_ms(lm.get("p50")),
            _fmt_ms(lm.get("p95")),
            _fmt_ms(lm.get("p99")),
            f"{summary.get('error_rate_pct', 0):.1f}%",
        ])

    peak = _peak_throughput_summary(llm_results, max_error_pct=max_error_pct)
    fastest = _fastest_latency_summary(llm_results, max_error_pct=max_error_pct)

    lines = [
        "# vLLM GPU Capacity Report",
        "",
        f"> GPU: {_gpu_label(gpu_summary) if gpu_summary else 'GPU metrics unavailable'}",
        "> Workload: OpenAI-compatible `/v1/chat/completions` requests against vLLM.",
        "> Planning basis: highest measured requests/second with error rate within threshold.",
        "> Fastest response time is reported separately from max throughput so the latency/throughput tradeoff is visible.",
        "",
        "## GPU Resource Usage",
        "",
        _gpu_summary_table(gpu_summary),
        "",
        "## vLLM Results",
        "",
        _md_table(["Concurrency", "req/s", "tok/s", "p50 ms", "p95 ms", "p99 ms", "Error"], rows),
        "",
    ]

    if peak:
        peak_c, peak_summary = peak
        peak_lm = peak_summary.get("latency_ms", {})
        lines += [
            "## Projection",
            "",
            (
                f"Highest measured sustained throughput on this GPU: **{peak_summary.get('throughput_rps', 0):.3f} req/s** "
                f"at concurrency **{peak_c}**."
            ),
            (
                f"At that throughput point p95 latency was {_fmt_ms(peak_lm.get('p95'))} ms and error rate was "
                f"{peak_summary.get('error_rate_pct', 0):.1f}%."
            ),
            "",
        ]
        if fastest:
            fastest_c, fastest_summary = fastest
            fastest_lm = fastest_summary.get("latency_ms", {})
            lines += [
                (
                    f"Fastest measured response time: p95 **{_fmt_ms(fastest_lm.get('p95'))} ms** "
                    f"at concurrency **{fastest_c}**."
                ),
                (
                    f"At that latency point throughput was {fastest_summary.get('throughput_rps', 0):.3f} req/s, "
                    f"{fastest_summary.get('tokens_out_per_sec', 0):.0f} tok/s, "
                    f"and error rate was {fastest_summary.get('error_rate_pct', 0):.1f}%."
                ),
                "",
            ]
        lines += [
            _projection_table(peak_summary.get("throughput_rps", 0), target_tiers),
            "",
            "Assumption: horizontal scaling is roughly linear across identical GPUs and the same prompt/output mix.",
            "Projection table is based on sustained requests/second, not in-flight concurrency.",
            "",
        ]
    else:
        lines += [
            "## Projection",
            "",
            "_No valid vLLM throughput result was available, so GPU count projection was not generated._",
            "",
        ]

    report = "\n".join(lines)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    return report
