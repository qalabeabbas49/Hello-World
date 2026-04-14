"""Render a benchmark markdown report from saved JSON results."""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from metrics.reporter import generate_report


DEFAULT_OPENAI_REPORT_MODEL = os.getenv("OPENAI_REPORT_MODEL", "gpt-5-mini")
DEFAULT_OPENAI_REASONING_EFFORT = os.getenv("OPENAI_REPORT_REASONING", "minimal")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        default="./results",
        help="Directory containing all_results.json and gpu_metrics.jsonl.",
    )
    parser.add_argument(
        "--ai",
        action="store_true",
        help="Generate an OpenAI-written narrative report in addition to the deterministic markdown report.",
    )
    parser.add_argument(
        "--ai-model",
        default=DEFAULT_OPENAI_REPORT_MODEL,
        help=f"OpenAI model for the AI-written report (default: {DEFAULT_OPENAI_REPORT_MODEL}).",
    )
    parser.add_argument(
        "--ai-reasoning-effort",
        default=DEFAULT_OPENAI_REASONING_EFFORT,
        help=f"Reasoning effort for the AI-written report (default: {DEFAULT_OPENAI_REASONING_EFFORT}).",
    )
    parser.add_argument(
        "--ai-output",
        default=None,
        help="Path for the AI-written markdown report (default: <results-dir>/report_ai.md).",
    )
    return parser.parse_args()


def _compact_results_summary(results: dict) -> dict:
    """Build a small, high-signal summary payload for the LLM prompt."""
    summary: dict[str, object] = {
        "whisper_compare": {},
        "llm": {},
        "e2e": {},
        "format": {},
    }

    compare = results.get("whisper_compare", {})
    if isinstance(compare, dict):
        for backend, models in compare.items():
            if not isinstance(models, dict):
                continue
            summary["whisper_compare"][backend] = {}
            for model, compute_map in models.items():
                if not isinstance(compute_map, dict):
                    continue
                summary["whisper_compare"][backend][model] = {}
                for compute_type, compute_results in compute_map.items():
                    if not isinstance(compute_results, dict):
                        continue
                    if isinstance(compute_results.get("workers"), dict):
                        summary["whisper_compare"][backend][model][compute_type] = {
                            "selected_workers": compute_results.get("selected_workers"),
                            "tested_workers": sorted(
                                int(w) for w in compute_results["workers"].keys() if str(w).isdigit()
                            ),
                        }
                    else:
                        summary["whisper_compare"][backend][model][compute_type] = {
                            "tested_workers": None,
                        }

    llm = results.get("llm", {})
    if isinstance(llm, dict):
        summary["llm"] = sorted(int(k) for k in llm.keys() if str(k).isdigit())

    e2e = results.get("e2e", {})
    if isinstance(e2e, dict):
        summary["e2e"] = sorted(int(k) for k in e2e.keys() if str(k).isdigit())

    fmt = results.get("format", {})
    if isinstance(fmt, dict):
        summary["format"] = {
            source: sorted(source_data.keys())
            for source, source_data in fmt.items()
            if isinstance(source_data, dict)
        }

    return summary


def _extract_response_text(data: dict) -> str:
    text = data.get("output_text")
    if isinstance(text, str) and text.strip():
        return text.strip()

    chunks: list[str] = []
    for item in data.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if not isinstance(content, dict):
                continue
            if content.get("type") in ("output_text", "text"):
                value = content.get("text")
                if isinstance(value, str) and value:
                    chunks.append(value)
    return "\n".join(chunks).strip()


def _generate_openai_report(
    base_report: str,
    results: dict,
    output_path: Path,
    model: str,
    reasoning_effort: str,
) -> Path:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for --ai report generation.")

    summary = _compact_results_summary(results)
    system_prompt = (
        "You are a benchmarking analyst writing polished markdown reports for engineering teams. "
        "Use only the provided benchmark data. Do not invent measurements. "
        "Call out the best-performing configurations, saturation patterns, bottlenecks, worker-sweep findings, "
        "capacity implications, and the next experiments worth running. "
        "Write a professional, concise, high-signal report in markdown."
    )
    user_prompt = (
        "Create an executive-quality markdown benchmark report based on the data below.\n\n"
        "Requirements:\n"
        "- Include sections: Executive Summary, Best Configurations, Saturation Findings, GPU Utilization Insights, Risks, Recommended Next Runs.\n"
        "- When worker sweep results exist, explicitly mention the selected worker count and why it appears best.\n"
        "- Preserve exact numbers when citing measurements.\n"
        "- Prefer short paragraphs and compact bullets.\n"
        "- End with a short decision-oriented recommendation.\n\n"
        "Compact result summary:\n"
        f"```json\n{json.dumps(summary, indent=2)}\n```\n\n"
        "Deterministic benchmark report:\n"
        f"```markdown\n{base_report}\n```"
    )
    payload = {
        "model": model,
        "reasoning": {"effort": reasoning_effort},
        "instructions": system_prompt,
        "input": user_prompt,
        "text": {"format": {"type": "text"}},
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API request failed: HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenAI API request failed: {exc}") from exc

    text = _extract_response_text(data)
    if not text:
        raise RuntimeError("OpenAI API returned no text output for the AI-written report.")

    output_path.write_text(text)
    return output_path


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_path = results_dir / "all_results.json"
    report_path = results_dir / "report.md"

    if not results_path.exists():
        print(f"No results at {results_path}. Run bench-all or bench-compare first.")
        return 1

    results = json.loads(results_path.read_text())
    report = generate_report(
        results,
        gpu_metrics_path=results_dir / "gpu_metrics.jsonl",
        output_path=report_path,
    )
    print(f"Report written to {report_path}")

    if args.ai:
        ai_output_path = Path(args.ai_output) if args.ai_output else results_dir / "report_ai.md"
        _generate_openai_report(
            base_report=report,
            results=results,
            output_path=ai_output_path,
            model=args.ai_model,
            reasoning_effort=args.ai_reasoning_effort,
        )
        print(f"AI report written to {ai_output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
