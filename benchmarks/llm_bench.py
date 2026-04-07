"""
LLM throughput benchmark.

Metrics captured:
  - End-to-end latency per request (request sent → final token received)
  - Aggregate tokens/second across all concurrent requests
  - TTFT (Time To First Token) via a streaming probe

Methodology:
  1. Generate LLM_CONCURRENCY_LEVELS × MIN_SAMPLES prompts from transcript templates.
  2. Warm up with WARMUP_REQUESTS sequential completions.
  3. For each concurrency level:
     a. Fire `concurrency` chat/completions simultaneously.
     b. Repeat for ceil(MIN_SAMPLES / concurrency) rounds.
     c. Sleep COOLDOWN_S.

Run directly:
  python -m benchmarks.llm_bench
  python -m benchmarks.llm_bench --url http://gpu-box:8002
"""
import asyncio
import argparse
import itertools
import json
import time
from pathlib import Path

import aiohttp

from benchmarks import config as cfg
from generators.transcript_gen import generate_prompt
from metrics.collector import MetricsCollector, RequestRecord


def _make_prompt_pool(size: int) -> list[dict]:
    return [generate_prompt(seed=i, model_name=cfg.LLM_MODEL_NAME) for i in range(size)]


async def _single_completion(
    session: aiohttp.ClientSession,
    payload: dict,
    request_id: str,
    collector: MetricsCollector,
) -> None:
    rec = RequestRecord(request_id=request_id, start_ts=time.perf_counter())
    try:
        timeout = aiohttp.ClientTimeout(total=cfg.LLM_TIMEOUT)
        async with session.post(
            f"{cfg.LLM_URL}/v1/chat/completions",
            json=payload,
            timeout=timeout,
        ) as resp:
            rec.end_ts     = time.perf_counter()
            rec.latency_ms = (rec.end_ts - rec.start_ts) * 1000
            if resp.status == 200:
                body           = await resp.json()
                usage          = body.get("usage", {})
                rec.tokens_in  = usage.get("prompt_tokens", 0)
                rec.tokens_out = usage.get("completion_tokens", 0)
            else:
                rec.success = False
                rec.error   = f"HTTP {resp.status}: {await resp.text()}"
    except Exception as exc:
        rec.end_ts     = time.perf_counter()
        rec.latency_ms = (rec.end_ts - rec.start_ts) * 1000
        rec.success    = False
        rec.error      = str(exc)
    await collector.record(rec)


async def _probe_ttft(session: aiohttp.ClientSession) -> float:
    """
    Measure Time To First Token via streaming.
    Returns TTFT in milliseconds, or -1 on error.
    """
    payload = generate_prompt(seed=999, model_name=cfg.LLM_MODEL_NAME)
    payload["stream"] = True

    t0 = time.perf_counter()
    try:
        async with session.post(
            f"{cfg.LLM_URL}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=cfg.LLM_TIMEOUT),
        ) as resp:
            if resp.status != 200:
                return -1.0
            async for chunk in resp.content:
                line = chunk.decode("utf-8", errors="ignore").strip()
                if line.startswith("data:") and "[DONE]" not in line:
                    return (time.perf_counter() - t0) * 1000
    except Exception:
        return -1.0
    return -1.0


async def _run_concurrency_level(
    concurrency: int,
    prompt_pool: list[dict],
) -> dict:
    test_name    = f"llm_c{concurrency}"
    prompt_cycle = itertools.cycle(prompt_pool)

    connector = aiohttp.TCPConnector(limit=concurrency + 10)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Warmup
        warmup_col = MetricsCollector(f"{test_name}_warmup")
        for i in range(cfg.WARMUP_REQUESTS):
            await _single_completion(session, next(prompt_cycle), f"w{i}", warmup_col)
        w = warmup_col.summarize()
        print(f"    warmup done  (avg {w.get('latency_ms', {}).get('mean', 0):.0f} ms)")

        # TTFT probe (at concurrency=1 only to avoid noise)
        ttft_ms = -1.0
        if concurrency == 1:
            ttft_ms = await _probe_ttft(session)
            print(f"    TTFT = {ttft_ms:.0f} ms")

        collector = MetricsCollector(test_name)
        n_rounds  = max(1, cfg.MIN_SAMPLES // concurrency)

        for round_i in range(n_rounds):
            tasks = [
                _single_completion(session, next(prompt_cycle), f"r{round_i}_req{j}", collector)
                for j in range(concurrency)
            ]
            await asyncio.gather(*tasks)

    summary = collector.summarize()
    summary["concurrency"] = concurrency
    summary["ttft_ms"]     = ttft_ms
    return summary


async def run_llm_bench(llm_url: str | None = None) -> dict:
    if llm_url:
        cfg.LLM_URL = llm_url

    print(f"\n=== LLM Benchmark  model={cfg.LLM_MODEL_NAME}  url={cfg.LLM_URL} ===")
    print(f"Generating {cfg.AUDIO_POOL_SIZE} prompt templates...")
    prompt_pool = _make_prompt_pool(cfg.AUDIO_POOL_SIZE)

    results: dict[str, dict] = {}
    for concurrency in cfg.LLM_CONCURRENCY:
        print(f"\n  concurrency = {concurrency}")
        summary = await _run_concurrency_level(concurrency, prompt_pool)
        results[str(concurrency)] = summary

        lm = summary.get("latency_ms", {})
        print(
            f"    rps={summary.get('throughput_rps', 0):.3f}  "
            f"tok/s={summary.get('tokens_out_per_sec', 0):.0f}  "
            f"p50={lm.get('p50', 0):.0f}ms  p99={lm.get('p99', 0):.0f}ms  "
            f"err={summary.get('error_rate_pct', 0):.1f}%"
        )

        if concurrency != cfg.LLM_CONCURRENCY[-1]:
            print(f"  cooling down {cfg.COOLDOWN_S}s...")
            await asyncio.sleep(cfg.COOLDOWN_S)

    return results


def _save(results: dict) -> None:
    out = Path(cfg.RESULTS_DIR)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "llm_bench.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {path}")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=None, help="Override LLM_URL")
    args = parser.parse_args()

    results = await run_llm_bench(llm_url=args.url)
    _save(results)


if __name__ == "__main__":
    asyncio.run(main())
