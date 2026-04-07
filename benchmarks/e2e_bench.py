"""
End-to-end (E2E) pipeline benchmark.

Simulates complete ambient scribing sessions under concurrent load:
  1. Per session: CHUNKS_PER_SESSION Whisper requests (47 s each) sent in parallel
     → simulates a doctor's appointment where chunks accumulate concurrently.
  2. After all chunks complete: 1 LLM chat completion (transcript → SOAP note).
  3. Measures total session wall time (Whisper phase + LLM phase + overhead).

Concurrency levels: number of simultaneous sessions (each with its own chunks + LLM call).

Results determine the mixed-workload saturation point — the key number for capacity planning.

Run directly:
  python -m benchmarks.e2e_bench
  python -m benchmarks.e2e_bench --sessions 20,50,100 --chunks 13
"""
import asyncio
import argparse
import itertools
import json
import time
from pathlib import Path

import aiohttp

from benchmarks import config as cfg
from generators.audio_gen import generate_pool
from generators.transcript_gen import generate_prompt
from metrics.collector import MetricsCollector, RequestRecord


async def _transcribe_chunk(
    session: aiohttp.ClientSession,
    audio_bytes: bytes,
    req_id: str,
) -> dict:
    """Transcribe one 47-second audio chunk. Returns {"text": ..., "ok": bool}."""
    form = aiohttp.FormData()
    form.add_field("file", audio_bytes, filename="chunk.wav", content_type="audio/wav")
    timeout = aiohttp.ClientTimeout(total=cfg.WHISPER_TIMEOUT)
    try:
        async with session.post(f"{cfg.WHISPER_URL}/transcribe", data=form, timeout=timeout) as resp:
            if resp.status == 200:
                body = await resp.json()
                return {"text": body.get("text", ""), "ok": True}
            return {"text": "", "ok": False, "error": f"HTTP {resp.status}"}
    except Exception as exc:
        return {"text": "", "ok": False, "error": str(exc)}


async def _generate_note(
    session: aiohttp.ClientSession,
    transcript: str,
) -> dict:
    """Send assembled transcript to LLM for SOAP note generation."""
    from generators.prompt_templates import SYSTEM_PROMPT
    payload = {
        "model": cfg.LLM_MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"TRANSCRIPT:\n{transcript}"},
        ],
        "max_tokens": 512,
        "temperature": 0.1,
        "stream": False,
    }
    timeout = aiohttp.ClientTimeout(total=cfg.LLM_TIMEOUT)
    try:
        async with session.post(
            f"{cfg.LLM_URL}/v1/chat/completions",
            json=payload,
            timeout=timeout,
        ) as resp:
            if resp.status == 200:
                body  = await resp.json()
                usage = body.get("usage", {})
                return {
                    "ok": True,
                    "tokens_out": usage.get("completion_tokens", 0),
                    "tokens_in":  usage.get("prompt_tokens", 0),
                }
            return {"ok": False, "error": f"HTTP {resp.status}"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


async def _run_single_session(
    session_id: int,
    http_session: aiohttp.ClientSession,
    audio_pool: list[bytes],
    n_chunks: int,
) -> dict:
    """
    Simulate one full doctor's appointment session.
    Returns timing breakdown and success status.
    """
    t_start = time.perf_counter()
    audio_cycle = itertools.cycle(audio_pool)

    # Phase 1: Parallel Whisper transcription of all chunks
    chunk_tasks = [
        _transcribe_chunk(
            http_session,
            next(audio_cycle),
            f"sess{session_id}_chunk{i}",
        )
        for i in range(n_chunks)
    ]
    chunk_results = await asyncio.gather(*chunk_tasks)
    t_whisper_done = time.perf_counter()

    # Assemble transcript from successful chunks
    transcript = " ".join(
        r["text"] for r in chunk_results if r.get("ok") and r.get("text")
    )
    whisper_errors = sum(1 for r in chunk_results if not r.get("ok"))

    # Phase 2: LLM note generation
    note_result = await _generate_note(http_session, transcript)
    t_end = time.perf_counter()

    return {
        "session_id":           session_id,
        "n_chunks":             n_chunks,
        "whisper_errors":       whisper_errors,
        "llm_ok":               note_result.get("ok", False),
        "whisper_time_ms":      (t_whisper_done - t_start)     * 1000,
        "llm_time_ms":          (t_end - t_whisper_done)       * 1000,
        "total_time_ms":        (t_end - t_start)              * 1000,
        "tokens_out":           note_result.get("tokens_out", 0),
        "success":              whisper_errors == 0 and note_result.get("ok", False),
    }


async def _run_session_count(
    n_sessions: int,
    audio_pool: list[bytes],
    n_chunks: int,
) -> dict:
    """Launch n_sessions concurrent E2E sessions and return aggregated metrics."""
    test_name = f"e2e_{n_sessions}sessions"
    collector = MetricsCollector(test_name)

    connector = aiohttp.TCPConnector(limit=n_sessions * n_chunks + 20)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Warmup: run 2 sessions sequentially
        for i in range(2):
            await _run_single_session(i, session, audio_pool, n_chunks)
        print(f"    warmup done")

        # Actual test
        collector.reset()
        session_tasks = [
            _run_session(i, session, audio_pool, n_chunks, collector)
            for i in range(n_sessions)
        ]
        await asyncio.gather(*session_tasks)

    return collector.summarize()


async def _run_session(
    session_id: int,
    http_session: aiohttp.ClientSession,
    audio_pool: list[bytes],
    n_chunks: int,
    collector: MetricsCollector,
) -> None:
    rec = RequestRecord(
        request_id=f"session_{session_id}",
        start_ts=time.perf_counter(),
    )
    result = await _run_single_session(session_id, http_session, audio_pool, n_chunks)
    rec.end_ts        = time.perf_counter()
    rec.latency_ms    = result["total_time_ms"]
    rec.tokens_out    = result.get("tokens_out", 0)
    rec.success       = result["success"]
    if not rec.success:
        rec.error = f"whisper_errors={result['whisper_errors']} llm_ok={result['llm_ok']}"
    await collector.record(rec)


async def run_e2e_bench(
    session_counts: list[int] | None = None,
    n_chunks: int | None = None,
) -> dict:
    if session_counts is None:
        session_counts = cfg.E2E_SESSION_COUNTS
    if n_chunks is None:
        n_chunks = cfg.CHUNKS_PER_SESSION

    print(f"\n=== E2E Benchmark  sessions={session_counts}  chunks_per_session={n_chunks} ===")
    print(f"Generating {cfg.AUDIO_POOL_SIZE} × {cfg.AUDIO_DURATION_S}s audio files...")
    audio_pool = generate_pool(cfg.AUDIO_POOL_SIZE, cfg.AUDIO_DURATION_S)
    print(f"  {len(audio_pool)} files ready")

    results: dict[str, dict] = {}
    for n_sessions in session_counts:
        print(f"\n  concurrent sessions = {n_sessions}")
        summary = await _run_session_count(n_sessions, audio_pool, n_chunks)
        results[str(n_sessions)] = summary

        lm = summary.get("latency_ms", {})
        print(
            f"    sessions/s={summary.get('throughput_rps', 0):.3f}  "
            f"p50={lm.get('p50', 0):.0f}ms  p99={lm.get('p99', 0):.0f}ms  "
            f"err={summary.get('error_rate_pct', 0):.1f}%"
        )

        if n_sessions != session_counts[-1]:
            print(f"  cooling down {cfg.COOLDOWN_S}s...")
            await asyncio.sleep(cfg.COOLDOWN_S)

    return results


def _save(results: dict) -> None:
    out = Path(cfg.RESULTS_DIR)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "e2e_bench.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {path}")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", default=None,
                        help="Comma-separated session counts, e.g. 20,50,100")
    parser.add_argument("--chunks", type=int, default=None,
                        help="Audio chunks per session (default from config)")
    args = parser.parse_args()

    session_counts = (
        [int(x) for x in args.sessions.split(",")]
        if args.sessions else None
    )
    results = await run_e2e_bench(session_counts=session_counts, n_chunks=args.chunks)
    _save(results)


if __name__ == "__main__":
    asyncio.run(main())
