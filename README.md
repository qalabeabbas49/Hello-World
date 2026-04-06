# GPU Stress Testing — Medical Ambient Scribing

End-to-end benchmark suite for a single NVIDIA H200 running:

- **Whisper** (faster-whisper / openai-whisper) — 47-second audio chunk transcription
- **LLM** (vLLM, ~20B parameters) — SOAP medical note generation

The benchmarks answer: **how many H200s do we need for 500 / 1 000 / 1 500 concurrent users?**

---

## Prerequisites

| Requirement | Notes |
|---|---|
| NVIDIA H200 (or H100) | `nvidia-smi` must be available |
| Docker + nvidia-container-toolkit | GPU passthrough to containers |
| Python 3.11+ | Client-side benchmark runner |
| ffmpeg | Opus audio generation + decoding |

```bash
pip install -r requirements.txt
```

---

## Quick Start

```bash
# 1. Configure
cp .env.example .env
# Edit LLM_MODEL_HOST_PATH to point at your model weights

# 2. Generate synthetic audio fixtures (WAV/FLAC/Opus at 16 kHz + 48 kHz)
make fixtures

# 3. Verify GPU access
make check-gpu

# 4. Run the full benchmark suite
make bench-all

# Results in ./results/report.md
```

---

## Benchmark Targets

| Target | What it measures | Key output |
|---|---|---|
| `make bench-compare` | Both backends × all Whisper models, c=1–100 | RTF / req/s / p50 / p95 matrices |
| `make bench-formats` | WAV / FLAC / Opus at 16 kHz + 48 kHz | Format decoding + resampling overhead |
| `make bench-real-audio` | Same as formats, using your actual recordings | Real-world performance |
| `make bench-llm` | LLM at c=1–50 | tokens/sec, TTFT, p50/p95/p99 |
| `make bench-e2e` | Full session: 13 Whisper chunks + 1 LLM note | sessions/sec → capacity projection |
| `make bench-all` | All of the above | Consolidated `report.md` |

### Key variables

```bash
make bench-compare BACKEND=faster_whisper MODEL=large-v2 WORKERS=16
make bench-formats FORMATS=wav,flac,opus RATES=16000,48000
make bench-real-audio REAL_AUDIO_DIR=/path/to/your/audio
make bench-all WORKERS=16 RESULTS=./results
```

### Compute type sweep (faster-whisper only)

```bash
# Compare float16 vs int8_float16 (~10-15% faster, <1% WER penalty)
python -m benchmarks.run_all --mode compare \
  --compute-types float16,int8_float16 \
  --output ./results
```

---

## Repository Layout

```
├── services/
│   ├── whisper/            FastAPI service, dual-backend (faster-whisper + openai-whisper)
│   │   ├── app.py          /transcribe  /health  /metrics
│   │   ├── batch_manager.py  WhisperPool factory + ffmpeg audio fallback
│   │   ├── config.py
│   │   └── Dockerfile
│   └── llm/
│       ├── entrypoint.sh   vLLM with --enable-prefix-caching
│       └── Dockerfile
│
├── benchmarks/
│   ├── config.py           URLs, model lists, concurrency levels, audio settings
│   ├── run_all.py          Master orchestrator (--mode whisper|compare|format|llm|e2e|all)
│   ├── whisper_bench.py    Concurrent Whisper load test
│   ├── llm_bench.py        LLM throughput test
│   ├── e2e_bench.py        End-to-end session simulation
│   └── audio_format_bench.py  Format × sample-rate sweep
│
├── generators/
│   ├── audio_gen.py        Synthetic audio: WAV/FLAC/Opus, 16k/48k, formant synthesis
│   ├── real_audio.py       RealAudioPool: scan directory, probe format+rate, serve bytes
│   ├── transcript_gen.py   Medical transcript templates
│   └── prompt_templates.py SOAP system prompt (enables vLLM prefix caching)
│
├── metrics/
│   ├── collector.py        AsyncIO-safe MetricsCollector + GPU label helpers
│   ├── gpu_monitor.py      nvidia-smi JSONL sidecar (500 ms polling)
│   └── reporter.py         Comparison matrices + capacity projections → report.md
│
├── fixtures/
│   ├── audio/              Synthetic audio (gitignored; regenerate with make fixtures)
│   └── real_audio/         Drop your own recordings here (gitignored)
│
├── results/                Benchmark output (gitignored)
├── docker-compose.yml              Mixed: Whisper + LLM + GPU monitor
├── docker-compose.whisper-only.yml Whisper isolation
├── docker-compose.llm-only.yml     LLM isolation
├── .env.example
└── Makefile
```

---

## Service Management

```bash
# Start individual services
make up-whisper BACKEND=faster_whisper MODEL=large-v2 WORKERS=16
make up-llm
make up-mixed   # both services (for E2E bench)

# Stop everything
make down
```

---

## Real Audio Files

Place your recordings under `./fixtures/real_audio/` (or set `REAL_AUDIO_DIR`).

**Organised layout** (auto-detected):
```
real_audio/
  wav_16000/   *.wav
  wav_48000/   *.wav
  flac_16000/  *.flac
  opus_48000/  *.opus
```

**Flat layout** also works — format inferred from extension, sample rate probed via soundfile/ffprobe.

Supported: WAV, FLAC, Opus (.opus / .ogg), MP3, M4A, WebM.

```bash
make bench-real-audio REAL_AUDIO_DIR=/path/to/your/audio
```

---

## Whisper Workers — Recommended Values (H200, 141 GB VRAM)

| Model | VRAM/instance | Whisper-only | Mixed (with LLM) |
|---|---|---|---|
| tiny | ~0.15 GB | 48+ | 48 |
| base | ~0.29 GB | 32+ | 32 |
| small | ~0.93 GB | 24 | 16 |
| medium | ~3.06 GB | 12 | 8 |
| large-v2 / large-v3 | ~6.17 GB | 6 | 4 |
| large-v3-turbo | ~3.10 GB | 12 | 8 |

Set via `WHISPER_WORKERS=N` in `.env` or `make up-whisper WORKERS=N`.

---

## Report

`results/report.md` contains:

1. GPU resource usage (peak/mean VRAM, utilisation, power, temperature)
2. Whisper comparison matrix — faster-whisper vs openai-whisper, all models × c=1/10/50
3. Detailed concurrency sweep per backend+model
4. VRAM reference table
5. Audio format × sample-rate matrix (WAV/FLAC/Opus × 16k/48k)
6. LLM throughput table
7. End-to-end pipeline results
8. **Capacity projections** — concurrent users → H200s required

```bash
# Re-render report from saved results without re-running benchmarks
make report
```

---

## Architecture Notes

**Whisper dual-backend** — one container, one image. Backend selected at runtime via `WHISPER_BACKEND` env var. Both share the same async pool pattern (`asyncio.Queue` of N model instances + `ThreadPoolExecutor`).

**Audio decoding** — `soundfile` handles WAV/FLAC/OGG Vorbis. ffmpeg subprocess pipe handles Opus/WebM/MP3/M4A. All audio resampled to 16 kHz mono server-side.

**LLM prefix caching** — The ~120-token SOAP system prompt is computed once; vLLM reuses the KV cache for every subsequent request, saving ~0.1s per request at high concurrency.

**GPU isolation** — No CUDA MIG. Separate Compose files omit the competing service. In mixed mode, `LLM_GPU_UTIL=0.45` claims ~63 GB, Whisper gets the remainder.
