# GPU Stress Test Report — Medical Ambient Scribing

> **GPU**: detected GPU · **LLM**: ~20B model via vLLM  
> **ASR backends**: faster-whisper (CTranslate2) vs openai-whisper (PyTorch) vs vLLM Whisper  
> **Audio chunks**: 47 s · **Concurrency sweep**: 1 / 10 / 50 / 100

---

## Contents

1. [GPU Resource Usage](#gpu-resource-usage)
2. [Whisper Comparison Matrix](#whisper-comparison-matrix)
3. [Whisper Detailed Results](#whisper-detailed-results)
4. [VRAM Reference](#vram-reference)
5. [Audio Format & Sample Rate Matrix](#audio-format--sample-rate-matrix)
6. [LLM Throughput](#llm-throughput)
7. [End-to-End Pipeline](#end-to-end-pipeline)
8. [Capacity Projections](#capacity-projections)
9. [Methodology](#methodology)

---

## GPU Resource Usage

_GPU metrics not available (run gpu_monitor sidecar)._

---

## Whisper Comparison Matrix

> **FW** = faster-whisper (CTranslate2)  |  **OW** = openai-whisper (PyTorch)  |  **VW** = vLLM Whisper  
> Columns: concurrency = [1, 10, 50, 100] simultaneous 47-second audio chunks

_No comparison data. Run `make bench-compare` or `python -m benchmarks.run_all --mode compare`._

---

## Whisper Detailed Results

### vLLM Whisper · `openai/whisper-large-v3`

| Concurrency | req/s | p50 ms | p90 ms | p95 ms | p99 ms | RTF    | Error % |
| ----------- | ----- | ------ | ------ | ------ | ------ | ------ | ------- |
| 20          | 7.36  | 593    | 11246  | 11490  | 11882  | 345.9× | 0.0%    |

---

## VRAM Reference

Per-instance VRAM footprint. H200 has 141 GB total.

| Model    | faster-whisper VRAM | openai-whisper VRAM | vLLM Whisper VRAM |
| -------- | ------------------- | ------------------- | ----------------- |
| medium   | ~3.06 GB            | ~3.06 GB            | ~3.06 GB+         |
| large-v3 | ~6.17 GB            | ~6.17 GB            | ~6.17 GB+         |

> faster-whisper (int8_float16): approximately half the FP16 VRAM above. vLLM adds scheduler/KV-cache overhead above model weights.  
> Actual parallelism depends on the measured GPU VRAM, worker count, vLLM queue/cache settings, backend, and compute type.

---

## Audio Format & Sample Rate Matrix

> Fixed concurrency = 10. Measures decoding overhead and resampling cost per format.
> **16 kHz** = Whisper-native (no resampling). **48 kHz** = browser/mic capture path (requires service resampling).

_No format results. Run `make bench-formats` or add `--mode format` to run_all._

---

## LLM Throughput

> Model: ~20B parameters, FP16 via vLLM · Input: ~800 tokens · Output: 512 max tokens  
> Prefix caching enabled (shared system prompt reused across all requests)

_No LLM results. Run `make bench-llm`._

---

## End-to-End Pipeline

> Mixed workload: 13 Whisper chunks (47 s each) → LLM SOAP note per session.  
> VRAM split depends on LLM_GPU_UTIL plus the Whisper worker count on this GPU.

_No E2E results. Run `make bench-e2e`._

---

## Capacity Projections

_Capacity projections require E2E results. Run `make bench-e2e` first._

---

## Methodology

### Audio generation
- Synthetic 47-second WAV (16 kHz mono, PCM-16, -3 dBFS peak).
- Formant synthesis (F1/F2 resonance + fricative bursts + lognormal pauses + -40 dBFS noise).
- No TTS dependency; generates in ~80 ms on CPU. Reproducible via integer seed.
- Exercises Whisper's encoder fully — unlike pure silence or pure noise.

### Benchmark procedure
- **Warmup**: 3 sequential requests before each concurrency level.
- **Test**: `asyncio.gather(N)` tasks repeated for ≥ 30 total samples.
- **Cooldown**: 10 s between levels (GPU thermal + scheduler drain).
- **GPU metrics**: nvidia-smi polled every 500 ms by sidecar container.

### Whisper backends
| Backend | Library | Precision | Batch | Notes |
|---------|---------|-----------|-------|-------|
| faster-whisper | CTranslate2 | FP16 / INT8 | Pool of N model instances | 1.4–2× faster than OW on GPU |
| openai-whisper | PyTorch | FP16 | Pool of N model instances | Reference implementation |

### Metrics glossary
- **RTF** (Real-Time Factor): `audio_duration ÷ transcription_latency`. RTF > 1 = faster than real-time.
- **TTFT**: Time to first token (LLM streaming latency).
- **tok/s**: Aggregate output tokens per second across all concurrent requests.
- **sessions/s**: Complete E2E sessions (all chunks + LLM note) per second.
