# Whisper-Only GPU Capacity Report

> GPU: GPU metrics unavailable
> Workload: concurrent 47-second audio chunk transcription requests.
> Planning basis: highest measured requests/second with error rate within threshold.
> Fastest response time is reported separately from max throughput so the latency/throughput tradeoff is visible.
> `Selected=yes` marks the chosen worker count for that backend/model/compute variant when worker sweep was used.

## GPU Resource Usage

_GPU metrics not available. Start the matching gpu_monitor sidecar for this benchmark._

## Whisper Results

| Backend        | Model    | Compute | Workers | Selected | Max req/s | Max req/s @ c | Peak GPU % | Peak VRAM MiB | Peak p95 ms | Fastest @ c | Fastest p95 ms | Peak error |
| -------------- | -------- | ------- | ------- | -------- | --------- | ------------- | ---------- | ------------- | ----------- | ----------- | -------------- | ---------- |
| openai_whisper | large-v3 | fp16    | 1       |          | 1.68      | 1             | —          | —             | 638         | 1           | 638            | 0.0%       |
| openai_whisper | large-v3 | fp16    | 2       | yes      | 1.79      | 50            | —          | —             | 26763       | 1           | 640            | 0.0%       |
| openai_whisper | large-v3 | fp16    | 4       |          | 1.70      | 1             | —          | —             | 636         | 1           | 636            | 0.0%       |
| openai_whisper | large-v3 | fp16    | 6       |          | 1.64      | 1             | —          | —             | 647         | 1           | 647            | 0.0%       |

## Projection

Highest measured sustained throughput on this GPU: **1.79 req/s** at concurrency **50** using `openai_whisper` / `large-v3` / `fp16` / `workers=2`.
At that throughput point p95 latency was 26763 ms and error rate was 0.0%.
Observed GPU metrics were not available for that run.

Fastest measured response time: p95 **640 ms** at concurrency **1** using `openai_whisper` / `large-v3` / `fp16` / `workers=2`.
At that latency point throughput was 1.68 req/s and error rate was 0.0%.

| Target requests/s | Measured req/s/GPU | GPUs needed | Utilization |
| ----------------- | ------------------ | ----------- | ----------- |
| 200               | 1.792              | 112         | 99.6%       |
| 400               | 1.792              | 224         | 99.6%       |
| 600               | 1.792              | 335         | 99.9%       |
| 1000              | 1.792              | 559         | 99.8%       |
| 1500              | 1.792              | 838         | 99.9%       |
| 2000              | 1.792              | 1117        | 99.9%       |
| 25000             | 1.792              | 13951       | 100.0%      |

Assumption: horizontal scaling is roughly linear across identical GPUs and the same request mix.
Projection table is based on sustained requests/second, not in-flight concurrency.
