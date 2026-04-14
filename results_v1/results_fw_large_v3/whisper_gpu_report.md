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
| faster_whisper | large-v3 | float16 | 1       |          | 1.90      | 100           | —          | —             | 50567       | 1           | 2282           | 0.0%       |
| faster_whisper | large-v3 | float16 | 2       |          | 2.80      | 50            | —          | —             | 16543       | 1           | 1907           | 0.0%       |
| faster_whisper | large-v3 | float16 | 4       |          | 1.88      | 50            | —          | —             | 26093       | 1           | 3080           | 0.0%       |
| faster_whisper | large-v3 | float16 | 6       | yes      | 2.93      | 50            | —          | —             | 13826       | 1           | 3571           | 0.0%       |

## Projection

Highest measured sustained throughput on this GPU: **2.93 req/s** at concurrency **50** using `faster_whisper` / `large-v3` / `float16` / `workers=6`.
At that throughput point p95 latency was 13826 ms and error rate was 0.0%.
Observed GPU metrics were not available for that run.

Fastest measured response time: p95 **3571 ms** at concurrency **1** using `faster_whisper` / `large-v3` / `float16` / `workers=6`.
At that latency point throughput was 1.26 req/s and error rate was 0.0%.

| Target requests/s | Measured req/s/GPU | GPUs needed | Utilization |
| ----------------- | ------------------ | ----------- | ----------- |
| 200               | 2.930              | 69          | 98.9%       |
| 400               | 2.930              | 137         | 99.6%       |
| 600               | 2.930              | 205         | 99.9%       |
| 1000              | 2.930              | 342         | 99.8%       |
| 1500              | 2.930              | 512         | 100.0%      |
| 2000              | 2.930              | 683         | 99.9%       |
| 25000             | 2.930              | 8533        | 100.0%      |

Assumption: horizontal scaling is roughly linear across identical GPUs and the same request mix.
Projection table is based on sustained requests/second, not in-flight concurrency.
