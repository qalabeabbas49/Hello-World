# Whisper-Only GPU Capacity Report

> GPU: GPU metrics unavailable
> Workload: concurrent 47-second audio chunk transcription requests.
> Planning basis: highest measured requests/second with error rate within threshold.
> Fastest response time is reported separately from max throughput so the latency/throughput tradeoff is visible.
> `Selected=yes` marks the chosen worker count for that backend/model/compute variant when worker sweep was used.

## GPU Resource Usage

_GPU metrics not available. Start the matching gpu_monitor sidecar for this benchmark._

## Whisper Results

| Backend        | Model | Compute | Workers | Selected | Max req/s | Max req/s @ c | Peak GPU % | Peak VRAM MiB | Peak p95 ms | Fastest @ c | Fastest p95 ms | Peak error |
| -------------- | ----- | ------- | ------- | -------- | --------- | ------------- | ---------- | ------------- | ----------- | ----------- | -------------- | ---------- |
| faster_whisper | tiny  | float16 | 10      | yes      | 3.78      | 50            | —          | —             | 13044       | 1           | 1097           | 0.0%       |
| faster_whisper | tiny  | float16 | 20      |          | 2.64      | 10            | —          | —             | 3943        | 1           | 1378           | 0.0%       |
| faster_whisper | tiny  | float16 | 30      |          | 3.10      | 50            | —          | —             | 15908       | 1           | 1075           | 0.0%       |
| faster_whisper | tiny  | float16 | 50      |          | 3.32      | 10            | —          | —             | 3046        | 1           | 1101           | 0.0%       |
| faster_whisper | tiny  | float16 | 128     |          | 3.43      | 100           | —          | —             | 28786       | 1           | 1369           | 0.0%       |

## Projection

Highest measured sustained throughput on this GPU: **3.78 req/s** at concurrency **50** using `faster_whisper` / `tiny` / `float16` / `workers=10`.
At that throughput point p95 latency was 13044 ms and error rate was 0.0%.
Observed GPU metrics were not available for that run.

Fastest measured response time: p95 **1097 ms** at concurrency **1** using `faster_whisper` / `tiny` / `float16` / `workers=10`.
At that latency point throughput was 2.37 req/s and error rate was 0.0%.

| Target requests/s | Measured req/s/GPU | GPUs needed | Utilization |
| ----------------- | ------------------ | ----------- | ----------- |
| 200               | 3.781              | 53          | 99.8%       |
| 400               | 3.781              | 106         | 99.8%       |
| 600               | 3.781              | 159         | 99.8%       |
| 1000              | 3.781              | 265         | 99.8%       |
| 1500              | 3.781              | 397         | 99.9%       |
| 2000              | 3.781              | 529         | 100.0%      |
| 25000             | 3.781              | 6612        | 100.0%      |

Assumption: horizontal scaling is roughly linear across identical GPUs and the same request mix.
Projection table is based on sustained requests/second, not in-flight concurrency.
