# Whisper Interim Report — faster_whisper / large-v3-turbo / float16

> GPU: GPU metrics unavailable
> Workload: concurrent 47-second audio chunk transcription requests.
> Planning basis: highest measured requests/second with error rate within threshold.
> Fastest response time is reported separately from max throughput so the latency/throughput tradeoff is visible.
> `Selected=yes` marks the chosen worker count for that backend/model/compute variant when worker sweep was used.

## GPU Resource Usage

_GPU metrics not available. Start the matching gpu_monitor sidecar for this benchmark._

## Whisper Results

| Backend        | Model          | Compute | Workers | Selected | Max req/s | Max req/s @ c | Peak GPU % | Peak VRAM MiB | Peak p95 ms | Fastest @ c | Fastest p95 ms | Peak error |
| -------------- | -------------- | ------- | ------- | -------- | --------- | ------------- | ---------- | ------------- | ----------- | ----------- | -------------- | ---------- |
| faster_whisper | large-v3-turbo | float16 | 1       |          | 3.23      | 50            | —          | —             | 14710       | 1           | 1256           | 0.0%       |
| faster_whisper | large-v3-turbo | float16 | 2       |          | 4.19      | 10            | —          | —             | 2380        | 1           | 804            | 0.0%       |
| faster_whisper | large-v3-turbo | float16 | 4       |          | 4.44      | 100           | —          | —             | 21357       | 1           | 1135           | 0.0%       |
| faster_whisper | large-v3-turbo | float16 | 6       |          | 4.78      | 50            | —          | —             | 9191        | 1           | 1244           | 0.0%       |
| faster_whisper | large-v3-turbo | float16 | 8       |          | 5.14      | 50            | —          | —             | 8965        | 1           | 1147           | 0.0%       |
| faster_whisper | large-v3-turbo | float16 | 10      | yes      | 6.16      | 100           | —          | —             | 15224       | 1           | 1070           | 0.0%       |
| faster_whisper | large-v3-turbo | float16 | 12      |          | 5.10      | 50            | —          | —             | 9261        | 1           | 1350           | 0.0%       |

## Projection

Highest measured sustained throughput on this GPU: **6.16 req/s** at concurrency **100** using `faster_whisper` / `large-v3-turbo` / `float16` / `workers=10`.
At that throughput point p95 latency was 15224 ms and error rate was 0.0%.
Observed GPU metrics were not available for that run.

Fastest measured response time: p95 **1070 ms** at concurrency **1** using `faster_whisper` / `large-v3-turbo` / `float16` / `workers=10`.
At that latency point throughput was 2.67 req/s and error rate was 0.0%.

| Target requests/s | Measured req/s/GPU | GPUs needed | Utilization |
| ----------------- | ------------------ | ----------- | ----------- |
| 200               | 6.160              | 33          | 98.4%       |
| 400               | 6.160              | 65          | 99.9%       |
| 600               | 6.160              | 98          | 99.4%       |
| 1000              | 6.160              | 163         | 99.6%       |
| 1500              | 6.160              | 244         | 99.8%       |
| 2000              | 6.160              | 325         | 99.9%       |
| 25000             | 6.160              | 4059        | 100.0%      |

Assumption: horizontal scaling is roughly linear across identical GPUs and the same request mix.
Projection table is based on sustained requests/second, not in-flight concurrency.
