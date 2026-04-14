# Whisper-Only GPU Capacity Report

> GPU: NVIDIA L40S (45.0 GiB VRAM)
> Workload: concurrent 47-second audio chunk transcription requests.
> Planning basis: highest measured requests/second with error rate within threshold.
> Fastest response time is reported separately from max throughput so the latency/throughput tradeoff is visible.
> `Selected=yes` marks the chosen worker count for that backend/model/compute variant when worker sweep was used.

## GPU Resource Usage

| Metric              | Peak / Value                | Mean  |
| ------------------- | --------------------------- | ----- |
| GPU(s)              | NVIDIA L40S (45.0 GiB VRAM) |       |
| GPU index           | 0                           |       |
| VRAM total (MiB)    | 46068                       |       |
| VRAM used (MiB)     | 16781                       | 12106 |
| GPU utilization (%) | 59.0                        | 13.1  |
| Power draw (W)      | 120                         | 86    |
| Temperature (C)     | 37                          | 32    |

## Whisper Results

| Backend        | Model | Compute | Workers | Selected | Max req/s | Max req/s @ c | Peak GPU % | Peak VRAM MiB | Peak p95 ms | Fastest @ c | Fastest p95 ms | Peak error |
| -------------- | ----- | ------- | ------- | -------- | --------- | ------------- | ---------- | ------------- | ----------- | ----------- | -------------- | ---------- |
| faster_whisper | tiny  | float16 | 10      |          | 3.79      | 100           | —          | —             | 24943       | 1           | 1114           | 0.0%       |
| faster_whisper | tiny  | float16 | 20      |          | 2.86      | 100           | —          | —             | 34205       | 1           | 1325           | 0.0%       |
| faster_whisper | tiny  | float16 | 30      |          | 3.31      | 100           | —          | —             | 29764       | 1           | 1389           | 0.0%       |
| faster_whisper | tiny  | float16 | 50      |          | 3.22      | 50            | —          | —             | 15102       | 1           | 1529           | 0.0%       |
| faster_whisper | tiny  | float16 | 128     | yes      | 4.03      | 10            | 59.0       | 16781         | 2419        | 1           | 1017           | 0.0%       |

## Projection

Highest measured sustained throughput on this GPU: **4.03 req/s** at concurrency **10** using `faster_whisper` / `tiny` / `float16` / `workers=128`.
At that throughput point p95 latency was 2419 ms and error rate was 0.0%.
Observed GPU peak during that run: 59.0% util, 16781 MiB VRAM.

Fastest measured response time: p95 **1017 ms** at concurrency **1** using `faster_whisper` / `tiny` / `float16` / `workers=128`.
At that latency point throughput was 2.60 req/s and error rate was 0.0%.

| Target requests/s | Measured req/s/GPU | GPUs needed | Utilization |
| ----------------- | ------------------ | ----------- | ----------- |
| 200               | 4.030              | 50          | 99.2%       |
| 400               | 4.030              | 100         | 99.2%       |
| 600               | 4.030              | 149         | 99.9%       |
| 1000              | 4.030              | 249         | 99.6%       |
| 1500              | 4.030              | 373         | 99.8%       |
| 2000              | 4.030              | 497         | 99.8%       |
| 25000             | 4.030              | 6203        | 100.0%      |

Assumption: horizontal scaling is roughly linear across identical GPUs and the same request mix.
Projection table is based on sustained requests/second, not in-flight concurrency.
