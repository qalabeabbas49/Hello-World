# vLLM Whisper Cross-GPU Capacity Report

> GPU: GPU metrics unavailable
> Workload: concurrent 47-second audio chunk transcription requests.
> Planning basis: highest measured requests/second with error rate within threshold.
> Fastest response time is reported separately from max throughput so the latency/throughput tradeoff is visible.
> `Selected=yes` marks the chosen worker count for that backend/model/compute variant when worker sweep was used.
> External queues (e.g. Redis/RabbitMQ) are not simulated; results size GPU workers and safe in-flight concurrency.
> Load generator: batch

## GPU Resource Usage

_GPU metrics not available. Start the matching gpu_monitor sidecar for this benchmark._

## Whisper Results

| Backend      | Model                   | Compute | Workers | Selected | Max req/s | Max req/s @ c | Peak GPU % | Peak VRAM MiB | Peak p95 ms | Fastest @ c | Fastest p95 ms | Peak error |
| ------------ | ----------------------- | ------- | ------- | -------- | --------- | ------------- | ---------- | ------------- | ----------- | ----------- | -------------- | ---------- |
| vllm_whisper | openai/whisper-large-v3 | default | —       | yes      | 0.97      | 1             | —          | —             | 3105        | 1           | 3105           | 0.0%       |

## Latency SLA operating point

Highest concurrency level where **p95 ≤ 12000 ms** and error rate ≤ 1% (per row).

| Backend      | Model                   | Compute | Workers | Max c @ SLA | p95 ms | req/s | Error |
| ------------ | ----------------------- | ------- | ------- | ----------- | ------ | ----- | ----- |
| vllm_whisper | openai/whisper-large-v3 | default | —       | 1           | 3105   | 0.97  | 0.0%  |

### Recommended operating point

`vllm_whisper` / `openai/whisper-large-v3` / `default` / workers `—` at concurrency **1** (p95 **3105 ms**, throughput **0.97 req/s**, error **0.0%**).

## Projection

Highest measured sustained throughput on this GPU: **0.97 req/s** at concurrency **1** using `vllm_whisper` / `openai/whisper-large-v3` / `default`.
At that throughput point p95 latency was 3105 ms and error rate was 0.0%.
Observed GPU metrics were not available for that run.

Fastest measured response time: p95 **3105 ms** at concurrency **1** using `vllm_whisper` / `openai/whisper-large-v3` / `default`.
At that latency point throughput was 0.97 req/s and error rate was 0.0%.

| Target requests/s | Measured req/s/GPU | GPUs needed | Utilization |
| ----------------- | ------------------ | ----------- | ----------- |
| 200               | 0.966              | 208         | 99.6%       |
| 400               | 0.966              | 415         | 99.8%       |
| 600               | 0.966              | 622         | 99.9%       |
| 1000              | 0.966              | 1036        | 100.0%      |
| 1500              | 0.966              | 1554        | 100.0%      |
| 2000              | 0.966              | 2072        | 100.0%      |
| 25000             | 0.966              | 25894       | 100.0%      |

Assumption: horizontal scaling is roughly linear across identical GPUs and the same request mix.
Projection table uses effective req/s/GPU when headroom is below 1; otherwise measured req/s.
Batch-mode benchmarks can over- or under-state sustained utilization versus steady load; see load generator notes above.
