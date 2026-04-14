# Whisper GPU Optimizer Report

> Objective: minimize latency first, while also measuring max sustained req/s on this GPU.

Tested configurations: 57
Valid benchmarked configurations: 57

## Top configs by latency (lower p95 is better)

| Rank | Backend | Model | Quantization | Workers | Beam | Best p95 ms @ c | Max req/s @ c |
| ---- | ------- | ----- | ------------ | ------- | ---- | --------------- | ------------- |
| 1 | faster_whisper | large-v3 | int8_float16 | 6 | 5 | 325 @ 1 | 7.22 @ 100 |
| 2 | faster_whisper | large-v3 | int8_float16 | 10 | 5 | 367 @ 1 | 6.80 @ 100 |
| 3 | openai_whisper | large-v3 | fp16 | 1 | 1 | 434 @ 1 | 2.49 @ 50 |
| 4 | openai_whisper | large-v3 | fp16 | 6 | 1 | 439 @ 1 | 2.48 @ 1 |
| 5 | openai_whisper | large-v3 | fp16 | 4 | 1 | 442 @ 1 | 2.52 @ 1 |
| 6 | openai_whisper | large-v3 | fp32 | 6 | 1 | 447 @ 1 | 2.38 @ 1 |
| 7 | openai_whisper | large-v3 | fp32 | 1 | 1 | 455 @ 1 | 2.44 @ 10 |
| 8 | openai_whisper | large-v3 | fp32 | 4 | 1 | 457 @ 1 | 2.32 @ 1 |
| 9 | openai_whisper | large-v3 | fp32 | 1 | 2 | 545 @ 1 | 1.94 @ 10 |
| 10 | openai_whisper | large-v3 | fp32 | 4 | 2 | 556 @ 1 | 1.91 @ 1 |
| 11 | openai_whisper | large-v3 | fp32 | 6 | 2 | 556 @ 1 | 1.90 @ 1 |
| 12 | openai_whisper | large-v3 | fp16 | 4 | 2 | 566 @ 1 | 1.89 @ 1 |
| 13 | openai_whisper | large-v3 | fp16 | 6 | 2 | 569 @ 1 | 1.89 @ 1 |
| 14 | openai_whisper | large-v3 | fp16 | 1 | 2 | 577 @ 1 | 1.90 @ 10 |
| 15 | openai_whisper | large-v3 | fp32 | 4 | 5 | 633 @ 1 | 1.69 @ 1 |

## Top configs by throughput (higher req/s is better)

| Rank | Backend | Model | Quantization | Workers | Beam | Best p95 ms @ c | Max req/s @ c |
| ---- | ------- | ----- | ------------ | ------- | ---- | --------------- | ------------- |
| 1 | faster_whisper | large-v3 | int8_float16 | 6 | 5 | 325 @ 1 | 7.22 @ 100 |
| 2 | faster_whisper | large-v3 | int8_float16 | 10 | 5 | 367 @ 1 | 6.80 @ 100 |
| 3 | faster_whisper | large-v3 | float16 | 6 | 5 | 1368 @ 1 | 4.17 @ 50 |
| 4 | faster_whisper | large-v3 | int8_float16 | 11 | 5 | 694 @ 1 | 4.05 @ 50 |
| 5 | faster_whisper | large-v3 | int8 | 6 | 5 | 2795 @ 1 | 3.17 @ 50 |
| 6 | faster_whisper | large-v3 | int8_float16 | 10 | 2 | 2060 @ 1 | 3.12 @ 50 |
| 7 | faster_whisper | large-v3 | int8_float16 | 1 | 5 | 865 @ 1 | 2.65 @ 25 |
| 8 | faster_whisper | large-v3 | int8 | 4 | 5 | 2524 @ 1 | 2.57 @ 100 |
| 9 | openai_whisper | large-v3 | fp16 | 4 | 1 | 442 @ 1 | 2.52 @ 1 |
| 10 | openai_whisper | large-v3 | fp16 | 1 | 1 | 434 @ 1 | 2.49 @ 50 |
| 11 | openai_whisper | large-v3 | fp16 | 6 | 1 | 439 @ 1 | 2.48 @ 1 |
| 12 | openai_whisper | large-v3 | fp32 | 1 | 1 | 455 @ 1 | 2.44 @ 10 |
| 13 | openai_whisper | large-v3 | fp32 | 6 | 1 | 447 @ 1 | 2.38 @ 1 |
| 14 | openai_whisper | large-v3 | fp32 | 4 | 1 | 457 @ 1 | 2.32 @ 1 |
| 15 | faster_whisper | large-v3 | int8 | 6 | 2 | 3083 @ 1 | 2.29 @ 50 |
