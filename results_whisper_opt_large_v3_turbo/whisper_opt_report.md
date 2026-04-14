# Whisper GPU Optimizer Report

> Objective: minimize latency first, while also measuring max sustained req/s on this GPU.

Tested configurations: 84
Valid benchmarked configurations: 84

## Top configs by latency (lower p95 is better)

| Rank | Backend | Model | Quantization | Workers | Beam | Best p95 ms @ c | Max req/s @ c |
| ---- | ------- | ----- | ------------ | ------- | ---- | --------------- | ------------- |
| 1 | openai_whisper | turbo | fp16 | 12 | 1 | 124 @ 1 | 8.34 @ 1 |
| 2 | openai_whisper | turbo | fp16 | 10 | 1 | 125 @ 1 | 8.28 @ 1 |
| 3 | openai_whisper | turbo | fp16 | 2 | 1 | 127 @ 1 | 8.14 @ 1 |
| 4 | openai_whisper | turbo | fp16 | 1 | 1 | 128 @ 1 | 8.26 @ 25 |
| 5 | openai_whisper | turbo | fp16 | 6 | 1 | 128 @ 1 | 8.13 @ 1 |
| 6 | openai_whisper | turbo | fp16 | 10 | 5 | 214 @ 1 | 5.29 @ 1 |
| 7 | openai_whisper | turbo | fp16 | 2 | 5 | 216 @ 1 | 5.44 @ 1 |
| 8 | openai_whisper | turbo | fp16 | 12 | 5 | 216 @ 1 | 5.47 @ 1 |
| 9 | openai_whisper | turbo | fp16 | 1 | 5 | 217 @ 1 | 5.53 @ 10 |
| 10 | openai_whisper | turbo | fp16 | 6 | 5 | 217 @ 1 | 5.43 @ 1 |
| 11 | openai_whisper | turbo | fp32 | 6 | 1 | 226 @ 1 | 4.49 @ 1 |
| 12 | openai_whisper | turbo | fp32 | 2 | 1 | 226 @ 1 | 4.52 @ 10 |
| 13 | openai_whisper | turbo | fp32 | 1 | 1 | 228 @ 1 | 4.51 @ 50 |
| 14 | openai_whisper | turbo | fp32 | 12 | 1 | 231 @ 1 | 4.44 @ 1 |
| 15 | openai_whisper | turbo | fp32 | 10 | 1 | 231 @ 1 | 4.50 @ 1 |

## Top configs by throughput (higher req/s is better)

| Rank | Backend | Model | Quantization | Workers | Beam | Best p95 ms @ c | Max req/s @ c |
| ---- | ------- | ----- | ------------ | ------- | ---- | --------------- | ------------- |
| 1 | openai_whisper | turbo | fp16 | 12 | 1 | 124 @ 1 | 8.34 @ 1 |
| 2 | openai_whisper | turbo | fp16 | 10 | 1 | 125 @ 1 | 8.28 @ 1 |
| 3 | openai_whisper | turbo | fp16 | 1 | 1 | 128 @ 1 | 8.26 @ 25 |
| 4 | faster_whisper | large-v3-turbo | int8_float16 | 6 | 5 | 699 @ 1 | 8.21 @ 25 |
| 5 | openai_whisper | turbo | fp16 | 2 | 1 | 127 @ 1 | 8.14 @ 1 |
| 6 | openai_whisper | turbo | fp16 | 6 | 1 | 128 @ 1 | 8.13 @ 1 |
| 7 | faster_whisper | large-v3-turbo | int8 | 12 | 5 | 623 @ 1 | 8.04 @ 100 |
| 8 | faster_whisper | large-v3-turbo | float16 | 10 | 5 | 875 @ 1 | 7.42 @ 100 |
| 9 | faster_whisper | large-v3-turbo | int8_float16 | 10 | 5 | 761 @ 1 | 7.35 @ 100 |
| 10 | faster_whisper | large-v3-turbo | int8_float16 | 12 | 5 | 780 @ 1 | 7.06 @ 100 |
| 11 | faster_whisper | large-v3-turbo | int8_float16 | 6 | 1 | 627 @ 1 | 7.01 @ 100 |
| 12 | faster_whisper | large-v3-turbo | int8 | 10 | 5 | 681 @ 1 | 6.60 @ 50 |
| 13 | faster_whisper | large-v3-turbo | int8 | 2 | 5 | 289 @ 1 | 6.49 @ 25 |
| 14 | faster_whisper | large-v3-turbo | int8 | 23 | 5 | 703 @ 1 | 6.47 @ 50 |
| 15 | faster_whisper | large-v3-turbo | int8 | 6 | 5 | 1195 @ 1 | 6.28 @ 50 |
