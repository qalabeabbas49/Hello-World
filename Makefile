# GPU Stress Testing — Medical Ambient Scribing
# ──────────────────────────────────────────────────────────────────────────────
# Prerequisites:
#   - Docker + nvidia-container-toolkit
#   - Python 3.11+  (`pip install -r requirements.txt`)
#   - Model weights placed at LLM_MODEL_HOST_PATH (see .env.example)
#
# Quick start:
#   cp .env.example .env                 # edit LLM_MODEL_HOST_PATH
#   make fixtures                        # generate synthetic 47 s audio
#   make bench-compare                   # ASR backends × all Whisper models
#   make bench-llm                       # LLM throughput sweep
#   make bench-e2e                       # end-to-end session sweep
#   make report                          # re-render report.md from saved results

.PHONY: help fixtures check-gpu \
        up-whisper up-llm up-vllm-whisper up-mixed down \
        bench-whisper bench-llm bench-e2e bench-compare bench-formats bench-real-audio bench-all \
        bench-whisper-gpu bench-whisper-opt bench-vllm-gpu \
        bench-vllm-sla bench-vllm-sweep bench-vllm-queue bench-vllm-scheduler-sweep report-vllm \
        report report-ai clean

# ── Configurable defaults ──────────────────────────────────────────────────────
MODEL          ?= large-v3
BACKEND        ?= faster_whisper
WORKERS        ?= 16
RESULTS        ?= ./results
SESSIONS       ?=
CHUNKS         ?=
FORMATS        ?= wav,flac,opus
RATES          ?= 16000,48000
REAL_AUDIO_DIR ?= ./fixtures/real_audio
CHUNK_DURATION ?= 47
COMPUTE_TYPES  ?= float16,int8_float16
TARGET_TIERS   ?= 200,400,600,1000,1500,2000,25000
# Empty = use WHISPER_CONCURRENCY_SWEEP_MODE (default dense) for whisper-gpu; set e.g. 1,10,50,100 for a short ladder.
WHISPER_GPU_CONCURRENCY ?=
WHISPER_CONCURRENCY_SWEEP_MODE ?= dense
WHISPER_CONCURRENCY_DENSE_MAX ?= 160
WHISPER_CONCURRENCY_DENSE_STEP ?= 8
WHISPER_BENCH_LOAD_MODE ?= steady
WHISPER_STEADY_STATE_DURATION_S ?= 90
WHISPER_PROJECTION_HEADROOM ?= 1.0
WHISPER_MAX_P95_MS ?=
VLLM_GPU_CONCURRENCY    ?= 1,10,50,100
WHISPER_GPU_BACKEND ?=
WHISPER_GPU_MODEL ?=
WHISPER_INPUT_MODE ?= file
WHISPER_GPU_INPUT_MODE ?= pcm_f32le
WHISPER_BEAM_SIZE ?= 5
VLLM_BEAM_SIZE ?= 1
VLLM_INPUT_MODE ?= pcm_f32le
VLLM_LOAD_MODE ?= steady
VLLM_STEADY_DURATION_S ?= 90
VLLM_SLA_P95_MS ?= 12000
VLLM_CONCURRENCY ?= 8,16,24,32,40,48,64,80,100
VLLM_SLA_CONCURRENCY ?= 4,8,12,16,20,24,28,32
VLLM_QUEUE_REDIS_URL ?= redis://localhost:6379/0
VLLM_QUEUE_STREAM ?= whisper_jobs
VLLM_QUEUE_GROUP ?= whisper_workers
VLLM_QUEUE_CONSUMERS ?= 8
VLLM_QUEUE_INGRESS_RPS ?= 8
VLLM_QUEUE_DURATION_S ?= 90
VLLM_WHISPER_MAX_BATCHED_TOKENS ?=
VLLM_WHISPER_SCHEDULER_DELAY_FACTOR ?=
VLLM_WHISPER_ENABLE_CHUNKED_PREFILL ?= 0
VLLM_SWEEP_MAX_NUM_SEQS ?= 128,256,384,512
VLLM_SWEEP_SCHEDULER_DELAY_FACTORS ?= 0.0,0.1,0.2
VLLM_SWEEP_CHUNKED_PREFILL ?= 0,1
WHISPER_GPU_UTIL ?= 0.90
WHISPER_MAX_WORKERS ?= 128
# full = sweep 1,2,4,... up to VRAM cap; max_only = single run at cap (faster; default for bench-whisper-gpu).
WHISPER_WORKER_SWEEP_MODE ?= max_only
WHISPER_WORKER_CANDIDATES ?=
WHISPER_OPT_BACKENDS ?= faster_whisper,openai_whisper
WHISPER_OPT_MODELS ?= medium,large-v3
WHISPER_OPT_FASTER_COMPUTE_TYPES ?= float16,int8_float16,int8
WHISPER_OPT_OPENAI_PRECISIONS ?= fp16,fp32
WHISPER_OPT_BEAM_SIZES ?= 1,2,5
WHISPER_OPT_CONCURRENCY ?= 1,4,8,16,32,64,100
WHISPER_OPT_INPUT_MODE ?= pcm_f32le
VLLM_GPU_UTIL ?= 0.95
VLLM_MAX_NUM_SEQS ?= 2048
VLLM_MAX_MODEL_LEN ?= 8192
VLLM_WHISPER_IMAGE ?= vllm/vllm-openai:latest
VLLM_WHISPER_MODEL ?= openai/whisper-large-v3
VLLM_WHISPER_CACHE_HOST_PATH ?= ./models
VLLM_WHISPER_GPU_UTIL ?= 0.95
VLLM_WHISPER_MAX_NUM_SEQS ?= 2048
VLLM_WHISPER_MAX_MODEL_LEN ?=
OPENAI_REPORT_MODEL ?= gpt-5-mini
OPENAI_REPORT_REASONING ?= minimal
HOST_UID ?= $(shell id -u)
HOST_GID ?= $(shell id -g)
ASR_MODEL = $(if $(filter vllm_whisper,$(BACKEND)),$(VLLM_WHISPER_MODEL),$(MODEL))

help:
	@echo ""
	@echo "  vLLM Whisper (recommended, simplified)"
	@echo "    make bench-vllm-sweep      Canonical steady-state concurrency sweep"
	@echo "    make bench-vllm-sla        SLA-focused run (p95 threshold)"
	@echo "    make bench-vllm-queue      Redis queue benchmark (producer/consumer)"
	@echo "    make bench-vllm-scheduler-sweep  Sweep scheduler/prefill knobs"
	@echo "    make report-vllm           Re-render vllm_whisper_report.md"
	@echo ""
	@echo "  Fixtures & checks"
	@echo "    make fixtures              Generate synthetic audio: WAV/FLAC/Opus at 16k+48k"
	@echo "    make check-gpu             Verify GPU driver, Docker GPU access"
	@echo ""
	@echo "  Service management"
	@echo "    make up-whisper            Start Whisper-only  (BACKEND=, MODEL=, WORKERS=)"
	@echo "    make up-llm                Start LLM-only"
	@echo "    make up-vllm-whisper       Start vLLM Whisper-only"
	@echo "    make up-mixed              Start both services (mixed workload)"
	@echo "    make down                  Stop all compose stacks"
	@echo ""
	@echo "  Benchmarks (manage their own docker lifecycle by default)"
	@echo "    make bench-whisper         One backend × one model  (BACKEND=, MODEL=)"
	@echo "    make bench-llm             LLM concurrency sweep"
	@echo "    make bench-e2e             E2E session sweep"
	@echo "    make bench-compare         ASR backends × ALL model sizes  [main concurrency sweep]"
	@echo "    make bench-formats         WAV/FLAC/Opus × 16k/48k format sweep (synthetic)"
	@echo "    make bench-real-audio      Same sweep with your own audio files (REAL_AUDIO_DIR=)"
	@echo "    make bench-all             bench-compare + bench-formats + bench-llm + bench-e2e"
	@echo "    make bench-whisper-gpu     Whisper-only GPU projection with auto worker sweep"
	@echo "    make bench-whisper-opt     Latency-first Whisper optimizer sweep on current GPU"
	@echo "    make bench-vllm-gpu        vLLM-only GPU capacity projection"
	@echo ""
	@echo "  Results"
	@echo "    make report                Re-render report.md from saved JSON results"
	@echo "    make report-ai             Generate report_ai.md with OpenAI (needs OPENAI_API_KEY)"
	@echo "    interim reports            Per-model Whisper reports are written under $(RESULTS)/interim/"
	@echo "    make clean                 Delete result files (keeps fixtures)"
	@echo ""
	@echo "  Key variables:"
	@echo "    VLLM_WHISPER_MODEL=$(VLLM_WHISPER_MODEL)"
	@echo "    VLLM_CONCURRENCY=$(VLLM_CONCURRENCY)  VLLM_SLA_CONCURRENCY=$(VLLM_SLA_CONCURRENCY)"
	@echo "    VLLM_BEAM_SIZE=$(VLLM_BEAM_SIZE)  VLLM_INPUT_MODE=$(VLLM_INPUT_MODE)"
	@echo "    VLLM_LOAD_MODE=$(VLLM_LOAD_MODE)  VLLM_STEADY_DURATION_S=$(VLLM_STEADY_DURATION_S)"
	@echo "    VLLM_SLA_P95_MS=$(VLLM_SLA_P95_MS)"
	@echo "    VLLM_QUEUE_REDIS_URL=$(VLLM_QUEUE_REDIS_URL)  VLLM_QUEUE_CONSUMERS=$(VLLM_QUEUE_CONSUMERS)"
	@echo "    VLLM_QUEUE_INGRESS_RPS=$(VLLM_QUEUE_INGRESS_RPS)  VLLM_QUEUE_DURATION_S=$(VLLM_QUEUE_DURATION_S)"
	@echo "    VLLM_WHISPER_MAX_BATCHED_TOKENS=$(VLLM_WHISPER_MAX_BATCHED_TOKENS)  VLLM_WHISPER_SCHEDULER_DELAY_FACTOR=$(VLLM_WHISPER_SCHEDULER_DELAY_FACTOR)"
	@echo "    VLLM_WHISPER_ENABLE_CHUNKED_PREFILL=$(VLLM_WHISPER_ENABLE_CHUNKED_PREFILL)"
	@echo "    BACKEND=$(BACKEND)  MODEL=$(MODEL)  WORKERS=$(WORKERS)"
	@echo "    FORMATS=$(FORMATS)  RATES=$(RATES)"
	@echo "    REAL_AUDIO_DIR=$(REAL_AUDIO_DIR)  CHUNK_DURATION=$(CHUNK_DURATION)s  RESULTS=$(RESULTS)"
	@echo "    TARGET_TIERS=$(TARGET_TIERS)  # interpreted as target requests/sec in GPU reports"
	@echo "    WHISPER_GPU_BACKEND=$(WHISPER_GPU_BACKEND)  WHISPER_GPU_MODEL=$(WHISPER_GPU_MODEL)"
	@echo "    WHISPER_INPUT_MODE=$(WHISPER_INPUT_MODE)  WHISPER_GPU_INPUT_MODE=$(WHISPER_GPU_INPUT_MODE)"
	@echo "    bench-whisper-gpu: WHISPER_BENCH_LOAD_MODE=$(WHISPER_BENCH_LOAD_MODE)  WHISPER_CONCURRENCY_SWEEP_MODE=$(WHISPER_CONCURRENCY_SWEEP_MODE)"
	@echo "    WHISPER_GPU_CONCURRENCY=$(WHISPER_GPU_CONCURRENCY)  WHISPER_MAX_P95_MS=$(WHISPER_MAX_P95_MS)  WHISPER_PROJECTION_HEADROOM=$(WHISPER_PROJECTION_HEADROOM)"
	@echo "    WHISPER_WORKER_SWEEP_MODE=$(WHISPER_WORKER_SWEEP_MODE)  # max_only skips low worker counts"

# ── Fixtures & preflight ───────────────────────────────────────────────────────
# Generates subdirs:  fixtures/audio/wav_16000/  wav_48000/  flac_16000/  …
fixtures:
	python -m generators.audio_gen \
	  --count 20 --duration 47 \
	  --formats $(FORMATS) --sample-rates $(RATES) \
	  --output ./fixtures/audio

check-gpu:
	@bash scripts/setup_gpu_isolation.sh

# ── Service management ────────────────────────────────────────────────────────
up-whisper:
	HOST_UID=$(HOST_UID) HOST_GID=$(HOST_GID) \
	  WHISPER_BACKEND=$(BACKEND) WHISPER_MODEL=$(MODEL) WHISPER_WORKERS=$(WORKERS) \
	  docker compose -f docker-compose.whisper-only.yml up -d --build
	@bash scripts/verify_services.sh whisper

up-llm:
	HOST_UID=$(HOST_UID) HOST_GID=$(HOST_GID) \
	  LLM_GPU_UTIL=$(VLLM_GPU_UTIL) LLM_MAX_NUM_SEQS=$(VLLM_MAX_NUM_SEQS) \
	  LLM_MAX_MODEL_LEN=$(VLLM_MAX_MODEL_LEN) \
	  docker compose -f docker-compose.llm-only.yml up -d --build
	@MAX_RETRIES=60 bash scripts/verify_services.sh llm

up-vllm-whisper:
	@mkdir -p $(RESULTS)
	HOST_UID=$(HOST_UID) HOST_GID=$(HOST_GID) \
	  VLLM_WHISPER_IMAGE=$(VLLM_WHISPER_IMAGE) \
	  VLLM_WHISPER_MODEL=$(VLLM_WHISPER_MODEL) \
	  VLLM_WHISPER_CACHE_HOST_PATH=$(VLLM_WHISPER_CACHE_HOST_PATH) \
	  VLLM_WHISPER_GPU_UTIL=$(VLLM_WHISPER_GPU_UTIL) \
	  VLLM_WHISPER_MAX_NUM_SEQS=$(VLLM_WHISPER_MAX_NUM_SEQS) \
	  VLLM_WHISPER_MAX_NUM_BATCHED_TOKENS=$(VLLM_WHISPER_MAX_BATCHED_TOKENS) \
	  VLLM_WHISPER_SCHEDULER_DELAY_FACTOR=$(VLLM_WHISPER_SCHEDULER_DELAY_FACTOR) \
	  VLLM_WHISPER_ENABLE_CHUNKED_PREFILL=$(VLLM_WHISPER_ENABLE_CHUNKED_PREFILL) \
	  VLLM_WHISPER_MAX_MODEL_LEN=$(VLLM_WHISPER_MAX_MODEL_LEN) \
	  docker compose -f docker-compose.vllm-whisper-only.yml up -d --build
	@MAX_RETRIES=60 bash scripts/verify_services.sh vllm-whisper

up-mixed:
	HOST_UID=$(HOST_UID) HOST_GID=$(HOST_GID) \
	  WHISPER_BACKEND=$(BACKEND) WHISPER_MODEL=$(MODEL) WHISPER_WORKERS=$(WORKERS) \
	  docker compose up -d --build
	@bash scripts/verify_services.sh all

down:
	-docker compose -f docker-compose.yml              down 2>/dev/null || true
	-docker compose -f docker-compose.whisper-only.yml down 2>/dev/null || true
	-docker compose -f docker-compose.llm-only.yml     down 2>/dev/null || true
	-docker compose -f docker-compose.vllm-whisper-only.yml down 2>/dev/null || true

# ── Benchmark targets ─────────────────────────────────────────────────────────

bench-vllm-sweep:
	@mkdir -p $(RESULTS)
	$(MAKE) up-vllm-whisper VLLM_WHISPER_MODEL=$(VLLM_WHISPER_MODEL)
	RESULTS_DIR=$(RESULTS) python -m benchmarks.run_all \
	  --mode sweep \
	  --model $(VLLM_WHISPER_MODEL) \
	  --whisper-concurrency $(VLLM_CONCURRENCY) \
	  --whisper-input-mode $(VLLM_INPUT_MODE) \
	  --whisper-beam-size $(VLLM_BEAM_SIZE) \
	  --whisper-load-mode $(VLLM_LOAD_MODE) \
	  --whisper-steady-duration-s $(VLLM_STEADY_DURATION_S) \
	  --whisper-max-p95-ms $(VLLM_SLA_P95_MS) \
	  --whisper-projection-headroom $(WHISPER_PROJECTION_HEADROOM) \
	  --target-tiers $(TARGET_TIERS) \
	  --output $(RESULTS)

bench-vllm-sla:
	@mkdir -p $(RESULTS)
	$(MAKE) up-vllm-whisper VLLM_WHISPER_MODEL=$(VLLM_WHISPER_MODEL)
	RESULTS_DIR=$(RESULTS) python -m benchmarks.run_all \
	  --mode sla \
	  --model $(VLLM_WHISPER_MODEL) \
	  --whisper-concurrency $(VLLM_SLA_CONCURRENCY) \
	  --whisper-input-mode $(VLLM_INPUT_MODE) \
	  --whisper-beam-size $(VLLM_BEAM_SIZE) \
	  --whisper-load-mode $(VLLM_LOAD_MODE) \
	  --whisper-steady-duration-s $(VLLM_STEADY_DURATION_S) \
	  --whisper-max-p95-ms $(VLLM_SLA_P95_MS) \
	  --whisper-projection-headroom $(WHISPER_PROJECTION_HEADROOM) \
	  --target-tiers $(TARGET_TIERS) \
	  --output $(RESULTS)

bench-vllm-queue:
	@mkdir -p $(RESULTS)
	$(MAKE) up-vllm-whisper VLLM_WHISPER_MODEL=$(VLLM_WHISPER_MODEL)
	RESULTS_DIR=$(RESULTS) python -m benchmarks.queue_bench \
	  --redis-url $(VLLM_QUEUE_REDIS_URL) \
	  --stream $(VLLM_QUEUE_STREAM) \
	  --group $(VLLM_QUEUE_GROUP) \
	  --consumers $(VLLM_QUEUE_CONSUMERS) \
	  --ingress-rps $(VLLM_QUEUE_INGRESS_RPS) \
	  --duration-s $(VLLM_QUEUE_DURATION_S) \
	  --model $(VLLM_WHISPER_MODEL) \
	  --whisper-base-url http://localhost:8003 \
	  --beam-size $(VLLM_BEAM_SIZE) \
	  --input-mode $(VLLM_INPUT_MODE) \
	  --reset-stream \
	  --output $(RESULTS)/queue_bench

bench-vllm-scheduler-sweep:
	@mkdir -p $(RESULTS)
	python -m benchmarks.vllm_scheduler_sweep \
	  --results-root $(RESULTS) \
	  --model $(VLLM_WHISPER_MODEL) \
	  --concurrency $(VLLM_SLA_CONCURRENCY) \
	  --beam-size $(VLLM_BEAM_SIZE) \
	  --input-mode $(VLLM_INPUT_MODE) \
	  --load-mode $(VLLM_LOAD_MODE) \
	  --steady-duration-s $(VLLM_STEADY_DURATION_S) \
	  --max-p95-ms $(VLLM_SLA_P95_MS) \
	  --max-num-seqs $(VLLM_SWEEP_MAX_NUM_SEQS) \
	  --scheduler-delay-factors $(VLLM_SWEEP_SCHEDULER_DELAY_FACTORS) \
	  --chunked-prefill $(VLLM_SWEEP_CHUNKED_PREFILL) \
	  --max-num-batched-tokens "$(VLLM_WHISPER_MAX_BATCHED_TOKENS)"

# Single backend+model Whisper sweep (brings up service, benchmarks, tears down)
bench-whisper:
ifeq ($(BACKEND),vllm_whisper)
	$(MAKE) up-vllm-whisper VLLM_WHISPER_MODEL=$(ASR_MODEL)
else
	$(MAKE) up-whisper BACKEND=$(BACKEND) MODEL=$(ASR_MODEL) WORKERS=$(WORKERS)
endif
	RESULTS_DIR=$(RESULTS) python -m benchmarks.run_all \
	  --mode whisper --model $(ASR_MODEL) --backend $(BACKEND) \
	  --whisper-input-mode $(WHISPER_INPUT_MODE) \
	  --whisper-beam-size $(WHISPER_BEAM_SIZE) \
	  --output $(RESULTS)
	$(MAKE) down

# LLM-only sweep
bench-llm:
	$(MAKE) up-llm
	RESULTS_DIR=$(RESULTS) python -m benchmarks.run_all --mode llm --output $(RESULTS)
	$(MAKE) down

# E2E mixed workload sweep (both services)
bench-e2e:
	$(MAKE) up-mixed BACKEND=$(BACKEND) MODEL=$(MODEL) WORKERS=4
	RESULTS_DIR=$(RESULTS) python -m benchmarks.run_all \
	  --mode e2e \
	  $(if $(SESSIONS),--sessions $(SESSIONS),) \
	  $(if $(CHUNKS),--chunks $(CHUNKS),) \
	  --output $(RESULTS)
	$(MAKE) down

# ── Main comparison sweep: ASR backends × all model sizes ──────────────────────
# run_all --mode compare manages docker compose internally per combination.
bench-compare:
	@mkdir -p $(RESULTS)
	RESULTS_DIR=$(RESULTS) python -m benchmarks.run_all \
	  --mode compare --workers $(WORKERS) --output $(RESULTS)
	@echo ""
	@echo "Comparison complete. Report: $(RESULTS)/report.md"

# Format + sample rate sweep (synthetic audio, Whisper-only service)
bench-formats:
	$(MAKE) up-whisper BACKEND=$(BACKEND) MODEL=$(MODEL) WORKERS=$(WORKERS)
	RESULTS_DIR=$(RESULTS) python -m benchmarks.run_all \
	  --mode format \
	  --audio-formats $(FORMATS) --audio-rates $(RATES) \
	  --output $(RESULTS)
	$(MAKE) down

# Same sweep with real audio files — auto-splits into CHUNK_DURATION-second WAV chunks
bench-real-audio:
	$(MAKE) up-whisper BACKEND=$(BACKEND) MODEL=$(MODEL) WORKERS=$(WORKERS)
	RESULTS_DIR=$(RESULTS) python -m benchmarks.run_all \
	  --mode format \
	  --audio-formats $(FORMATS) --audio-rates $(RATES) \
	  --real-audio-dir $(REAL_AUDIO_DIR) \
	  --chunk-duration $(CHUNK_DURATION) \
	  --output $(RESULTS)
	$(MAKE) down

# Full suite: compare + formats + LLM + E2E
bench-all:
	@mkdir -p $(RESULTS)
	RESULTS_DIR=$(RESULTS) python -m benchmarks.run_all \
	  --mode all --workers $(WORKERS) \
	  --audio-formats $(FORMATS) --audio-rates $(RATES) \
	  $(if $(SESSIONS),--sessions $(SESSIONS),) \
	  $(if $(CHUNKS),--chunks $(CHUNKS),) \
	  $(if $(wildcard $(REAL_AUDIO_DIR)),--real-audio-dir $(REAL_AUDIO_DIR),) \
	  --chunk-duration $(CHUNK_DURATION) \
	  --output $(RESULTS)
	@echo ""
	@echo "Full suite complete. Report: $(RESULTS)/report.md"

# Whisper-only extensive capacity projection for the GPU this runs on.
bench-whisper-gpu:
	@mkdir -p $(RESULTS)
	LABEL_FILE="$(abspath $(RESULTS))/gpu_label.txt" RESULTS_DIR=$(RESULTS) \
	  WHISPER_GPU_UTIL=$(WHISPER_GPU_UTIL) WHISPER_MAX_WORKERS=$(WHISPER_MAX_WORKERS) \
	  WHISPER_CONCURRENCY_SWEEP_MODE=$(WHISPER_CONCURRENCY_SWEEP_MODE) \
	  WHISPER_CONCURRENCY_DENSE_MAX=$(WHISPER_CONCURRENCY_DENSE_MAX) \
	  WHISPER_CONCURRENCY_DENSE_STEP=$(WHISPER_CONCURRENCY_DENSE_STEP) \
	  WHISPER_LOAD_MODE=$(WHISPER_BENCH_LOAD_MODE) \
	  WHISPER_STEADY_STATE_DURATION_S=$(WHISPER_STEADY_STATE_DURATION_S) \
	  WHISPER_PROJECTION_HEADROOM=$(WHISPER_PROJECTION_HEADROOM) \
	  WHISPER_WORKER_SWEEP_MODE=$(WHISPER_WORKER_SWEEP_MODE) \
	  VLLM_WHISPER_IMAGE=$(VLLM_WHISPER_IMAGE) \
	  VLLM_WHISPER_CACHE_HOST_PATH=$(VLLM_WHISPER_CACHE_HOST_PATH) \
	  VLLM_WHISPER_GPU_UTIL=$(VLLM_WHISPER_GPU_UTIL) \
	  VLLM_WHISPER_MAX_NUM_SEQS=$(VLLM_WHISPER_MAX_NUM_SEQS) \
	  VLLM_WHISPER_MAX_MODEL_LEN=$(VLLM_WHISPER_MAX_MODEL_LEN) \
	  python -m benchmarks.run_all \
	    --mode whisper-gpu \
	    $(if $(WHISPER_GPU_BACKEND),--backend $(WHISPER_GPU_BACKEND),) \
	    $(if $(WHISPER_GPU_MODEL),--model $(WHISPER_GPU_MODEL),) \
	    --workers $(WORKERS) \
	    --auto-whisper-workers \
	    --compute-types $(COMPUTE_TYPES) \
	    --whisper-input-mode $(WHISPER_GPU_INPUT_MODE) \
	    --whisper-beam-size $(WHISPER_BEAM_SIZE) \
	    $(if $(strip $(WHISPER_GPU_CONCURRENCY)),--whisper-concurrency $(WHISPER_GPU_CONCURRENCY),) \
	    $(if $(strip $(WHISPER_MAX_P95_MS)),--whisper-max-p95-ms $(WHISPER_MAX_P95_MS),) \
	    --whisper-load-mode $(WHISPER_BENCH_LOAD_MODE) \
	    --whisper-steady-duration-s $(WHISPER_STEADY_STATE_DURATION_S) \
	    --whisper-projection-headroom $(WHISPER_PROJECTION_HEADROOM) \
	    --whisper-worker-sweep-mode $(WHISPER_WORKER_SWEEP_MODE) \
	    $(if $(WHISPER_WORKER_CANDIDATES),--whisper-worker-candidates $(WHISPER_WORKER_CANDIDATES),) \
	    --target-tiers $(TARGET_TIERS) \
	    --output $(RESULTS)
	@echo ""
	@echo "Whisper GPU report: $(RESULTS)/whisper_gpu_report.md"

# Whisper-only optimizer sweep: backend/model/quantization/beam/workers.
bench-whisper-opt:
	@mkdir -p $(RESULTS)
	LABEL_FILE="$(abspath $(RESULTS))/gpu_label.txt" RESULTS_DIR=$(RESULTS) \
	  WHISPER_GPU_UTIL=$(WHISPER_GPU_UTIL) WHISPER_MAX_WORKERS=$(WHISPER_MAX_WORKERS) \
	  python -m benchmarks.run_all \
	    --mode whisper-opt \
	    --workers auto \
	    --auto-whisper-workers \
	    --whisper-input-mode $(WHISPER_OPT_INPUT_MODE) \
	    --whisper-concurrency $(WHISPER_OPT_CONCURRENCY) \
	    --whisper-opt-backends $(WHISPER_OPT_BACKENDS) \
	    --whisper-opt-models $(WHISPER_OPT_MODELS) \
	    --whisper-opt-faster-compute-types $(WHISPER_OPT_FASTER_COMPUTE_TYPES) \
	    --whisper-opt-openai-precisions $(WHISPER_OPT_OPENAI_PRECISIONS) \
	    --whisper-opt-beam-sizes $(WHISPER_OPT_BEAM_SIZES) \
	    $(if $(WHISPER_WORKER_CANDIDATES),--whisper-worker-candidates $(WHISPER_WORKER_CANDIDATES),) \
	    --output $(RESULTS)
	@echo ""
	@echo "Whisper optimizer report: $(RESULTS)/whisper_opt_report.md"

# vLLM-only capacity projection for the GPU this runs on.
bench-vllm-gpu:
	$(MAKE) up-llm VLLM_GPU_UTIL=$(VLLM_GPU_UTIL) VLLM_MAX_NUM_SEQS=$(VLLM_MAX_NUM_SEQS) VLLM_MAX_MODEL_LEN=$(VLLM_MAX_MODEL_LEN)
	LABEL_FILE="$(abspath $(RESULTS))/gpu_label.txt" RESULTS_DIR=$(RESULTS) \
	  python -m benchmarks.run_all \
	    --mode vllm-gpu \
	    --llm-concurrency $(VLLM_GPU_CONCURRENCY) \
	    --target-tiers $(TARGET_TIERS) \
	    --output $(RESULTS)
	$(MAKE) down
	@echo ""
	@echo "vLLM GPU report: $(RESULTS)/vllm_gpu_report.md"

# ── Report ────────────────────────────────────────────────────────────────────
report:
	@python -m scripts.render_report --results-dir "$(RESULTS)"

report-vllm:
	RESULTS_DIR=$(RESULTS) python -m benchmarks.run_all \
	  --mode whisper \
	  --model $(VLLM_WHISPER_MODEL) \
	  --whisper-concurrency $(VLLM_CONCURRENCY) \
	  --whisper-input-mode $(VLLM_INPUT_MODE) \
	  --whisper-beam-size $(VLLM_BEAM_SIZE) \
	  --whisper-load-mode $(VLLM_LOAD_MODE) \
	  --whisper-steady-duration-s $(VLLM_STEADY_DURATION_S) \
	  --whisper-max-p95-ms $(VLLM_SLA_P95_MS) \
	  --whisper-projection-headroom $(WHISPER_PROJECTION_HEADROOM) \
	  --target-tiers $(TARGET_TIERS) \
	  --output $(RESULTS)

report-ai:
	@python -m scripts.render_report \
	  --results-dir "$(RESULTS)" \
	  --ai \
	  --ai-model "$(OPENAI_REPORT_MODEL)" \
	  --ai-reasoning-effort "$(OPENAI_REPORT_REASONING)"

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	@rm -f $(RESULTS)/*.json $(RESULTS)/*.jsonl $(RESULTS)/*.md
	@echo "Results cleared (fixtures preserved)."
