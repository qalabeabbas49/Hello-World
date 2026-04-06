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
#   make bench-compare                   # both backends × all Whisper models
#   make bench-llm                       # LLM throughput sweep
#   make bench-e2e                       # end-to-end session sweep
#   make report                          # re-render report.md from saved results

.PHONY: help fixtures check-gpu \
        up-whisper up-llm up-mixed down \
        bench-whisper bench-llm bench-e2e bench-compare bench-formats bench-real-audio bench-all \
        report clean

# ── Configurable defaults ──────────────────────────────────────────────────────
MODEL          ?= large-v2
BACKEND        ?= faster_whisper
WORKERS        ?= 16
RESULTS        ?= ./results
SESSIONS       ?=
CHUNKS         ?=
FORMATS        ?= wav,flac,opus
RATES          ?= 16000,48000
REAL_AUDIO_DIR ?= ./fixtures/real_audio

help:
	@echo ""
	@echo "  Fixtures & checks"
	@echo "    make fixtures              Generate synthetic audio: WAV/FLAC/Opus at 16k+48k"
	@echo "    make check-gpu             Verify GPU driver, Docker GPU access"
	@echo ""
	@echo "  Service management"
	@echo "    make up-whisper            Start Whisper-only  (BACKEND=, MODEL=, WORKERS=)"
	@echo "    make up-llm                Start LLM-only"
	@echo "    make up-mixed              Start both services (mixed workload)"
	@echo "    make down                  Stop all compose stacks"
	@echo ""
	@echo "  Benchmarks (manage their own docker lifecycle by default)"
	@echo "    make bench-whisper         One backend × one model  (BACKEND=, MODEL=)"
	@echo "    make bench-llm             LLM concurrency sweep"
	@echo "    make bench-e2e             E2E session sweep"
	@echo "    make bench-compare         Both backends × ALL model sizes  [main concurrency sweep]"
	@echo "    make bench-formats         WAV/FLAC/Opus × 16k/48k format sweep (synthetic)"
	@echo "    make bench-real-audio      Same sweep with your own audio files (REAL_AUDIO_DIR=)"
	@echo "    make bench-all             bench-compare + bench-formats + bench-llm + bench-e2e"
	@echo ""
	@echo "  Results"
	@echo "    make report                Re-render report.md from saved JSON results"
	@echo "    make clean                 Delete result files (keeps fixtures)"
	@echo ""
	@echo "  Key variables:"
	@echo "    BACKEND=$(BACKEND)  MODEL=$(MODEL)  WORKERS=$(WORKERS)"
	@echo "    FORMATS=$(FORMATS)  RATES=$(RATES)"
	@echo "    REAL_AUDIO_DIR=$(REAL_AUDIO_DIR)  RESULTS=$(RESULTS)"

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
	WHISPER_BACKEND=$(BACKEND) WHISPER_MODEL=$(MODEL) WHISPER_WORKERS=$(WORKERS) \
	  docker compose -f docker-compose.whisper-only.yml up -d --build
	@bash scripts/verify_services.sh whisper

up-llm:
	docker compose -f docker-compose.llm-only.yml up -d --build
	@bash scripts/verify_services.sh llm

up-mixed:
	WHISPER_BACKEND=$(BACKEND) WHISPER_MODEL=$(MODEL) WHISPER_WORKERS=$(WORKERS) \
	  docker compose up -d --build
	@bash scripts/verify_services.sh all

down:
	-docker compose -f docker-compose.yml              down 2>/dev/null || true
	-docker compose -f docker-compose.whisper-only.yml down 2>/dev/null || true
	-docker compose -f docker-compose.llm-only.yml     down 2>/dev/null || true

# ── Benchmark targets ─────────────────────────────────────────────────────────

# Single backend+model Whisper sweep (brings up service, benchmarks, tears down)
bench-whisper:
	$(MAKE) up-whisper BACKEND=$(BACKEND) MODEL=$(MODEL) WORKERS=$(WORKERS)
	RESULTS_DIR=$(RESULTS) python -m benchmarks.run_all \
	  --mode whisper --model $(MODEL) --backend $(BACKEND) --output $(RESULTS)
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

# ── Main comparison sweep: both backends × all model sizes ─────────────────────
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

# Same sweep with real audio files (service must be running or will be started by up-whisper)
bench-real-audio:
	$(MAKE) up-whisper BACKEND=$(BACKEND) MODEL=$(MODEL) WORKERS=$(WORKERS)
	RESULTS_DIR=$(RESULTS) python -m benchmarks.run_all \
	  --mode format \
	  --audio-formats $(FORMATS) --audio-rates $(RATES) \
	  --real-audio-dir $(REAL_AUDIO_DIR) \
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
	  --output $(RESULTS)
	@echo ""
	@echo "Full suite complete. Report: $(RESULTS)/report.md"

# ── Report ────────────────────────────────────────────────────────────────────
report:
	@python - <<'EOF'
import json, sys
from pathlib import Path
from metrics.reporter import generate_report
p = Path("$(RESULTS)/all_results.json")
if not p.exists():
    print(f"No results at {p}. Run bench-all or bench-compare first.")
    sys.exit(1)
results = json.loads(p.read_text())
generate_report(
    results,
    gpu_metrics_path="$(RESULTS)/gpu_metrics.jsonl",
    output_path="$(RESULTS)/report.md",
)
print("Report written to $(RESULTS)/report.md")
EOF

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	@rm -f $(RESULTS)/*.json $(RESULTS)/*.jsonl $(RESULTS)/*.md
	@echo "Results cleared (fixtures preserved)."
