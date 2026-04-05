# GPU Stress Testing Makefile
# Targets assume you are in the repo root and have Docker + nvidia-container-toolkit installed.
#
# Quick start:
#   cp .env.example .env          # edit LLM_MODEL_HOST_PATH
#   make fixtures                 # generate synthetic audio files
#   make bench-whisper MODEL=large-v2
#   make bench-llm
#   make bench-e2e

.PHONY: help fixtures \
        up-whisper up-llm up-mixed down \
        bench-whisper bench-llm bench-e2e bench-all \
        report clean

# ── Defaults ──────────────────────────────────────────────────────────────────
MODEL   ?= large-v2
WORKERS ?= 4
RESULTS ?= ./results

help:
	@echo ""
	@echo "  make fixtures              Generate synthetic 47s WAV files"
	@echo ""
	@echo "  make up-whisper            Start Whisper-only service (MODEL=<size>)"
	@echo "  make up-llm                Start LLM-only service"
	@echo "  make up-mixed              Start both services (mixed workload)"
	@echo "  make down                  Stop all services"
	@echo ""
	@echo "  make bench-whisper         Whisper concurrency sweep (service must be up)"
	@echo "  make bench-llm             LLM concurrency sweep (service must be up)"
	@echo "  make bench-e2e             E2E session sweep (both services must be up)"
	@echo "  make bench-all             Full suite: all Whisper models + LLM + E2E"
	@echo ""
	@echo "  make report                Re-render report from existing results"
	@echo "  make clean                 Remove result files"
	@echo ""
	@echo "  Variables:  MODEL=$(MODEL)  WORKERS=$(WORKERS)  RESULTS=$(RESULTS)"

# ── Fixtures ──────────────────────────────────────────────────────────────────
fixtures:
	@echo "Generating synthetic audio fixtures..."
	python -m generators.audio_gen --count 20 --duration 47 --output ./fixtures/audio
	@echo "Done."

# ── Service management ────────────────────────────────────────────────────────
up-whisper:
	WHISPER_MODEL=$(MODEL) WHISPER_WORKERS=$(WORKERS) \
	  docker compose -f docker-compose.whisper-only.yml up -d --build
	@echo "Waiting for Whisper service..."
	@bash scripts/verify_services.sh whisper

up-llm:
	docker compose -f docker-compose.llm-only.yml up -d --build
	@echo "Waiting for LLM service..."
	@bash scripts/verify_services.sh llm

up-mixed:
	WHISPER_MODEL=$(MODEL) WHISPER_WORKERS=$(WORKERS) \
	  docker compose up -d --build
	@echo "Waiting for services..."
	@bash scripts/verify_services.sh all

down:
	-docker compose -f docker-compose.yml              down 2>/dev/null
	-docker compose -f docker-compose.whisper-only.yml down 2>/dev/null
	-docker compose -f docker-compose.llm-only.yml     down 2>/dev/null

# ── Benchmark targets ─────────────────────────────────────────────────────────
bench-whisper: up-whisper
	RESULTS_DIR=$(RESULTS) python -m benchmarks.run_all --mode whisper --model $(MODEL) --output $(RESULTS)
	$(MAKE) down

bench-llm: up-llm
	RESULTS_DIR=$(RESULTS) python -m benchmarks.run_all --mode llm --output $(RESULTS)
	$(MAKE) down

bench-e2e: up-mixed
	RESULTS_DIR=$(RESULTS) python -m benchmarks.run_all --mode e2e --output $(RESULTS)
	$(MAKE) down

# Full suite — runs each Whisper model size in turn, then LLM, then E2E
bench-all:
	@for model in tiny base small medium large-v2 large-v3; do \
		echo ""; \
		echo "###############################################"; \
		echo "  Whisper model: $$model"; \
		echo "###############################################"; \
		$(MAKE) bench-whisper MODEL=$$model RESULTS=$(RESULTS)/whisper_$$model; \
	done
	$(MAKE) bench-llm   RESULTS=$(RESULTS)
	$(MAKE) bench-e2e   RESULTS=$(RESULTS)

# ── Report ────────────────────────────────────────────────────────────────────
report:
	python - <<'EOF'
import json, sys
from pathlib import Path
from metrics.reporter import generate_report
p = Path("$(RESULTS)/all_results.json")
if not p.exists():
    print(f"No results found at {p}. Run bench-all first.")
    sys.exit(1)
results = json.loads(p.read_text())
generate_report(results, gpu_metrics_path="$(RESULTS)/gpu_metrics.jsonl",
                output_path="$(RESULTS)/report.md")
print("Report written to $(RESULTS)/report.md")
EOF

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	rm -rf $(RESULTS)/*.json $(RESULTS)/*.jsonl $(RESULTS)/*.md
	@echo "Results cleared."
