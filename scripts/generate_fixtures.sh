#!/bin/bash
# Generate synthetic audio fixtures for benchmark runs.
# Run once before benchmarking to avoid on-the-fly generation overhead.
set -euo pipefail

COUNT="${1:-20}"
DURATION="${2:-47}"
OUTDIR="${3:-./fixtures/audio}"

echo "Generating $COUNT × ${DURATION}s WAV files → $OUTDIR"
python -m generators.audio_gen --count "$COUNT" --duration "$DURATION" --output "$OUTDIR"
echo "Done. $(ls "$OUTDIR"/*.wav 2>/dev/null | wc -l) files in $OUTDIR"
