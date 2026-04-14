"""
GPU metrics sidecar — polls nvidia-smi every POLL_INTERVAL_MS milliseconds
and appends structured JSONL records to OUTPUT_FILE.

Runs as a standalone process (inside the gpu_monitor container).
Uses nvidia-smi rather than pynvml to avoid an extra dependency and because
500 ms polling resolution is sufficient for our benchmark granularity.

Record schema (one JSON object per line):
  ts              Unix timestamp (float)
  gpu_index       GPU index string
  gpu_name        GPU model name
  gpu_util_pct    GPU compute utilization %
  mem_util_pct    Memory controller utilization %
  mem_used_mib    VRAM used (MiB)
  mem_free_mib    VRAM free (MiB)
  mem_total_mib   VRAM total (MiB)
  sm_clock_mhz    SM clock frequency
  mem_clock_mhz   Memory clock frequency
  power_w         Power draw (W)
  temp_c          GPU temperature (°C)
  label           Optional test label injected via MONITOR_LABEL env var
"""
import asyncio
import json
import os
import subprocess
import time
from pathlib import Path

POLL_INTERVAL = float(os.getenv("POLL_INTERVAL_MS", "500")) / 1000.0
OUTPUT_FILE   = Path(os.getenv("OUTPUT_FILE", "/results/gpu_metrics.jsonl"))
LABEL         = os.getenv("MONITOR_LABEL", "")

# Optional label file — benchmark scripts write the current test phase here.
# gpu_monitor reads it on each poll so GPU records are tagged with the active phase.
# e.g.  echo "fw_large-v2_c50" > /results/gpu_label.txt
LABEL_FILE    = Path(os.getenv("LABEL_FILE", "/results/gpu_label.txt"))


def _read_label() -> str:
    """Read dynamic label from file, falling back to the static LABEL env var."""
    try:
        return LABEL_FILE.read_text().strip() or LABEL
    except FileNotFoundError:
        return LABEL
    except Exception:
        return LABEL

_SMI_QUERY = (
    "timestamp,index,name,"
    "utilization.gpu,utilization.memory,"
    "memory.used,memory.free,memory.total,"
    "clocks.current.sm,clocks.current.memory,"
    "power.draw,temperature.gpu"
)


def _parse_line(line: str, ts: float) -> dict:
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 12:
        return {"ts": ts, "parse_error": line}
    return {
        "ts":            ts,
        "gpu_index":     parts[1],
        "gpu_name":      parts[2],
        "gpu_util_pct":  _safe_float(parts[3]),
        "mem_util_pct":  _safe_float(parts[4]),
        "mem_used_mib":  _safe_float(parts[5]),
        "mem_free_mib":  _safe_float(parts[6]),
        "mem_total_mib": _safe_float(parts[7]),
        "sm_clock_mhz":  _safe_float(parts[8]),
        "mem_clock_mhz": _safe_float(parts[9]),
        "power_w":       _safe_float(parts[10]),
        "temp_c":        _safe_float(parts[11]),
        # label is injected by the caller after _parse_line() returns
    }


def _safe_float(s: str) -> float:
    try:
        return float(s)
    except ValueError:
        return -1.0


async def poll_forever() -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    print(f"GPU monitor started → {OUTPUT_FILE}  (interval={POLL_INTERVAL*1000:.0f}ms)")

    with OUTPUT_FILE.open("w", buffering=1) as fh:   # line-buffered, fresh file per monitor run
        while True:
            ts = time.time()
            try:
                result = subprocess.run(
                    ["nvidia-smi",
                     f"--query-gpu={_SMI_QUERY}",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=3,
                )
                current_label = _read_label()
                for raw_line in result.stdout.strip().splitlines():
                    record = _parse_line(raw_line, ts)
                    record["label"] = current_label
                    fh.write(json.dumps(record) + "\n")
            except FileNotFoundError:
                fh.write(json.dumps({"ts": ts, "error": "nvidia-smi not found"}) + "\n")
            except Exception as exc:
                fh.write(json.dumps({"ts": ts, "error": str(exc)}) + "\n")

            await asyncio.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    asyncio.run(poll_forever())
