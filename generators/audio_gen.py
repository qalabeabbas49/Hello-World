"""
Synthetic 47-second audio generator for Whisper benchmarking.

Design goals:
- Produce speech-like spectrograms so Whisper's encoder is exercised fully
  (pure silence → trivially fast; pure noise → atypical load).
- No external TTS dependency; generates ~80 ms per file on CPU.
- Reproducible via integer seeds (same seed → identical bytes).
- Output: 16 kHz mono WAV, PCM_16, targeting ~-3 dBFS peak.

Approach: voiced formant synthesis (F1/F2 resonances) + fricative bursts +
lognormal inter-word pauses + -40 dBFS ambient noise floor.
"""
import argparse
import io
import sys
from pathlib import Path

import numpy as np
import scipy.signal as signal
import soundfile as sf

SAMPLE_RATE = 16_000          # Whisper's native rate; no resampling overhead
DURATION_S  = 47.0
N_SAMPLES   = int(SAMPLE_RATE * DURATION_S)

# F1/F2 formant pairs (Hz) approximating common English vowels
VOWEL_FORMANTS = [
    (800, 1200),   # /a/ as in "father"
    (400, 2000),   # /i/ as in "see"
    (600,  900),   # /o/ as in "go"
    (700, 1100),   # /ɛ/ as in "bed"
]


def _voiced_segment(n_samples: int, f0: float, formants: tuple[int, int]) -> np.ndarray:
    """Sawtooth glottal pulse train filtered through two formant bandpasses."""
    t       = np.arange(n_samples) / SAMPLE_RATE
    glottal = signal.sawtooth(2 * np.pi * f0 * t, width=0.8).astype(np.float32)
    out     = np.zeros(n_samples, dtype=np.float32)
    for fc in formants:
        low  = max(fc - 200, 80)
        high = min(fc + 200, 7_900)
        b, a = signal.butter(4, [low, high], btype="band", fs=SAMPLE_RATE)
        out += signal.lfilter(b, a, glottal).astype(np.float32)
    peak = np.max(np.abs(out))
    return out / (peak + 1e-9)


def _unvoiced_segment(n_samples: int) -> np.ndarray:
    """Band-limited noise burst simulating fricative consonants."""
    noise = np.random.randn(n_samples).astype(np.float32)
    b, a  = signal.butter(4, [3_000, 7_000], btype="band", fs=SAMPLE_RATE)
    return (signal.lfilter(b, a, noise) * 0.3).astype(np.float32)


def generate_audio(duration_s: float = DURATION_S, seed: int = 0) -> bytes:
    """
    Return bytes of a 16 kHz mono WAV file with realistic speech characteristics.

    Args:
        duration_s: Length of audio in seconds (default 47).
        seed:       RNG seed for reproducibility.

    Returns:
        Raw WAV bytes (PCM_16, 16 kHz, mono).
    """
    rng       = np.random.default_rng(seed)
    n_samples = int(SAMPLE_RATE * duration_s)
    audio     = np.zeros(n_samples, dtype=np.float32)
    pos       = 0

    while pos < n_samples:
        # Word duration: 200–600 ms voiced or unvoiced segment
        word_len = int(rng.uniform(0.2, 0.6) * SAMPLE_RATE)
        word_len = min(word_len, n_samples - pos)
        if word_len == 0:
            break

        f0       = float(rng.uniform(90, 220))
        formants = VOWEL_FORMANTS[int(rng.integers(len(VOWEL_FORMANTS)))]

        if rng.random() < 0.72:
            segment = _voiced_segment(word_len, f0, formants)
        else:
            segment = _unvoiced_segment(word_len)

        amplitude = float(rng.uniform(0.3, 0.85))
        audio[pos : pos + word_len] += segment[:word_len] * amplitude
        pos += word_len

        # Inter-word pause drawn from a lognormal distribution (50–400 ms)
        pause_s   = float(rng.lognormal(mean=np.log(0.12), sigma=0.6))
        pause_s   = np.clip(pause_s, 0.05, 0.40)
        pause_len = int(min(pause_s * SAMPLE_RATE, n_samples - pos))
        pos      += pause_len

    # Ambient noise floor at ~-40 dBFS
    audio += rng.standard_normal(n_samples).astype(np.float32) * 0.01

    # Normalize to -3 dBFS peak (0.707 linear)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.707

    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def generate_pool(count: int = 10, duration_s: float = DURATION_S) -> list[bytes]:
    """Pre-generate `count` unique audio files for benchmark cycling."""
    return [generate_audio(duration_s=duration_s, seed=i) for i in range(count)]


# ── CLI entry point ───────────────────────────────────────────────────────────
def _cli():
    parser = argparse.ArgumentParser(description="Generate synthetic benchmark audio files.")
    parser.add_argument("--count",    type=int,   default=10,         help="Number of files to generate")
    parser.add_argument("--duration", type=float, default=DURATION_S, help="Duration in seconds (default 47)")
    parser.add_argument("--output",   type=str,   default="./fixtures/audio", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.count} × {args.duration}s WAV files → {out_dir}")
    for i in range(args.count):
        wav_bytes = generate_audio(duration_s=args.duration, seed=i)
        path = out_dir / f"audio_{i:04d}.wav"
        path.write_bytes(wav_bytes)
        print(f"  {path}  ({len(wav_bytes) / 1024:.0f} KB)")
    print("Done.")


if __name__ == "__main__":
    _cli()
