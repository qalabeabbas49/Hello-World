"""
Synthetic audio generator — supports WAV, FLAC, and Opus at 16 kHz or 48 kHz.

Design goals:
- Produce speech-like spectrograms (formant synthesis) so Whisper's encoder
  is exercised fully — unlike pure silence or noise.
- No external TTS dependency; each 47 s file generates in ~80–150 ms on CPU.
- Reproducible via integer seeds.
- Formats: WAV (PCM_16), FLAC (PCM_16), Opus (OGG container, libopus 32 kbps).
- Sample rates: 16 kHz (Whisper-native, no service resampling) and 48 kHz
  (common for browser/microphone capture — tests the service resampling path).

Approach: voiced F1/F2 formant synthesis + fricative bursts +
lognormal inter-word pauses + -40 dBFS ambient noise floor.
"""
import argparse
import io
import subprocess
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import scipy.signal as sig_mod
import soundfile as sf

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_DURATION_S    = 47.0
DEFAULT_SAMPLE_RATE   = 16_000

SUPPORTED_FORMATS      = ["wav", "flac", "opus"]
SUPPORTED_SAMPLE_RATES = [16_000, 48_000]

# F1/F2 formant pairs (Hz) approximating common English vowels
VOWEL_FORMANTS = [
    (800, 1200),   # /a/
    (400, 2000),   # /i/
    (600,  900),   # /o/
    (700, 1100),   # /ɛ/
]

SAMPLE_RATE = DEFAULT_SAMPLE_RATE   # kept for backwards-compatibility
DURATION_S  = DEFAULT_DURATION_S
N_SAMPLES   = int(SAMPLE_RATE * DURATION_S)


# ── Core audio synthesis ──────────────────────────────────────────────────────

def _voiced_segment(n_samples: int, f0: float, formants: tuple[int, int], sr: int) -> np.ndarray:
    t       = np.arange(n_samples) / sr
    glottal = sig_mod.sawtooth(2 * np.pi * f0 * t, width=0.8).astype(np.float32)
    out     = np.zeros(n_samples, dtype=np.float32)
    for fc in formants:
        low  = max(fc - 200, 80)
        high = min(fc + 200, sr // 2 - 100)
        b, a = sig_mod.butter(4, [low, high], btype="band", fs=sr)
        out += sig_mod.lfilter(b, a, glottal).astype(np.float32)
    peak = np.max(np.abs(out))
    return out / (peak + 1e-9)


def _unvoiced_segment(n_samples: int, sr: int) -> np.ndarray:
    noise = np.random.randn(n_samples).astype(np.float32)
    high  = min(7_000, sr // 2 - 100)
    b, a  = sig_mod.butter(4, [3_000, high], btype="band", fs=sr)
    return (sig_mod.lfilter(b, a, noise) * 0.3).astype(np.float32)


def _synthesize(duration_s: float = DEFAULT_DURATION_S,
                sample_rate: int = DEFAULT_SAMPLE_RATE,
                seed: int = 0) -> np.ndarray:
    """Return a float32 numpy array of speech-like audio."""
    rng       = np.random.default_rng(seed)
    n_samples = int(sample_rate * duration_s)
    audio     = np.zeros(n_samples, dtype=np.float32)
    pos       = 0

    while pos < n_samples:
        word_len = int(rng.uniform(0.2, 0.6) * sample_rate)
        word_len = min(word_len, n_samples - pos)
        if word_len == 0:
            break

        f0       = float(rng.uniform(90, 220))
        formants = VOWEL_FORMANTS[int(rng.integers(len(VOWEL_FORMANTS)))]

        segment  = (_voiced_segment(word_len, f0, formants, sample_rate)
                    if rng.random() < 0.72
                    else _unvoiced_segment(word_len, sample_rate))

        audio[pos : pos + word_len] += segment[:word_len] * float(rng.uniform(0.3, 0.85))
        pos += word_len

        pause_s   = float(rng.lognormal(mean=np.log(0.12), sigma=0.6))
        pause_s   = float(np.clip(pause_s, 0.05, 0.40))
        pos      += int(min(pause_s * sample_rate, n_samples - pos))

    # Ambient noise floor ~-40 dBFS
    audio += rng.standard_normal(n_samples).astype(np.float32) * 0.01

    # Normalize to -3 dBFS peak
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.707
    return audio


# ── Format encoders ───────────────────────────────────────────────────────────

def _to_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _to_flac(audio: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="FLAC", subtype="PCM_16")
    return buf.getvalue()


def _to_opus(audio: np.ndarray, sample_rate: int) -> bytes:
    """
    Encode to OGG/Opus via ffmpeg (libopus).
    ffmpeg reads raw PCM from stdin, writes OGG container to stdout.
    Bitrate 32 kbps is realistic for medical audio (speech-quality).

    Requires: ffmpeg with libopus support (standard in most distros).
    """
    pcm_bytes = (audio * 32_767).astype(np.int16).tobytes()
    result    = subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "s16le", "-ar", str(sample_rate), "-ac", "1",
            "-i", "pipe:0",
            "-c:a", "libopus",
            "-b:a", "32k",
            "-application", "voip",   # optimised for speech
            "-f", "ogg",
            "pipe:1",
        ],
        input=pcm_bytes,
        capture_output=True,
        check=True,
    )
    return result.stdout


# ── Public API ────────────────────────────────────────────────────────────────

def generate_audio(
    duration_s: float = DEFAULT_DURATION_S,
    seed: int = 0,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    fmt: str = "wav",
) -> bytes:
    """
    Generate synthetic speech audio as bytes.

    Args:
        duration_s:  Duration in seconds (default 47).
        seed:        RNG seed — same seed → identical bytes.
        sample_rate: 16000 (Whisper-native) or 48000 (browser-capture typical).
        fmt:         "wav" | "flac" | "opus"

    Returns:
        Encoded audio bytes.
    """
    fmt = fmt.lower()
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format {fmt!r}. Choose from {SUPPORTED_FORMATS}.")
    if sample_rate not in SUPPORTED_SAMPLE_RATES:
        raise ValueError(f"Unsupported sample rate {sample_rate}. Choose from {SUPPORTED_SAMPLE_RATES}.")

    audio = _synthesize(duration_s, sample_rate, seed)

    if fmt == "wav":
        return _to_wav(audio, sample_rate)
    elif fmt == "flac":
        return _to_flac(audio, sample_rate)
    elif fmt == "opus":
        return _to_opus(audio, sample_rate)


def generate_audio_array(
    duration_s: float = DEFAULT_DURATION_S,
    seed: int = 0,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> np.ndarray:
    """
    Generate synthetic speech audio as a float32 mono numpy array.

    This is useful for decode-free GPU benchmarks where we want to bypass
    per-request container decoding/resampling overhead.
    """
    if sample_rate not in SUPPORTED_SAMPLE_RATES:
        raise ValueError(f"Unsupported sample rate {sample_rate}. Choose from {SUPPORTED_SAMPLE_RATES}.")
    return _synthesize(duration_s, sample_rate, seed)


def generate_pool(
    count: int = 10,
    duration_s: float = DEFAULT_DURATION_S,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    fmt: str = "wav",
) -> list[bytes]:
    """Pre-generate `count` unique audio files for benchmark cycling."""
    return [generate_audio(duration_s=duration_s, seed=i, sample_rate=sample_rate, fmt=fmt) for i in range(count)]


def generate_pool_decoded(
    count: int = 10,
    duration_s: float = DEFAULT_DURATION_S,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> list[np.ndarray]:
    """Pre-generate `count` decoded float32 mono arrays for benchmark cycling."""
    return [generate_audio_array(duration_s=duration_s, seed=i, sample_rate=sample_rate) for i in range(count)]


def generate_pool_pcm_f32le(
    count: int = 10,
    duration_s: float = DEFAULT_DURATION_S,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> list[bytes]:
    """
    Pre-generate `count` raw float32 little-endian PCM payloads for decode-free
    Whisper benchmarks.
    """
    return [arr.astype(np.float32, copy=False).tobytes() for arr in generate_pool_decoded(count, duration_s, sample_rate)]


def _content_type(fmt: str) -> str:
    return {
        "wav":  "audio/wav",
        "flac": "audio/flac",
        "opus": "audio/ogg",
    }.get(fmt, "application/octet-stream")


# ── CLI entry point ───────────────────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic benchmark audio.")
    parser.add_argument("--count",       type=int,   default=10,                  help="Number of files")
    parser.add_argument("--duration",    type=float, default=DEFAULT_DURATION_S,  help="Duration seconds")
    parser.add_argument("--output",      type=str,   default="./fixtures/audio",  help="Output directory")
    parser.add_argument("--formats",     type=str,   default="wav",               help="Comma-separated: wav,flac,opus")
    parser.add_argument("--sample-rates",type=str,   default="16000",             help="Comma-separated: 16000,48000")
    args = parser.parse_args()

    fmts  = [f.strip() for f in args.formats.split(",")]
    rates = [int(r.strip()) for r in args.sample_rates.split(",")]

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = args.count * len(fmts) * len(rates)
    print(f"Generating {total} files ({args.count} seeds × {len(fmts)} formats × {len(rates)} rates) → {out_dir}")

    for fmt in fmts:
        for rate in rates:
            subdir = out_dir / f"{fmt}_{rate}"
            subdir.mkdir(parents=True, exist_ok=True)
            for i in range(args.count):
                ext  = "ogg" if fmt == "opus" else fmt
                path = subdir / f"audio_{i:04d}.{ext}"
                path.write_bytes(generate_audio(duration_s=args.duration, seed=i,
                                                sample_rate=rate, fmt=fmt))
                print(f"  {path}  ({path.stat().st_size // 1024} KB)")
    print("Done.")


if __name__ == "__main__":
    _cli()
