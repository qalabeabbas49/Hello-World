"""
Real audio file loader and chunker for benchmark testing.

Scans a user-provided directory for audio files, splits them into
production-sized 47-second chunks (matching the real streaming pipeline),
and makes them available as benchmark pools.

Supported input formats: WAV, FLAC, Opus (.opus / .ogg), MP3, M4A, WebM.
All formats are decoded to PCM internally and re-encoded as WAV chunks so
the benchmark client does not depend on external codec libraries at send time.
The Whisper service receives standard 47-second WAV chunks — exactly what
the production streaming pipeline sends.

Folder structure (expected but not required):
  <real_audio_dir>/
    ├── wav_16000/          ← WAV files at 16 kHz
    │   ├── session_001.wav
    │   └── ...
    ├── wav_48000/          ← WAV files at 48 kHz
    ├── flac_16000/
    ├── flac_48000/
    ├── opus_16000/
    └── opus_48000/

If flat (no subdirectories), format and sample rate are detected from
the file header / extension. All files in the flat root are pooled together.

Usage:
    pool = RealAudioPool.from_directory("./my_audio_files")
    pool.print_summary()

    # Get 47-second WAV chunks ready to send to Whisper
    chunks = pool.get_chunks(chunk_duration_s=47.0)
    for chunk in chunks:
        send_to_whisper(chunk.data, duration_s=chunk.duration_s)
"""
import io
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# Maps file extension → (format_name, content_type)
EXT_MAP: dict[str, tuple[str, str]] = {
    ".wav":  ("wav",  "audio/wav"),
    ".flac": ("flac", "audio/flac"),
    ".opus": ("opus", "audio/ogg"),
    ".ogg":  ("opus", "audio/ogg"),
    ".mp3":  ("mp3",  "audio/mpeg"),
    ".m4a":  ("m4a",  "audio/mp4"),
    ".webm": ("webm", "audio/webm"),
    ".aac":  ("aac",  "audio/aac"),
}


@dataclass
class AudioFile:
    path:         Path
    fmt:          str          # "wav" | "flac" | "opus" | "mp3" | ...
    sample_rate:  int | None   # None if unknown / not probed
    duration_s:   float | None # None if not probed
    content_type: str
    size_bytes:   int

    def read_bytes(self) -> bytes:
        return self.path.read_bytes()

    def __repr__(self) -> str:
        sr  = f"{self.sample_rate // 1000}k" if self.sample_rate else "?"
        dur = f"{self.duration_s:.1f}s" if self.duration_s else "?"
        return f"AudioFile({self.path.name}, {self.fmt}, {sr}, {dur}, {self.size_bytes // 1024}KB)"


@dataclass
class AudioChunk:
    """A single 47-second (or shorter final) chunk ready to POST to the Whisper service."""
    data:         bytes   # WAV-encoded PCM at the source sample rate
    duration_s:   float   # actual chunk duration (last chunk may be < chunk_duration_s)
    source_path:  Path    # originating file
    chunk_index:  int     # 0-based index within the source file
    total_chunks: int     # total number of chunks from this file


def _load_to_pcm(path: Path) -> tuple[np.ndarray, int]:
    """
    Load any supported audio file to a float32 mono PCM array.

    Returns:
        (samples, sample_rate) where samples is shape (N,).

    Strategy:
      1. soundfile  — handles WAV, FLAC, OGG Vorbis natively
      2. ffmpeg pipe — fallback for Opus, MP3, M4A, WebM
    """
    try:
        arr, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        return arr, sr
    except Exception:
        pass

    # ffmpeg fallback: decode to raw f32le at original sample rate
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", str(path)],
            capture_output=True, text=True, timeout=10,
        )
        import json as _json
        streams = _json.loads(probe.stdout).get("streams", [])
        audio   = next((s for s in streams if s.get("codec_type") == "audio"), None)
        sr      = int(audio["sample_rate"]) if audio else 16000
    except Exception:
        sr = 16000

    result = subprocess.run(
        ["ffmpeg", "-v", "quiet", "-i", str(path),
         "-f", "f32le", "-ac", "1", "-ar", str(sr), "pipe:1"],
        capture_output=True, timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed on {path}: {result.stderr.decode(errors='replace')[:200]}"
        )
    arr = np.frombuffer(result.stdout, dtype=np.float32).copy()
    return arr, sr


def _pcm_to_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """Encode a float32 PCM array as 16-bit WAV bytes (in-memory, no temp file)."""
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _split_into_chunks(
    samples:          np.ndarray,
    sample_rate:      int,
    source_path:      Path,
    chunk_duration_s: float = 47.0,
    min_duration_s:   float = 5.0,
) -> list[AudioChunk]:
    """
    Split a PCM array into fixed-length WAV chunks matching the production
    streaming pipeline (47 seconds per chunk by default).

    Args:
        samples:          Mono float32 PCM array.
        sample_rate:      Sample rate of the PCM data.
        source_path:      Original file path (for metadata only).
        chunk_duration_s: Target chunk length in seconds (default 47 s).
        min_duration_s:   Discard trailing chunks shorter than this threshold
                          to avoid sending near-silent tail clips.

    Returns:
        List of AudioChunk objects, each containing a WAV-encoded byte string.
    """
    chunk_samples = int(chunk_duration_s * sample_rate)
    total_samples = len(samples)

    offsets = list(range(0, total_samples, chunk_samples))
    total   = len(offsets)
    chunks: list[AudioChunk] = []

    for idx, start in enumerate(offsets):
        end    = min(start + chunk_samples, total_samples)
        clip   = samples[start:end]
        dur_s  = len(clip) / sample_rate

        if dur_s < min_duration_s:
            logger.debug(
                "Dropping trailing chunk %d/%d (%.1fs < min %.1fs) from %s",
                idx, total, dur_s, min_duration_s, source_path.name,
            )
            continue

        chunks.append(AudioChunk(
            data         = _pcm_to_wav_bytes(clip, sample_rate),
            duration_s   = round(dur_s, 3),
            source_path  = source_path,
            chunk_index  = idx,
            total_chunks = total,
        ))

    return chunks


def _probe_audio(path: Path) -> tuple[int | None, float | None]:
    """
    Probe sample rate and duration from an audio file using soundfile.
    Falls back gracefully for formats soundfile doesn't support.
    """
    try:
        import soundfile as sf
        info = sf.info(str(path))
        return info.samplerate, info.duration
    except Exception:
        pass

    # For Opus/WebM/MP3 that soundfile can't probe, try ffprobe
    try:
        import subprocess, json
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_streams",
                str(path),
            ],
            capture_output=True, text=True, timeout=5,
        )
        data    = json.loads(result.stdout)
        streams = data.get("streams", [])
        audio   = next((s for s in streams if s.get("codec_type") == "audio"), None)
        if audio:
            sr  = int(audio.get("sample_rate", 0)) or None
            dur = float(audio.get("duration",  0)) or None
            return sr, dur
    except Exception:
        pass

    return None, None


def _infer_sr_from_path(path: Path) -> int | None:
    """
    Try to extract sample rate from directory name or filename.
    e.g.  wav_16000/file.wav → 16000
          audio_48k.wav       → 48000
    """
    tokens = path.parent.name + "_" + path.stem
    for rate in (16000, 48000, 44100, 22050, 8000):
        if str(rate) in tokens or str(rate // 1000) + "k" in tokens.lower():
            return rate
    return None


class RealAudioPool:
    """
    Manages a collection of real audio files for use in benchmark runs.

    Attributes:
        files: All discovered AudioFile objects.
    """

    def __init__(self, files: list[AudioFile]):
        self.files = files

    @classmethod
    def from_directory(
        cls,
        directory: str | Path,
        probe: bool = True,
        max_files: int | None = None,
    ) -> "RealAudioPool":
        """
        Scan `directory` recursively for audio files.

        Args:
            directory: Path to the folder containing audio files.
            probe:     If True, probe each file for sample rate and duration.
                       Disable for large collections (slow for Opus/MP3).
            max_files: Cap on total files loaded (for quick sanity runs).
        """
        root = Path(directory)
        if not root.exists():
            raise FileNotFoundError(f"Audio directory not found: {root}")

        found: list[AudioFile] = []

        for ext, (fmt, content_type) in EXT_MAP.items():
            for path in sorted(root.rglob(f"*{ext}")):
                sr, dur = None, None
                if probe:
                    sr_path = _infer_sr_from_path(path)
                    sr_probe, dur = _probe_audio(path)
                    sr = sr_path or sr_probe

                found.append(AudioFile(
                    path=path,
                    fmt=fmt,
                    sample_rate=sr,
                    duration_s=dur,
                    content_type=content_type,
                    size_bytes=path.stat().st_size,
                ))

                if max_files and len(found) >= max_files:
                    break
            if max_files and len(found) >= max_files:
                break

        if not found:
            raise ValueError(
                f"No audio files found in {root}. "
                f"Supported extensions: {list(EXT_MAP)}"
            )

        logger.info("RealAudioPool: loaded %d files from %s", len(found), root)
        return cls(found)

    def get_pool(
        self,
        fmt: str | None = None,
        sample_rate: int | None = None,
        max_files: int | None = None,
    ) -> list[bytes]:
        """
        Return a list of audio byte strings filtered by format and/or sample rate.

        Args:
            fmt:         Filter by format ("wav", "flac", "opus", etc.) or None for all.
            sample_rate: Filter by sample rate (16000, 48000, etc.) or None for all.
            max_files:   Maximum number of files to return.

        Returns:
            List of raw audio bytes, ready to POST to the Whisper service.
        """
        candidates = self.files
        if fmt:
            candidates = [f for f in candidates if f.fmt == fmt.lower()]
        if sample_rate:
            # Include files where SR is known and matches, or where SR is unknown
            candidates = [f for f in candidates
                          if f.sample_rate is None or f.sample_rate == sample_rate]
        if not candidates:
            raise ValueError(
                f"No files match fmt={fmt!r} sample_rate={sample_rate}. "
                f"Available: {self.summary()}"
            )
        if max_files:
            candidates = candidates[:max_files]
        return [f.read_bytes() for f in candidates]

    def get_pool_with_meta(
        self,
        fmt: str | None = None,
        sample_rate: int | None = None,
        max_files: int | None = None,
    ) -> list[tuple[bytes, AudioFile]]:
        """Same as get_pool() but returns (bytes, AudioFile) tuples for metadata access."""
        candidates = self.files
        if fmt:
            candidates = [f for f in candidates if f.fmt == fmt.lower()]
        if sample_rate:
            candidates = [f for f in candidates
                          if f.sample_rate is None or f.sample_rate == sample_rate]
        if max_files:
            candidates = candidates[:max_files]
        return [(f.read_bytes(), f) for f in candidates]

    def get_chunks(
        self,
        fmt: str | None = None,
        sample_rate: int | None = None,
        max_files: int | None = None,
        chunk_duration_s: float = 47.0,
        min_chunk_duration_s: float = 5.0,
    ) -> list[AudioChunk]:
        """
        Load audio files, split each into production-sized chunks, and return
        the full list of AudioChunk objects ready to POST to the Whisper service.

        Each chunk is WAV-encoded PCM at the original file's sample rate.
        This matches the production streaming pipeline where audio is segmented
        into fixed-length chunks before being sent to Whisper.

        Args:
            fmt:                  Filter by original format ("wav", "flac", "opus", ...).
            sample_rate:          Filter by original sample rate.
            max_files:            Cap on source files to load (chunks are generated from these).
            chunk_duration_s:     Chunk length in seconds (default 47 s — production value).
            min_chunk_duration_s: Drop trailing chunks shorter than this threshold.

        Returns:
            Flat list of AudioChunk objects (may be many more than max_files).
        """
        candidates = self.files
        if fmt:
            candidates = [f for f in candidates if f.fmt == fmt.lower()]
        if sample_rate:
            candidates = [f for f in candidates
                          if f.sample_rate is None or f.sample_rate == sample_rate]
        if not candidates:
            raise ValueError(
                f"No files match fmt={fmt!r} sample_rate={sample_rate}. "
                f"Available: {self.summary()}"
            )
        if max_files:
            candidates = candidates[:max_files]

        all_chunks: list[AudioChunk] = []
        for audio_file in candidates:
            try:
                samples, sr = _load_to_pcm(audio_file.path)
                file_chunks = _split_into_chunks(
                    samples, sr, audio_file.path,
                    chunk_duration_s=chunk_duration_s,
                    min_duration_s=min_chunk_duration_s,
                )
                all_chunks.extend(file_chunks)
                logger.debug(
                    "%s → %d chunks (%.1fs each)",
                    audio_file.path.name, len(file_chunks), chunk_duration_s,
                )
            except Exception as exc:
                logger.warning("Could not chunk %s: %s", audio_file.path.name, exc)

        if not all_chunks:
            raise ValueError(
                f"No chunks generated from {len(candidates)} file(s). "
                "Check that the files are valid audio and ffmpeg is installed."
            )

        logger.info(
            "get_chunks: %d source files → %d × %.0fs chunks",
            len(candidates), len(all_chunks), chunk_duration_s,
        )
        return all_chunks

    def summary(self) -> dict:
        """Return a dict summarising the pool by format and sample rate."""
        by_fmt: dict[str, dict] = {}
        for f in self.files:
            if f.fmt not in by_fmt:
                by_fmt[f.fmt] = {"count": 0, "sample_rates": set(), "total_kb": 0}
            by_fmt[f.fmt]["count"]        += 1
            by_fmt[f.fmt]["total_kb"]     += f.size_bytes // 1024
            if f.sample_rate:
                by_fmt[f.fmt]["sample_rates"].add(f.sample_rate)
        # Convert sets to sorted lists for display
        for v in by_fmt.values():
            v["sample_rates"] = sorted(v["sample_rates"])
        return by_fmt

    def print_summary(self) -> None:
        print(f"\nReal audio pool: {len(self.files)} files")
        for fmt, info in self.summary().items():
            rates = ", ".join(f"{r//1000}k" for r in info["sample_rates"]) or "unknown rate"
            print(f"  {fmt:6s}  {info['count']:4d} files  {info['total_kb']:6d} KB  [{rates}]")

    def content_type(self, fmt: str) -> str:
        for f in self.files:
            if f.fmt == fmt:
                return f.content_type
        return "application/octet-stream"
