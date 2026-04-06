"""
Real audio file loader for benchmark testing.

Scans a user-provided directory for audio files and makes them available
as benchmark pools — a direct replacement for the synthetic audio pool.

Supported input formats: WAV, FLAC, Opus (.opus / .ogg), MP3, M4A, WebM.
Files are read as raw bytes and sent as-is to the Whisper service.
The service's ffmpeg layer handles all decoding, so no preprocessing here.

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
    bytes_list = pool.get_pool(fmt="wav", sample_rate=16000, max_files=10)
"""
import io
import logging
from dataclasses import dataclass
from pathlib import Path

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
