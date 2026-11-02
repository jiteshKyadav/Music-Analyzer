"""
duration.py
===========

Audio duration detection module for the Music Analyzer.

This module inspects only the header/metadata of an audio file and never
decodes the full waveform, which makes it fast, memory-cheap, and safe to
call on very long tracks.

Public API
----------
- ``get_duration(path)``      returns the exact duration in seconds as a
                              ``float`` (full precision, no rounding).
- ``format_duration(seconds)`` returns the same duration formatted as a
                              human-readable string ``HH:MM:SS.mmm``.

Both functions are pure and side-effect free. Import them directly into
the main analyzer pipeline; nothing runs on import.

Supported formats
-----------------
- WAV  (parsed by ``soundfile``)
- MP3  (parsed by ``mutagen``, handles both CBR and VBR correctly)

Adding FLAC later is a one-line change: append ``".flac"`` to
``_SOUNDFILE_EXTENSIONS`` — ``soundfile`` already supports it natively.
"""

from __future__ import annotations

import math
from pathlib import Path

from utils.validation import (
    AudioPath,
    validate_path,
)

import soundfile as sf
from mutagen.mp3 import MP3, HeaderNotFoundError

# ``soundfile.LibsndfileError`` was introduced in soundfile 0.12. On older
# installs the backend raises plain ``RuntimeError``; the getattr keeps the
# module importable across versions without behavioural change.
_SoundfileError = getattr(sf, "LibsndfileError", RuntimeError)

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

_SOUNDFILE_EXTENSIONS = frozenset({".wav"})
_MUTAGEN_EXTENSIONS = frozenset({".mp3"})

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _duration_via_soundfile(path: Path) -> float:
    """Read a WAV (or other soundfile-supported) header and compute duration.

    ``soundfile.info`` reads only the header, so this is O(1) in file size.
    """
    info = sf.info(str(path))
    if info.samplerate <= 0 or info.frames <= 0:
        raise ValueError(
            f"Invalid audio header: frames={info.frames}, "
            f"samplerate={info.samplerate}"
        )
    return info.frames / info.samplerate


def _duration_via_mutagen(path: Path) -> float:
    """Read an MP3 frame/metadata header and return its reported duration.

    ``mutagen.mp3.MP3`` parses Xing/VBRI headers when present, so both CBR
    and VBR files return an accurate duration.
    """
    audio = MP3(str(path))
    if audio.info is None or audio.info.length is None:
        raise ValueError(f"MP3 header did not report a duration: {path}")
    return float(audio.info.length)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_duration(path: AudioPath) -> float:
    """Return the exact duration of an audio file, in seconds.

    The value is a ``float`` with full precision (no rounding, no
    truncation). Use :func:`format_duration` separately if a human-readable
    string is needed — the raw float is never overwritten.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a supported audio file (WAV or MP3).

    Returns
    -------
    float
        Duration in seconds. Example: ``125.987654321``.

    Raises
    ------
    FileNotFoundError
        The file does not exist.
    PermissionError
        The file exists but cannot be read.
    ValueError
        The file is empty, has an unsupported extension, or reports a
        non-positive / invalid duration.
    RuntimeError
        The audio backend failed to parse the file (corrupted or otherwise
        unreadable). The original exception is preserved as ``__cause__``.
    """
    path = validate_path(path)
    extension = path.suffix.lower()

    try:
        if extension in _SOUNDFILE_EXTENSIONS:
            duration = _duration_via_soundfile(path)
        else:  # extension in _MUTAGEN_EXTENSIONS
            duration = _duration_via_mutagen(path)
    except (FileNotFoundError, PermissionError, ValueError):
        # Preserve semantic exceptions raised by helpers.
        raise
    except (HeaderNotFoundError, _SoundfileError) as exc:
        raise RuntimeError(
            f"Audio backend could not parse file '{path}': {exc}"
        ) from exc
    except Exception as exc:  # last-resort safety net
        raise RuntimeError(
            f"Unexpected error while reading '{path}': {exc}"
        ) from exc

    if duration <= 0:
        raise ValueError(
            f"Detected non-positive duration ({duration}s) — file may be "
            f"corrupted: {path}"
        )

    return float(duration)


def format_duration(seconds: float, include_milliseconds: bool = True) -> str:
    """Format a duration in seconds as ``HH:MM:SS`` or ``HH:MM:SS.mmm``.

    This function never mutates or replaces the underlying float; it only
    produces a display string.

    Parameters
    ----------
    seconds : float
        Non-negative duration in seconds.
    include_milliseconds : bool, default True
        If ``True``, append ``.mmm`` (three-digit milliseconds); precision
        is preserved down to the millisecond. If ``False``, return
        ``HH:MM:SS`` (seconds are rounded to the nearest integer).

    Returns
    -------
    str
        Formatted duration.

    Raises
    ------
    ValueError
        ``seconds`` is negative or NaN.
    """
    if math.isnan(seconds):
        raise ValueError("Duration is NaN.")
    if seconds < 0:
        raise ValueError(f"Duration must be non-negative, got {seconds}.")

    if include_milliseconds:
        total_ms = int(round(seconds * 1000))
        hours, remainder = divmod(total_ms, 3_600_000)
        minutes, remainder = divmod(remainder, 60_000)
        secs, millis = divmod(remainder, 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

    total_secs = int(round(seconds))
    hours, remainder = divmod(total_secs, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


# ---------------------------------------------------------------------------
# Manual/CLI entry point — only runs when this file is executed directly.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Print the duration of a WAV or MP3 audio file."
    )
    parser.add_argument("path", help="Path to a WAV or MP3 file")
    parser.add_argument(
        "--no-ms",
        action="store_true",
        help="Format HH:MM:SS without milliseconds",
    )
    args = parser.parse_args()

    try:
        duration_seconds = get_duration(args.path)
    except (FileNotFoundError, PermissionError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    formatted = format_duration(duration_seconds, include_milliseconds=not args.no_ms)

    print("========== AUDIO INFORMATION ==========")
    print(f"Audio File          : {args.path}")
    print(f"Duration (seconds)  : {duration_seconds}")
    print(f"Duration (HH:MM:SS) : {formatted}")
    print("=======================================")
