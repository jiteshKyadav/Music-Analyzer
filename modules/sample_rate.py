"""
sample_rate.py
==============

Sample rate detection module for the Music Analyzer.

This module inspects only the header/metadata of an audio file and never
decodes the full waveform, which makes it fast, memory-cheap, and safe to
call on very long tracks.

Public API
----------
- ``get_sample_rate(path)`` returns the exact sample rate in Hz as an
                            ``int`` (no estimation, no resampling).

The function is pure and side-effect free. Import it directly into the
main analyzer pipeline; nothing runs on import.

Supported formats
-----------------
- WAV  (parsed by ``soundfile``)
- MP3  (parsed by ``mutagen``)

Adding FLAC later is a one-line change: append ``".flac"`` to
``_SOUNDFILE_EXTENSIONS`` — ``soundfile`` already supports it natively.
"""

from __future__ import annotations

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


def _sample_rate_via_soundfile(path: Path) -> int:
    """Read a WAV (or other soundfile-supported) header and return sample rate.

    ``soundfile.info`` reads only the header, so this is O(1) in file size.
    """
    info = sf.info(str(path))
    if info.samplerate <= 0:
        raise ValueError(
            f"Invalid audio header: samplerate={info.samplerate}"
        )
    return int(info.samplerate)


def _sample_rate_via_mutagen(path: Path) -> int:
    """Read an MP3 header and return its reported sample rate.

    ``mutagen.mp3.MP3`` parses the MPEG frame header, which encodes the
    sample rate directly (32000 / 44100 / 48000 Hz for MPEG-1;
    16000 / 22050 / 24000 for MPEG-2; 8000 / 11025 / 12000 for MPEG-2.5).
    No decoding is performed.
    """
    audio = MP3(str(path))
    # ``MP3()`` either parses the MPEG frame header successfully — populating
    # ``info.sample_rate`` — or raises ``HeaderNotFoundError``. A ``None``
    # value is not a documented state, so no defensive check is needed; the
    # ``<= 0`` guard below is the meaningful sanity check.
    sample_rate = int(audio.info.sample_rate)
    if sample_rate <= 0:
        raise ValueError(f"Invalid MP3 sample rate: {sample_rate}")
    return sample_rate


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_sample_rate(path: AudioPath) -> int:
    """Return the exact sample rate of an audio file, in hertz.

    The value is read directly from the file's header — no decoding, no
    estimation, no resampling. For a 44.1 kHz WAV the return value is
    exactly ``44100``.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a supported audio file (WAV or MP3).

    Returns
    -------
    int
        Sample rate in hertz (e.g. ``44100``, ``48000``).

    Raises
    ------
    FileNotFoundError
        The file does not exist.
    PermissionError
        The file exists but cannot be read.
    ValueError
        The file is empty, has an unsupported extension, or reports an
        invalid / non-positive sample rate.
    RuntimeError
        The audio backend failed to parse the file (corrupted or otherwise
        unreadable). The original exception is preserved as ``__cause__``.
    """
    path = validate_path(path)
    extension = path.suffix.lower()
    
    try:
        if extension in _SOUNDFILE_EXTENSIONS:
            return _sample_rate_via_soundfile(path)
        # extension in _MUTAGEN_EXTENSIONS
        return _sample_rate_via_mutagen(path)
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


# ---------------------------------------------------------------------------
# Manual/CLI entry point — only runs when this file is executed directly.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Print the sample rate of a WAV or MP3 audio file."
    )
    parser.add_argument("path", help="Path to a WAV or MP3 file")
    args = parser.parse_args()

    try:
        sample_rate = get_sample_rate(args.path)
    except (FileNotFoundError, PermissionError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print("========== AUDIO INFORMATION ==========")
    print(f"Audio File          : {args.path}")
    print(f"Sample Rate (Hz)    : {sample_rate}")
    print("=======================================")
