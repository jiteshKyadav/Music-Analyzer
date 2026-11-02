"""
loudness.py
===========

Integrated loudness (LUFS) measurement module for the Music Analyzer.

Implements the ITU-R BS.1770-4 loudness algorithm as specified by
EBU R128, via the ``pyloudnorm`` library. The reported value is
**Integrated Loudness** in LUFS — not peak, not RMS, not dBFS, not
average amplitude.

Public API
----------
- ``get_loudness(path)`` returns the integrated loudness of an audio
                        file in LUFS as a ``float``.

The function is pure and side-effect free. Import it directly into the
main analyzer pipeline; nothing runs on import.

Supported formats
-----------------
- WAV
- MP3  MP3 decoding depends on the linked libsndfile build. Support was
       introduced in libsndfile 1.1.0 (2022), but the codec is a
       compile-time option and is not enabled in every OS package or
       wheel. When it is unavailable, ``soundfile.read`` raises a
       ``LibsndfileError``, which this module wraps as ``RuntimeError``
       with the original exception preserved as ``__cause__``.

Adding FLAC later requires updating the shared validation module.

Standards references
--------------------
- ITU-R BS.1770-4  — Algorithms to measure audio programme loudness
- EBU R128         — Loudness normalisation and permitted maximum level

The default ``pyln.Meter`` configuration implements BS.1770-4 exactly:
K-weighting pre-filter, 400 ms measurement blocks with 75 % overlap, and
the two-stage gating (-70 LUFS absolute, -10 LU relative). These are not
overridden anywhere in this module.
"""

from __future__ import annotations

from pathlib import Path

from utils.validation import (
    AudioPath,
    validate_path,
)

import numpy as np
import pyloudnorm as pyln
import soundfile as sf

# ``soundfile.LibsndfileError`` was introduced in soundfile 0.12. On older
# installs the backend raises plain ``RuntimeError``; the getattr keeps the
# module importable across versions without behavioural change.
_SoundfileError = getattr(sf, "LibsndfileError", RuntimeError)

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------


# WAV and MP3 both decode through ``soundfile.read``; ``pyloudnorm`` then
# operates on the resulting float samples regardless of source format, so
# no per-format backend split is needed in this module.


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Decode an audio file to floating-point samples.

    Uses ``soundfile.read`` with an explicit ``dtype="float64"`` request,
    which returns samples in the range [-1.0, 1.0] and preserves the
    file's native sample rate — no resampling is performed. The explicit
    dtype guarantees the precision ``pyloudnorm`` expects regardless of
    any future change to soundfile's default. Mono files come back as
    shape ``(frames,)`` and multichannel files as ``(frames, channels)``,
    which is exactly the layout ``pyloudnorm`` expects.

    The file is decoded **exactly once**; callers must not reload it.

    Returns
    -------
    tuple[numpy.ndarray, int]
        The decoded sample array and the native sample rate in hertz.

    Raises
    ------
    ValueError
        The decoded stream is empty, reports a non-positive sample rate,
        or contains non-finite samples (NaN or Inf).
    """
    data, sample_rate = sf.read(str(path), always_2d=False, dtype="float64")
    if data.size == 0:
        raise ValueError(f"Audio file decoded to zero samples: {path}")
    if sample_rate <= 0:
        raise ValueError(
            f"Invalid sample rate reported by decoder: {sample_rate}"
        )
    # Guard against pathological decoder output. NaN or Inf samples would
    # silently propagate through the K-weighting filter and produce a
    # meaningless loudness value, so we reject them before measurement.
    if not np.isfinite(data).all():
        raise ValueError(
            f"Decoded audio contains non-finite samples (NaN or Inf): {path}"
        )
    return data, int(sample_rate)


def _get_integrated_loudness(data: np.ndarray, sample_rate: int) -> float:
    """Measure integrated loudness (LUFS) per ITU-R BS.1770-4.

    Constructs the default ``pyln.Meter`` for the given sample rate — the
    reference BS.1770 implementation with K-weighting, 400 ms blocks,
    75 % overlap, and two-stage gating. No parameters are overridden.

    Silent input yields ``-inf``, which is a valid measurement per the
    standard and is returned unchanged.
    """
    meter = pyln.Meter(sample_rate)
    loudness = meter.integrated_loudness(data)
    return float(loudness)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_loudness(path: AudioPath) -> float:
    """Return the ITU-R BS.1770-4 integrated loudness of an audio file, in LUFS.

    The file is decoded once via ``soundfile.read`` and measured by the
    default ``pyloudnorm`` meter, which implements the standard exactly.
    The returned value is a signed float in LUFS; fully silent input
    yields ``-inf``, per the standard.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a supported audio file (WAV or MP3).

    Returns
    -------
    float
        Integrated loudness in LUFS (e.g. ``-13.84``).

    Raises
    ------
    FileNotFoundError
        The file does not exist.
    PermissionError
        The file exists but cannot be read.
    ValueError
        The file is empty, has an unsupported extension, or decodes to no
        samples / an invalid sample rate.
    RuntimeError
        The audio backend failed to decode the file, or the loudness
        measurement itself failed. The original exception is preserved as
        ``__cause__``.
    """
    path = validate_path(path)

    # --- Decode stage ------------------------------------------------------
    try:
        data, sample_rate = _load_audio(path)
    except (FileNotFoundError, PermissionError, ValueError):
        # Preserve semantic exceptions raised by the loader / validator.
        raise
    except _SoundfileError as exc:
        raise RuntimeError(
            f"Audio backend could not decode file '{path}': {exc}"
        ) from exc
    except Exception as exc:  # last-resort safety net for the decode stage
        raise RuntimeError(
            f"Unexpected error while decoding '{path}': {exc}"
        ) from exc

    # --- Measurement stage -------------------------------------------------
    try:
        return _get_integrated_loudness(data, sample_rate)
    except Exception as exc:  # covers pyloudnorm-side / meter failures
        raise RuntimeError(
            f"Loudness measurement failed for '{path}': {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Manual/CLI entry point — only runs when this file is executed directly.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description=(
            "Print the ITU-R BS.1770-4 integrated loudness (LUFS) of a "
            "WAV or MP3 audio file."
        )
    )
    parser.add_argument("path", help="Path to a WAV or MP3 file")
    args = parser.parse_args()

    try:
        lufs = get_loudness(args.path)
    except (FileNotFoundError, PermissionError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print("========== AUDIO INFORMATION ==========")
    print(f"Audio File          : {args.path}")
    print(f"Integrated Loudness : {lufs:.2f} LUFS")
    print("=======================================")