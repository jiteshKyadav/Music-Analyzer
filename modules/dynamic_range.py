"""
dynamic_range.py
================

Dynamic Range measurement module for the Music Analyzer.

The metric reported by this module is **Peak to Loudness Ratio (PLR)**:

    Dynamic Range (dB) = True Peak (dBTP) − Integrated Loudness (LUFS)

Both terms are computed per ITU-R BS.1770-4 / EBU R128:

- **Integrated Loudness** is measured by ``pyloudnorm`` — the reference
  Python implementation of BS.1770-4 (K-weighting, 400 ms blocks with
  75 % overlap, two-stage gating).
- **True Peak** is measured following the signal chain in BS.1770-4
  annex 2: 4× polyphase oversampling (with the anti-alias low-pass
  filter built into ``scipy.signal.resample_poly``), then maximum
  absolute magnitude across all channels and time, then dBTP via
  ``20 · log10``. The 4× factor matches the convention used by
  ``ffmpeg``'s ebur128 filter and other reference implementations.

Public API
----------
- ``get_dynamic_range(path)`` returns PLR in dB as a ``float``.

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
from scipy.signal import resample_poly

# ``soundfile.LibsndfileError`` was introduced in soundfile 0.12. On older
# installs the backend raises plain ``RuntimeError``; the getattr keeps the
# module importable across versions without behavioural change.
_SoundfileError = getattr(sf, "LibsndfileError", RuntimeError)

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------


# WAV and MP3 both decode through ``soundfile.read``; the measurement chain
# then operates on the resulting float samples regardless of source format,
# so no per-format backend split is needed in this module.


# BS.1770-4 annex 2 specifies oversampling to at least 192 kHz. A 4× factor
# reaches 192 kHz from 48 kHz sources and 176.4 kHz from 44.1 kHz sources;
# the small shortfall at 44.1 kHz is the widely accepted industry compromise
# used by ffmpeg's ebur128 filter, libebur128, and Essentia's true-peak
# detector, and is the reason 4× is treated here as the canonical factor.
_TRUE_PEAK_OVERSAMPLE_FACTOR = 4


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Decode an audio file to floating-point samples.

    Uses ``soundfile.read`` with an explicit ``dtype="float64"`` request,
    which returns samples in the range [-1.0, 1.0] and preserves the
    file's native sample rate — no resampling is performed here. The
    explicit dtype guarantees the precision downstream measurements
    expect regardless of any future change to soundfile's default. Mono
    files come back as shape ``(frames,)`` and multichannel files as
    ``(frames, channels)``.

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
    # silently propagate through the K-weighting filter and the oversampler
    # and produce meaningless measurements, so we reject them up front.
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


def _get_true_peak(data: np.ndarray) -> float:
    """Measure True Peak in dBTP per ITU-R BS.1770-4 annex 2.

    Implements the standard's True Peak signal chain:

    1. Oversample the signal by 4× using a polyphase filter. This is done
       with ``scipy.signal.resample_poly``, whose built-in FIR low-pass
       satisfies the anti-alias requirement of the annex.
    2. Take the maximum absolute magnitude across all channels and time.
       Per BS.1770, the True Peak of a multichannel programme is the
       highest single-channel peak, which ``np.max(np.abs(...))``
       computes directly on the flattened absolute values.
    3. Convert to dBTP as ``20 · log10(peak)``.

    Fully silent input has zero magnitude; the log10 of zero would be
    ``-inf``. A small floor is applied to the peak magnitude so that
    the return type stays a finite float in that edge case.
    """
    # Resample along the sample-time axis. For a 1-D mono array this is
    # axis 0 (the only axis); for a 2-D (frames, channels) array this is
    # also axis 0 — resample_poly then processes each channel independently.
    oversampled = resample_poly(
        data, up=_TRUE_PEAK_OVERSAMPLE_FACTOR, down=1, axis=0
    )
    peak_magnitude = float(np.max(np.abs(oversampled)))
    # Floor at 1e-12 (-240 dBTP) to keep the log defined for absolute
    # silence. Real signals are many orders of magnitude above this.
    peak_magnitude = max(peak_magnitude, 1e-12)
    return 20.0 * float(np.log10(peak_magnitude))


def _get_dynamic_range(true_peak_dbtp: float, integrated_lufs: float) -> float:
    """Compute PLR (dB) from True Peak (dBTP) and Integrated Loudness (LUFS)."""
    return float(true_peak_dbtp - integrated_lufs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_dynamic_range(path: AudioPath) -> float:
    """Return the Peak to Loudness Ratio (PLR) of an audio file, in dB.

    The file is decoded once via ``soundfile.read``; the same sample
    array is then passed to both the loudness meter and the True Peak
    chain, so the audio is never loaded twice.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a supported audio file (WAV or MP3).

    Returns
    -------
    float
        Dynamic Range in dB, defined as
        ``True Peak (dBTP) - Integrated Loudness (LUFS)``.
        Example: ``12.84``. For fully silent input the loudness term is
        ``-inf`` and the return is ``+inf``, signalling "no measurable
        programme content"; callers should treat this as an unmeasurable
        file rather than a real dynamic range.

    Raises
    ------
    FileNotFoundError
        The file does not exist.
    PermissionError
        The file exists but cannot be read.
    ValueError
        The file is empty, has an unsupported extension, or decodes to
        no samples / an invalid sample rate / non-finite samples.
    RuntimeError
        The audio backend failed to decode the file, or one of the
        measurement stages failed. The original exception is preserved
        as ``__cause__``.
    """
    path = validate_path(path)

    # --- Decode stage (runs exactly once) ---------------------------------
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
    # Both measurements operate on the SAME decoded sample array — no
    # reload, no duplicate decoding, no unnecessary copies.
    try:
        loudness_lufs = _get_integrated_loudness(data, sample_rate)
        true_peak_dbtp = _get_true_peak(data)
    except Exception as exc:  # covers pyloudnorm-side / resampler failures
        raise RuntimeError(
            f"Dynamic range measurement failed for '{path}': {exc}"
        ) from exc

    return _get_dynamic_range(true_peak_dbtp, loudness_lufs)


# ---------------------------------------------------------------------------
# Manual/CLI entry point — only runs when this file is executed directly.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description=(
            "Print the Dynamic Range (PLR: True Peak dBTP minus Integrated "
            "Loudness LUFS) of a WAV or MP3 audio file, per ITU-R BS.1770-4."
        )
    )
    parser.add_argument("path", help="Path to a WAV or MP3 file")
    args = parser.parse_args()

    try:
        dynamic_range = get_dynamic_range(args.path)
    except (FileNotFoundError, PermissionError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print("========== AUDIO INFORMATION ==========")
    print(f"Audio File          : {args.path}")
    print(f"Dynamic Range (PLR) : {dynamic_range:.2f} dB")
    print("=======================================")
