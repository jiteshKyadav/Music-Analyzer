"""
Audio preprocessing entry point for the Music Analyzer project.

This module performs lightweight audio validation before feature
extraction begins. It delegates path validation to `utils.validation`
and performs minimal audio decoding only to detect unsuitable input
such as completely silent or very short audio.

## Public API

- `validate_audio(path)`

If validation succeeds, the function returns `True`.
Otherwise, it raises the appropriate exception.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from utils.validation import AudioPath, validate_path


# Minimum duration required for reliable BPM and key analysis.
MIN_ANALYSIS_DURATION = 5.0


def validate_audio(path: AudioPath) -> bool:
    """
    Validate an audio file before analysis.

    Validation includes:
    - Path/file validation via `validate_path`
    - Audio decoding
    - Non-empty decoded audio
    - Valid sample rate
    - Completely silent audio detection
    - Minimum duration check

    Parameters
    ----------
    path : str | pathlib.Path
        Path to an audio file.

    Returns
    -------
    bool
        Always returns True if validation succeeds.

    Raises
    ------
    FileNotFoundError
        Audio file does not exist.

    PermissionError
        Audio file cannot be accessed.

    ValueError
        File is empty, unsupported, unreadable, completely silent,
        or shorter than the minimum analysis duration.

    RuntimeError
        Audio backend cannot decode the file.
    """

    # --------------------------------------------------
    # 1. Path-level validation
    # --------------------------------------------------
    validated_path = validate_path(path)

    # --------------------------------------------------
    # 2. Decode audio for content-level validation
    # --------------------------------------------------
    try:
        audio, sample_rate = sf.read(
            str(validated_path),
            always_2d=False,
            dtype="float64",
        )
    except Exception as exc:
        raise RuntimeError(
            f"Audio file could not be decoded: {validated_path}"
        ) from exc

    # --------------------------------------------------
    # 3. Validate decoded audio
    # --------------------------------------------------
    if audio.size == 0:
        raise ValueError(
            f"Audio file decoded to zero samples: {validated_path}"
        )

    if sample_rate <= 0:
        raise ValueError(
            f"Invalid sample rate reported by decoder: {sample_rate}"
        )

    if not np.isfinite(audio).all():
        raise ValueError(
            f"Decoded audio contains non-finite samples (NaN or Inf): "
            f"{validated_path}"
        )

    # --------------------------------------------------
    # 4. Calculate duration
    # --------------------------------------------------
    duration = audio.shape[0] / sample_rate

    # --------------------------------------------------
    # 5. Completely silent audio
    # --------------------------------------------------
    if np.all(audio == 0):
        raise ValueError(
            f"Audio is completely silent; analysis cannot be performed: "
            f"{validated_path}"
        )

    # --------------------------------------------------
    # 6. Minimum duration
    # --------------------------------------------------
    if duration < MIN_ANALYSIS_DURATION:
        raise ValueError(
            f"Audio is too short for reliable analysis "
            f"({duration:.2f}s). Minimum required duration is "
            f"{MIN_ANALYSIS_DURATION:.1f}s: {validated_path}"
        )

    return True