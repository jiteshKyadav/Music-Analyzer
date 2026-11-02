"""
validation.py
=============

Shared validation utilities for the Music Analyzer project.

This module contains common validation logic used by all audio-analysis
modules before processing begins.

Validation performed
--------------------
- File exists
- Regular file
- Read permission
- Non-empty
- Supported extension

The module intentionally does NOT decode audio or detect corruption.
Decoding belongs to the individual analysis modules.
"""

from __future__ import annotations

from pathlib import Path

AudioPath = str | Path

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({
    ".wav",
    ".mp3",
})


def validate_path(path: AudioPath) -> Path:
    """
    Validate an audio file path and return a normalized Path object.

    Parameters
    ----------
    path : str | Path
        Path to an audio file.

    Returns
    -------
    pathlib.Path
        Validated Path object.

    Raises
    ------
    FileNotFoundError
        File does not exist.

    ValueError
        Path is not a file,
        file is empty,
        or extension is unsupported.

    PermissionError
        File cannot be accessed.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Audio file does not exist: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Path is not a regular file: {path}"
        )

    try:
        size = path.stat().st_size
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot access file: {path}"
        ) from exc

    if size == 0:
        raise ValueError(
            f"Audio file is empty (0 bytes): {path}"
        )

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension '{extension}'. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}."
        )

    return path