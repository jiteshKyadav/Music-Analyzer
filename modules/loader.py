"""
modules/loader.py

Audio loading module for the Essentia Key Detection project.
"""

from pathlib import Path
import essentia.standard as es


TARGET_SAMPLE_RATE = 44100


def load_audio(file_path: str):
    """
    Load an audio file in mono.

    Returns
    -------
    audio : numpy.ndarray
        Mono audio signal.

    sample_rate : int
        Analysis sample rate.

    duration : float
        Audio duration in seconds.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not path.is_file():
        raise FileNotFoundError(f"Not a valid file: {file_path}")

    try:
        loader = es.MonoLoader(
            filename=str(path),
            sampleRate=TARGET_SAMPLE_RATE
        )

        audio = loader()

    except Exception as e:
        raise RuntimeError(f"Essentia failed to load audio.\n{e}")

    if len(audio) == 0:
        raise RuntimeError("Loaded audio is empty.")

    duration = len(audio) / TARGET_SAMPLE_RATE

    return audio, TARGET_SAMPLE_RATE, duration