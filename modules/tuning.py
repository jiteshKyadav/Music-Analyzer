"""
modules/tuning.py

Tuning frequency estimation using Essentia.

Essentia Version:
    2.1-beta6-dev
"""

import essentia.standard as es
import numpy as np

DEFAULT_TUNING = 440.0


def estimate_tuning_frequency(
    audio,
    frame_size: int = 4096,
    hop_size: int = 2048
) -> float:
    """
    Estimate tuning frequency of an audio signal.

    Parameters
    ----------
    audio : numpy.ndarray
        Mono audio signal.

    frame_size : int
        Frame size used by TuningFrequencyExtractor.

    hop_size : int
        Hop size used by TuningFrequencyExtractor.

    Returns
    -------
    float
        Estimated tuning frequency in Hz.

    Notes
    -----
    Falls back to 440 Hz if tuning estimation fails.
    """

    try:

        extractor = es.TuningFrequencyExtractor(
            frameSize=frame_size,
            hopSize=hop_size
        )

        tuning = extractor(audio)

        # Convert to numpy array
        tuning = np.asarray(tuning, dtype=np.float32)

        # Remove invalid values
        tuning = tuning[tuning > 0]

        if len(tuning) == 0:
            return DEFAULT_TUNING

        # Median is more robust than mean
        estimated_tuning = float(np.median(tuning))

        return estimated_tuning

    except Exception as e:

        print(f"[WARNING] Tuning estimation failed: {e}")
        print(f"[INFO] Using default tuning: {DEFAULT_TUNING} Hz")

        return DEFAULT_TUNING