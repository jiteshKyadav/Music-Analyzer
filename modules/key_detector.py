"""
modules/key_detector.py

Musical Key Detection using Essentia KeyExtractor.

Essentia Version:
    2.1-beta6-dev
"""

import essentia.standard as es


SUPPORTED_PROFILES = [
    "bgate",
    "edma",
    "temperley",
    "krumhansl"
]


def detect_key(
    audio,
    sample_rate=44100,
    tuning_frequency=440.0,
    profile_type="bgate",
    frame_size=4096,
    hop_size=4096,
    hpcp_size=12,
    window_type="hann",
    weight_type="cosine",
    average_detuning_correction=True
):
    """
    Detect musical key using Essentia KeyExtractor.

    Returns
    -------
    dict
    """

    if profile_type not in SUPPORTED_PROFILES:
        raise ValueError(
            f"Unsupported profile '{profile_type}'. "
            f"Choose from {SUPPORTED_PROFILES}"
        )

    extractor = es.KeyExtractor(
        sampleRate=sample_rate,
        profileType=profile_type,
        frameSize=frame_size,
        hopSize=hop_size,
        hpcpSize=hpcp_size,
        windowType=window_type,
        weightType=weight_type,
        tuningFrequency=tuning_frequency,
        averageDetuningCorrection=average_detuning_correction
    )

    # Official API
    key, scale, strength = extractor(audio)

    return {
        "key": key,
        "scale": scale,
        "strength": float(strength),

        "profile": profile_type,

        "tuning_frequency": float(tuning_frequency),

        "sample_rate": sample_rate,

        "frame_size": frame_size,
        "hop_size": hop_size,
        "hpcp_size": hpcp_size,

        "window_type": window_type,
        "weight_type": weight_type
    }


def detect_key_all_profiles(
    audio,
    sample_rate,
    tuning_frequency
):
    """
    Run KeyExtractor with all supported profiles.

    Returns
    -------
    list
    """

    results = []

    for profile in SUPPORTED_PROFILES:

        try:

            result = detect_key(
                audio=audio,
                sample_rate=sample_rate,
                tuning_frequency=tuning_frequency,
                profile_type=profile
            )

            results.append(result)

        except Exception as e:

            results.append({
                "profile": profile,
                "error": str(e)
            })

    return results


def get_best_profile(results):
    """
    Return profile with highest strength.
    """

    valid = [
        r for r in results
        if "strength" in r
    ]

    if not valid:
        return None

    return max(valid, key=lambda x: x["strength"])