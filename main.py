from pathlib import Path

from modules.bpm import BPMDetector
from modules.key_detector import detect_key_all_profiles, get_best_profile
from modules.loader import load_audio
from modules.tuning import estimate_tuning_frequency

from modules.duration import get_duration, format_duration
from modules.sample_rate import get_sample_rate
from modules.loudness import get_loudness
from modules.dynamic_range import get_dynamic_range

from utils.preprocessor import validate_audio
from modules.genre import predict_genre

def analyze_audio(file_path, progress_callback=None):
    """Run the complete Music Analyzer pipeline.

    ``progress_callback`` is optional so the existing CLI/backend API remains
    compatible. When supplied, it receives ``(percent, status)`` updates.
    """

    def report(percent, status):
        if progress_callback is not None:
            progress_callback(percent, status)

    # --------------------------------------------------
    # 1. Validate
    # --------------------------------------------------
    report(3, "Validating audio...")
    validate_audio(file_path)
    path = Path(file_path)

    print("\nLoading audio...")

    # --------------------------------------------------
    # 2. Basic information
    # --------------------------------------------------
    report(8, "Reading audio information...")
    duration = get_duration(path)
    sample_rate = get_sample_rate(path)

    print("Done.")
    print(f"Duration   : {duration:.2f} sec")
    print(f"Sample Rate: {sample_rate} Hz")
    report(15, "Audio information loaded")

    # --------------------------------------------------
    # 3. Loudness
    # --------------------------------------------------
    report(18, "Calculating loudness...")
    print("\nCalculating loudness...")
    loudness = get_loudness(path)
    report(30, "Loudness calculated")

    # --------------------------------------------------
    # 4. Dynamic Range
    # --------------------------------------------------
    report(33, "Calculating dynamic range...")
    print("Calculating dynamic range...")
    dynamic_range = get_dynamic_range(path)
    report(45, "Dynamic range calculated")

    # --------------------------------------------------
    # 5. BPM
    # --------------------------------------------------
    report(48, "Analyzing BPM...")
    print("\nRunning BPM analysis...")

    bpm_detector = BPMDetector()
    bpm_result = bpm_detector.analyze(str(path))
    report(62, "BPM analysis complete")

    # --------------------------------------------------
    # 6. Load audio for Key analysis
    # --------------------------------------------------
    report(65, "Preparing key analysis...")
    print("\nLoading audio for key analysis...")

    audio, analysis_sr, _ = load_audio(str(path))
    report(70, "Audio prepared for key analysis")

    # --------------------------------------------------
    # 7. Tuning estimation
    # --------------------------------------------------
    report(72, "Estimating tuning frequency...")
    print("Estimating tuning frequency...")

    tuning_frequency = estimate_tuning_frequency(audio)

    print(f"Tuning : {tuning_frequency:.2f} Hz")
    report(78, "Tuning estimated")

    # --------------------------------------------------
    # 8. Key detection — all profiles
    # --------------------------------------------------
    report(80, "Detecting key and scale...")
    print("\nRunning all key profiles...")

    key_results = detect_key_all_profiles(
        audio=audio,
        sample_rate=analysis_sr,
        tuning_frequency=tuning_frequency
    )
    report(94, "Key and scale detected")

    # --------------------------------------------------
    # 9. Best key profile
    # --------------------------------------------------
    report(97, "Selecting best result...")
    best_key = get_best_profile(key_results)

    # --------------------------------------------------
    # 10. Genre
    # --------------------------------------------------
    report(98, "Predicting genre...")
    print("\nRunning genre analysis...")

    genre_result = predict_genre(path)

    print(
        f"Genre      : {genre_result['genre']}"
    )
    print(
        f"Confidence : {genre_result['confidence']:.4f}"
    )

    report(99, "Genre predicted")

    # --------------------------------------------------
    # 11. Final result
    # --------------------------------------------------
    results = {
        "file": str(path),
        "duration": duration,
        "sample_rate": sample_rate,
        "loudness": loudness,
        "dynamic_range": dynamic_range,

        "bpm": bpm_result,

        "key_profiles": key_results,
        "best_key": best_key,

        "tuning_frequency": tuning_frequency,
        "genre": genre_result,
    }

    report(100, "Analysis complete")
    return results


def print_final_results(results):
    print("\n" + "=" * 70)
    print("                 MUSIC ANALYZER")
    print("=" * 70)

    print("\n[AUDIO]")
    print(f"File              : {results['file']}")
    print(f"Duration          : {format_duration(results['duration'])}")
    print(f"Sample Rate       : {results['sample_rate']} Hz")

    print("\n[AUDIO FEATURES]")
    print(f"Integrated Loudness: {results['loudness']:.2f} LUFS")
    print(f"Dynamic Range     : {results['dynamic_range']:.2f} dB")

    print("\n[RHYTHM]")
    print(f"BPM               : {results['bpm']['bpm']:.2f}")
    print(f"Confidence        : {results['bpm']['confidence']:.3f}")
    print(f"Custom Confidence : {results['bpm']['custom_confidence']:.3f}")

    print("\n[KEY]")
    best = results["best_key"]

    if best:
        print(f"Key               : {best['key']}")
        print(f"Scale             : {best['scale']}")
        print(f"Strength          : {best['strength']:.4f}")
        print(f"Profile           : {best['profile']}")
    else:
        print("Key               : Not detected")
    print(f"Tuning Frequency  : {results['tuning_frequency']:.2f} Hz")

    print("\n" + "=" * 70)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Music Analyzer"
    )

    parser.add_argument(
        "audio_file",
        help="Path to WAV or MP3 audio file"
    )

    args = parser.parse_args()

    try:
        results = analyze_audio(args.audio_file)
        print_final_results(results)

    except Exception as exc:
        print(f"\nERROR: {exc}")


if __name__ == "__main__":
    main()




# source .venv/bin/activate
# python -m py_compile gui/app.py
# python gui/app.py

