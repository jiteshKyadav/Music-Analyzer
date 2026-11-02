"""
modules/printer.py

Formatted console output for Essentia Key Analyzer.
"""


def print_analysis_results(
    audio_file,
    duration,
    sample_rate,
    result
):
    """
    Print formatted key detection results.
    """

    print("\n" + "=" * 60)
    print("ESSENTIA KEY ANALYZER")
    print("=" * 60)

    print(f"Audio File        : {audio_file}")
    print(f"Duration          : {duration:.2f} sec")
    print(f"Sample Rate       : {sample_rate} Hz")

    print("-" * 60)

    print(f"Detected Key      : {result['key']}")
    print(f"Detected Scale    : {result['scale']}")
    print(f"Strength          : {result['strength']:.4f}")

    print("-" * 60)

    print(f"Tuning Frequency  : {result['tuning_frequency']:.2f} Hz")
    print(f"Profile           : {result['profile']}")

    print("-" * 60)

    print(f"Frame Size        : {result['frame_size']}")
    print(f"Hop Size          : {result['hop_size']}")
    print(f"HPCP Size         : {result['hpcp_size']}")
    print(f"Window Type       : {result['window_type']}")
    print(f"Weight Type       : {result['weight_type']}")

    print("=" * 60)


def print_profile_comparison(results):
    """
    Print comparison of all profiles.
    """

    print("\n")
    print("=" * 80)
    print("PROFILE COMPARISON")
    print("=" * 80)

    print(
        f"{'PROFILE':<15}"
        f"{'KEY':<8}"
        f"{'SCALE':<10}"
        f"{'STRENGTH'}"
    )

    print("-" * 80)

    best = None
    best_strength = -1

    for r in results:

        if "error" in r:
            print(f"{r['profile']:<15} ERROR")
            continue

        print(
            f"{r['profile']:<15}"
            f"{r['key']:<8}"
            f"{r['scale']:<10}"
            f"{r['strength']:.4f}"
        )

        if r["strength"] > best_strength:
            best_strength = r["strength"]
            best = r

    print("-" * 80)

    if best:
        print(
            f"\nBest Profile : {best['profile']}"
        )
        print(
            f"Detected Key : {best['key']} {best['scale']}"
        )
        print(
            f"Strength     : {best['strength']:.4f}"
        )

    print("=" * 80)