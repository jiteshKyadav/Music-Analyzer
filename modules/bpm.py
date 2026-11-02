#!/usr/bin/env python3
"""
Professional BPM Detector using Essentia
Focus: Maximum accuracy through proper use of RhythmExtractor2013
"""

import os
import sys
import time
import logging
from typing import Tuple, List, Dict, Any, Optional
from pathlib import Path

import numpy as np
import essentia
import essentia.standard as es
LOG_Level = logging.INFO
# Configure logging
logging.basicConfig(
    level=LOG_Level,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("BPMDetector")


class BPMDetector:
    """
    Professional BPM detector using Essentia's RhythmExtractor2013.
    Implements robust preprocessing, validation, and result verification.
    """
    
    # Valid BPM range for music
    MIN_BPM = 40
    MAX_BPM = 220
    
    # Confidence threshold for reliable estimates
    CONFIDENCE_THRESHOLD = 0.3
    
    # Short clip threshold in seconds
    SHORT_CLIP_DURATION = 5.0
    
    # Silence threshold in dB (relative to full scale)
    SILENCE_THRESHOLD_DB = -60.0
    
    def __init__(self, target_sample_rate: int = 44100):
        """
        Initialize the BPM detector.
        
        Args:
            target_sample_rate: Target sample rate for analysis (RhythmExtractor2013 works best at 44100 Hz)
        """
        self.target_sample_rate = target_sample_rate
        logger.info(f"BPM Detector initialized (target sample rate: {target_sample_rate} Hz)")
    
    def load_audio(self, filename: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file and return mono audio array with actual sample rate.
        
        Args:
            filename: Path to audio file
            
        Returns:
            Tuple of (audio_array, sample_rate)
            
        Raises:
            FileNotFoundError: If file doesn't exist
            RuntimeError: If loading fails
        """
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"Audio file not found: {filename}")
        
        logger.info(f"Loading audio file: {filename}")
        
        try:
            # Use AudioLoader to get actual sample rate and other metadata
            # Returns: audio, sample_rate, number_channels, bit_rate, codec, md5
            loader = es.AudioLoader(filename=filename)
            audio, sample_rate, _, _, _, _ = loader()
            
            # Convert to mono if stereo
            if len(audio.shape) > 1 and audio.shape[1] > 1:
                logger.info(f"Converting stereo to mono (original channels: {audio.shape[1]})")
                audio = np.mean(audio, axis=1)
            
            duration = len(audio) / sample_rate
            
            logger.info(f"File: {filename}")
            logger.info(f"Duration: {duration:.2f} seconds")
            logger.info(f"Actual sample rate: {sample_rate} Hz")
            logger.info(f"Samples: {len(audio)}")
            
            return audio, int(sample_rate)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load audio: {e}") from e
    
    def preprocess_audio(self, audio: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, int]:
        """
        Preprocess audio for optimal rhythm analysis.
        
        Steps:
        1. DC offset removal
        2. Amplitude normalization
        3. Silence trimming
        4. Resampling to target sample rate if needed
        
        Args:
            audio: Input audio array
            sample_rate: Current sample rate
            
        Returns:
            Tuple of (processed_audio, target_sample_rate)
        """
        logger.info("Preprocessing audio...")
        
        # 1. DC offset removal
        dc_offset = np.mean(audio)
        audio = audio - dc_offset
        logger.debug(f"Removed DC offset: {dc_offset:.6f}")
        
        # 2. Amplitude normalization to [-1, 1]
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak
            logger.debug(f"Normalized amplitude (peak was {peak:.4f})")
        
        # 3. Silence trimming
        audio = self._trim_silence(audio, self.SILENCE_THRESHOLD_DB)
        logger.debug(f"After silence trimming: {len(audio)} samples")
        
        # 4. Resample ONLY if sample rate differs from target
        if sample_rate != self.target_sample_rate:
            logger.info(f"Resampling from {sample_rate} Hz to {self.target_sample_rate} Hz")
            resampler = es.Resample(
                inputSampleRate=sample_rate,
                outputSampleRate=self.target_sample_rate
            )
            audio = resampler(audio)
            sample_rate = self.target_sample_rate
        else:
            logger.info(f"Sample rate already at target ({self.target_sample_rate} Hz), no resampling needed")
        
        logger.info(f"Preprocessing complete. Final length: {len(audio)} samples")
        return audio, sample_rate
    
    def _trim_silence(self, audio: np.ndarray, threshold_db: float) -> np.ndarray:
        """
        Trim leading and trailing silence from audio.
        
        Args:
            audio: Input audio array
            threshold_db: Silence threshold in dB (relative to full scale)
            
        Returns:
            Trimmed audio array
        """
        if len(audio) == 0:
            return audio
        
        # Convert dB threshold to linear amplitude
        threshold = 10 ** (threshold_db / 20.0)
        
        # Find non-silent samples
        non_silent_indices = np.where(np.abs(audio) > threshold)[0]
        
        if len(non_silent_indices) == 0:
            logger.warning("Audio appears to be completely silent")
            return audio
        
        start_idx = non_silent_indices[0]
        end_idx = non_silent_indices[-1] + 1
        
        return audio[start_idx:end_idx]
    
    def analyze_rhythm(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Perform rhythm analysis using RhythmExtractor2013.
        
        This algorithm internally uses both multifeature and degara methods
        and returns the most reliable estimate.
        
        Args:
            audio: Preprocessed audio array
            
        Returns:
            Dictionary containing rhythm analysis results
        """
        logger.info("Analyzing rhythm with RhythmExtractor2013...")
        
        try:
            # RhythmExtractor2013 returns 5 values:
            # bpm, beats, confidence, estimates, bpm_intervals
            rhythm_extractor = es.RhythmExtractor2013()

            bpm, beats, confidence, estimates, bpm_intervals = rhythm_extractor(audio)

            logger.info(f"First 30 beat timestamps: {beats[:30]}")
            logger.info(f"Total beats: {len(beats)}")
            
            logger.info(f"RhythmExtractor2013 completed:")
            logger.info(f"  BPM: {bpm:.2f}")
            logger.info(f"  Confidence: {confidence:.4f}")
            logger.info(f"  Number of beats: {len(beats)}")
            logger.info(f"Number of BPM estimates: {len(estimates)}")
            
            return {
                'bpm': bpm,
                'beats': beats,
                'confidence': confidence,
                'estimates': estimates,
                'bpm_intervals': bpm_intervals
            }
            
        except Exception as e:
            raise RuntimeError(f"Rhythm analysis failed: {e}") from e
    
    def analyze_estimate_distribution(self, estimates: List[float]) -> Dict[str, float]:
        """
        Analyze the distribution of BPM estimates.
        
        Args:
            estimates: List of BPM estimates from RhythmExtractor2013
            
        Returns:
            Dictionary containing distribution statistics
        """
        estimates = np.asarray(estimates)

        if estimates.size == 0:
            return {
                'median': 0.0,
                'variance': 0.0,
                'spread': 0.0,
                'agreement': 0.0
            }
        
        estimates_array = np.array(estimates)
        median = np.median(estimates_array)
        variance = np.var(estimates_array)
        spread = np.max(estimates_array) - np.min(estimates_array)
        
        # Calculate agreement as percentage of estimates within ±2 BPM of median
        if len(estimates) > 1:
            within_range = np.sum(np.abs(estimates_array - median) <= 2.0)
            agreement = (within_range / len(estimates)) * 100.0
        else:
            agreement = 100.0
        
        logger.info(f"Estimate distribution analysis:")
        logger.info(f"  Median: {median:.2f}")
        logger.info(f"  Variance: {variance:.4f}")
        logger.info(f"  Spread: {spread:.2f}")
        logger.info(f"  Agreement: {agreement:.1f}%")
        
        return {
            'median': median,
            'variance': variance,
            'spread': spread,
            'agreement': agreement
        }
    
    def analyze_beat_stability(self, beats: List[float]) -> Dict[str, float]:
        """
        Analyze beat interval stability.
        
        Args:
            beats: List of beat positions in seconds
            
        Returns:
            Dictionary containing beat stability metrics
        """
        if len(beats) < 2:
            return {
                'mean_interval': 0.0,
                'median_interval': 0.0,
                'std_interval': 0.0,
                'cv': 0.0,
                'interval_range': 0.0,
                'stability_score': 0.0
            }
        
        intervals = np.diff(beats)
        mean_interval = np.mean(intervals)
        median_interval = np.median(intervals)
        std_interval = np.std(intervals)
        cv = std_interval / mean_interval if mean_interval > 0 else float('inf')
        interval_range = np.max(intervals) - np.min(intervals)
        
        # Stability score: lower CV = higher stability (0 to 1)
        stability_score = max(0.0, 1.0 - cv)
        
        logger.info(f"Beat stability analysis:")
        logger.info(f"  Mean interval: {mean_interval:.4f}s")
        logger.info(f"  Median interval: {median_interval:.4f}s")
        logger.info(f"  Std interval: {std_interval:.4f}s")
        logger.info(f"  Coefficient of variation: {cv:.4f}")
        logger.info(f"  Interval range: {interval_range:.4f}s")
        logger.info(f"  Stability score: {stability_score:.4f}")
        
        return {
            'mean_interval': mean_interval,
            'median_interval': median_interval,
            'std_interval': std_interval,
            'cv': cv,
            'interval_range': interval_range,
            'stability_score': stability_score
        }
    def analyze_half_tempo_candidate(self, beats: List[float]) -> Optional[Dict[str, float]]:
        """
        Analyze whether every second beat produces a more stable tempo.

        Args:
            beats: Beat timestamps returned by RhythmExtractor2013.

        Returns:
            Dictionary with half-tempo analysis or None if insufficient beats.
        """

        if len(beats) < 8:
            return None

        # Keep every second beat
        half_beats = beats[::2]

        # Reuse existing stability analysis
        stats = self.analyze_beat_stability(half_beats)

        candidate_bpm = 60.0 / stats["median_interval"] if stats["median_interval"] > 0 else 0.0

        logger.info("Half-tempo candidate:")
        logger.info(f"  Candidate BPM : {candidate_bpm:.2f}")
        logger.info(f"  Candidate CV  : {stats['cv']:.4f}")
        logger.info(f"  Stability     : {stats['stability_score']:.4f}")

        return {
            "candidate_bpm": candidate_bpm,
            "candidate_cv": stats["cv"],
            "candidate_stability": stats["stability_score"]
        }
        
    def detect_tempo_ambiguity(self, bpm: float, estimates: List[float]) -> Tuple[bool, Optional[float]]:
        """
        Detect half/double tempo ambiguity.
        
        Args:
            bpm: Detected BPM
            estimates: List of BPM estimates
            
        Returns:
            Tuple of (is_ambiguous, alternative_bpm)
        """
        estimates = np.asarray(estimates)

        if estimates.size == 0:
            return False, None
        
        # Check if any estimate is approximately half or double
        for estimate in estimates:
            if estimate == 0:
                continue
            
            ratio = bpm / estimate
            # Check for half tempo (ratio ≈ 2.0)
            if abs(ratio - 2.0) < 0.1:
                return True, estimate
            # Check for double tempo (ratio ≈ 0.5)
            if abs(ratio - 0.5) < 0.1:
                return True, estimate
        
        return False, None
    
    def compute_confidence_score(
        self,
        rhythm_result: Dict[str, Any],
        estimate_stats: Dict[str, float],
        beat_stats: Dict[str, float]
    ) -> float:
        """
        Compute normalized confidence score (0.0 - 1.0).

        Combines:
            - Essentia confidence
            - Estimate agreement
            - Beat stability
            - Estimate variance
            - Beat count
        """

        # ------------------------------------------------------------
        # 1. Normalize Essentia confidence
        # Observed values are roughly between 0 and 3.
        # ------------------------------------------------------------
        raw_confidence = rhythm_result["confidence"]
        confidence_score = min(raw_confidence / 3.0, 1.0)

        # ------------------------------------------------------------
        # 2. Estimate agreement
        # Already 0-100
        # ------------------------------------------------------------
        agreement_score = estimate_stats["agreement"] / 100.0

        # ------------------------------------------------------------
        # 3. Beat stability
        # Already between 0 and 1
        # ------------------------------------------------------------
        stability_score = beat_stats["stability_score"]

        # ------------------------------------------------------------
        # 4. Variance
        # Lower variance = higher confidence
        # ------------------------------------------------------------
        variance = estimate_stats["variance"]

        variance_score = max(
            0.0,
            min(
                1.0,
                1.0 - (variance / 10.0)
            )
        )

        # ------------------------------------------------------------
        # 5. Beat count
        # Saturates at 200 beats.
        # ------------------------------------------------------------
        beat_count = len(rhythm_result["beats"])

        beat_count_score = min(
            beat_count / 200.0,
            1.0
        )

        # ------------------------------------------------------------
        # Weighted confidence
        # ------------------------------------------------------------

        custom_confidence = (

            confidence_score * 0.15 +

            agreement_score * 0.30 +

            stability_score * 0.30 +

            variance_score * 0.15 +

            beat_count_score * 0.10

        )

        logger.info("Custom confidence computation:")

        logger.info(f"  Essentia score   : {confidence_score:.3f}")

        logger.info(f"  Agreement score  : {agreement_score:.3f}")

        logger.info(f"  Stability score  : {stability_score:.3f}")

        logger.info(f"  Variance score   : {variance_score:.3f}")

        logger.info(f"  Beat count score : {beat_count_score:.3f}")

        logger.info(f"  Final confidence : {custom_confidence:.3f}")

        return custom_confidence
    
    def validate_bpm(self, bpm: float, duration: float, num_beats: int) -> Tuple[bool, List[str]]:
        """
        Validate BPM estimate and return warnings.
        
        Args:
            bpm: Detected BPM
            duration: Audio duration in seconds
            num_beats: Number of detected beats
            
        Returns:
            Tuple of (is_valid, warnings)
        """
        warnings = []
        
        # Check BPM range
        if bpm < self.MIN_BPM:
            warnings.append(f"BPM ({bpm:.2f}) is below minimum ({self.MIN_BPM})")
        if bpm > self.MAX_BPM:
            warnings.append(f"BPM ({bpm:.2f}) is above maximum ({self.MAX_BPM})")
        
        # Check for sufficient beats
        if num_beats < 10:
            warnings.append(f"Only {num_beats} beats detected - low confidence")
        
        # Check for short clips
        if duration < self.SHORT_CLIP_DURATION:
            warnings.append(f"Audio duration ({duration:.2f}s) is very short")
        
        # Check for reasonable beat density
        if duration > 0:
            beats_per_second = num_beats / duration
            if beats_per_second < 0.5:
                warnings.append(f"Low beat density ({beats_per_second:.2f} beats/sec)")
            if beats_per_second > 5.0:
                warnings.append(f"High beat density ({beats_per_second:.2f} beats/sec)")
        
        is_valid = len(warnings) == 0
        
        if warnings:
            logger.warning(f"BPM validation warnings:")
            for warning in warnings:
                logger.warning(f"  - {warning}")
        else:
            logger.info("BPM validation passed")
        
        return is_valid, warnings
    
    def analyze(self, filename: str) -> Dict[str, Any]:
        """
        Perform complete BPM analysis.
        
        Args:
            filename: Path to audio file
            
        Returns:
            Dictionary containing all analysis results
        """
        start_time = time.time()
        
        logger.info(f"Starting BPM analysis of: {filename}")
        
        # Load audio
        audio, sample_rate = self.load_audio(filename)
        duration = len(audio) / sample_rate
        
        # Preprocess audio
        processed_audio, target_sr = self.preprocess_audio(audio, sample_rate)
        
        # Analyze rhythm
        rhythm_result = self.analyze_rhythm(processed_audio)
        
        # Analyze estimate distribution
        estimate_stats = self.analyze_estimate_distribution(rhythm_result['estimates'])
        
        # Analyze beat stability
        beat_stats = self.analyze_beat_stability(rhythm_result['beats'])
        half_tempo = self.analyze_half_tempo_candidate(rhythm_result["beats"])
        # Detect tempo ambiguity
        is_ambiguous, alternative_bpm = self.detect_tempo_ambiguity(
            rhythm_result['bpm'], 
            rhythm_result['estimates']
        )
        # ------------------------------------------------------------
        # Automatic Half-Tempo Correction
        # ------------------------------------------------------------

        if half_tempo is not None:

            original_bpm = rhythm_result["bpm"]
            candidate_bpm = half_tempo["candidate_bpm"]

            original_cv = beat_stats["cv"]
            candidate_cv = half_tempo["candidate_cv"]

            # Calculate percentage improvement in beat stability
            cv_improvement = (
                (original_cv - candidate_cv) / original_cv
                if original_cv > 0
                else 0.0
            )

            logger.info(f"CV Improvement : {cv_improvement * 100:.1f}%")
            logger.info("=" * 60)
            logger.info("HALF-TEMPO DIAGNOSTICS")
            logger.info(f"Original BPM      : {original_bpm:.2f}")
            logger.info(f"Candidate BPM     : {candidate_bpm:.2f}")
            logger.info(f"Essentia Conf     : {rhythm_result['confidence']:.3f}")
            logger.info(f"Agreement         : {estimate_stats['agreement']:.2f}%")
            logger.info(f"Beat Count        : {len(rhythm_result['beats'])}")
            logger.info(f"Original CV       : {original_cv:.4f}")
            logger.info(f"Candidate CV      : {candidate_cv:.4f}")
            logger.info(f"CV Improvement    : {cv_improvement*100:.2f}%")
            logger.info("=" * 60)
            if (
                original_bpm > 140
                and 60 <= candidate_bpm <= 120
                and cv_improvement > 0.20
            ):
                logger.info("=" * 60)
                logger.info("Half-tempo correction accepted")
                logger.info(f"Original BPM  : {original_bpm:.2f}")
                logger.info(f"Corrected BPM : {candidate_bpm:.2f}")
                logger.info(f"Original CV   : {original_cv:.4f}")
                logger.info(f"Half CV       : {candidate_cv:.4f}")
                logger.info("=" * 60)

                rhythm_result["bpm"] = candidate_bpm
                        
        # Compute custom confidence
        custom_confidence = self.compute_confidence_score(
            rhythm_result, 
            estimate_stats, 
            beat_stats
        )
        
        # Validate BPM
        is_valid, warnings = self.validate_bpm(
            rhythm_result['bpm'], 
            duration, 
            len(rhythm_result['beats'])
        )
        if half_tempo is not None and rhythm_result["bpm"] == half_tempo["candidate_bpm"]:
            warnings.append(
                "Half-tempo correction applied using beat stability analysis."
            )        
        processing_time = time.time() - start_time
        
        # Compile results
        results = {
            'filename': filename,
            'duration': duration,
            'sample_rate': sample_rate,
            'target_sample_rate': target_sr,
            'bpm': rhythm_result['bpm'],
            'confidence': rhythm_result['confidence'],
            'custom_confidence': custom_confidence,
            'beats': rhythm_result['beats'],
            'num_beats': len(rhythm_result['beats']),
            'estimates': rhythm_result['estimates'],
            'estimate_stats': estimate_stats,
            'beat_stats': beat_stats,
            'half_tempo': half_tempo,
            'is_ambiguous': is_ambiguous,
            'alternative_bpm': alternative_bpm,
            'is_valid': is_valid,
            'warnings': warnings,
            'processing_time': processing_time
        }
        
        return results
    
    def print_report(self, results: Dict[str, Any]) -> None:
        """
        Print a clean analysis report.

        Args:
            results: Dictionary containing BPM analysis results.
        """

        print("\n" + "=" * 60)
        print("ESSENTIA BPM ANALYSIS REPORT")
        print("=" * 60)

        print("\n[AUDIO]")
        print(f"File               : {results['filename']}")
        print(f"Duration           : {results['duration']:.2f} sec")
        print(f"Original SR        : {results['sample_rate']} Hz")
        print(f"Analysis SR        : {results['target_sample_rate']} Hz")

        print("\n[RHYTHM]")
        print(f"Estimated BPM      : {results['bpm']:.2f}")
        print(f"Confidence         : {results['confidence']:.3f}")
        print(f"Custom Confidence  : {results['custom_confidence']:.3f}")
        print(f"Detected Beats     : {results['num_beats']}")

        if results["estimate_stats"]:
            stats = results["estimate_stats"]
            print("\n[ESTIMATE DISTRIBUTION]")
            print(f"Median BPM         : {stats['median']:.2f}")
            print(f"Variance           : {stats['variance']:.3f}")
            print(f"Spread             : {stats['spread']:.2f}")
            print(f"Agreement          : {stats['agreement']:.2f}%")

        if results["beat_stats"]:
            beat = results["beat_stats"]
            print("\n[BEAT STABILITY]")
            print(f"Mean Interval      : {beat['mean_interval']:.4f} sec")
            print(f"Median Interval    : {beat['median_interval']:.4f} sec")
            print(f"Std Interval       : {beat['std_interval']:.4f} sec")
            print(f"CV                 : {beat['cv']:.4f}")
            print(f"Stability Score    : {beat['stability_score']:.4f}")

        print("\n[VALIDATION]")
        print(f"Valid Result       : {'YES' if results['is_valid'] else 'NO'}")

        if results["is_ambiguous"]:
            print(
                f"Tempo Ambiguity    : YES ({results['alternative_bpm']:.2f} BPM candidate)"
            )
        else:
            print("Tempo Ambiguity    : NO")

        if results["warnings"]:
            print("\n[WARNINGS]")
            for warning in results["warnings"]:
                print(f" - {warning}")

        print("\n[PERFORMANCE]")
        print(f"Processing Time    : {results['processing_time']:.3f} sec")

        print("=" * 60)


def main():
    """
    Command line entry point.
    """

    if len(sys.argv) != 2:
        print("Usage:")
        print("python3 bpm.py <audio_file>")
        sys.exit(1)

    audio_file = sys.argv[1]

    try:
        detector = BPMDetector()

        results = detector.analyze(audio_file)

        detector.print_report(results)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(1)

    except Exception as e:
        logger.exception("BPM analysis failed.")
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()