"""
genre.py
=======

Genre prediction module for the Music Analyzer.

This module performs music-genre classification using the trained
EXP-19 PANNs Cnn14 model.

Model
-----
- Architecture: PANNs Transfer_Cnn14
- Training experiment: EXP-19
- Task: 7-class genre classification
- Sample rate: 32000 Hz
- Input crop: 10 seconds
- Test/inference crops: 0, 5, 10, 15, 20 seconds
- Aggregation: average crop probabilities

Supported genres
----------------
- Electronic
- Experimental
- Folk
- Hip-Hop
- Instrumental
- International
- Rock

Important
---------
Pop is intentionally NOT part of this classifier because EXP-19 was
trained as a true 7-class task with Pop completely excluded.

Public API
----------
- ``predict_genre(path)`` returns a dictionary containing the predicted
  genre and its confidence.

The model is loaded lazily and cached in memory so the checkpoint is not
reloaded for every audio file analyzed by the GUI.
"""

from pathlib import Path
import sys

import numpy as np
import torch

from utils.preprocessor import validate_audio


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path.home() / "Final_Project" / "Music Analyzer"

DL_DIR = PROJECT_ROOT / "DeepLearning"

PANN_DIR = DL_DIR / "panns" / "panns-transfer"

CHECKPOINT_PATH = (
    DL_DIR
    / "dl_artifacts"
    / "fma8_finetune"
    / "EXP-19_PopExcluded_7Class"
    / "best.pth"
)


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

SAMPLE_RATE = 32000

WINDOW_SIZE = 1024
HOP_SIZE = 320
MEL_BINS = 64
FMIN = 50
FMAX = 14000

CROP_SECONDS = 10
CROP_SAMPLES = SAMPLE_RATE * CROP_SECONDS

# Five deterministic 10-second crops:
#
# 0-10
# 5-15
# 10-20
# 15-25
# 20-30
#
CROP_STARTS = (
    0,
    5 * SAMPLE_RATE,
    10 * SAMPLE_RATE,
    15 * SAMPLE_RATE,
    20 * SAMPLE_RATE,
)

NUM_CLASSES = 7

CLASS_NAMES = (
    "Electronic",
    "Experimental",
    "Folk",
    "Hip-Hop",
    "Instrumental",
    "International",
    "Rock",
)


# ============================================================================
# DEVICE
# ============================================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================================
# INTERNAL MODEL CACHE
# ============================================================================

_MODEL = None


# ============================================================================
# PANNs IMPORT
# ============================================================================

def _import_panns():
    """
    Import the PANNs Transfer_Cnn14 architecture.

    The PANNs repository is kept inside the project's DeepLearning
    directory, so it is added to sys.path before importing.
    """

    panns_path = str(PANN_DIR)
    pytorch_path = str(PANN_DIR / "pytorch")

    if panns_path not in sys.path:
        sys.path.insert(0, panns_path)

    if pytorch_path not in sys.path:
        sys.path.insert(0, pytorch_path)

    try:
        from pytorch.models import Transfer_Cnn14
    except Exception as exc:
        raise RuntimeError(
            "Unable to import PANNs Transfer_Cnn14. "
            f"Expected PANNs directory: '{PANN_DIR}'."
        ) from exc

    return Transfer_Cnn14


# ============================================================================
# MODEL CREATION
# ============================================================================

def _build_model():
    """
    Build the exact 7-class PANNs Transfer_Cnn14 architecture used by EXP-19.
    """

    Transfer_Cnn14 = _import_panns()

    model = Transfer_Cnn14(
        sample_rate=SAMPLE_RATE,
        window_size=WINDOW_SIZE,
        hop_size=HOP_SIZE,
        mel_bins=MEL_BINS,
        fmin=FMIN,
        fmax=FMAX,
        classes_num=NUM_CLASSES,
        freeze_base=False,
    )

    return model.to(DEVICE)


# ============================================================================
# CHECKPOINT KEY NORMALIZATION
# ============================================================================

def _normalize_checkpoint_key(key):
    """
    Normalize common wrapper prefixes used in the EXP-19 checkpoint.

    The actual Transfer_Cnn14 model expects keys such as:

        base.conv_block1.conv1.weight
        base.fc1.weight
        fc_transfer.weight

    Therefore the internal ``base.`` prefix must be preserved.

    Only the DataParallel ``module.`` wrapper is removed.
    """

    key = str(key)

    while key.startswith("module."):
        key = key[len("module."):]

    return key

# ============================================================================
# CHECKPOINT LOADING
# ============================================================================

def _load_model():
    """
    Load the EXP-19 best checkpoint.

    The model is loaded only once and then cached for subsequent calls.

    Checkpoint tensors are matched by normalized key and compatible shape.
    """

    global _MODEL

    if _MODEL is not None:
        return _MODEL

    # ------------------------------------------------------------------------
    # Check checkpoint exists
    # ------------------------------------------------------------------------

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            "EXP-19 genre checkpoint not found: "
            f"'{CHECKPOINT_PATH}'"
        )

    # ------------------------------------------------------------------------
    # Build exact EXP-19 architecture
    # ------------------------------------------------------------------------

    model = _build_model()

    # ------------------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            "Invalid EXP-19 checkpoint format. "
            "Expected a checkpoint dictionary."
        )

    if "model" not in checkpoint:
        raise RuntimeError(
            "EXP-19 checkpoint does not contain a 'model' state dictionary."
        )

    raw_state_dict = checkpoint["model"]

    if not isinstance(raw_state_dict, dict):
        raise RuntimeError(
            "EXP-19 checkpoint 'model' entry is not a state dictionary."
        )

    # ------------------------------------------------------------------------
    # Normalize checkpoint keys.
    #
    # IMPORTANT:
    # Keep ``base.`` because the Transfer_Cnn14 model itself contains
    # ``base.*`` keys.
    # ------------------------------------------------------------------------

    normalized_state_dict = {}

    for key, value in raw_state_dict.items():

        normalized_key = _normalize_checkpoint_key(key)

        if normalized_key not in normalized_state_dict:
            normalized_state_dict[normalized_key] = value

    # ------------------------------------------------------------------------
    # Match checkpoint against current model.
    # ------------------------------------------------------------------------

    model_state = model.state_dict()

    compatible_state = {}
    shape_mismatches = []

    for key, value in normalized_state_dict.items():

        if key not in model_state:
            continue

        if not isinstance(value, torch.Tensor):
            continue

        expected_shape = tuple(
            model_state[key].shape
        )

        found_shape = tuple(
            value.shape
        )

        if expected_shape != found_shape:

            shape_mismatches.append(
                (
                    key,
                    expected_shape,
                    found_shape,
                )
            )

            continue

        compatible_state[key] = value

    # ------------------------------------------------------------------------
    # Safety gate.
    # ------------------------------------------------------------------------

    missing_keys = [
        key
        for key in model_state
        if key not in compatible_state
    ]

    unexpected_keys = [
        key
        for key in normalized_state_dict
        if key not in model_state
    ]

    # ------------------------------------------------------------------------
    # Shape mismatch must never be silently ignored.
    # ------------------------------------------------------------------------

    if shape_mismatches:

        mismatch_preview = "\n".join(
            (
                f"  {key}: expected {expected}, "
                f"found {found}"
            )
            for key, expected, found in shape_mismatches[:10]
        )

        raise RuntimeError(
            "EXP-19 checkpoint contains shape mismatches:\n"
            f"{mismatch_preview}"
        )

    # ------------------------------------------------------------------------
    # EXP-19 should load the complete 7-class model.
    # ------------------------------------------------------------------------

    if missing_keys:

        missing_preview = "\n".join(
            f"  {key}"
            for key in missing_keys[:20]
        )

        raise RuntimeError(
            "EXP-19 checkpoint is missing model tensors.\n"
            f"Missing tensors: {len(missing_keys)}\n"
            f"{missing_preview}"
        )

    # ------------------------------------------------------------------------
    # Load checkpoint.
    # ------------------------------------------------------------------------

    load_result = model.load_state_dict(
        compatible_state,
        strict=False,
    )

    if load_result.missing_keys:
        raise RuntimeError(
            "EXP-19 checkpoint loading reported missing keys after "
            "compatibility filtering:\n"
            f"{load_result.missing_keys}"
        )

    if load_result.unexpected_keys:
        raise RuntimeError(
            "EXP-19 checkpoint loading reported unexpected keys:\n"
            f"{load_result.unexpected_keys}"
        )

    # ------------------------------------------------------------------------
    # Verify classifier output dimension.
    # ------------------------------------------------------------------------

    classifier_output = None

    for key, tensor in model.state_dict().items():

        if key.endswith("fc_transfer.weight"):

            classifier_output = int(
                tensor.shape[0]
            )

            break

    if classifier_output != NUM_CLASSES:

        raise RuntimeError(
            "EXP-19 classifier dimension check failed. "
            f"Expected {NUM_CLASSES}, "
            f"found {classifier_output}."
        )

    # ------------------------------------------------------------------------
    # Evaluation mode.
    # ------------------------------------------------------------------------

    model.eval()

    _MODEL = model

    print(
        f"EXP-19 genre model loaded: "
        f"{len(compatible_state)} tensors"
    )

    print(
        f"Device              : {DEVICE}"
    )

    print(
        f"Checkpoint           : {CHECKPOINT_PATH}"
    )

    print(
        f"Classes              : {NUM_CLASSES}"
    )

    return _MODEL

# ============================================================================
# AUDIO PREPARATION
# ============================================================================

def _prepare_waveform(path):
    """
    Load audio and convert it into the waveform format expected by PANNs.

    The existing Music Analyzer loader is used for decoding. The decoded
    waveform is then explicitly resampled to the exact 32000 Hz required
    by the EXP-19 PANNs model.

    Returns
    -------
    np.ndarray
        Mono float32 waveform at exactly 32000 Hz.
    """

    from modules.loader import load_audio
    import librosa

    path = Path(path)

    # validate_audio() validates the file; it does not return the path.
    validate_audio(path)

    # ------------------------------------------------------------------------
    # Decode using the existing Music Analyzer loader.
    # ------------------------------------------------------------------------

    audio, analysis_sr, _ = load_audio(str(path))

    if audio is None:
        raise RuntimeError(
            f"Unable to decode audio file: '{path}'"
        )

    audio = np.asarray(audio)

    # ------------------------------------------------------------------------
    # Mono conversion
    # ------------------------------------------------------------------------

    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    # ------------------------------------------------------------------------
    # Safety checks before resampling.
    # ------------------------------------------------------------------------

    if audio.size == 0:
        raise RuntimeError(
            f"Decoded audio is empty: '{path}'"
        )

    if not np.isfinite(audio).all():
        raise RuntimeError(
            f"Decoded audio contains NaN or Inf values: '{path}'"
        )

    # ------------------------------------------------------------------------
    # Explicitly resample to the exact EXP-19 sample rate.
    #
    # The existing loader returned 44100 Hz for this MP3. EXP-19 was trained
    # at 32000 Hz, so the genre module must enforce 32000 Hz independently.
    # ------------------------------------------------------------------------

    analysis_sr = int(analysis_sr)

    if analysis_sr <= 0:
        raise RuntimeError(
            f"Invalid audio sample rate returned by loader: {analysis_sr} Hz."
        )

    if analysis_sr != SAMPLE_RATE:

        print(
            f"Genre analysis: resampling "
            f"{analysis_sr} Hz -> {SAMPLE_RATE} Hz"
        )

        audio = librosa.resample(
            audio,
            orig_sr=analysis_sr,
            target_sr=SAMPLE_RATE,
        )

        audio = np.asarray(
            audio,
            dtype=np.float32,
        )

    # ------------------------------------------------------------------------
    # Final safety checks.
    # ------------------------------------------------------------------------

    if audio.size == 0:
        raise RuntimeError(
            f"Audio became empty after resampling: '{path}'"
        )

    if not np.isfinite(audio).all():
        raise RuntimeError(
            f"Audio contains NaN or Inf values after resampling: '{path}'"
        )

    return audio

# ============================================================================
# CROP CREATION
# ============================================================================

def _make_crops(waveform):
    """
    Create the five deterministic 10-second EXP-19 inference crops.

    If the decoded audio is shorter than a required crop, zero-padding is
    used only for the missing portion.
    """

    crops = []

    for start in CROP_STARTS:

        end = start + CROP_SAMPLES

        crop = waveform[start:end]

        if len(crop) < CROP_SAMPLES:

            padded = np.zeros(
                CROP_SAMPLES,
                dtype=np.float32,
            )

            padded[:len(crop)] = crop

            crop = padded

        if len(crop) != CROP_SAMPLES:
            raise RuntimeError(
                "Unable to create a valid 10-second "
                "genre-analysis crop."
            )

        crops.append(crop)

    return np.stack(
        crops,
        axis=0,
    )


# ============================================================================
# GENRE PREDICTION
# ============================================================================

@torch.no_grad()
def predict_genre(path):
    """
    Predict the genre of an audio file using the EXP-19 model.

    Parameters
    ----------
    path:
        Path to the audio file.

    Returns
    -------
    dict
        Example::

            {
                "genre": "Electronic",
                "confidence": 0.4216,
            }

    Notes
    -----
    Five deterministic 10-second crops are evaluated and their probability
    distributions are averaged before selecting the final genre.
    """

    # ------------------------------------------------------------------------
    # Keep the real path.
    #
    # validate_audio() only validates. It must not replace `path`.
    # ------------------------------------------------------------------------

    path = Path(path)

    validate_audio(path)

    # ------------------------------------------------------------------------
    # Prepare waveform
    # ------------------------------------------------------------------------

    waveform = _prepare_waveform(path)

    # ------------------------------------------------------------------------
    # Create deterministic 5-crop TTA input
    # ------------------------------------------------------------------------

    crops = _make_crops(waveform)

    # ------------------------------------------------------------------------
    # Load cached model
    # ------------------------------------------------------------------------

    model = _load_model()

    # ------------------------------------------------------------------------
    # NumPy -> PyTorch
    # ------------------------------------------------------------------------

    x = torch.from_numpy(
        crops
    ).to(
        DEVICE,
        dtype=torch.float32,
    )

    # ------------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------------

    output = model(x)

    # PANNs Transfer_Cnn14 returns a dictionary.
    if not isinstance(output, dict):
        raise RuntimeError(
            "EXP-19 genre model returned an invalid output type. "
            f"Expected dict, got {type(output).__name__}."
        )

    if "clipwise_output" not in output:
        raise RuntimeError(
            "EXP-19 genre model output does not contain "
            "'clipwise_output'."
        )

    logits = output["clipwise_output"]

    if not isinstance(logits, torch.Tensor):
        raise RuntimeError(
            "EXP-19 'clipwise_output' is not a torch.Tensor."
        )

    # ------------------------------------------------------------------------
    # Verify output shape.
    # ------------------------------------------------------------------------

    expected_shape = (
        len(CROP_STARTS),
        NUM_CLASSES,
    )

    if tuple(logits.shape) != expected_shape:
        raise RuntimeError(
            "Unexpected EXP-19 model output shape: "
            f"{tuple(logits.shape)}. "
            f"Expected {expected_shape}."
        )

    # ------------------------------------------------------------------------
    # EXP-19 standalone inference used PANNs clipwise_output as
    # log-probability and converted it to probability using exp().
    # ------------------------------------------------------------------------

    crop_probabilities = torch.exp(logits)

    if not torch.isfinite(crop_probabilities).all():
        raise RuntimeError(
            "EXP-19 genre model produced non-finite probabilities."
        )

    # ------------------------------------------------------------------------
    # Average probabilities across the five deterministic crops.
    # ------------------------------------------------------------------------

    mean_probabilities = crop_probabilities.mean(
        dim=0
    )

    if not torch.isfinite(mean_probabilities).all():
        raise RuntimeError(
            "EXP-19 mean genre probabilities are non-finite."
        )

    # ------------------------------------------------------------------------
    # Final prediction.
    # ------------------------------------------------------------------------

    predicted_index = int(
        torch.argmax(mean_probabilities).item()
    )

    if not 0 <= predicted_index < NUM_CLASSES:
        raise RuntimeError(
            f"Invalid predicted class index: {predicted_index}"
        )

    confidence = float(
        mean_probabilities[predicted_index].item()
    )

    genre = CLASS_NAMES[predicted_index]

    return {
        "genre": genre,
        "confidence": confidence,
    }


# ============================================================================
# OPTIONAL CLI
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Predict music genre using the EXP-19 "
            "7-class PANNs model."
        )
    )

    parser.add_argument(
        "audio_file",
        help="Path to WAV or MP3 audio file",
    )

    args = parser.parse_args()

    try:

        result = predict_genre(
            args.audio_file
        )

        print("\n" + "=" * 60)
        print("GENRE ANALYSIS")
        print("=" * 60)
        print(f"Genre      : {result['genre']}")
        print(f"Confidence : {result['confidence']:.4f}")
        print("=" * 60)

    except Exception as exc:

        print(
            f"\nERROR: {exc}"
        )


if __name__ == "__main__":
    main()