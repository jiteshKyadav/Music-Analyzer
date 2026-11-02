import os
import sys
import json
import random
import h5py
import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.expanduser(
    "~/Final_Project/Music Analyzer"
)

DL_DIR = os.path.join(
    PROJECT_ROOT,
    "DeepLearning"
)

PANN_DIR = os.path.join(
    DL_DIR,
    "panns",
    "panns-transfer"
)

HDF5_PATH = os.path.join(
    DL_DIR,
    "dl_artifacts",
    "fma8",
    "features",
    "waveform.h5"
)

EXP_DIR = os.path.join(
    DL_DIR,
    "dl_artifacts",
    "fma8_finetune",
    "EXP-19_PopExcluded_7Class"
)

CHECKPOINT_PATH = os.path.join(
    EXP_DIR,
    "best.pth"
)

# ============================================================
# IMPORT PANNs
# ============================================================

sys.path.insert(0, PANN_DIR)
sys.path.insert(
    0,
    os.path.join(PANN_DIR, "pytorch")
)

from pytorch.models import Transfer_Cnn14


# ============================================================
# CONFIG
# ============================================================

SEED = 0

SAMPLE_RATE = 32000

TRACK_SECONDS = 30

TRACK_SAMPLES = (
    SAMPLE_RATE * TRACK_SECONDS
)

CROP_SECONDS = 10

CROP_SAMPLES = (
    SAMPLE_RATE * CROP_SECONDS
)

NUM_CLASSES = 7

BATCH_SIZE = 4

NUM_WORKERS = 0

PIN_MEMORY = True

DEVICE = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)

# Original FMA8 label:
# 0 Electronic
# 1 Experimental
# 2 Folk
# 3 Hip-Hop
# 4 Instrumental
# 5 International
# 6 Pop              <-- EXCLUDED
# 7 Rock             --> NEW LABEL 6

POP_ORIGINAL_LABEL = 6

ORIGINAL_TO_NEW_LABEL = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    7: 6,
}

CLASS_NAMES = [
    "Electronic",
    "Experimental",
    "Folk",
    "Hip-Hop",
    "Instrumental",
    "International",
    "Rock",
]

# EXP-09 deterministic 5-crop test TTA
TEST_CROP_STARTS = [
    0,
    5,
    10,
    15,
    20,
]


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# ============================================================
# DATASET
# ============================================================

class FMA7TestDataset(Dataset):

    def __init__(
        self,
        hdf5_path,
        split="test",
    ):

        self.hdf5_path = hdf5_path

        self.split = split

        self.h5 = None

        with h5py.File(
            self.hdf5_path,
            "r"
        ) as h5:

            splits = np.asarray(
                h5["split"][:]
            )

            targets = np.asarray(
                h5["target"][:]
            )

            original_labels = np.argmax(
                targets,
                axis=1
            ).astype(np.int64)

            split_mask = (
                splits.astype(str) == split
            )

            original_indices = np.where(
                split_mask
            )[0]

            retained_indices = []

            retained_labels = []

            removed_pop = 0

            invalid_labels = []

            for real_idx in original_indices:

                original_label = int(
                    original_labels[real_idx]
                )

                # ------------------------------------------------
                # COMPLETELY EXCLUDE POP
                # ------------------------------------------------

                if (
                    original_label
                    == POP_ORIGINAL_LABEL
                ):

                    removed_pop += 1

                    continue

                # ------------------------------------------------
                # VALID 7 CLASS LABELS
                # ------------------------------------------------

                if (
                    original_label
                    not in ORIGINAL_TO_NEW_LABEL
                ):

                    invalid_labels.append(
                        original_label
                    )

                    continue

                new_label = (
                    ORIGINAL_TO_NEW_LABEL[
                        original_label
                    ]
                )

                retained_indices.append(
                    real_idx
                )

                retained_labels.append(
                    new_label
                )

            if len(invalid_labels) > 0:

                raise RuntimeError(
                    f"Unexpected labels found: "
                    f"{sorted(set(invalid_labels))}"
                )

            if len(retained_indices) == 0:

                raise RuntimeError(
                    "No non-Pop test samples found."
                )

            self.indices = np.asarray(
                retained_indices,
                dtype=np.int64
            )

            self.labels = np.asarray(
                retained_labels,
                dtype=np.int64
            )

            # ------------------------------------------------
            # SAFETY
            # ------------------------------------------------

            if np.any(
                self.labels < 0
            ) or np.any(
                self.labels >= NUM_CLASSES
            ):

                raise RuntimeError(
                    "Invalid 7-class labels detected."
                )

            print()
            print(
                f"test: original = "
                f"{len(original_indices)}"
            )

            print(
                f"test: Pop removed = "
                f"{removed_pop}"
            )

            print(
                f"test: retained 7-class samples = "
                f"{len(self.indices)}"
            )

            print(
                "test: label distribution:"
            )

            for cls in range(NUM_CLASSES):

                count = int(
                    np.sum(
                        self.labels == cls
                    )
                )

                print(
                    f"  {cls} = "
                    f"{CLASS_NAMES[cls]:15s} : "
                    f"{count}"
                )

    def _open(self):

        if self.h5 is None:

            self.h5 = h5py.File(
                self.hdf5_path,
                "r"
            )

    def __len__(self):

        return len(self.indices)

    def __getitem__(self, idx):

        self._open()

        real_idx = int(
            self.indices[idx]
        )

        waveform = np.asarray(
            self.h5["waveform"][real_idx],
            dtype=np.float32
        )

        waveform = (
            waveform / 32768.0
        )

        if waveform.shape[0] != TRACK_SAMPLES:

            raise RuntimeError(
                f"Unexpected waveform length: "
                f"{waveform.shape}"
            )

        crops = []

        for start_sec in TEST_CROP_STARTS:

            start = (
                start_sec * SAMPLE_RATE
            )

            end = (
                start +
                CROP_SAMPLES
            )

            crop = waveform[
                start:end
            ]

            if crop.shape[0] != CROP_SAMPLES:

                raise RuntimeError(
                    f"Invalid crop length: "
                    f"{crop.shape}"
                )

            crops.append(crop)

        crops = np.stack(
            crops,
            axis=0
        )

        crops = torch.from_numpy(
            crops
        ).float()

        label = torch.tensor(
            int(self.labels[idx]),
            dtype=torch.long
        )

        return crops, label

    def __del__(self):

        try:

            if self.h5 is not None:

                self.h5.close()

        except Exception:

            pass


# ============================================================
# MODEL
# ============================================================

def build_model():

    model = Transfer_Cnn14(
        sample_rate=32000,
        window_size=1024,
        hop_size=320,
        mel_bins=64,
        fmin=50,
        fmax=14000,
        classes_num=NUM_CLASSES,
        freeze_base=False,
    )

    return model


# ============================================================
# LOAD EXP-19 CHECKPOINT
# ============================================================

def load_checkpoint(
    model,
    checkpoint_path,
):

    print()
    print("=" * 70)
    print("LOADING EXP-19 BEST CHECKPOINT")
    print("=" * 70)

    print(
        f"Checkpoint:\n"
        f"{checkpoint_path}"
    )

    if not os.path.exists(
        checkpoint_path
    ):

        raise FileNotFoundError(
            checkpoint_path
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu"
    )

    if "model" not in checkpoint:

        raise RuntimeError(
            "Checkpoint does not contain "
            "'model'."
        )

    state_dict = checkpoint["model"]

    cleaned_state = {}

    for key, value in state_dict.items():

        new_key = key

        if new_key.startswith(
            "module."
        ):

            new_key = new_key[
                len("module.") :
            ]

        cleaned_state[
            new_key
        ] = value

    model_state = model.state_dict()

    compatible = {}

    unexpected = []

    shape_mismatch = []

    for key, value in cleaned_state.items():

        if key not in model_state:

            unexpected.append(key)

            continue

        if (
            model_state[key].shape
            != value.shape
        ):

            shape_mismatch.append(
                (
                    key,
                    tuple(value.shape),
                    tuple(
                        model_state[key].shape
                    ),
                )
            )

            continue

        compatible[key] = value

    missing = [
        key
        for key in model_state
        if key not in compatible
    ]

    model.load_state_dict(
        compatible,
        strict=False
    )

    print(
        f"Checkpoint epoch        : "
        f"{checkpoint.get('epoch', 'N/A')}"
    )

    print(
        f"Best validation F1      : "
        f"{checkpoint.get('best_val_macro_f1', 'N/A')}"
    )

    print(
        f"Compatible tensors      : "
        f"{len(compatible)}"
    )

    print(
        f"Missing tensors         : "
        f"{len(missing)}"
    )

    print(
        f"Unexpected tensors      : "
        f"{len(unexpected)}"
    )

    print(
        f"Shape mismatches        : "
        f"{len(shape_mismatch)}"
    )

    if len(shape_mismatch) > 0:

        print()
        print(
            "SHAPE MISMATCH DETAILS:"
        )

        for item in shape_mismatch:

            print(
                f"  {item[0]} : "
                f"{item[1]} -> {item[2]}"
            )

    # --------------------------------------------------------
    # EXP-19 CHECKPOINT SAFETY
    # --------------------------------------------------------

    classifier_weight = (
        model.fc_transfer.weight
    )

    classifier_bias = (
        model.fc_transfer.bias
    )

    if (
        classifier_weight.shape[0]
        != NUM_CLASSES
    ):

        raise RuntimeError(
            "Classifier output dimension "
            "is not 7."
        )

    if (
        classifier_bias.shape[0]
        != NUM_CLASSES
    ):

        raise RuntimeError(
            "Classifier bias dimension "
            "is not 7."
        )

    print()
    print(
        "EXP-19 7-CLASS CHECKPOINT GATE: PASS"
    )

    return checkpoint


# ============================================================
# BATCH PREDICTION
# ============================================================

@torch.no_grad()
def predict_dataset(
    model,
    loader,
):

    model.eval()

    all_targets = []

    all_predictions = []

    all_probabilities = []

    total_loss = 0.0

    total_samples = 0

    for crops, targets in loader:

        # ----------------------------------------------------
        # crops:
        # [B, 5, samples]
        # ----------------------------------------------------

        crops = crops.to(
            DEVICE,
            non_blocking=True
        )

        targets = targets.to(
            DEVICE,
            non_blocking=True
        )

        batch_size = (
            crops.shape[0]
        )

        num_crops = (
            crops.shape[1]
        )

        flat_crops = crops.reshape(
            batch_size * num_crops,
            -1
        )

        outputs = model(
            flat_crops
        )

        logits = outputs[
            "clipwise_output"
        ]

        # ----------------------------------------------------
        # EXP-09 STYLE 5-CROP TTA
        # ----------------------------------------------------

        probabilities = torch.exp(
            logits
        )

        probabilities = probabilities.reshape(
            batch_size,
            num_crops,
            NUM_CLASSES
        )

        mean_probabilities = (
            probabilities.mean(
                dim=1
            )
        )

        predictions = (
            mean_probabilities.argmax(
                dim=1
            )
        )

        # ----------------------------------------------------
        # LOSS
        # ----------------------------------------------------

        log_probabilities = torch.log(
            mean_probabilities.clamp_min(
                1e-12
            )
        )

        log_pt = (
            log_probabilities.gather(
                1,
                targets.unsqueeze(1)
            )
            .squeeze(1)
        )

        pt = torch.exp(
            log_pt
        )

        loss = -(
            (1.0 - pt) ** 3.0
        ) * log_pt

        total_loss += (
            loss.sum().item()
        )

        total_samples += (
            batch_size
        )

        all_targets.extend(
            targets.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_probabilities.append(
            mean_probabilities.cpu().numpy()
        )

    mean_loss = (
        total_loss /
        total_samples
    )

    all_targets = np.asarray(
        all_targets,
        dtype=np.int64
    )

    all_predictions = np.asarray(
        all_predictions,
        dtype=np.int64
    )

    all_probabilities = np.concatenate(
        all_probabilities,
        axis=0
    )

    return (
        mean_loss,
        all_targets,
        all_predictions,
        all_probabilities,
    )


# ============================================================
# METRICS
# ============================================================

def print_metrics(
    targets,
    predictions,
    loss,
):

    accuracy = accuracy_score(
        targets,
        predictions
    )

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            targets,
            predictions,
            labels=np.arange(NUM_CLASSES),
            zero_division=0,
        )
    )

    macro_precision = (
        precision.mean()
    )

    macro_recall = (
        recall.mean()
    )

    macro_f1 = (
        f1.mean()
    )

    cm = confusion_matrix(
        targets,
        predictions,
        labels=np.arange(NUM_CLASSES)
    )

    print()
    print("=" * 70)
    print("EXP-19 TEST RESULTS")
    print("=" * 70)

    print(
        f"Test Loss          : {loss:.6f}"
    )

    print(
        f"Accuracy            : {accuracy:.6f}"
    )

    print(
        f"Macro Precision     : "
        f"{macro_precision:.6f}"
    )

    print(
        f"Macro Recall        : "
        f"{macro_recall:.6f}"
    )

    print(
        f"Macro F1            : "
        f"{macro_f1:.6f}"
    )

    print()
    print(
        "PER-CLASS METRICS"
    )

    print(
        f"{'Class':18s}"
        f"{'Precision':>12s}"
        f"{'Recall':>12s}"
        f"{'F1':>12s}"
        f"{'Support':>12s}"
    )

    for i in range(NUM_CLASSES):

        print(
            f"{CLASS_NAMES[i]:18s}"
            f"{precision[i]:12.6f}"
            f"{recall[i]:12.6f}"
            f"{f1[i]:12.6f}"
            f"{support[i]:12d}"
        )

    print()
    print(
        "CONFUSION MATRIX"
    )

    print(
        "Rows = True"
    )

    print(
        "Cols = Predicted"
    )

    print()

    header = " " * 18

    for name in CLASS_NAMES:

        header += (
            f"{name[:8]:>10s}"
        )

    print(header)

    for i in range(NUM_CLASSES):

        row = (
            f"{CLASS_NAMES[i]:18s}"
        )

        for j in range(NUM_CLASSES):

            row += (
                f"{cm[i, j]:10d}"
            )

        print(row)

    return {
        "test_loss": float(loss),
        "accuracy": float(accuracy),
        "macro_precision": float(
            macro_precision
        ),
        "macro_recall": float(
            macro_recall
        ),
        "macro_f1": float(
            macro_f1
        ),
        "per_class": {
            CLASS_NAMES[i]: {
                "precision": float(
                    precision[i]
                ),
                "recall": float(
                    recall[i]
                ),
                "f1": float(
                    f1[i]
                ),
                "support": int(
                    support[i]
                ),
            }
            for i in range(NUM_CLASSES)
        },
        "confusion_matrix": cm.tolist(),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "EXP-19: POP-EXCLUDED 7-CLASS PANNs TEST"
    )
    print("=" * 70)

    print()
    print(
        "BASE EXPERIMENT : EXP-09"
    )

    print(
        "CHECKPOINT       : EXP-19 best.pth"
    )

    print(
        "EXCLUDED CLASS   : Pop"
    )

    print(
        "NUM CLASSES      : 7"
    )

    print()

    for i, name in enumerate(
        CLASS_NAMES
    ):

        print(
            f"  {i} = {name}"
        )

    print()
    print(
        f"Device        : {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU           : "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"Sample rate   : "
        f"{SAMPLE_RATE}"
    )

    print(
        f"Track length  : "
        f"{TRACK_SECONDS} sec"
    )

    print(
        f"Crop length   : "
        f"{CROP_SECONDS} sec"
    )

    print(
        "Test crops    : "
        "0-10, 5-15, 10-20, 15-25, 20-30 sec"
    )

    print(
        "Pop           : COMPLETELY EXCLUDED"
    )

    print(
        "Checkpoint    : best.pth"
    )

    # --------------------------------------------------------
    # SEED
    # --------------------------------------------------------

    set_seed(SEED)

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("LOADING 7-CLASS TEST DATASET")
    print("=" * 70)

    dataset = FMA7TestDataset(
        HDF5_PATH,
        split="test"
    )

    print()
    print("=" * 70)
    print("TEST DATASET SAFETY")
    print("=" * 70)

    print(
        f"Test samples after Pop removal : "
        f"{len(dataset)}"
    )

    print(
        "Pop samples are excluded."
    )

    print(
        "All predictions are 7-class."
    )

    # --------------------------------------------------------
    # DATALOADER
    # --------------------------------------------------------

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            PIN_MEMORY
            and DEVICE.type == "cuda"
        ),
        drop_last=False,
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("BUILDING EXP-19 7-CLASS MODEL")
    print("=" * 70)

    model = build_model()

    model = model.to(
        DEVICE
    )

    # --------------------------------------------------------
    # CHECKPOINT
    # --------------------------------------------------------

    checkpoint = load_checkpoint(
        model,
        CHECKPOINT_PATH
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RUNNING EXP-19 TEST")
    print("=" * 70)

    test_loss, targets, predictions, probabilities = (
        predict_dataset(
            model,
            loader
        )
    )

    # --------------------------------------------------------
    # FINAL SAFETY
    # --------------------------------------------------------

    if len(targets) != len(dataset):

        raise RuntimeError(
            "Prediction count does not "
            "match test dataset size."
        )

    if np.any(
        targets < 0
    ) or np.any(
        targets >= NUM_CLASSES
    ):

        raise RuntimeError(
            "Invalid target labels detected."
        )

    if np.any(
        predictions < 0
    ) or np.any(
        predictions >= NUM_CLASSES
    ):

        raise RuntimeError(
            "Invalid predicted labels detected."
        )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    results = print_metrics(
        targets,
        predictions,
        test_loss
    )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    results_path = os.path.join(
        EXP_DIR,
        "test_results_7class.json"
    )

    with open(
        results_path,
        "w"
    ) as f:

        json.dump(
            {
                "experiment": "EXP-19",
                "base_experiment": "EXP-09",
                "excluded_class": "Pop",
                "num_classes": 7,
                "class_names": CLASS_NAMES,
                "checkpoint": CHECKPOINT_PATH,
                "test_samples": int(
                    len(dataset)
                ),
                "test_crop_starts": TEST_CROP_STARTS,
                "results": results,
            },
            f,
            indent=2
        )

    print()
    print("=" * 70)
    print("EXP-19 TEST COMPLETE")
    print("=" * 70)

    print(
        f"Test Macro F1 : "
        f"{results['macro_f1']:.6f}"
    )

    print()
    print(
        "Results saved:"
    )

    print(
        results_path
    )

    print()
    print(
        "POP WAS COMPLETELY EXCLUDED "
        "FROM THE TEST TASK."
    )


if __name__ == "__main__":

    main()