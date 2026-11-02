import os
import sys
import time
import random
import json
import signal

import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ================================================================
# EXP-19: 7-CLASS PANNs Cnn14
#         POP EXCLUDED
#
# BASE:
#   EXP-09
#
# ONLY MAJOR CHANGE:
#   Pop class is completely excluded.
#
# INCLUDED CLASSES:
#   0 = Electronic
#   1 = Experimental
#   2 = Folk
#   3 = Hip-Hop
#   4 = Instrumental
#   5 = International
#   6 = Rock
#
# ORIGINAL FMA LABELS:
#   0 = Electronic
#   1 = Experimental
#   2 = Folk
#   3 = Hip-Hop
#   4 = Instrumental
#   5 = International
#   6 = Pop       <-- EXCLUDED
#   7 = Rock      <-- remapped to new label 6
#
# EVERYTHING ELSE REMAINS CONTROLLED AGAINST EXP-09.
#
# IMPORTANT:
#   Pop samples are excluded from BOTH training and validation.
#
#   Test set is NEVER used during training.
#
#   Test evaluation is intentionally NOT included in this script.
# ================================================================


# ================================================================
# PATHS
# ================================================================

PROJECT_ROOT = os.path.expanduser(
    "~/Final_Project/Music Analyzer"
)

PANN_DIR = os.path.join(
    PROJECT_ROOT,
    "DeepLearning",
    "panns",
    "panns-transfer"
)

HDF5_PATH = os.path.join(
    PROJECT_ROOT,
    "DeepLearning",
    "dl_artifacts",
    "fma8",
    "features",
    "waveform.h5"
)

PRETRAINED_CHECKPOINT = os.path.join(
    PANN_DIR,
    "checkpoints",
    "Cnn14_mAP=0.431.pth"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "DeepLearning",
    "dl_artifacts",
    "fma8_finetune",
    "EXP-19_PopExcluded_7Class"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

BEST_PATH = os.path.join(
    OUTPUT_DIR,
    "best.pth"
)

LAST_PATH = os.path.join(
    OUTPUT_DIR,
    "last.pth"
)

HISTORY_PATH = os.path.join(
    OUTPUT_DIR,
    "history.json"
)


# ================================================================
# PANNs IMPORT
# ================================================================

sys.path.insert(
    0,
    PANN_DIR
)

sys.path.insert(
    0,
    os.path.join(
        PANN_DIR,
        "pytorch"
    )
)

from pytorch.models import Transfer_Cnn14


# ================================================================
# CONFIG
# ================================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

SEED = 0

SAMPLE_RATE = 32000

TRACK_SECONDS = 30

TRACK_SAMPLES = (
    SAMPLE_RATE *
    TRACK_SECONDS
)

CROP_SECONDS = 10

CROP_SAMPLES = (
    SAMPLE_RATE *
    CROP_SECONDS
)

MAX_RANDOM_START = (
    TRACK_SAMPLES -
    CROP_SAMPLES
)

NUM_VAL_CROPS = 3

BATCH_SIZE = 4

MAX_EPOCHS = 50

BACKBONE_LR = 1e-5

HEAD_LR = 3e-4

WEIGHT_DECAY = 1e-4

PATIENCE = 8

GRAD_CLIP = 5.0

# ---------------------------------------------------------------
# EXP-19 = 7 CLASS
# ---------------------------------------------------------------

NUM_CLASSES = 7

FOCAL_GAMMA = 3.0

FOCAL_ALPHA = 1.0


# ---------------------------------------------------------------
# NEW 7-CLASS LABEL SPACE
# ---------------------------------------------------------------

CLASS_NAMES = [
    "Electronic",
    "Experimental",
    "Folk",
    "Hip-Hop",
    "Instrumental",
    "International",
    "Rock",
]


# ---------------------------------------------------------------
# ORIGINAL FMA LABELS TO KEEP
# ---------------------------------------------------------------

# Original:
# 0 Electronic
# 1 Experimental
# 2 Folk
# 3 Hip-Hop
# 4 Instrumental
# 5 International
# 6 Pop          -> excluded
# 7 Rock         -> new 6

ORIGINAL_TO_NEW_LABEL = {
    0: 0,  # Electronic
    1: 1,  # Experimental
    2: 2,  # Folk
    3: 3,  # Hip-Hop
    4: 4,  # Instrumental
    5: 5,  # International
    7: 6,  # Rock
}

POP_ORIGINAL_LABEL = 6


# ================================================================
# EXPECTED TOTAL COUNTS
# ================================================================

# Original dataset totals:
# Train = 5599
# Val   = 1200
# Test  = 1198
#
# Pop is excluded.
#
# We deliberately do NOT hard-code the resulting 7-class counts
# because the exact original per-class train distribution is not
# required for correctness. The script verifies that:
#
#   1. original split count is correct
#   2. no Pop samples survive
#   3. all retained labels belong to the 7 allowed classes
#
EXPECTED_COUNTS = {
    "train": 5599,
    "val": 1200,
    "test": 1198,
}


# ================================================================
# GLOBAL INTERRUPT FLAG
# ================================================================

STOP_REQUESTED = False


def signal_handler(
    signum,
    frame
):

    global STOP_REQUESTED

    STOP_REQUESTED = True

    print()
    print("=" * 70)
    print("STOP REQUEST RECEIVED")
    print("=" * 70)
    print(
        "Training will stop safely after the current epoch."
    )
    print(
        "The latest completed epoch checkpoint will be preserved."
    )
    print("=" * 70)


signal.signal(
    signal.SIGINT,
    signal_handler
)

signal.signal(
    signal.SIGTERM,
    signal_handler
)


# ================================================================
# REPRODUCIBILITY
# ================================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# ================================================================
# DATASET
# ================================================================

class FMA8Dataset(Dataset):

    def __init__(
        self,
        hdf5_path,
        split,
        training=False
    ):

        self.hdf5_path = hdf5_path

        self.split = split

        self.training = training

        self.h5 = None

        # --------------------------------------------------------
        # READ SPLIT + TARGETS
        # --------------------------------------------------------

        with h5py.File(
            self.hdf5_path,
            "r"
        ) as f:

            splits = np.asarray(
                f["split"]
            )

            if splits.dtype.kind == "S":

                splits = np.char.decode(
                    splits
                )

            targets = np.asarray(
                f["target"]
            )

            original_indices = np.where(
                splits == split
            )[0]

            expected = EXPECTED_COUNTS[split]

            if len(original_indices) != expected:

                raise RuntimeError(
                    f"{split} count mismatch: "
                    f"found {len(original_indices)}, "
                    f"expected {expected}"
                )

            # ----------------------------------------------------
            # REMOVE POP
            # ----------------------------------------------------

            retained_indices = []

            retained_labels = []

            removed_pop = 0

            invalid_labels = []

            for real_idx in original_indices:

                target = np.asarray(
                    targets[real_idx],
                    dtype=np.float32
                )

                original_label = int(
                    np.argmax(target)
                )

                # Pop = original label 6
                if original_label == POP_ORIGINAL_LABEL:

                    removed_pop += 1

                    continue

                # Anything other than the 7 desired classes
                if original_label not in ORIGINAL_TO_NEW_LABEL:

                    invalid_labels.append(
                        original_label
                    )

                    continue

                new_label = ORIGINAL_TO_NEW_LABEL[
                    original_label
                ]

                retained_indices.append(
                    real_idx
                )

                retained_labels.append(
                    new_label
                )

            if len(invalid_labels) > 0:

                raise RuntimeError(
                    f"Unexpected labels found in {split}: "
                    f"{sorted(set(invalid_labels))}"
                )

            if len(retained_indices) == 0:

                raise RuntimeError(
                    f"No non-Pop samples found in {split}."
                )

            self.indices = np.asarray(
                retained_indices,
                dtype=np.int64
            )

            self.labels = np.asarray(
                retained_labels,
                dtype=np.int64
            )

        # --------------------------------------------------------
        # VERIFY NO POP REMAINS
        # --------------------------------------------------------

        if np.any(
            self.labels < 0
        ) or np.any(
            self.labels >= NUM_CLASSES
        ):

            raise RuntimeError(
                f"Invalid remapped labels found in {split}."
            )

        # --------------------------------------------------------
        # PRINT DATASET INFORMATION
        # --------------------------------------------------------

        print()
        print(
            f"{split}: original = "
            f"{len(original_indices)}"
        )

        print(
            f"{split}: Pop removed = "
            f"{removed_pop}"
        )

        print(
            f"{split}: retained 7-class samples = "
            f"{len(self.indices)}"
        )

        print(
            f"{split}: label distribution:"
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

        # --------------------------------------------------------
        # FINAL SAFETY ASSERTION
        # --------------------------------------------------------
        # Pop was already removed using the ORIGINAL label above.
        # After remapping, label 6 is ROCK, so checking self.labels == 6
        # would incorrectly flag valid Rock samples as Pop.

    def _open(self):

        if self.h5 is None:

            self.h5 = h5py.File(
                self.hdf5_path,
                "r"
            )

    def __len__(self):

        return len(
            self.indices
        )

    def __getitem__(
        self,
        idx
    ):

        self._open()

        real_idx = self.indices[idx]

        waveform = np.asarray(
            self.h5["waveform"][real_idx],
            dtype=np.float32
        )

        waveform /= 32768.0

        # --------------------------------------------------------
        # EXACT 960000-SAMPLE SAFETY
        # --------------------------------------------------------

        if len(waveform) != TRACK_SAMPLES:

            raise RuntimeError(
                f"Invalid full waveform length for "
                f"index {real_idx}: "
                f"found {len(waveform)}, "
                f"expected {TRACK_SAMPLES}"
            )

        label = int(
            self.labels[idx]
        )

        # --------------------------------------------------------
        # TRAINING
        # --------------------------------------------------------

        if self.training:

            start = np.random.randint(
                0,
                MAX_RANDOM_START + 1
            )

            end = (
                start +
                CROP_SAMPLES
            )

            crop = waveform[
                start:end
            ]

            if len(crop) != CROP_SAMPLES:

                raise RuntimeError(
                    f"Invalid random crop length: "
                    f"{len(crop)}"
                )

            return (
                torch.from_numpy(
                    crop.copy()
                ),
                torch.tensor(
                    label,
                    dtype=torch.long
                )
            )

        # --------------------------------------------------------
        # VALIDATION
        # --------------------------------------------------------

        crops = []

        for crop_id in range(
            NUM_VAL_CROPS
        ):

            start = (
                crop_id *
                CROP_SAMPLES
            )

            end = (
                start +
                CROP_SAMPLES
            )

            crop = waveform[
                start:end
            ]

            if len(crop) != CROP_SAMPLES:

                raise RuntimeError(
                    f"Invalid validation crop length: "
                    f"{len(crop)}"
                )

            crops.append(
                crop
            )

        crops = np.stack(
            crops,
            axis=0
        )

        return (
            torch.from_numpy(
                crops.copy()
            ),
            torch.tensor(
                label,
                dtype=torch.long
            )
        )

    def __del__(self):

        try:

            if self.h5 is not None:

                self.h5.close()

        except Exception:

            pass


# ================================================================
# FOCAL LOSS
# ================================================================

class FocalLoss(nn.Module):

    def __init__(
        self,
        gamma=2.0,
        alpha=1.0
    ):

        super().__init__()

        self.gamma = gamma

        self.alpha = alpha

    def forward(
        self,
        log_probs,
        targets
    ):

        log_pt = (
            log_probs.gather(
                1,
                targets.unsqueeze(1)
            )
            .squeeze(1)
        )

        pt = torch.exp(
            log_pt
        )

        focal_factor = (
            1.0 - pt
        ).pow(
            self.gamma
        )

        loss = (
            -self.alpha
            *
            focal_factor
            *
            log_pt
        )

        return loss.mean()


# ================================================================
# METRICS
# ================================================================

def calculate_metrics(
    y_true,
    y_pred
):

    y_true = np.asarray(
        y_true
    )

    y_pred = np.asarray(
        y_pred
    )

    accuracy = (
        y_true == y_pred
    ).mean()

    f1s = []

    precisions = []

    recalls = []

    for cls in range(
        NUM_CLASSES
    ):

        tp = np.sum(
            (y_true == cls)
            &
            (y_pred == cls)
        )

        fp = np.sum(
            (y_true != cls)
            &
            (y_pred == cls)
        )

        fn = np.sum(
            (y_true == cls)
            &
            (y_pred != cls)
        )

        precision = (
            tp / (tp + fp)
            if tp + fp > 0
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn > 0
            else 0.0
        )

        f1 = (
            2.0 *
            precision *
            recall
            /
            (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        precisions.append(
            precision
        )

        recalls.append(
            recall
        )

        f1s.append(
            f1
        )

    return {
        "accuracy": float(
            accuracy
        ),

        "macro_precision": float(
            np.mean(
                precisions
            )
        ),

        "macro_recall": float(
            np.mean(
                recalls
            )
        ),

        "macro_f1": float(
            np.mean(
                f1s
            )
        ),
    }


# ================================================================
# MODEL
# ================================================================

def load_model():

    print("=" * 70)

    print(
        "LOADING AUDIOSET-PRETRAINED PANNs Cnn14"
    )

    print("=" * 70)

    print(
        "Checkpoint:"
    )

    print(
        PRETRAINED_CHECKPOINT
    )

    model = Transfer_Cnn14(
        sample_rate=32000,
        window_size=1024,
        hop_size=320,
        mel_bins=64,
        fmin=50,
        fmax=14000,
        classes_num=NUM_CLASSES,
        freeze_base=False
    )

    checkpoint = torch.load(
        PRETRAINED_CHECKPOINT,
        map_location="cpu"
    )

    if (
        isinstance(
            checkpoint,
            dict
        )
        and
        "model" in checkpoint
    ):

        state_dict = checkpoint[
            "model"
        ]

    else:

        state_dict = checkpoint

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        new_key = key

        if new_key.startswith(
            "module."
        ):

            new_key = new_key[
                len("module.") :
            ]

        if new_key.startswith(
            "base."
        ):

            new_key = new_key[
                len("base.") :
            ]

        cleaned_state_dict[
            new_key
        ] = value

    base_state = model.base.state_dict()

    compatible = {}

    for key, value in cleaned_state_dict.items():

        if (
            key in base_state
            and
            base_state[key].shape
            ==
            value.shape
        ):

            compatible[key] = value

    if len(compatible) == 0:

        raise RuntimeError(
            "Could not load compatible "
            "Cnn14 pretrained parameters."
        )

    missing, unexpected = (
        model.base.load_state_dict(
            compatible,
            strict=False
        )
    )

    print(
        f"Pretrained tensors loaded : "
        f"{len(compatible)}"
    )

    print(
        f"Missing backbone tensors   : "
        f"{len(missing)}"
    )

    print(
        f"Unexpected tensors         : "
        f"{len(unexpected)}"
    )

    # ------------------------------------------------------------
    # STRICT PRETRAINED GATE
    # ------------------------------------------------------------

    if len(compatible) != 84:

        raise RuntimeError(
            "PRETRAINED LOADING GATE FAILED: "
            f"expected exactly 84 compatible tensors, "
            f"got {len(compatible)}"
        )

    if len(missing) != 0:

        raise RuntimeError(
            "PRETRAINED LOADING GATE FAILED: "
            f"{len(missing)} backbone tensors missing."
        )

    if len(unexpected) != 0:

        raise RuntimeError(
            "PRETRAINED LOADING GATE FAILED: "
            f"{len(unexpected)} unexpected tensors."
        )

    print()
    print(
        "PRETRAINED 84/84 GATE: PASS"
    )

    model.to(
        DEVICE
    )

    return model


# ================================================================
# TRAINABLE LAYERS
# ================================================================

def configure_trainable_layers(
    model
):

    for parameter in (
        model.parameters()
    ):

        parameter.requires_grad = False

    trainable_prefixes = [

        "base.conv_block5.",

        "base.conv_block6.",

        "base.fc1.",

        "fc_transfer.",
    ]

    for name, parameter in (
        model.named_parameters()
    ):

        for prefix in trainable_prefixes:

            if name.startswith(
                prefix
            ):

                parameter.requires_grad = True

                break

    print()

    print("=" * 70)

    print(
        "TRAINABLE LAYERS"
    )

    print("=" * 70)

    total_trainable = 0

    total_frozen = 0

    for name, parameter in (
        model.named_parameters()
    ):

        count = parameter.numel()

        if parameter.requires_grad:

            total_trainable += count

            print(
                f"TRAIN : {name}"
            )

        else:

            total_frozen += count

    print()

    print(
        f"Trainable parameters : "
        f"{total_trainable:,}"
    )

    print(
        f"Frozen parameters    : "
        f"{total_frozen:,}"
    )

    print(
        f"Total parameters     : "
        f"{total_trainable + total_frozen:,}"
    )

    # ------------------------------------------------------------
    # EXPECTED EXP-09 BACKBONE COUNTS
    # ------------------------------------------------------------

    if total_trainable != 75_001_863:

        raise RuntimeError(
            "TRAINABLE PARAMETER COUNT MISMATCH: "
            f"expected 75,001,863, "
            f"got {total_trainable}"
        )

    if total_frozen != 6_849_551:

        raise RuntimeError(
            "FROZEN PARAMETER COUNT MISMATCH: "
            f"expected 6,849,551, "
            f"got {total_frozen}"
        )

    if (
        total_trainable +
        total_frozen
        !=
        81_851_414
    ):

        raise RuntimeError(
            "TOTAL PARAMETER COUNT MISMATCH: "
            f"expected 81,851,414, "
            f"got "
            f"{total_trainable + total_frozen}"
        )

    print()
    print(
        "PARAMETER COUNT GATE: PASS"
    )


# ================================================================
# FROZEN BATCHNORM
# ================================================================

def freeze_batchnorm_running_stats(
    model
):

    bn_count = 0

    for module in model.modules():

        if isinstance(
            module,
            (
                nn.BatchNorm1d,
                nn.BatchNorm2d,
                nn.BatchNorm3d,
            )
        ):

            module.eval()

            bn_count += 1

    return bn_count


def set_training_mode(
    model
):

    model.train()

    return freeze_batchnorm_running_stats(
        model
    )


# ================================================================
# OPTIMIZER
# ================================================================

def build_optimizer(
    model
):

    backbone_params = []

    head_params = []

    for name, parameter in (
        model.named_parameters()
    ):

        if not parameter.requires_grad:

            continue

        if name.startswith(
            "base."
        ):

            backbone_params.append(
                parameter
            )

        else:

            head_params.append(
                parameter
            )

    print()

    print(
        f"Trainable backbone parameters : "
        f"{sum(p.numel() for p in backbone_params):,}"
    )

    print(
        f"Trainable head parameters     : "
        f"{sum(p.numel() for p in head_params):,}"
    )

    optimizer = torch.optim.AdamW(
        [
            {
                "params": backbone_params,
                "lr": BACKBONE_LR,
            },
            {
                "params": head_params,
                "lr": HEAD_LR,
            },
        ],
        weight_decay=WEIGHT_DECAY
    )

    return optimizer


# ================================================================
# ATOMIC CHECKPOINT SAVE
# ================================================================

def atomic_torch_save(
    checkpoint,
    final_path
):

    temp_path = (
        final_path +
        ".tmp"
    )

    torch.save(
        checkpoint,
        temp_path
    )

    os.replace(
        temp_path,
        final_path
    )


# ================================================================
# CHECKPOINT BUILDER
# ================================================================

def make_checkpoint(
    model,
    optimizer,
    epoch,
    best_val_f1
):

    return {

        "epoch": epoch,

        "model":
            model.state_dict(),

        "optimizer":
            optimizer.state_dict(),

        "best_val_macro_f1":
            best_val_f1,

        "config": {

            "experiment":
                "EXP-19",

            "base_experiment":
                "EXP-09",

            "excluded_class":
                "Pop",

            "num_classes":
                NUM_CLASSES,

            "class_names":
                CLASS_NAMES,

            "sample_rate":
                SAMPLE_RATE,

            "track_seconds":
                TRACK_SECONDS,

            "track_samples":
                TRACK_SAMPLES,

            "crop_seconds":
                CROP_SECONDS,

            "crop_samples":
                CROP_SAMPLES,

            "training_crop":
                "random_continuous",

            "validation_crops":
                "0-10,10-20,20-30",

            "num_val_crops":
                NUM_VAL_CROPS,

            "backbone_lr":
                BACKBONE_LR,

            "head_lr":
                HEAD_LR,

            "weight_decay":
                WEIGHT_DECAY,

            "batch_size":
                BATCH_SIZE,

            "seed":
                SEED,

            "frozen_batchnorm":
                True,

            "loss":
                "focal",

            "focal_gamma":
                FOCAL_GAMMA,

            "focal_alpha":
                FOCAL_ALPHA,

            "pop_excluded":
                True,
        },
    }


# ================================================================
# HISTORY SAVE
# ================================================================

def save_history(
    history
):

    temp_path = (
        HISTORY_PATH +
        ".tmp"
    )

    with open(
        temp_path,
        "w"
    ) as f:

        json.dump(
            history,
            f,
            indent=2
        )

    os.replace(
        temp_path,
        HISTORY_PATH
    )


# ================================================================
# RESUME CHECKPOINT
# ================================================================

def load_resume_checkpoint(
    model,
    optimizer
):

    if not os.path.exists(
        LAST_PATH
    ):

        print()
        print(
            "No last.pth found."
        )

        print(
            "Starting EXP-19 from scratch."
        )

        return (
            0,
            -1.0,
            []
        )

    print()
    print("=" * 70)
    print("RESUME CHECKPOINT FOUND")
    print("=" * 70)

    print(
        LAST_PATH
    )

    checkpoint = torch.load(
        LAST_PATH,
        map_location=DEVICE
    )

    required_keys = [
        "epoch",
        "model",
        "optimizer",
        "best_val_macro_f1",
    ]

    for key in required_keys:

        if key not in checkpoint:

            raise RuntimeError(
                f"Invalid checkpoint: "
                f"missing '{key}'"
            )

    # ------------------------------------------------------------
    # CONFIG SAFETY
    # ------------------------------------------------------------

    saved_config = checkpoint.get(
        "config",
        {}
    )

    if saved_config.get(
        "experiment"
    ) != "EXP-19":

        raise RuntimeError(
            "Checkpoint experiment mismatch. "
            "This is not an EXP-19 checkpoint."
        )

    if saved_config.get(
        "num_classes"
    ) != NUM_CLASSES:

        raise RuntimeError(
            "Checkpoint NUM_CLASSES mismatch."
        )

    if saved_config.get(
        "pop_excluded"
    ) is not True:

        raise RuntimeError(
            "Checkpoint does not confirm Pop exclusion."
        )

    model.load_state_dict(
        checkpoint["model"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer"]
    )

    start_epoch = int(
        checkpoint["epoch"]
    )

    best_val_f1 = float(
        checkpoint[
            "best_val_macro_f1"
        ]
    )

    history = []

    if os.path.exists(
        HISTORY_PATH
    ):

        try:

            with open(
                HISTORY_PATH,
                "r"
            ) as f:

                history = json.load(
                    f
                )

        except Exception:

            print(
                "WARNING: history.json "
                "could not be loaded."
            )

            history = []

    print(
        f"Last completed epoch : "
        f"{start_epoch}"
    )

    print(
        f"Best validation F1    : "
        f"{best_val_f1:.6f}"
    )

    print(
        f"History records        : "
        f"{len(history)}"
    )

    print()
    print(
        f"RESUMING FROM EPOCH "
        f"{start_epoch + 1}"
    )

    print("=" * 70)

    return (
        start_epoch,
        best_val_f1,
        history
    )


# ================================================================
# TRAIN ONE EPOCH
# ================================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion
):

    set_training_mode(
        model
    )

    total_loss = 0.0

    total_samples = 0

    for waveforms, labels in loader:

        waveforms = waveforms.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        # --------------------------------------------------------
        # POP CONTAMINATION SAFETY
        # --------------------------------------------------------

        if torch.any(
            labels < 0
        ) or torch.any(
            labels >= NUM_CLASSES
        ):

            raise RuntimeError(
                "Invalid label detected in training batch."
            )

        optimizer.zero_grad(
            set_to_none=True
        )

        output = model(
            waveforms
        )

        logits = output[
            "clipwise_output"
        ]

        loss = criterion(
            logits,
            labels
        )

        if not torch.isfinite(
            loss
        ):

            raise RuntimeError(
                f"Non-finite training loss: "
                f"{loss.item()}"
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP
        )

        optimizer.step()

        batch_size = (
            labels.size(0)
        )

        total_loss += (
            loss.item()
            *
            batch_size
        )

        total_samples += (
            batch_size
        )

    return (
        total_loss /
        total_samples
    )


# ================================================================
# VALIDATION
# ================================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion
):

    model.eval()

    freeze_batchnorm_running_stats(
        model
    )

    total_loss = 0.0

    total_samples = 0

    all_true = []

    all_pred = []

    for crops, labels in loader:

        batch_size = (
            crops.size(0)
        )

        crops = crops.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        # --------------------------------------------------------
        # POP CONTAMINATION SAFETY
        # --------------------------------------------------------

        if torch.any(
            labels < 0
        ) or torch.any(
            labels >= NUM_CLASSES
        ):

            raise RuntimeError(
                "Invalid label detected in validation batch."
            )

        crops_flat = crops.reshape(
            batch_size *
            NUM_VAL_CROPS,
            CROP_SAMPLES
        )

        output = model(
            crops_flat
        )

        logits = output[
            "clipwise_output"
        ]

        logits = logits.reshape(
            batch_size,
            NUM_VAL_CROPS,
            NUM_CLASSES
        )

        probabilities = torch.exp(
            logits
        )

        mean_probabilities = (
            probabilities.mean(
                dim=1
            )
        )

        track_log_probs = torch.log(
            mean_probabilities.clamp(
                min=1e-8
            )
        )

        loss = criterion(
            track_log_probs,
            labels
        )

        if not torch.isfinite(
            loss
        ):

            raise RuntimeError(
                f"Non-finite validation loss: "
                f"{loss.item()}"
            )

        total_loss += (
            loss.item()
            *
            batch_size
        )

        total_samples += (
            batch_size
        )

        predictions = torch.argmax(
            mean_probabilities,
            dim=1
        )

        all_true.extend(
            labels.cpu().numpy()
        )

        all_pred.extend(
            predictions.cpu().numpy()
        )

    metrics = calculate_metrics(
        all_true,
        all_pred
    )

    metrics["loss"] = (
        total_loss /
        total_samples
    )

    return metrics


# ================================================================
# MAIN
# ================================================================

def main():

    print("=" * 70)

    print(
        "EXP-19: POP-EXCLUDED 7-CLASS PANNs"
    )

    print("=" * 70)

    print()
    print(
        "BASE EXPERIMENT : EXP-09"
    )

    print(
        "EXCLUDED CLASS  : Pop"
    )

    print(
        "NUM CLASSES     : 7"
    )

    print()

    for idx, name in enumerate(
        CLASS_NAMES
    ):

        print(
            f"  {idx} = {name}"
        )

    print()

    print(
        f"Device        : {DEVICE}"
    )

    if DEVICE == "cuda":

        print(
            f"GPU           : "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"Sample rate   : {SAMPLE_RATE}"
    )

    print(
        f"Track length  : {TRACK_SECONDS} sec"
    )

    print(
        f"Track samples : {TRACK_SAMPLES}"
    )

    print(
        f"Crop length   : {CROP_SECONDS} sec"
    )

    print(
        "Training crop : continuous random"
    )

    print(
        "Validation    : 3 deterministic crops"
    )

    print(
        "Validation    : Pop excluded"
    )

    print(
        f"Batch size    : {BATCH_SIZE}"
    )

    print(
        f"Max epochs    : {MAX_EPOCHS}"
    )

    print(
        f"Backbone LR   : {BACKBONE_LR}"
    )

    print(
        f"Head LR       : {HEAD_LR}"
    )

    print(
        f"Weight decay  : {WEIGHT_DECAY}"
    )

    print(
        f"Focal gamma   : {FOCAL_GAMMA}"
    )

    print(
        f"Focal alpha   : {FOCAL_ALPHA}"
    )

    print(
        f"Seed          : {SEED}"
    )

    print()
    print(
        "TEST SET WILL NOT BE USED."
    )

    # ------------------------------------------------------------
    # DATASET
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("LOADING DATASET")
    print("=" * 70)

    train_dataset = FMA8Dataset(
        HDF5_PATH,
        "train",
        training=True
    )

    val_dataset = FMA8Dataset(
        HDF5_PATH,
        "val",
        training=False
    )

    print()
    print("=" * 70)
    print("DATASET SAFETY")
    print("=" * 70)

    print(
        f"Training samples after Pop removal : "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation samples after Pop removal : "
        f"{len(val_dataset)}"
    )

    print(
        "Pop samples are excluded."
    )

    print(
        "Test samples are not loaded."
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(
            DEVICE == "cuda"
        ),
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            DEVICE == "cuda"
        ),
        drop_last=False
    )

    # ------------------------------------------------------------
    # MODEL
    # ------------------------------------------------------------

    model = load_model()

    configure_trainable_layers(
        model
    )

    bn_count = (
        freeze_batchnorm_running_stats(
            model
        )
    )

    print()
    print(
        f"BatchNorm layers controlled : "
        f"{bn_count}"
    )

    if bn_count != 13:

        raise RuntimeError(
            f"BatchNorm count mismatch: "
            f"expected 13, got {bn_count}"
        )

    print(
        "BATCHNORM GATE: PASS"
    )

    optimizer = build_optimizer(
        model
    )

    criterion = FocalLoss(
        gamma=FOCAL_GAMMA,
        alpha=FOCAL_ALPHA
    )

    print()
    print("=" * 70)
    print("LOSS FUNCTION")
    print("=" * 70)

    print(
        "Focal Loss"
    )

    print(
        f"gamma = {FOCAL_GAMMA}"
    )

    print(
        f"alpha = {FOCAL_ALPHA}"
    )

    # ------------------------------------------------------------
    # RESUME
    # ------------------------------------------------------------

    (
        start_epoch,
        best_val_f1,
        history
    ) = load_resume_checkpoint(
        model,
        optimizer
    )

    # ------------------------------------------------------------
    # IF ALREADY COMPLETE
    # ------------------------------------------------------------

    if start_epoch >= MAX_EPOCHS:

        print()
        print("=" * 70)
        print("TRAINING ALREADY COMPLETE")
        print("=" * 70)

        print(
            f"Last epoch : {start_epoch}"
        )

        print(
            f"Best F1    : {best_val_f1:.6f}"
        )

        return

    # ------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING EXP-19")
    print("=" * 70)

    print()
    print(
        "Pop is COMPLETELY EXCLUDED."
    )

    print(
        "Only 7 genres are being trained."
    )

    print(
        "Validation also contains only 7 genres."
    )

    print()

    start_time = time.time()

    final_epoch = start_epoch

    try:

        for epoch in range(
            start_epoch + 1,
            MAX_EPOCHS + 1
        ):

            epoch_start = time.time()

            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion
            )

            val_metrics = evaluate(
                model,
                val_loader,
                criterion
            )

            record = {

                "epoch": epoch,

                "train_loss": float(
                    train_loss
                ),

                "val_loss": float(
                    val_metrics[
                        "loss"
                    ]
                ),

                "val_accuracy": float(
                    val_metrics[
                        "accuracy"
                    ]
                ),

                "val_macro_precision": float(
                    val_metrics[
                        "macro_precision"
                    ]
                ),

                "val_macro_recall": float(
                    val_metrics[
                        "macro_recall"
                    ]
                ),

                "val_macro_f1": float(
                    val_metrics[
                        "macro_f1"
                    ]
                ),
            }

            history.append(
                record
            )

            current_f1 = (
                val_metrics[
                    "macro_f1"
                ]
            )

            # ----------------------------------------------------
            # BEST
            # ----------------------------------------------------

            if current_f1 > best_val_f1:

                best_val_f1 = current_f1

                marker = "<-- BEST"

                best_checkpoint = (
                    make_checkpoint(
                        model,
                        optimizer,
                        epoch,
                        best_val_f1
                    )
                )

                atomic_torch_save(
                    best_checkpoint,
                    BEST_PATH
                )

                print()
                print(
                    "BEST CHECKPOINT SAVED"
                )

            else:

                marker = ""

            # ----------------------------------------------------
            # LAST
            # ----------------------------------------------------

            last_checkpoint = (
                make_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    best_val_f1
                )
            )

            atomic_torch_save(
                last_checkpoint,
                LAST_PATH
            )

            save_history(
                history
            )

            final_epoch = epoch

            epoch_time = (
                time.time()
                -
                epoch_start
            )

            print(
                f"Epoch {epoch:3d}/{MAX_EPOCHS} | "
                f"Loss {train_loss:.4f} | "
                f"Val Acc "
                f"{val_metrics['accuracy']:.4f} | "
                f"Val F1 "
                f"{current_f1:.6f} "
                f"{marker} | "
                f"{epoch_time / 60:.2f} min"
            )

            # ----------------------------------------------------
            # EARLY STOPPING
            # ----------------------------------------------------

            epochs_without_improvement = 0

            for previous in reversed(
                history
            ):

                if (
                    previous[
                        "val_macro_f1"
                    ]
                    >
                    best_val_f1
                    -
                    1e-12
                ):

                    break

                epochs_without_improvement += 1

            if (
                epochs_without_improvement
                >= PATIENCE
            ):

                print()
                print(
                    f"Early stopping at epoch "
                    f"{epoch}."
                )

                break

            # ----------------------------------------------------
            # SAFE STOP
            # ----------------------------------------------------

            if STOP_REQUESTED:

                print()
                print("=" * 70)
                print(
                    "SAFE STOP COMPLETE"
                )
                print("=" * 70)

                print(
                    f"Last completed epoch : "
                    f"{final_epoch}"
                )

                print(
                    "Checkpoint saved."
                )

                print(
                    "Run the same script again "
                    "to resume."
                )

                break

    except KeyboardInterrupt:

        print()
        print("=" * 70)
        print(
            "KEYBOARD INTERRUPT"
        )
        print("=" * 70)

        print(
            "The last completed epoch "
            "checkpoint remains available."
        )

    # ------------------------------------------------------------
    # FINAL
    # ------------------------------------------------------------

    elapsed = (
        time.time()
        -
        start_time
    )

    print()
    print("=" * 70)
    print("EXP-19 TRAINING STATUS")
    print("=" * 70)

    print(
        f"Last completed epoch : "
        f"{final_epoch}"
    )

    print(
        f"Best validation F1    : "
        f"{best_val_f1:.6f}"
    )

    print(
        f"Session time          : "
        f"{elapsed / 60:.2f} min"
    )

    print()
    print(
        "Best checkpoint:"
    )

    print(
        BEST_PATH
    )

    print()
    print(
        "Last checkpoint:"
    )

    print(
        LAST_PATH
    )

    print()
    print(
        "History:"
    )

    print(
        HISTORY_PATH
    )

    print()
    print(
        "TEST SET WAS NOT USED."
    )

    print()
    print(
        "EXP-19 CLASSIFICATION SPACE:"
    )

    for idx, name in enumerate(
        CLASS_NAMES
    ):

        print(
            f"  {idx} = {name}"
        )

    print()
    print(
        "POP WAS COMPLETELY EXCLUDED."
    )

    print("=" * 70)


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":

    main()