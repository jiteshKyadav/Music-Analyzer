# Deep Learning Artifacts – Music Analyzer

This directory contains the datasets and trained model artifacts required to train and evaluate the Music Analyzer genre classification model.

The large files are hosted externally on Google Drive instead of being uploaded to GitHub.

## 1. Download the Artifacts

**[Download Music Analyzer – dl_artifacts (Google Drive)](https://drive.google.com/drive/folders/1vBziDjGhhBTkHaay8BCDuRAVr8e3LbSy?usp=drive_link)**

The Google Drive folder contains the following resources:

### Dataset – `fma8/`

| File | Description |
|---|---|
| `features/waveform.h5` | Preprocessed FMA-small audio waveforms and associated labels and split information used by the training and evaluation scripts. |

The audio waveforms are stored at 32 kHz, with 30 seconds of audio per track.

### Fine-tuned Model – `fma8_finetune/`

The EXP-19 experiment directory contains:

| File | Description |
|---|---|
| `best.pth` | Best model checkpoint selected using validation Macro F1. |
| `last.pth` | Latest training checkpoint, including optimizer and training state for resuming training. |
| `history.json` | Training history containing epoch-wise metrics. |

**Experiment:** EXP-19 – Pop Excluded 7-Class PANNs Cnn14.

## 2. Expected Directory Structure

After downloading and extracting the artifacts, arrange them as follows:

```text
DeepLearning/
│
├── dl_artifacts/
│   │
│   ├── fma8/
│   │   └── features/
│   │       └── waveform.h5
│   │
│   └── fma8_finetune/
│       └── EXP-19_PopExcluded_7Class/
│           ├── best.pth
│           ├── last.pth
│           └── history.json
│
├── panns/
│   └── panns-transfer/
│       └── checkpoints/
│           └── Cnn14_mAP=0.431.pth
│
├── EXP-19_PopExcluded_7Class.py
└── EXP-19_PopExcluded_7Class_TEST.py
```

Keep the directory names and file names unchanged because the scripts expect these paths.

## 3. PANNs Setup

Music Analyzer uses the PANNs Cnn14 architecture with transfer learning.

The PANNs source code and its pretrained AudioSet checkpoint are separate from the Google Drive artifacts.

**Official PANNs repository:**

https://github.com/qiuqiangkong/audioset_tagging_cnn

Clone the repository into the expected directory:

```bash
mkdir -p DeepLearning/panns

git clone https://github.com/qiuqiangkong/audioset_tagging_cnn.git DeepLearning/panns/panns-transfer
```

Download the pretrained Cnn14 checkpoint:

**Pretrained checkpoint:** `Cnn14_mAP=0.431.pth`

Official download source:

https://zenodo.org/records/3960586

Place the downloaded checkpoint inside:

```text
DeepLearning/panns/panns-transfer/checkpoints/
```

The final checkpoint path should be:

```text
DeepLearning/panns/panns-transfer/checkpoints/Cnn14_mAP=0.431.pth
```

Refer to the official PANNs repository for its installation requirements and implementation details.

## 4. Dataset

- **FMA-small:** https://os.unil.cloud.switch.ch/fma/fma_small.zip
- **Official Dataset Documentation:** https://github.com/mdeff/fma
  
## 5. Python Dependencies

The training and evaluation scripts use the following main Python libraries:

- Python
- PyTorch
- NumPy
- h5py
- scikit-learn (for the evaluation script)

The PANNs implementation also has its own dependencies.

Install the dependencies required by the official PANNs repository and ensure that the Python environment has the libraries used by the Music Analyzer scripts.

GPU acceleration is recommended for training. CPU execution may be possible but can be considerably slower.

## 6. Running the Training Script

Before running the scripts, open:

`EXP-19_PopExcluded_7Class.py`

Update the `PROJECT_ROOT` variable to the absolute path of your local Music Analyzer project directory.

Verify that the following resources are available:

- FMA waveform dataset.
- PANNs source code.
- Pretrained Cnn14 checkpoint.

Run the training script:

```bash
python DeepLearning/EXP-19_PopExcluded_7Class.py
```

The training script:

- Excludes the Pop genre from training and validation.
- Uses seven music genres.
- Uses random 10-second audio crops for training.
- Uses three deterministic crops for validation.
- Uses Focal Loss.
- Selects the best checkpoint using validation Macro F1.
- Keeps the test set separate from training and model selection.
- Saves training checkpoints and history in the EXP-19 experiment directory.

## 7. Running Test Evaluation

After training, or after downloading the supplied EXP-19 checkpoint, open:

`EXP-19_PopExcluded_7Class_TEST.py`

Update its `PROJECT_ROOT` variable to your local project directory.

Run:

```bash
python DeepLearning/EXP-19_PopExcluded_7Class_TEST.py
```

The evaluation script:

- Loads the EXP-19 best checkpoint.
- Excludes Pop samples from the test task.
- Uses five deterministic 10-second crops per track.
- Averages crop-level probabilities for final predictions.
- Calculates accuracy, macro precision, macro recall and macro F1.
- Generates per-class metrics and a confusion matrix.
- Saves the evaluation results as a JSON file.

The test set is used only for final evaluation, not for training or model selection.

## 8. Important Notes

1. Download the artifacts from the Google Drive link before running the scripts.
2. Do not rename the expected dataset, checkpoint or experiment directories.
3. Update `PROJECT_ROOT` in both scripts to match your local project location.
4. Keep the original dataset splits unchanged when reproducing the reported experiment.
5. The supplied `best.pth` checkpoint can be used for evaluation without retraining.
6. The `last.pth` checkpoint is intended for resuming training.
7. Do not use the test set to tune model parameters or select checkpoints.
8. The scripts and artifacts represent the EXP-19 seven-class experiment. Older experiments and unrelated evaluation scripts are not part of this final pipeline.

---

**Music Analyzer – Deep Learning Module**

PANNs Cnn14 | Transfer Learning | FMA-small | PyTorch
