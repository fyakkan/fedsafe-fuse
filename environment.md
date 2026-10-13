# Environment notes

## Target runtime

**Google Colab Free.** GPU = T4 (when available); falls back to CPU if quota exhausted. Session caps:
- ~12 h max session length
- ~90 min idle disconnect
- One concurrent session per account

**No Pro / Pro+ / Kaggle backup.** All training must survive disconnect via 5-round checkpoint cadence.

## Python

Python 3.10 (Colab default as of 2026-05). Pin versions in `requirements.txt`; Colab already provides torch/torchvision/numpy/pandas/matplotlib so `pip install` will mostly upgrade and add `flwr`, `medmnist`, `opacus`, `einops`.

## Local machine

- macOS (Darwin 24.6.0). No GPU.
- Used for: LaTeX, plotting from downloaded CSVs, code editing, git operations.
- **Not** used for: training, federated simulation, DLG attack — those run on Colab.

## Reproducibility

- Seed = 42 across `random`, `numpy`, `torch`, `torch.cuda`.
- Deterministic CuDNN where it doesn't kill performance.
- Hyperparameters logged into every results CSV row.
- Checkpoints saved every 5 federated rounds.

## Known constraints

- BraTS 2020 retrieved from a **Kaggle redistribution** (`awsaf49/brats20-dataset-training-validation`), not the official Synapse release. Stated in the paper's Experimental Setup. Citations: Menze 2015, Bakas 2017.
- Round-2 runs (notebooks 13–15) use seeds {42, 1, 2, 3, 4}; protocol fixed in advance, no model selection.
- Earlier-version items (DLG surrogate, 4.9M-parameter MobileNetV3 backbone, FROP, batch-mean "DP-SGD") are archived and no longer part of the paper.
