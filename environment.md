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

- BraTS 2020 retrieved from a **Kaggle redistribution**, not the official Synapse release. Disclosed in paper Methods. Citations: Menze 2015, Bakas 2017.
- DLG attack (Zhu et al. 2019) run against a **small encoder-only surrogate at batch size 1** rather than the full 8M-param fusion model. Disclosed in paper privacy section.
