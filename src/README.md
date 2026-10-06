# `src/` — Python source

Reference implementation of FedSafe-Fuse. The Colab notebooks in `../notebooks/`
embed these modules inline so they run self-contained on a fresh runtime.

```
src/
├── models/
│   ├── fedsafe.py    # dual MobileNetV3-Small + 2-layer Transformer + conv decoder (~4.9M params)
│   └── ifcnn.py      # IFCNN baseline (Zhang et al. 2021)
├── fed/
│   └── trainer.py    # in-process FedAvg simulator: standard / FROP ('fipca') modes
│                     #   ('dpsgd' is deprecated: batch-mean clipping, not DP-SGD)
├── conformal.py      # split-conformal SACP: exact order-statistic quantile, patient-level
│                     #   guarantee, 3-region Mondrian thresholds, federated quantile, CRC
├── losses.py         # L1 + beta*SSIM losses (input-composite and held-out target) + scorers
└── data.py           # MedMNIST + BraTS dataset wrappers and partition loader
```

All random seeds are fixed to 42.

Unit tests: `python3 -m pytest tests -q` (CPU, a few seconds).
