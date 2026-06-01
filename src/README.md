# `src/` — Python source

Reference implementation of FedSafe-Fuse. The Colab notebooks in `../notebooks/`
embed these modules inline so they run self-contained on a fresh runtime.

```
src/
├── models/
│   ├── fedsafe.py    # dual MobileNetV3-Small + 2-layer Transformer + conv decoder (~4.9M params)
│   └── ifcnn.py      # IFCNN baseline (Zhang et al. 2021)
├── fed/
│   └── trainer.py    # in-process FedAvg simulator: standard / DP-SGD / FIPCA modes
├── losses.py         # L1 + beta*SSIM composite loss + SSIM/PSNR scorers
└── data.py           # MedMNIST + BraTS dataset wrappers and partition loader
```

All random seeds are fixed to 42.
