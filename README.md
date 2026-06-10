# FedSafe-Fuse

**Privacy-Preserving Federated Medical Image Fusion with Spatially-Aware Conformal Uncertainty.**
Term project for *Advanced Applications & Performance Evaluation*.
Author: **Furkan Yakkan**, Abdullah Gül University, Kayseri / Türkiye.

The paper is at `paper/main.pdf` (8 pages, IEEE conference format).
A ready-to-submit zip is built by `scripts/build_submission_zip.py` (`submission.zip`).

## What's inside

FedSafe-Fuse is a federated framework whose **primary contributions are two privacy/uncertainty
mechanisms**; the fusion backbone is the vehicle through which they are demonstrated and stress-tested
(non-IID, K=3 hospitals):

1. **FIPCA (orthonormal rank-d gradient projection)** (`src/fed/trainer.py`) — replaces full
   weight-delta transmission with a 4d-byte rank-d coefficient vector, providing privacy against
   gradient inversion at four orders of magnitude lower bandwidth than DP-SGD.
2. **Spatially-Aware Conformal Prediction** (`notebooks/04_sacp.ipynb`) — Canny-edge region split +
   regional (1-α) quantile thresholds, with empirically validated coverage on **both** synthetic
   MedMNIST and real BraTS data.
3. **Lightweight Transformer–CNN fusion backbone** (`src/models/fedsafe.py`) — dual MobileNetV3-Small
   encoders + 2-layer Transformer + conv decoder, **4.9 M** trainable parameters.

## Headline numbers

**Synthetic MedMNIST OrganAMNIST (K=3, T=50):**

| Method                       | Privacy            | SSIM     | Bytes/round |
|------------------------------|--------------------|----------|-------------|
| FedAvg + IFCNN (B2)          | none               | 0.972    | 3.81 MB     |
| FedAvg + DP-SGD (B1)         | DP, σ=0.5          | 0.235    | 59.18 MB    |
| FedSafe-Fuse + FIPCA (ours)  | rank-32 projection | 0.239    | **384 B**   |

**Real multi-modal BraTS 2020 (T1/T2, HGG/LGG K=3):** FedAvg + IFCNN converges to **SSIM 0.995,
PSNR 38.9 dB** — the federated fusion task and pipeline are sound on real clinical imagery.

**Privacy.** Under DLG (Zhu et al. 2019), reconstruction SSIM = 0.9999 with raw gradients vs ≤ 0.013
under both DP-SGD and FIPCA — **FIPCA matches DP-SGD's privacy at 154,109× lower communication.**
This holds on three axes: (i) the tractable surrogate, (ii) the **full 4.9 M-parameter backbone**
(even raw gradients barely reconstruct, mean SSIM ≈ 0.06), justified by gradient-leakage proxies that
*shrink* with model size, and (iii) a **formal Rényi-DP bound** for the noise baseline
(**ε ≈ 6.2** at δ=10⁻⁵, computed by `scripts/dp_accounting_fedsafe.py`).

## Repository layout

```
.
├── paper/                  IEEE LaTeX source
│   ├── main.tex            full paper (compile with pdflatex + bibtex)
│   ├── refs.bib            17 BibTeX entries
│   └── tables/             9 auto-generated LaTeX result tables
├── src/                    Reference Python implementation
│   ├── models/             FedSafe-Fuse and IFCNN
│   ├── fed/                In-process FedAvg simulator (standard / DP-SGD / FIPCA + error feedback)
│   ├── losses.py           L1 + β·SSIM composite + SSIM/PSNR scorers
│   └── data.py             MedMNIST + BraTS Dataset wrappers
├── notebooks/              Colab notebooks, one per phase
│   ├── 01_data.ipynb         Data acquisition + non-IID partitioning
│   ├── 02_smoketest.ipynb    Architecture smoke test + GPU timing
│   ├── 03_fedtrain.ipynb     Federated training (3 methods, MedMNIST)
│   ├── 04_sacp.ipynb         SACP calibration + coverage validation
│   ├── 05_attack.ipynb       DLG attack (surrogate) + comm-cost summary
│   ├── 06_ablations.ipynb    FIPCA rank ablation
│   ├── 10a_quality.ipynb     FIPCA + error feedback (negative result)
│   ├── 10a2_quality.ipynb    FIPCA + basis resampling (divergence diagnosis)
│   ├── 10b_brats.ipynb       Real BraTS T1/T2 federated run + SACP coverage
│   └── 10c_dlg_full.ipynb    DLG vs full 4.9 M backbone + leakage proxies
├── scripts/
│   ├── build_paper_artifacts.py   Regenerate tables + ablation figure from CSVs
│   ├── build_submission_zip.py    Package the submission .zip
│   └── dp_accounting_fedsafe.py   Standalone Rényi-DP (ε, δ) accountant
├── results/                CSV outputs from every Colab run
├── figures/                PNG figures used in the paper
└── partitions/             non-IID partition manifests + calibration hold-out
```

## Reproducing the results

All training and inference run on Google Colab Free with a T4 GPU. The local machine only needs
`pdflatex`, `python3`, `matplotlib`, `pandas`, and `scipy` (the DP accountant) for the build scripts.

| Phase | Notebook                       | T4 wall-clock | Output(s)                                            |
|-------|--------------------------------|---------------|------------------------------------------------------|
| 1     | `01_data.ipynb`                | ~50 min       | `partitions/*.json`, sanity figure                   |
| 2     | `02_smoketest.ipynb`           | ~5 min        | smoke-test CSV + timing                              |
| 3     | `03_fedtrain.ipynb`            | ~20 min       | `results/round1_main.csv`, convergence + comm figures, checkpoints |
| 4     | `04_sacp.ipynb`                | ~8 min        | `results/coverage_table.csv`, SACP scores, heatmaps  |
| 5     | `05_attack.ipynb`              | ~5 min        | `results/attack_table.csv`, comm-cost summary, attack figures |
| 6     | `06_ablations.ipynb`           | ~12 min       | `results/ablation_fipca_rank.csv`, ablation figure   |
| 10b   | `10b_brats.ipynb`              | ~25 min       | `results/brats_train.csv`, `results/coverage_table_brats.csv` |
| 10c   | `10c_dlg_full.ipynb`           | ~15 min       | `results/dlg_full_backbone.csv`, `results/leakage_proxies.csv` |

The formal DP accounting runs **locally** (no GPU): `python3 scripts/dp_accounting_fedsafe.py` →
`results/dp_accounting.csv`.

After the notebooks complete, build the paper locally:

```bash
python3 scripts/build_paper_artifacts.py   # regenerate the 9 LaTeX tables + cleaned figure
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
python3 scripts/build_submission_zip.py    # produce submission.zip
```

All random seeds are fixed to 42. Locked hyperparameters and any deviations from the original
proposal are documented in the paper (Experimental Setup and Limitations).

## Design notes

Disclosed in the paper (Experimental Setup, Methodology, and Limitations):

1. **FIPCA is an orthonormal random projection**, not the data-aware incremental PCA of the proposal.
   The communication and privacy properties carry over; a learned data-aware basis remains the
   principled path to also recovering fusion quality. We report a **negative result**: error feedback
   cannot rescue quality under a fixed basis (provably inert) or a randomly-resampled one (divergent).
2. **In-process FedAvg simulator** instead of Flower — identical round structure and algorithm.
3. **Per-client samples capped at 40 per local epoch** and **BraTS run at 12 cases/client** to fit the
   ~4-hour Colab-Free T4 budget; single seed (42).

## License

MIT — see `LICENSE`.

## Acknowledgements

This work was developed with the assistance of **Claude (Anthropic)** as a programming and writing
assistant. All technical content, experimental results, and the final manuscript were reviewed,
validated, and edited by the author, who takes full responsibility for accuracy, originality, and
academic integrity.
