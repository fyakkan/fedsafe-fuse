# FedSafe-Fuse

**Privacy-Preserving Federated Medical Image Fusion with Spatially-Aware Conformal Uncertainty.**
Round 1 term project for *Advanced Applications & Performance Evaluation*.
Author: **Furkan Yakkan**, Abdullah Gül University, Kayseri / Türkiye.

The paper draft is at `paper/main.pdf` (6 pages, IEEE conference format).
A ready-to-submit zip is at `submission.zip` (558 KB).

## What's inside

FedSafe-Fuse combines three components on a non-IID K=3 federated setting:

1. **Lightweight Transformer–CNN fusion backbone** (`src/models/fedsafe.py`) — dual MobileNetV3-Small encoders + 2-layer Transformer + conv decoder, **4.9 M** trainable parameters.
2. **FIPCA (orthonormal rank-d gradient projection)** (`src/fed/trainer.py`) — replaces full weight-delta transmission with a 4d-byte rank-d coefficient vector, providing privacy against gradient inversion at four orders of magnitude lower bandwidth than DP-SGD.
3. **Spatially-Aware Conformal Prediction** (`notebooks/04_sacp.ipynb`) — Canny-edge region split + regional (1-α) quantile thresholds, with empirically validated coverage on a held-out MedMNIST test set.

## Headline Round-1 numbers (MedMNIST OrganAMNIST, K=3, T=50)

| Method                       | Privacy            | SSIM     | Bytes/round |
|------------------------------|--------------------|----------|-------------|
| FedAvg + IFCNN (B2)          | none               | 0.972    | 3.81 MB     |
| FedAvg + DP-SGD (B1)         | DP, σ=0.5          | 0.235    | 59.18 MB    |
| FedSafe-Fuse + FIPCA (ours)  | rank-32 projection | 0.239    | **384 B**   |

**Privacy under DLG (Zhu et al. 2019), measured on a 10K-param surrogate:** reconstruction SSIM = 0.9999 with raw gradients vs ≤ 0.013 under both DP-SGD and FIPCA. **FIPCA matches DP-SGD's privacy at 154,109× lower communication cost.**

## Repository layout

```
.
├── paper/                  IEEE LaTeX source
│   ├── main.tex            full paper (compile with pdflatex + bibtex)
│   ├── main.pdf            compiled paper (6 pages)
│   ├── refs.bib            14 BibTeX entries
│   └── tables/             4 auto-generated LaTeX result tables
├── src/                    Reference Python implementation
│   ├── models/             FedSafe-Fuse and IFCNN
│   ├── fed/                In-process FedAvg simulator (standard / DP-SGD / FIPCA)
│   ├── losses.py           L1 + β·SSIM composite + SSIM/PSNR scorers
│   └── data.py             MedMNIST + BraTS Dataset wrappers
├── notebooks/              Colab notebooks, one per phase
│   ├── 01_data.ipynb         Data acquisition + non-IID partitioning
│   ├── 02_smoketest.ipynb    Architecture smoke test + GPU timing
│   ├── 03_fedtrain.ipynb     Federated training (3 methods)
│   ├── 04_sacp.ipynb         SACP calibration + coverage validation
│   ├── 05_attack.ipynb       DLG attack + comm-cost summary
│   └── 06_ablations.ipynb    FIPCA rank ablation
├── scripts/
│   ├── build_paper_artifacts.py   Regenerate tables + ablation figure from CSVs
│   └── build_submission_zip.py    Package the submission .zip
├── results/                CSV outputs from every Colab run
├── figures/                PNG figures used in the paper
├── partitions/             non-IID partition manifests + calibration hold-out
└── submission.zip          Final submission package
```

## Reproducing the Round-1 results

All training and inference runs on Google Colab Free with a T4 GPU. The local Mac only needs `pdflatex`, `python3`, `matplotlib`, and `pandas` for the paper-building scripts.

| Phase | Notebook                       | Wall-clock on T4 | Output(s)                                                                |
|-------|--------------------------------|------------------|---------------------------------------------------------------------------|
| 1     | `notebooks/01_data.ipynb`      | ~50 min (one-time, includes BraTS download) | `partitions/*.json`, `figures/phase1_sanity.png` |
| 2     | `notebooks/02_smoketest.ipynb` | ~5 min           | `results/smoke_test.csv`, `results/timing_extrapolation.txt`              |
| 3     | `notebooks/03_fedtrain.ipynb`  | ~20 min          | `results/round1_main.csv`, `figures/convergence_curves.png`, `figures/comm_cost_per_round.png`, checkpoints |
| 4     | `notebooks/04_sacp.ipynb`      | ~8 min           | `results/coverage_table.csv`, `results/sacp_scores.npz`, `figures/uncertainty_heatmaps.png` |
| 5     | `notebooks/05_attack.ipynb`    | ~5 min           | `results/attack_table.csv`, `results/comm_cost_summary.csv`, `figures/{attack_reconstructions,privacy_vs_comm}.png` |
| 6     | `notebooks/06_ablations.ipynb` | ~12 min          | `results/ablation_fipca_rank.csv`, `figures/ablation_fipca_rank.png`      |

After all notebooks complete, run locally:

```bash
python3 scripts/build_paper_artifacts.py   # regenerate LaTeX tables + cleaned figure
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
python3 scripts/build_submission_zip.py    # produce submission.zip
```

All random seeds are fixed to 42. The locked hyperparameters and any scope deviations from the original proposal are documented in the paper itself (Experimental Setup and Limitations sections).

## Notes on Round-1 scope cuts

These are disclosed in the paper itself (Experimental Setup and Limitations sections):

1. **Per-client samples capped at 40–100 per local epoch** (vs full client data) — fits the ~4-hour Colab Free T4 session budget.
2. **FIPCA implemented as orthonormal random projection** (vs the data-aware incremental PCA basis in the original proposal). The communication-cost and privacy-against-DLG properties carry over; learned data-aware basis is Round-2 work.
3. **In-process FedAvg simulator** instead of Flower — same round structure and algorithm, no gRPC overhead. Plug-in to real Flower is trivial.
4. **MedMNIST only**. BraTS 2020 was extracted and partitioned (365 cases, 17,568 slices stored on the project's Drive mirror) but federated training on BraTS is Round-2 work.

## License

MIT — see `LICENSE`.

## Acknowledgements

This work was developed with the assistance of **Claude (Anthropic)** as a programming and writing assistant. All technical content, experimental results, and final manuscript were reviewed, validated, and edited by the author, who takes full responsibility for accuracy, originality, and academic integrity.
