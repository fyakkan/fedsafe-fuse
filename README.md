# FedSafe-Fuse

**Federated Conformal Uncertainty for Held-Out-Sequence Brain MRI Synthesis:
Patient-Level Guarantees and Tumour-Boundary Risk Control.**
Author: **Furkan Yakkan**, Abdullah Gül University, Kayseri / Türkiye.

The paper is at `paper/main.pdf` (IEEE conference format). The point-by-point response to the
Round-2 review is `paper/RESPONSE_TO_REVIEW_ROUND2.md`.

## What the paper does

A federated IFCNN (K=3 tumour-grade-skewed clients, FedAvg) predicts **FLAIR from T1 + T2** on
BraTS 2020 (365 patients). FLAIR is never an input, so the residual `|f − FLAIR|` is a genuine
predictive error, and conformal intervals around `f` are meaningful.

1. **Exact patient-level guarantee:** with probability ≥ 1−α a new patient has ≥ 90 % of its brain
   pixels inside `f ± τ`. The patient is the exchangeable unit, so pixel dependence does not matter.
2. **Count-only federated calibration:** clients release integer counts only; the threshold is
   bit-identical to pooled calibration (135/135 checks, ≤ 62 rounds).
3. **Conformal risk control on the tumour boundary:** expected tumour-boundary miscoverage ≤ α,
   using tumour labels at calibration only.
4. **Approximate pixel-level per-region (Mondrian) thresholds**, plus a global baseline. Negative
   finding: input-defined interior/boundary regions do not narrow intervals.

## Headline numbers (54 test patients, 5 FedAvg seeds)

| Result | Value |
|---|---|
| FLAIR prediction, FedAvg vs centralised (MAE in brain) | 0.119 ± 0.002 vs 0.117 ± 0.002 (best trivial baseline 0.159) |
| Patient-level coverage, α = 0.10 (target ≥ 0.90) | 0.941, Clopper–Pearson 95 % CI [0.84, 0.99] |
| Smallest client (14 patients) alone vs pooled, α = 0.10 | τ 0.479 vs 0.342; infeasible at α = 0.05 |
| Tumour-boundary miscoverage, no CRC → CRC (α = 0.10) | 0.109 → 0.079 |
| Strong input degradation, clean-calibrated vs matched (boundary) | 0.867 vs 0.916 coverage |

All numbers come from `results/*.csv`; the paper tables are generated from them by
`scripts/build_round2_tables.py`.

## Repository layout

```
.
├── paper/                    IEEE LaTeX source (main.tex, refs.bib, tables/r2_*.tex)
├── src/
│   ├── conformal.py          order-statistic quantile, patient-level guarantee, input-only regions,
│   │                         Mondrian thresholds, count-only federated quantile, conformal risk control
│   ├── models/ifcnn.py       IFCNN + InputMasked (output zeroed outside (T1>0)|(T2>0))
│   ├── fed/trainer.py        in-process FedAvg simulator (supervised_target=True for FLAIR)
│   └── losses.py             L1 + (1−SSIM) losses, SSIM/PSNR scorers
├── tests/                    unit tests (python3 -m pytest tests -q)
├── notebooks/
│   ├── 13_brats_flair_data.ipynb   T1/T2/FLAIR/seg/brain-mask extraction + patient-level splits
│   ├── 14_fed_flair_train.ipynb    FedAvg vs centralised IFCNN, 5 seeds
│   └── 15_fed_conformal.ipynb      all conformal analyses
├── scripts/
│   ├── make_nb13.py / make_nb14.py / make_nb15.py   generate the notebooks (embed src/ verbatim)
│   └── build_round2_tables.py      CSV → paper tables
├── results/ figures/ partitions/
```

Notebooks `01`–`12`, `scripts/build_paper_artifacts.py` and `scripts/dp_accounting_fedsafe.py` belong
to the earlier version of this project (MedMNIST, FROP/"FIPCA" compression, a batch-mean-clipped
"DP-SGD" baseline, DLG attacks). Those claims are **withdrawn**: the FROP model did not train, and the
DP baseline was not valid DP-SGD (no valid ε). They are kept only for the record.

## Reproducing

Colab T4: run notebooks 13 (≈ 35 min, CPU is fine), 14 (≈ 3 h, resumable) and 15 (≈ 1 h, resumable)
in order; each writes to `MyDrive/FedSafeFuse/`. Then locally:

```bash
python3 -m pytest tests -q
python3 scripts/build_round2_tables.py
cd paper && latexmk -pdf main.tex
```

## License

MIT — see `LICENSE`.

## Acknowledgements

This work was developed with the assistance of **Claude (Anthropic)** as a programming and writing
assistant. All technical content, experimental results, and the final manuscript were reviewed,
validated, and edited by the author, who takes full responsibility for accuracy, originality, and
academic integrity.
