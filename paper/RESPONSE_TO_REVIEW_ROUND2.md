# Response to the Round-2 Review (Reviewer 2)

Dear Professor,

Thank you for this careful, code-verified review. I re-checked each of its three deciding facts against the repository and **all three are correct**:

- **Fact A.** The FROP model's SSIM is 0.23936 at both round 1 and round 50.
- **Fact B.** `fedsafe_standard` trains only from 0.239 to 0.278, and IFCNN + FROP falls from 0.239 to 0.234.
- **Fact C.** The SACP target `0.5(m+p)` is computable from the inputs.

The two code defects are also confirmed: BatchNorm buffers were projected through FROP, and "DP-SGD" clipped the batch-mean gradient.

Following your recommendation not to keep all three contributions, I took **Option B**, "federated spatially-aware conformal uncertainty with a held-out target", and rebuilt the experiments from scratch. The revised manuscript has a new title: *"Federated Conformal Uncertainty for Held-Out-Sequence Brain MRI Synthesis: Patient-Level Guarantees and Tumour-Boundary Risk Control"*. It reports the results honestly, including two negative findings.

**Summary of what changed**

- **Task.** Inputs are T1 and T2; the target is **FLAIR**, which is never available at inference. All 365 BraTS 2020 patients are used (17,568 slices) with **patient-level splits**: each of the 3 grade-skewed clients splits its patients 65/10/25 % into train, val and calibration (78 calibration patients in total), and a separate central test set holds 54 patients.
- **Model.** IFCNN in every arm, trained with FedAvg and centralised, 5 seeds each, under a protocol fixed in advance with no model selection. The MobileNetV3 backbone, MedMNIST, FROP-as-method, "DP-SGD" and DLG are removed from the paper.
- **Conformal methods.**
  - An exact order-statistic quantile.
  - An **exact patient-level guarantee**.
  - **Pixel-level Mondrian** thresholds explicitly labelled as approximate, plus a global baseline.
  - **Count-only federated calibration**, bit-identical to pooled calibration.
  - **Conformal risk control on the tumour-boundary band**.
  - An input-degradation study.
- **Reproducibility.** 19 unit tests (`tests/`). Every paper table is generated from the result CSVs by `scripts/build_round2_tables.py`, and every new reference was verified against Crossref or arXiv metadata.

---

## 1. Fatal

**F1 — FROP transmits and learns nothing.** *Accepted; repositioned as a negative result (your option b).*

- `_flatten_state` now projects `model.parameters()` only. BatchNorm buffers are averaged uncompressed and counted in bytes/round (`src/fed/trainer.py`). A regression test checks that `running_mean` and `num_batches_tracked` move under FROP; it fails under the old behaviour.
- New §V-G, "Negative result", states the following:
  - With a fixed basis the global model is always the initialisation plus a *d*-dimensional perturbation (d/N ≈ 6.5×10⁻⁶).
  - The logged SSIM (0.23936 → 0.23936) and IFCNN + FROP (0.239 → 0.234) show it does not train.
  - The BN bug is disclosed.
  - All privacy and communication claims for FROP are **withdrawn**, citing Yue et al. (USENIX Security 2023).
- The "rank-d bottleneck, not the backbone" sentence, the rank ablation and every "matches DP-SGD" phrase are deleted. The extra Table I rows you suggested are not needed, because FROP is no longer an evaluated method; its numbers are quoted in §V-G.

**F2 — The DP-SGD baseline is not DP-SGD; ε ≈ 6.2 is invalid.** *Accepted; removed.*

- The mechanism is removed from the paper together with the DP-accounting table; no ε appears anywhere.
- In code, `mode='dpsgd'` now raises a warning that it is not DP-SGD and is kept only to reproduce archived numbers.
- The paper now claims **no privacy property beyond federated data locality** (§V-G).
- I did not re-run Opacus DP-SGD. That experiment belongs to Option A and would not support any claim in the re-scoped paper.

**F3 — The SACP target is observable at inference.** *Fixed with your options 1 and 3.*

- **Option 1:** the target is now held-out FLAIR, extracted together with the segmentation and an input-only brain mask (`notebooks/13_brats_flair_data.ipynb`). The score `|f − FLAIR|` is therefore a genuine predictive residual.
- **Option 3:** conformal risk control (Angelopoulos et al., CRC) on the tumour-boundary band, using segmentation labels at calibration only. Tumour-boundary miscoverage at α = 0.10 drops from **0.109** without CRC to **0.079 ± 0.003** with CRC (pooled calibration).
- The trivial predictors are reported (Table I): zero, copy-T2, ½(T1+T2) — the old `f = r` — and a linear fit.
  - FedAvg reaches MAE_brain **0.119 ± 0.002** and SSIM_brain **0.659 ± 0.009**.
  - The best trivial predictor reaches 0.159 and 0.368.
  - Whole-image SSIM is reported alongside brain-restricted metrics, so it is not presented as a fusion-quality score.

---

## 2. Major

**M1 — Missing joint privacy–utility evaluation and compression baselines.** These requests belong to Option A. Under Option B the paper makes **no compression or privacy claim**: the `privacy_vs_comm` figure is removed, and FROP survives only as the §V-G negative result. PowerSGD and top-k were therefore not added.

**M2 — Weak gradient-inversion evaluation.** All DLG results and the "compression as privacy" reading are **removed**, as are the leakage proxies. Yue et al. 2023 is cited for why compression is not privacy. Inverting-Gradients attacks were not run, since there is no longer a privacy claim to test.

**M3 — Guarantee, unit and validation.** Each sub-point:

- **(i) Quantile.** The order statistic ⌈(n+1)(1−α)⌉ is used everywhere (`src/conformal.py`), with a Monte-Carlo unit test that checks the coverage.
- **(ii) Two guarantee levels.**
  - *Exact:* conformalisation at the **patient** level (Proposition 1). For a new patient, P(≥ 1−β of the region's pixels covered) ≥ 1−α. I used the patient rather than the image as the unit because slices of one patient are dependent.
  - *Approximate:* pixel-level per-region thresholds, named **Mondrian** (Vovk et al. 2005).
  - The abstract now states which statement is exact and which is empirical.
- **(iii) Patient leakage.** The split is patient-level throughout: calibration uses 78 client-held patients, testing uses 54 *different* patients, and an assertion forbids overlap. The SSIM/PSNR evaluation set is the same 54-patient test set, which is fully disjoint from calibration.
- **(iv) Three regions.** Background (outside the input-only brain mask), tissue interior, and boundary. The Canny 50/100 boundary covered 58–63 % of the brain, so the primary definition is now the fixed 25 % top-gradient share of ½(T1+T2) inside the brain. Canny is kept as a sensitivity check. Pixel fractions are reported (`results/conformal_region_fractions.csv`). Because background is its own region, the "width inversion" artefact can no longer occur.
- **(v) Global-τ baseline.** Table IV compares a single global threshold with the per-region thresholds. The honest finding is that **input-defined regions do not improve efficiency**: the brain-average half-width is 0.267 vs 0.268, because errors concentrate in lesions rather than at input edges. The paper states this, and it motivates the CRC component.
- **(vi) Half-width.** τ is reported, not 2τ.
- **(vii) Models and seeds.** Every conformal result uses converged models (validation loss flat by round 25) and all 5 FedAvg seeds, reported as mean ± std. The constant-model MedMNIST coverage is dropped.
- **Over-coverage.** Pixel-level coverage is above nominal (0.93–0.95 at a 0.90 target), and the paper reports it as exactly that; "meets or exceeds" no longer hides it. Patient-level Clopper–Pearson intervals contain the target at every α.

**M4 — The fusion task and metrics are not fusion.**

- MedMNIST, every "MRI + PET" mention and the "standard practice" sentence are removed.
- The task is named for what it is: held-out-sequence (cross-sequence) synthesis from multi-sequence MRI. Limitation L5 discusses the framing.
- Because a real held-out target now exists, I report reference-based metrics (SSIM and PSNR, whole-image and brain-restricted, plus MAE_brain). No-reference fusion metrics (EN, SF, Q_AB/F…) and AANLIB PET/SPECT pairs were not added, since the paper no longer claims MRI–PET or SPECT fusion.

**M5 — Backbone confounded with mechanism.** IFCNN is now the backbone for all arms; the MobileNetV3 backbone is gone. IFCNN's output is zeroed outside the input-derived brain mask (`InputMasked`, unit-tested), so the model is not penalised on the 73 % air background where trivial predictors are exactly 0. A short centralised "sanity gate" in notebook 14 must beat ½(T1+T2) before the long runs start.

**M6 — Bandwidth framing.** The 154,109× ratio and all bandwidth claims are removed. The only communication figure given is FedAvg's upload, 3.8 MB per round for three clients, which is not a claim.

**M7 — Citation–claim mismatches.**

| Issue | Action |
|---|---|
| `angelopoulos2022cv` (classification) | Replaced by Angelopoulos et al., *Image-to-Image Regression with Distribution-Free UQ*, ICML 2022 (arXiv:2202.05265) |
| `tang2023semantic` (SeAFusion) | Replaced by MATR, Tang et al., IEEE TIP 31:5134–5149, 2022 (doi:10.1109/TIP.2022.3193288) |
| "To our knowledge no prior work…" | Deleted. FLMMIF (Meng et al., Front. AI 2026, doi:10.3389/frai.2026.1857709), FEMMIF (Alabduljabbar et al., IJIST 36(4) 2026, doi:10.1002/ima.70375) and Lu et al., ICML 2023 (arXiv:2305.17564) are cited |
| Missing work | Added: Teneggi et al. (K-RCPS, ICML 2023), Conformal Risk Control (arXiv:2208.02814), Lei et al. JASA 2018, Bonawitz et al. CCS 2017, Clopper–Pearson 1934. Vovk et al. 2005 (book) is cited for Mondrian; the 2003 tech report is not used |
| Unverified (SwinFusion, CDDFuse) | Not cited |

All new entries were checked against Crossref (DOIs) or the arXiv API. Conformal Risk Control is cited as an arXiv preprint because I could not confirm its venue from metadata.

**M8 — Statistics.**

- The FROP "± 0.000" and the "0.239 vs 0.234" comparison are gone, along with every single-seed and single-image result.
- Every number in Tables I–VI is a 5-seed mean ± std.
- Coverages carry Clopper–Pearson intervals (patient level) or patient-cluster bootstrap intervals (pixel level).
- Small subgroups are flagged: there are only 9 LGG test patients, the paper says the data cannot resolve differences of that size, and Client 1's CRC risk of 0.103 is reported as slightly above α.

---

## 3. Minor

- **Energy d/N vs √(d/N).** Resolved: the leakage-proxy analysis is removed, and §V-G states d/N ≈ 6.5×10⁻⁶ explicitly.
- **Threat model and secure aggregation.** The federated calibration section says the server sees only Σ_k c_k(t) under secure aggregation, and that the pooled ranks of probed thresholds are revealed. Limitation L4 states that secure aggregation is assumed, not implemented.
- **MedMNIST eval set ⊂ coverage set.** Not applicable: MedMNIST is dropped, and the BraTS test set is disjoint from calibration.
- **Canny thresholds.** Replaced as the primary region definition by a fixed-share gradient region. Boundary-pixel fractions are reported for both definitions.
- **`environment.md` and README.** Both are harmonised with the paper: no "8M-param" or "FIPCA" wording, no 154,109×, and the Kaggle data source is stated in Methods. The archived notebooks are labelled as withdrawn.

## 4. Section-5 text changes

1. The title no longer says "Privacy-Preserving"; it now reads "Brain MRI Synthesis".
2. Every "MRI + PET" / "PET input" mention is removed.
3. The synthetic-pairing "standard practice" sentence is deleted.
4. The SeAFusion sentence is replaced by MATR.
5. "No prior work…" is deleted; FLMMIF, FEMMIF and Lu et al. are cited.
6. "Limiting factor is the rank-d bottleneck" and every "matches DP-SGD" phrase are deleted.
7. "Confirming the pipeline" is deleted.
8. The abstract states the exact patient-level guarantee, and pixel-level thresholds are labelled approximate (§III-D).
9. Calibration is now **federated**, with count-only messages bit-identical to pooled calibration (135/135 checks).
10. The `dp_accounting` table is deleted.

## 5. Results that did not go as hoped (reported in the paper)

- **Input-defined spatial regions do not narrow intervals.** The boundary threshold is *smaller* than the interior one, and the brain-average width is unchanged.
- **The expected grade-shift failure of per-client calibration did not appear.** HGG-only calibration covered the LGG test patients as well as pooled calibration (8 of 9 in every scheme). Federated pooling helps chiefly the smallest client: its intervals are 29 % narrower with pooling, and α = 0.05 becomes feasible.
- **Patient-level intervals are wide**, a half-width of about 0.34 on a [0,1] scale (Limitation L2).

## 6. Reproducibility

| Step | Artifact |
|---|---|
| Data extraction and splits | `notebooks/13_brats_flair_data.ipynb` → `partitions/brats_K3_v2.json` |
| Training (FedAvg + centralised, 5 seeds) | `notebooks/14_fed_flair_train.ipynb` → `results/flair_test_summary.csv` |
| Conformal analyses | `notebooks/15_fed_conformal.ipynb` → `results/conformal_*.csv` |
| Paper tables | `scripts/build_round2_tables.py` (CSV → `paper/tables/r2_*.tex`) |
| Tests | `python3 -m pytest tests -q` (19 pass) |

The notebooks are generated by `scripts/make_nb1{3,4,5}.py`, which embed the unit-tested `src/` modules verbatim.

Sincerely,
Furkan Yakkan
Abdullah Gül University
