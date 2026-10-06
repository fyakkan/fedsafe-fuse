"""Generate notebooks/13_brats_flair_data.ipynb (embeds src/conformal.py verbatim)."""
import json
import os
import sys

REPO = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..")
conformal_src = open(os.path.join(REPO, "src/conformal.py")).read()

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})


md(r"""
# 13 — BraTS T1/T2/FLAIR re-extraction + patient-level splits (Round-2, Phase 2)

**Why.** Round-2 review (F3): the old SACP target `0.5·(T1+T2)` is computable from the inputs, so its conformal interval is vacuous. The new task predicts **FLAIR from T1 + T2**; FLAIR is unavailable at inference, so `|f − FLAIR|` is a genuine predictive residual. The old processed `.npz` files only kept T1/T2, so we re-extract from the raw volumes already on Drive.

**What changes vs. `01_data` (all disclosed in the paper):**
- Adds **FLAIR**, the **segmentation** (needed for conformal risk control on tumour boundaries) and a **brain mask** `(T1>0) | (T2>0)` (computable at inference; used for the background region).
- **Per-volume** normalisation: each modality is clipped to its [0.5, 99.5] brain-voxel percentiles and scaled to [0,1]; background stays exactly 0. (Old: per-slice percentiles, which make the target's intensity scale slice-dependent.)
- Resize 240→128 with `INTER_AREA` (images) / `INTER_NEAREST` (masks). Images stored as float16.
- Same slice rule as before: keep axial slices with ≥1 % tumour.
- **Patient-level splits (review M3-iii):** the 54 old calibration cases become the central **test** set; each client's patients are split 65 % train / 10 % val (monitoring only, never for model selection) / 25 % **calibration**. No patient appears in two splits (asserted).

**Runtime.** CPU runtime is fine. ~30–60 min (dominated by reading NIfTI from Drive). **Resumable** — re-running skips finished cases.

**Outputs (on Drive under `MyDrive/FedSafeFuse/`):**
| Drive path | Download to local |
|---|---|
| `brats/processed_v2/<case>.npz` | (stays on Drive) |
| `partitions/brats_K3_v2.json` | `partitions/brats_K3_v2.json` |
| `results/region_fractions_v2.csv` | `results/region_fractions_v2.csv` |
| `results/flair_trivial_baselines.csv` | `results/flair_trivial_baselines.csv` |
| `figures/phase2_flair_sanity.png` | `figures/phase2_flair_sanity.png` |
""")

code(r"""
# 0. Setup
!pip install -q nibabel scikit-image opencv-python-headless
import os, sys, json, time, glob, random
import numpy as np, pandas as pd, cv2, nibabel as nib
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED); np.random.seed(SEED)

from google.colab import drive
if not os.path.exists('/content/drive/MyDrive'):
    drive.mount('/content/drive')
DRIVE_ROOT = '/content/drive/MyDrive/FedSafeFuse'
RAW_DIR    = f'{DRIVE_ROOT}/brats/raw'
OUT_DIR    = f'{DRIVE_ROOT}/brats/processed_v2'
for sub in ['brats/processed_v2', 'partitions', 'results', 'figures']:
    os.makedirs(f'{DRIVE_ROOT}/{sub}', exist_ok=True)
print('OK')
""")

md("Embedded verbatim from `src/conformal.py` (unit-tested locally; do not edit here).")
code("%%writefile /content/conformal.py\n" + conformal_src)
code(r"""
sys.path.insert(0, '/content')
import importlib, conformal as cp
importlib.reload(cp)
print('conformal.py loaded:', [f for f in dir(cp) if not f.startswith('_')][:6], '...')
""")

code(r"""
# 1. Locate raw BraTS 2020 (downloaded by 01_data) + reuse the old partition for grades/clients
anchor = glob.glob(f'{RAW_DIR}/**/BraTS20_Training_001', recursive=True)
assert anchor, ('Raw BraTS not found on Drive. Re-run cell 5 of 01_data.ipynb (Kaggle download) first.')
TRAIN_DIR = os.path.dirname(anchor[0])
print('Training dir:', TRAIN_DIR)
print('Files in case 001:', sorted(os.listdir(anchor[0])))

with open(f'{DRIVE_ROOT}/partitions/brats_K3.json') as f:
    old = json.load(f)
grade_by_id = {m['case_id']: m['grade'] for m in old['all_meta']}
old_nslices = {m['case_id']: m['n_slices'] for m in old['all_meta']}
print(f"Old partition: {len(old['all_meta'])} cases, clients={[len(c) for c in old['client_cases']]}, "
      f"calib={len(old['calibration_cases'])}, slices={sum(old_nslices.values())}")
""")

code(r"""
# 2. Extraction functions + smoke test
def find_modality(case_dir, modality):
    for fname in os.listdir(case_dir):
        lower = fname.lower()
        if not (lower.endswith('.nii') or lower.endswith('.nii.gz')):
            continue
        stem = lower.rsplit('.nii', 1)[0]
        if modality == 'seg':
            if 'seg' in stem:
                return os.path.join(case_dir, fname)
        elif stem.endswith('_' + modality):
            return os.path.join(case_dir, fname)
    return None

def volume_norm(vol, lo_pct=0.5, hi_pct=99.5):
    # Per-volume robust [0,1] scaling over that modality's non-zero (brain) voxels; background stays 0.
    fg = vol > 0
    if fg.sum() < 100:
        raise ValueError('modality volume is (almost) empty')
    lo, hi = np.percentile(vol[fg], [lo_pct, hi_pct])
    out = np.clip((vol - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    out[~fg] = 0.0
    return out.astype(np.float32)

def rs_img(x):  return cv2.resize(x, (128, 128), interpolation=cv2.INTER_AREA)
def rs_mask(x): return cv2.resize(x.astype(np.uint8), (128, 128), interpolation=cv2.INTER_NEAREST)

def extract_case(case_dir):
    # Returns (data, reason). data is None when the case is skipped; reason says why.
    paths = {m: find_modality(case_dir, m) for m in ['t1', 't2', 'flair', 'seg']}
    if not all(paths.values()):
        return None, 'missing ' + ','.join(m for m, p in paths.items() if not p)
    raw = {m: nib.load(p).get_fdata().astype(np.float32) for m, p in paths.items()}
    if len({v.shape for v in raw.values()}) != 1:
        return None, 'shape mismatch ' + str({m: v.shape for m, v in raw.items()})
    brain_vol = (raw['t1'] > 0) | (raw['t2'] > 0)          # computable at inference (no FLAIR)
    norm = {m: volume_norm(raw[m]) for m in ['t1', 't2', 'flair']}
    seg = raw['seg'].astype(np.uint8)
    keep = [s for s in range(seg.shape[2]) if (seg[..., s] > 0).mean() >= 0.01]
    if not keep:
        return None, 'no slice with >=1% tumour (also skipped by 01_data)'
    out = {
        'slice_idx': np.array(keep, dtype=np.int32),
        't1':    np.stack([rs_img(norm['t1'][..., s])    for s in keep]).astype(np.float16),
        't2':    np.stack([rs_img(norm['t2'][..., s])    for s in keep]).astype(np.float16),
        'flair': np.stack([rs_img(norm['flair'][..., s]) for s in keep]).astype(np.float16),
        'seg':   np.stack([rs_mask(seg[..., s])          for s in keep]),
        'brain': np.stack([rs_mask(brain_vol[..., s])    for s in keep]),
    }
    return out, None

case_dirs = sorted(glob.glob(f'{TRAIN_DIR}/BraTS20_Training_*'))
print(f'Found {len(case_dirs)} case dirs')
d0, why0 = extract_case(case_dirs[0])
assert d0 is not None, f'Smoke test failed on first case: {why0}'
cid0 = os.path.basename(case_dirs[0])
print(f"Smoke OK {cid0}: {len(d0['slice_idx'])} slices (old: {old_nslices.get(cid0)}) | "
      f"t1 {d0['t1'].shape} {d0['t1'].dtype} | seg labels {np.unique(d0['seg'])} | "
      f"brain frac {d0['brain'].mean():.3f} | flair range [{d0['flair'].min():.2f},{d0['flair'].max():.2f}]")
""")

code(r"""
# 3. Extract all cases (resumable)
meta, skipped = [], []
t0 = time.time()
for i, cdir in enumerate(case_dirs):
    cid = os.path.basename(cdir)
    out_npz = f'{OUT_DIR}/{cid}.npz'
    if os.path.exists(out_npz):
        try:
            with np.load(out_npz) as d:
                n_done = int(d['t1'].shape[0]); _ = d['flair'].shape
            meta.append({'case_id': cid, 'grade': grade_by_id.get(cid, 'UNK'), 'n_slices': n_done})
            continue
        except Exception as e:   # partial file from an interrupted session -> redo this case
            print(f'  {cid}: unreadable existing file ({e}); re-extracting')
            os.remove(out_npz)
    try:
        data, why = extract_case(cdir)
    except Exception as e:       # one bad case must never stop the run
        data, why = None, f'error: {type(e).__name__}: {e}'
    if data is None:
        skipped.append((cid, why))
        continue
    tmp = out_npz[:-4] + '.tmp.npz'      # atomic write: a crash never leaves a half-written .npz
    np.savez_compressed(tmp, **data)
    os.replace(tmp, out_npz)
    meta.append({'case_id': cid, 'grade': grade_by_id.get(cid, 'UNK'), 'n_slices': int(len(data['slice_idx']))})
    if (i + 1) % 20 == 0:
        print(f'  {i+1}/{len(case_dirs)} | {time.time()-t0:.0f}s')

print(f'\nDone in {time.time()-t0:.0f}s: {len(meta)} cases, {sum(m["n_slices"] for m in meta)} slices '
      f'(old: {len(old_nslices)} cases, {sum(old_nslices.values())} slices)')
print(f'Skipped {len(skipped)} cases:')
for cid, why in skipped:
    print(f'  {cid}: {why}')
missing_old = sorted(set(old_nslices) - {m['case_id'] for m in meta})
print(f'Cases in old partition but not extracted now: {len(missing_old)}', missing_old[:10])
diff = [(m['case_id'], m['n_slices'], old_nslices.get(m['case_id'])) for m in meta
        if old_nslices.get(m['case_id']) not in (None, m['n_slices'])]
print(f'Cases whose slice count differs from old extraction: {len(diff)}', diff[:5])
""")

code(r"""
# 4. Patient-level splits: test = old calibration cases; per-client train/val/calib
have = {m['case_id'] for m in meta}
nsl  = {m['case_id']: m['n_slices'] for m in meta}
rng = np.random.default_rng(SEED)
FRAC_TRAIN, FRAC_VAL = 0.65, 0.10   # remainder (0.25) = calibration

test_cases = sorted(c for c in old['calibration_cases'] if c in have)
clients = []
for k, cases in enumerate(old['client_cases']):
    cases = [c for c in cases if c in have]
    perm = list(rng.permutation(cases))
    n_tr = int(round(FRAC_TRAIN * len(perm))); n_va = int(round(FRAC_VAL * len(perm)))
    clients.append({'train': sorted(perm[:n_tr]), 'val': sorted(perm[n_tr:n_tr + n_va]), 'calib': sorted(perm[n_tr + n_va:])})

all_lists = [test_cases] + [cl[s] for cl in clients for s in ('train', 'val', 'calib')]
flat = [c for l in all_lists for c in l]
assert len(flat) == len(set(flat)), 'patient appears in more than one split'

def summary(cases):
    return {'cases': len(cases), 'slices': int(sum(nsl[c] for c in cases)),
            'HGG': sum(grade_by_id.get(c) == 'HGG' for c in cases),
            'LGG': sum(grade_by_id.get(c) == 'LGG' for c in cases)}

part = {
    'dataset': 'brats2020_kaggle_awsaf49_v2_t1_t2_flair',
    'seed': SEED, 'num_clients': len(clients),
    'task': 'inputs (t1, t2) -> held-out target flair',
    'preprocessing': 'per-volume [0.5,99.5] brain-percentile norm; INTER_AREA 240->128; slices with >=1% tumour',
    'test_cases': test_cases,
    'clients': clients,
    'summary': {'test': summary(test_cases),
                **{f'client{k}_{s}': summary(cl[s]) for k, cl in enumerate(clients) for s in ('train', 'val', 'calib')}},
    'all_meta': meta,
}
with open(f'{DRIVE_ROOT}/partitions/brats_K3_v2.json', 'w') as f:
    json.dump(part, f, indent=2)
print(pd.DataFrame(part['summary']).T.to_string())
""")

code(r"""
# 5. Region fractions (background / interior / boundary) per split  [review minor: report pixel fractions]
def load_case(cid):
    with np.load(f'{OUT_DIR}/{cid}.npz') as d:
        return {k: d[k] for k in d.files}

def case_regions(d):
    guide = 0.5 * (d['t1'].astype(np.float32) + d['t2'].astype(np.float32))   # inputs only
    return [cp.region_map(guide[i], d['brain'][i]) for i in range(len(guide))]

rows = []
for name, cases in [('test', test_cases)] + [(f'client{k}_calib', cl['calib']) for k, cl in enumerate(clients)]:
    labs, tb = [], [0, 0]
    for c in cases:
        d = load_case(c); L = case_regions(d); labs += L
        for i, lab in enumerate(L):   # tumour pixels falling in each brain region
            tum = d['seg'][i] > 0
            tb[0] += int((tum & (lab == cp.REGION_BOUNDARY)).sum()); tb[1] += int(tum.sum())
    fr = cp.region_fractions(labs)
    rows.append({'split': name, 'n_slices': len(labs), **{f'frac_{k}': round(v, 4) for k, v in fr.items()},
                 'tumour_px_in_boundary': round(tb[0] / max(tb[1], 1), 4)})
rf = pd.DataFrame(rows)
rf.to_csv(f'{DRIVE_ROOT}/results/region_fractions_v2.csv', index=False)
print(rf.to_string(index=False))
""")

code(r"""
# 6. Trivial baselines vs FLAIR on the TEST patients (must be beaten in Phase 3; review F3 'f = r' row)
from skimage.metrics import structural_similarity as ssim_fn

def lin_fit(cases):
    # Least-squares FLAIR ~ a*T1 + b*T2 + c on brain pixels of the given cases (fit on client TRAIN cases only).
    X, y = [], []
    for c in cases:
        d = load_case(c); m = d['brain'].astype(bool)
        X.append(np.stack([d['t1'][m], d['t2'][m], np.ones(m.sum(), np.float16)], 1).astype(np.float32))
        y.append(d['flair'][m].astype(np.float32))
    X, y = np.concatenate(X), np.concatenate(y)
    return np.linalg.lstsq(X, y, rcond=None)[0]

train_all = [c for cl in clients for c in cl['train']]
fit_cases = rng.choice(train_all, size=min(60, len(train_all)), replace=False)
coef = lin_fit(fit_cases)
print(f'Linear FLAIR ~ a*T1 + b*T2 + c, fitted on {len(fit_cases)} train patients:', np.round(coef, 4))

baselines = {
    'zero':           lambda t1, t2: np.zeros_like(t1),
    'T2':             lambda t1, t2: t2,
    'mean(T1,T2)':    lambda t1, t2: 0.5 * (t1 + t2),
    'linear(T1,T2)':  lambda t1, t2: np.clip(coef[0] * t1 + coef[1] * t2 + coef[2], 0, 1) * (t1 + t2 > 0),
}
rows = []
for c in test_cases:
    d = load_case(c)
    t1, t2, fl = (d[k].astype(np.float32) for k in ('t1', 't2', 'flair'))
    for name, f in baselines.items():
        p = f(t1, t2)
        mae  = float(np.mean([np.abs(p[i] - fl[i])[d['brain'][i] > 0].mean() for i in range(len(fl))]))
        ssim = float(np.mean([ssim_fn(fl[i], p[i], data_range=1.0) for i in range(len(fl))]))
        mse  = float(np.mean((p - fl) ** 2))
        rows.append({'case': c, 'baseline': name, 'mae_brain': mae, 'ssim': ssim, 'psnr': 10 * np.log10(1 / max(mse, 1e-12))})
bl = pd.DataFrame(rows)
summ = bl.groupby('baseline')[['ssim', 'psnr', 'mae_brain']].agg(['mean', 'std']).round(4)
summ.to_csv(f'{DRIVE_ROOT}/results/flair_trivial_baselines.csv')
print('Per-patient mean (std) over', len(test_cases), 'test patients:')
print(summ.to_string())
""")

code(r"""
# 7. Sanity figure: 3 test slices -> T1 | T2 | FLAIR (target) | seg | regions
n_show = min(3, len(test_cases))
fig, ax = plt.subplots(n_show, 5, figsize=(15, 3 * n_show), squeeze=False)
picks = rng.choice(test_cases, size=n_show, replace=False)
for r, c in enumerate(picks):
    d = load_case(c); i = len(d['t1']) // 2
    lab = case_regions({k: d[k][i:i+1] for k in ('t1', 't2', 'brain')})[0]
    for j, (img, title, kw) in enumerate([
        (d['t1'][i], 'T1 (input)', dict(cmap='gray', vmin=0, vmax=1)),
        (d['t2'][i], 'T2 (input)', dict(cmap='gray', vmin=0, vmax=1)),
        (d['flair'][i], 'FLAIR (held-out target)', dict(cmap='gray', vmin=0, vmax=1)),
        (d['seg'][i], 'tumour seg', dict(cmap='viridis', vmin=0, vmax=4)),
        (lab, 'regions: bg / interior / boundary', dict(cmap='viridis', vmin=0, vmax=2)),
    ]):
        ax[r, j].imshow(img.astype(np.float32), **kw); ax[r, j].axis('off')
        ax[r, j].set_title(f'{c[-3:]} {title}' if j == 0 else title, fontsize=9)
plt.tight_layout()
fig_path = f'{DRIVE_ROOT}/figures/phase2_flair_sanity.png'
plt.savefig(fig_path, dpi=130, bbox_inches='tight'); plt.show()
print('Saved', fig_path)
""")

md(r"""
## Done — paste back to chat
1. Output of **cell 3** (case/slice totals, skipped cases, slice-count differences vs. old extraction).
2. The split table from **cell 4**.
3. The tables from **cells 5 and 6** (region fractions, trivial baselines).
4. Whether `phase2_flair_sanity.png` looks right (FLAIR bright where tumour oedema is; boundary region inside the brain, not on the skull/air rim).

Then download `partitions/brats_K3_v2.json`, the two CSVs and the PNG into the local repo.
""")

nb = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"},
                 "colab": {"provenance": []}},
    "nbformat": 4, "nbformat_minor": 5,
}
for c in nb["cells"]:
    c["source"] = c["source"].splitlines(keepends=True)
with open(os.path.join(REPO, "notebooks/13_brats_flair_data.ipynb"), "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", len(cells), "cells")
