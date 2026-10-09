"""Generate notebooks/15_fed_conformal.ipynb (embeds src/ modules verbatim).

Run: python3 scripts/make_nb15.py   (re-run whenever an embedded src/ file changes)
"""
import json
import os
import sys

REPO = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..")
EMBED = {"conformal.py": "src/conformal.py", "ifcnn.py": "src/models/ifcnn.py"}

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})


md(r"""
# 15 — Federated spatially-aware conformal calibration (Round-2, Phase 4)

Inference only — uses the 5 FedAvg checkpoints from notebook 14. Score = per-pixel residual `|f(T1,T2) − FLAIR|` (FLAIR is never an input).

**What is computed (all on the 54 held-out test patients, every FedAvg seed):**
1. **Regions** (input-only): background (outside `(T1>0)|(T2>0)`), tissue *interior*, *boundary* = top-25 % smoothed-gradient brain pixels of ½(T1+T2). The old Canny 50/100 definition is kept as a sensitivity check (clean inputs).
2. **Pixel-level Mondrian thresholds** (approximate guarantee) per region vs. one **global brain threshold** — does splitting improve region-conditional coverage?
3. **Patient-level thresholds** (exact guarantee: with prob ≥ 1−α a new patient has ≥ 1−β of its region pixels covered; β = 0.1).
4. **Calibration schemes:** *pooled* (= federated: identical bits, verified, clients only release counts) vs. *per-client* (client 0 HGG-only, client 1 mixed, client 2 LGG-only); coverage also split by test grade.
5. **Conformal risk control** on the **tumour-boundary band** (labels used at calibration only): expected fraction of uncovered tumour-boundary pixels ≤ α.
6. **Input degradation** (blur + noise on T1/T2): matched calibration vs. calibrate-clean/test-degraded.

**Runtime.** GPU recommended (T4): ≈ 3–5 min per (seed, degradation) → 15 units ≈ 1 h. **Resumable per unit**: just *Run all* again after a disconnect.

**Outputs (Drive `MyDrive/FedSafeFuse/results/`):** `conformal_runs/*` (per unit), `conformal_summary.csv`, `conformal_crc_summary.csv`, `conformal_fedcheck.csv`, `conformal_region_fractions.csv`, figure `figures/phase4_conformal_example.png`.
""")

code(r"""
# 0. Config (do not change between sessions)
SEEDS         = [42, 1, 2, 3, 4]
ALPHAS        = [0.05, 0.10, 0.15]
BETA          = 0.10                 # patient-level: fraction of a patient's region pixels allowed outside
CRC_ALPHA     = 0.10
LAMBDAS       = [round(0.01 * i, 2) for i in range(1, 1001)]   # CRC grid 0.01 .. 10.00
BOUNDARY_FRAC = 0.25
DEGRADATIONS  = {'clean': (0.0, 0.00), 'mild': (0.75, 0.02), 'strong': (1.5, 0.05)}  # (blur sigma, noise sd)
N_BOOT        = 2000
REQUIRE_GPU   = True
""")

code(r"""
# 1. Setup + mount Drive
import os, sys, json, time, random, shutil, importlib, zlib
import numpy as np, pandas as pd, cv2, torch
import matplotlib.pyplot as plt

if REQUIRE_GPU and not torch.cuda.is_available():
    raise RuntimeError('No GPU. Runtime -> Change runtime type -> T4 GPU, then Run all again.')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device', DEVICE)

from google.colab import drive
if not os.path.exists('/content/drive/MyDrive'):
    drive.mount('/content/drive')
DRIVE_ROOT = '/content/drive/MyDrive/FedSafeFuse'
LOCAL   = '/content/local_flair'
SRC_DIR = '/content/fedsrc'
RUN_DIR = f'{DRIVE_ROOT}/results/conformal_runs'
for d in [LOCAL, SRC_DIR, RUN_DIR, f'{DRIVE_ROOT}/figures']:
    os.makedirs(d, exist_ok=True)

def atomic_csv(df, path):
    tmp = path + '.tmp'; df.to_csv(tmp, index=False); os.replace(tmp, path)

def atomic_json(obj, path):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f)
    os.replace(tmp, path)
print('OK')
""")

md("Embedded verbatim from `src/` (unit-tested locally; do not edit here).")
for name, rel in EMBED.items():
    code(f"%%writefile /content/fedsrc/{name}\n" + open(os.path.join(REPO, rel)).read())

code(r"""
sys.path.insert(0, SRC_DIR)
import conformal as cp, ifcnn
importlib.reload(cp); importlib.reload(ifcnn)
from ifcnn import IFCNN, InputMasked
def build_model():
    return InputMasked(IFCNN())
REG = {'background': cp.REGION_BACKGROUND, 'interior': cp.REGION_INTERIOR, 'boundary': cp.REGION_BOUNDARY}
print('OK')
""")

code(r"""
# 2. Partition, checkpoints, copy calib + test case files (resumable)
with open(f'{DRIVE_ROOT}/partitions/brats_K3_v2.json') as f:
    part = json.load(f)
grade = {m['case_id']: m['grade'] for m in part['all_meta']}
calib_by_client = [cl['calib'] for cl in part['clients']]
calib_cases = [c for l in calib_by_client for c in l]
test_cases  = part['test_cases']
print(f'calib per client={[len(l) for l in calib_by_client]} (pooled {len(calib_cases)}) | test={len(test_cases)} '
      f"(HGG {sum(grade[c]=='HGG' for c in test_cases)}, LGG {sum(grade[c]=='LGG' for c in test_cases)})")

for s in SEEDS:
    p = f'{DRIVE_ROOT}/checkpoints/flair_fedavg_seed{s}.pt'
    if not os.path.exists(p):
        raise FileNotFoundError(f'{p} missing - finish notebook 14 first')

t0 = time.time(); copied = 0
for cid in calib_cases + test_cases:
    src, dst = f'{DRIVE_ROOT}/brats/processed_v2/{cid}.npz', f'{LOCAL}/{cid}.npz'
    if not os.path.exists(src):
        raise FileNotFoundError(f'{src} missing - re-run notebook 13')
    if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
        continue
    shutil.copy(src, dst + '.tmp'); os.replace(dst + '.tmp', dst); copied += 1
print(f'Copied {copied} files in {time.time()-t0:.0f}s')

def load_case(cid):
    with np.load(f'{LOCAL}/{cid}.npz') as d:
        return {k: d[k] for k in d.files}
CASES = {c: load_case(c) for c in calib_cases + test_cases}
print('slices: calib', sum(len(CASES[c]['t1']) for c in calib_cases), '| test', sum(len(CASES[c]['t1']) for c in test_cases))
""")

code(r"""
# 3. Degraded inputs + input-only regions (cached per degradation; same for every seed)
def degrade(d, cid, level):
    sig, sd = DEGRADATIONS[level]
    t1, t2 = d['t1'].astype(np.float32), d['t2'].astype(np.float32)
    brain = (t1 > 0) | (t2 > 0)
    if sig == 0 and sd == 0:
        return t1, t2
    rng = np.random.default_rng(zlib.crc32(f'{cid}|{level}'.encode()))   # same noise for every seed
    out = []
    for x in (t1, t2):
        y = np.stack([cv2.GaussianBlur(s, (0, 0), sig) for s in x]) if sig > 0 else x.copy()
        y = y + rng.normal(0, sd, size=y.shape).astype(np.float32)
        y = np.where(brain, np.clip(y, 1e-3, 1.0), 0.0).astype(np.float32)   # keep the input mask unchanged
        out.append(y)
    return out[0], out[1]

def regions_for(t1, t2, method):
    brain = (t1 > 0) | (t2 > 0); guide = 0.5 * (t1 + t2)
    if method == 'gradient':
        return np.stack([cp.region_map_gradient(guide[i], brain[i], BOUNDARY_FRAC) for i in range(len(t1))])
    return np.stack([cp.region_map(guide[i], brain[i]) for i in range(len(t1))])

INPUTS, LABELS, BANDS = {}, {}, {}
t0 = time.time()
for c, d in CASES.items():
    BANDS[c] = np.stack([cp.tumour_boundary_band(s) for s in d['seg']])
for level in DEGRADATIONS:
    for c, d in CASES.items():
        t1, t2 = degrade(d, c, level)
        INPUTS[(c, level)] = (t1, t2)
        LABELS[(c, level, 'gradient')] = regions_for(t1, t2, 'gradient')
        if level == 'clean':
            LABELS[(c, level, 'canny')] = regions_for(t1, t2, 'canny')
print(f'inputs/regions ready in {time.time()-t0:.0f}s')

rows = []
for (c, level, method), lab in LABELS.items():
    for split, cases in (('calib', calib_cases), ('test', test_cases)):
        if c in cases:
            rows.append({'degradation': level, 'region_method': method, 'split': split, 'case': c,
                         **{n: int((lab == r).sum()) for n, r in REG.items()},
                         'tumour_band_px': int(BANDS[c].sum()),
                         'band_in_boundary': int((BANDS[c] & (lab == cp.REGION_BOUNDARY)).sum())})
rf = pd.DataFrame(rows).groupby(['degradation', 'region_method', 'split']).sum(numeric_only=True)
tot = rf[list(REG)].sum(axis=1)
rf_out = pd.DataFrame({f'frac_{n}': rf[n] / tot for n in REG})
rf_out['boundary_share_of_brain'] = rf['boundary'] / (rf['boundary'] + rf['interior'])
rf_out['tumour_band_in_boundary'] = rf['band_in_boundary'] / rf['tumour_band_px']
atomic_csv(rf_out.reset_index(), f'{DRIVE_ROOT}/results/conformal_region_fractions.csv')
print(rf_out.round(4).to_string())
""")

code(r"""
# 4. Per-patient residual structures for one (seed, degradation)
@torch.no_grad()
def predict(model, t1, t2, bs=64):
    out = []
    for i in range(0, len(t1), bs):
        a = torch.from_numpy(t1[i:i+bs])[:, None].to(DEVICE); b = torch.from_numpy(t2[i:i+bs])[:, None].to(DEVICE)
        out.append(model(a, b)[:, 0].float().cpu().numpy())
    return np.concatenate(out)

def patient_struct(res, lab, band):
    P = {n: np.sort(res[lab == r]) for n, r in REG.items()}
    P['brain'] = np.sort(res[lab > 0])
    P['band_res'], P['band_lab'] = res[band], lab[band]
    return P

def build_structs(seed, level):
    ck = torch.load(f'{DRIVE_ROOT}/checkpoints/flair_fedavg_seed{seed}.pt', map_location=DEVICE)
    model = build_model().to(DEVICE); model.load_state_dict(ck['state_dict']); model.eval()
    S = {}
    for c, d in CASES.items():
        t1, t2 = INPUTS[(c, level)]
        res = np.abs(predict(model, t1, t2) - d['flair'].astype(np.float32))
        for method in ('gradient', 'canny'):
            if (c, level, method) in LABELS:
                S[(c, method)] = patient_struct(res, LABELS[(c, level, method)], BANDS[c])
    return S
print('OK')
""")

code(r"""
# 5. Analysis helpers
SCHEMES = {'pooled': calib_cases, **{f'client{k}': l for k, l in enumerate(calib_by_client)}}
SUBGROUPS = {'all': test_cases, 'HGG': [c for c in test_cases if grade[c] == 'HGG'],
             'LGG': [c for c in test_cases if grade[c] == 'LGG']}

def pooled_sorted(S, cases, method, region):
    return np.sort(np.concatenate([S[(c, method)][region] for c in cases]))

def pixel_cov(S, cases, method, region, tau):
    cov = np.array([np.searchsorted(S[(c, method)][region], tau, side='right') for c in cases], dtype=np.float64)
    tot = np.array([S[(c, method)][region].size for c in cases], dtype=np.float64)
    keep = tot > 0
    cov, tot = cov[keep], tot[keep]
    if tot.size == 0:
        return np.nan, np.nan, np.nan, 0
    lo, hi = cp.cluster_bootstrap_coverage(cov, tot, n_boot=N_BOOT, seed=0)
    return float(cov.sum() / tot.sum()), lo, hi, int(tot.size)

def patient_scores_of(S, cases, method, region):
    return np.array([cp.within_unit_quantile(S[(c, method)][region], BETA) if S[(c, method)][region].size else np.nan
                     for c in cases])

def patient_cov(scores, tau):
    s = scores[~np.isnan(scores)]
    k = int((s <= tau).sum()); n = int(s.size)
    lo, hi = cp.clopper_pearson(k, n) if n else (np.nan, np.nan)
    return (k / n if n else np.nan), lo, hi, n

def thresholds(S, method):
    # Calibration thresholds for every scheme/level/region/alpha (json-serialisable).
    T = {}
    for sch, cases in SCHEMES.items():
        for region in ('interior', 'boundary', 'background', 'brain'):
            srt = pooled_sorted(S, cases, method, region)
            sc = patient_scores_of(S, cases, method, region)
            for a in ALPHAS:
                T[f'pixel|{sch}|{region}|{a}'] = cp.conformal_quantile(srt, a) if srt.size else float('nan')
                T[f'patient|{sch}|{region}|{a}'] = (cp.patient_level_threshold(sc, a)
                                                    if np.any(~np.isnan(sc)) else float('nan'))
            T[f'n|{sch}|{region}'] = int(np.sum(~np.isnan(sc)))
    return T

def coverage_rows(S, method, T, seed, level, calib_condition):
    rows = []
    base = dict(seed=seed, degradation=level, region_method=method, calib_condition=calib_condition)
    for sch in SCHEMES:
        for a in ALPHAS:
            for region in ('interior', 'boundary', 'background'):
                for ttype, tau in (('mondrian', T[f'pixel|{sch}|{region}|{a}']), ('global_brain', T[f'pixel|{sch}|brain|{a}'])):
                    if ttype == 'global_brain' and region == 'background':
                        continue
                    for sg, cases in SUBGROUPS.items():
                        cov, lo, hi, n = pixel_cov(S, cases, method, region, tau)
                        rows.append({**base, 'level': 'pixel', 'scheme': sch, 'threshold_type': ttype, 'alpha': a,
                                     'region': region, 'subgroup': sg, 'tau': tau, 'n_cal': T[f'n|{sch}|{region}'],
                                     'coverage': cov, 'ci_lo': lo, 'ci_hi': hi, 'n_test': n})
            for region in ('interior', 'boundary', 'brain'):
                tau = T[f'patient|{sch}|{region}|{a}']
                for sg, cases in SUBGROUPS.items():
                    cov, lo, hi, n = patient_cov(patient_scores_of(S, cases, method, region), tau)
                    rows.append({**base, 'level': 'patient', 'scheme': sch, 'threshold_type': 'mondrian', 'alpha': a,
                                 'region': region, 'subgroup': sg, 'tau': tau, 'n_cal': T[f'n|{sch}|{region}'],
                                 'coverage': cov, 'ci_lo': lo, 'ci_hi': hi, 'n_test': n})
    return rows

def crc_rows(S, T, seed, level):
    # Tumour-boundary miscoverage risk; interval shape = pooled pixel-Mondrian tau per region (alpha = CRC_ALPHA).
    lam = np.array(LAMBDAS)
    rows = []
    for sch, cases in SCHEMES.items():
        tau_map = np.array([max(T[f'pixel|{sch}|{n}|{CRC_ALPHA}'], 1e-3) for n in ('background', 'interior', 'boundary')])
        def curve(c):
            P = S[(c, 'gradient')]
            if P['band_res'].size == 0:
                return None
            return cp.miscoverage_loss_curve(P['band_res'], tau_map[P['band_lab']], np.ones(P['band_res'].size, bool), lam)
        cal = [x for x in (curve(c) for c in cases) if x is not None]
        tst = np.array([x for x in (curve(c) for c in test_cases) if x is not None])
        if not cal or not len(tst):
            rows.append({'seed': seed, 'degradation': level, 'scheme': sch, 'alpha': CRC_ALPHA, 'n_cal': len(cal),
                         'lambda_hat': np.nan, 'fed_equals_central': np.nan, 'test_risk': np.nan,
                         'ci_lo': np.nan, 'ci_hi': np.nan, 'test_risk_lambda1': np.nan})
            continue
        lam_hat = cp.crc_lambda(np.array(cal), lam, CRC_ALPHA)
        fed_hat = cp.crc_from_sums(np.sum([cal[i] for i in range(len(cal))], axis=0), len(cal), lam, CRC_ALPHA)
        if np.isfinite(lam_hat):
            risk = tst[:, np.searchsorted(lam, lam_hat)]
            bs = np.random.default_rng(0).integers(0, len(risk), (N_BOOT, len(risk)))
            lo, hi = np.quantile(risk[bs].mean(1), [0.025, 0.975])
            r_mean = float(risk.mean())
        else:
            r_mean, lo, hi = 0.0, 0.0, 0.0
        rows.append({'seed': seed, 'degradation': level, 'scheme': sch, 'alpha': CRC_ALPHA, 'n_cal': len(cal),
                     'lambda_hat': lam_hat, 'fed_equals_central': bool(fed_hat == lam_hat),
                     'test_risk': r_mean, 'ci_lo': float(lo), 'ci_hi': float(hi),
                     'test_risk_lambda1': float(tst[:, np.searchsorted(lam, 1.0)].mean())})
    return rows

def fedcheck_rows(S, seed, level):
    rows = []
    for region in ('interior', 'boundary', 'brain'):
        per_client = [np.concatenate([S[(c, 'gradient')][region] for c in l]) for l in calib_by_client]
        pooled = np.concatenate(per_client)
        if pooled.size == 0:
            continue
        for a in ALPHAS:
            q_c = cp.conformal_quantile(pooled, a)
            q_f, rounds = cp.federated_conformal_quantile([cp.count_fn(x) for x in per_client],
                                                          [x.size for x in per_client], a, upper=1.0, return_rounds=True)
            rows.append({'seed': seed, 'degradation': level, 'region': region, 'alpha': a, 'tau_central': q_c,
                         'tau_federated': q_f, 'identical': bool(q_c == q_f), 'rounds': rounds, 'n_pixels': int(pooled.size)})
    return rows
print('OK')
""")

code(r"""
# 6. Main loop over (seed, degradation) — resumable; 'clean' first because degraded units also test clean thresholds
def unit_paths(seed, level):
    b = f'{RUN_DIR}/seed{seed}_{level}'
    return {'cov': b + '_coverage.csv', 'crc': b + '_crc.csv', 'fed': b + '_fedcheck.csv', 'thr': b + '_thresholds.json'}

def unit_done(seed, level):
    p = unit_paths(seed, level)
    try:
        for k in ('cov', 'crc', 'fed'):
            pd.read_csv(p[k])
        with open(p['thr']) as f:
            json.load(f)
        return True
    except Exception:
        return False

order = [(s, l) for s in SEEDS for l in DEGRADATIONS]
todo = [u for u in order if not unit_done(*u)]
print(f'{len(todo)} unit(s) to do')
for i, (seed, level) in enumerate(todo):
    t0 = time.time()
    S = build_structs(seed, level)
    T = {m: thresholds(S, m) for m in ('gradient', 'canny') if (test_cases[0], m) in S}
    cov = []
    for m, Tm in T.items():
        cov += coverage_rows(S, m, Tm, seed, level, 'matched')
    if level != 'clean':
        with open(unit_paths(seed, 'clean')['thr']) as f:
            T_clean = json.load(f)['gradient']
        cov += coverage_rows(S, 'gradient', T_clean, seed, level, 'clean')
    p = unit_paths(seed, level)
    atomic_csv(pd.DataFrame(cov), p['cov'])
    atomic_csv(pd.DataFrame(crc_rows(S, T['gradient'], seed, level)), p['crc'])
    atomic_csv(pd.DataFrame(fedcheck_rows(S, seed, level)), p['fed'])
    atomic_json(T, p['thr'])          # written last = unit-complete marker
    del S
    print(f'  [{i+1}/{len(todo)}] seed {seed} {level}: {(time.time()-t0)/60:.1f} min')
print('All units finished.')
""")

code(r"""
# 7. Summaries (mean ± std over seeds)
cov = pd.concat([pd.read_csv(unit_paths(s, l)['cov']) for s, l in order])
crc = pd.concat([pd.read_csv(unit_paths(s, l)['crc']) for s, l in order])
fed = pd.concat([pd.read_csv(unit_paths(s, l)['fed']) for s, l in order])
keys = ['degradation', 'region_method', 'calib_condition', 'level', 'scheme', 'threshold_type', 'alpha', 'region', 'subgroup']
summ = cov.groupby(keys).agg(tau_mean=('tau', 'mean'), tau_std=('tau', 'std'), n_cal=('n_cal', 'first'),
                             cov_mean=('coverage', 'mean'), cov_std=('coverage', 'std'),
                             ci_lo_mean=('ci_lo', 'mean'), ci_hi_mean=('ci_hi', 'mean'), n_test=('n_test', 'first')).reset_index()
atomic_csv(summ, f'{DRIVE_ROOT}/results/conformal_summary.csv')
crc_s = crc.groupby(['degradation', 'scheme']).agg(n_cal=('n_cal', 'first'), lambda_mean=('lambda_hat', 'mean'),
        lambda_std=('lambda_hat', 'std'), risk_mean=('test_risk', 'mean'), risk_std=('test_risk', 'std'),
        risk_lambda1=('test_risk_lambda1', 'mean'), fed_equal=('fed_equals_central', 'all')).reset_index()
atomic_csv(crc_s, f'{DRIVE_ROOT}/results/conformal_crc_summary.csv')
atomic_csv(fed, f'{DRIVE_ROOT}/results/conformal_fedcheck.csv')

pd.set_option('display.width', 250)
def view(title, q, cols):
    print(f'\n=== {title} ===')
    print(summ.query(q)[cols].round(4).to_string(index=False))

C = "degradation=='clean' and calib_condition=='matched' and subgroup=='all'"
print(f"Federated == pooled quantile in all {len(fed)} checks: {bool(fed.identical.all())} (max rounds {fed.rounds.max()})")
view('A. Pixel-level, pooled calib, alpha=0.10: Mondrian vs global-brain threshold',
     C + " and level=='pixel' and scheme=='pooled' and alpha==0.10",
     ['region_method', 'threshold_type', 'region', 'tau_mean', 'tau_std', 'cov_mean', 'cov_std', 'ci_lo_mean', 'ci_hi_mean'])
view('B. Patient-level (exact) guarantee, pooled calib, gradient regions',
     C + " and level=='patient' and scheme=='pooled' and region_method=='gradient'",
     ['alpha', 'region', 'n_cal', 'tau_mean', 'cov_mean', 'cov_std', 'ci_lo_mean', 'ci_hi_mean', 'n_test'])
view('C. Calibration scheme x test grade, patient-level brain, alpha=0.10',
     "degradation=='clean' and calib_condition=='matched' and level=='patient' and region=='brain' and alpha==0.10 and region_method=='gradient'",
     ['scheme', 'subgroup', 'n_cal', 'tau_mean', 'cov_mean', 'cov_std', 'ci_lo_mean', 'ci_hi_mean', 'n_test'])
view('C2. Per-client feasibility at alpha=0.05 (inf = too few calibration patients)',
     C + " and level=='patient' and region=='brain' and alpha==0.05 and region_method=='gradient'",
     ['scheme', 'n_cal', 'tau_mean', 'cov_mean'])
view('D. Degradation: pixel-level boundary/interior, pooled, alpha=0.10 (matched vs calibrated on clean)',
     "level=='pixel' and scheme=='pooled' and alpha==0.10 and subgroup=='all' and threshold_type=='mondrian' and region!='background' and region_method=='gradient'",
     ['degradation', 'calib_condition', 'region', 'tau_mean', 'cov_mean', 'cov_std', 'ci_lo_mean'])
print('\n=== E. Conformal risk control on the tumour-boundary band (alpha = %.2f) ===' % CRC_ALPHA)
print(crc_s.round(4).to_string(index=False))
""")

code(r"""
# 8. Example figure (seed 42, clean, pooled thresholds, alpha = 0.10)
seed, a = SEEDS[0], 0.10
with open(unit_paths(seed, 'clean')['thr']) as f:
    T = json.load(f)['gradient']
ck = torch.load(f'{DRIVE_ROOT}/checkpoints/flair_fedavg_seed{seed}.pt', map_location=DEVICE)
model = build_model().to(DEVICE); model.load_state_dict(ck['state_dict']); model.eval()
c = sorted(test_cases, key=lambda x: -BANDS[x].sum())[0]
d = CASES[c]; i = int(np.argmax(BANDS[c].reshape(len(BANDS[c]), -1).sum(1)))
t1, t2 = INPUTS[(c, 'clean')]
p = predict(model, t1[i:i+1], t2[i:i+1])[0]; fl = d['flair'][i].astype(np.float32); err = np.abs(p - fl)
lab = LABELS[(c, 'clean', 'gradient')][i]
half = np.choose(lab, [T[f'pixel|pooled|{n}|{a}'] for n in ('background', 'interior', 'boundary')])
miss = (err > half) & (lab > 0)
panels = [(fl, 'FLAIR (target)', 'gray', 0, 1), (p, 'FedAvg prediction', 'gray', 0, 1), (err, '|error|', 'magma', 0, 0.5),
          (lab, 'regions bg/int/bnd', 'viridis', 0, 2), (half, f'half-width tau (alpha={a})', 'magma', 0, 0.5),
          (miss.astype(float) + 0.5 * BANDS[c][i], 'miss (1) / tumour band (0.5)', 'viridis', 0, 1.5)]
fig, ax = plt.subplots(1, 6, figsize=(20, 3.6))
for j, (img, title, cm, lo, hi) in enumerate(panels):
    ax[j].imshow(img, cmap=cm, vmin=lo, vmax=hi); ax[j].set_title(title, fontsize=9); ax[j].axis('off')
plt.suptitle(f'{c} slice {int(d["slice_idx"][i])}', fontsize=10); plt.tight_layout()
out = f'{DRIVE_ROOT}/figures/phase4_conformal_example.png'
plt.savefig(out, dpi=130, bbox_inches='tight'); plt.show(); print('Saved', out)
""")

md(r"""
## Done — paste back to chat
1. The region-fraction table (cell 3).
2. Everything printed by **cell 7** (federated check line + tables A–E).
3. Whether `phase4_conformal_example.png` looks sensible.

Then download `results/conformal_summary.csv`, `conformal_crc_summary.csv`, `conformal_fedcheck.csv`, `conformal_region_fractions.csv` and the PNG.
""")

nb = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"},
                 "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"}},
    "nbformat": 4, "nbformat_minor": 5,
}
for c in nb["cells"]:
    c["source"] = c["source"].splitlines(keepends=True)
with open(os.path.join(REPO, "notebooks/15_fed_conformal.ipynb"), "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", len(cells), "cells")
