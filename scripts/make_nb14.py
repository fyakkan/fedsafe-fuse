"""Generate notebooks/14_fed_flair_train.ipynb (embeds src/ modules verbatim).

Run: python3 scripts/make_nb14.py   (re-run whenever an embedded src/ file changes)
"""
import json
import os
import sys

REPO = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..")
EMBED = {
    "ifcnn.py": "src/models/ifcnn.py",
    "losses.py": "src/losses.py",
    "trainer.py": "src/fed/trainer.py",
}

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})


md(r"""
# 14 — Federated FLAIR prediction from T1 + T2 (Round-2, Phase 3)

**What.** Train **IFCNN** (output zeroed outside the input-derived brain mask `(T1>0)|(T2>0)`; same backbone for every arm; review M5) to predict the held-out **FLAIR** from **T1 + T2** with **FedAvg** over the 3 grade-skewed clients, plus a **centralised** reference trained on the pooled client-train patients with the identical trainer (K = 1). **5 seeds** each. Final metrics on the **54 test patients** use exactly the Phase-2 baseline definitions, so the model is compared like-for-like with `zero`, `T2`, `mean(T1,T2)` and `linear(T1,T2)`.

**Protocol (fixed in advance; no model selection).** T = 30 rounds, E = 1 local epoch of 1 000 random slices per client (centralised: 3 000 slices per round = same total samples), batch 16, Adam lr 1e-3, loss L1 + (1 − SSIM) against FLAIR. Validation patients are used only for the printed learning curve; the round-30 model is always the one evaluated.

**Needs a GPU runtime** (Runtime → Change runtime type → T4 GPU).

**Runtime.** Copy ≈ 5 min; sanity gate ≈ 3 min; each run ≈ 15–20 min on a T4 → 10 runs ≈ 3 h. **Resumable per run**: finished (method, seed) runs are skipped, so if the session dies just *Run all* again. A run interrupted mid-way restarts from round 1 (≤ 20 min lost).

**Outputs (Drive `MyDrive/FedSafeFuse/`):**
| Drive path | Download to local |
|---|---|
| `checkpoints/flair_<method>_seed<s>.pt` | (stays on Drive; Phase 4 uses them) |
| `results/flair_runs/<method>_seed<s>_history.csv` | — |
| `results/flair_runs/<method>_seed<s>_test.csv` | — |
| `results/flair_test_summary.csv` | `results/flair_test_summary.csv` |
| `results/flair_train_history.csv` | `results/flair_train_history.csv` |
| `figures/phase3_flair_examples.png` | `figures/phase3_flair_examples.png` |
""")

code(r"""
# 0. Config  (do not change between sessions, or resumed runs will not be comparable)
SEEDS      = [42, 1, 2, 3, 4]
METHODS    = ['fedavg', 'central']
T_ROUNDS   = 30
E_LOCAL    = 1
SAMPLES_PER_CLIENT = 1000      # per local epoch; centralised uses K * this per round
BATCH      = 16
LR         = 1e-3
LOG_EVERY  = 5
GATE_ROUNDS = 5                # sanity gate: short centralised run before the long runs
REQUIRE_GPU = True
""")

code(r"""
# 1. Setup + mount Drive
import os, sys, json, time, glob, random, shutil, importlib
import numpy as np, pandas as pd
import torch
import matplotlib.pyplot as plt

if REQUIRE_GPU and not torch.cuda.is_available():
    raise RuntimeError('No GPU. Runtime -> Change runtime type -> T4 GPU, then Run all again.')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Torch', torch.__version__, '| device', DEVICE, torch.cuda.get_device_name(0) if DEVICE == 'cuda' else '')

from google.colab import drive
if not os.path.exists('/content/drive/MyDrive'):
    drive.mount('/content/drive')
DRIVE_ROOT = '/content/drive/MyDrive/FedSafeFuse'
LOCAL      = '/content/local_flair'
SRC_DIR    = '/content/fedsrc'
RUN_DIR    = f'{DRIVE_ROOT}/results/flair_runs'
CKPT_DIR   = f'{DRIVE_ROOT}/checkpoints'
for d in [LOCAL, SRC_DIR, RUN_DIR, CKPT_DIR, f'{DRIVE_ROOT}/figures']:
    os.makedirs(d, exist_ok=True)

def atomic_csv(df, path):
    tmp = path + '.tmp'; df.to_csv(tmp, index=False); os.replace(tmp, path)

def atomic_torch_save(obj, path):
    tmp = path + '.tmp'; torch.save(obj, tmp); os.replace(tmp, path)
print('OK')
""")

md("Embedded verbatim from `src/` (unit-tested locally; do not edit here).")
for name, rel in EMBED.items():
    code(f"%%writefile /content/fedsrc/{name}\n" + open(os.path.join(REPO, rel)).read())

code(r"""
sys.path.insert(0, SRC_DIR)
import ifcnn, losses, trainer
for m in (ifcnn, losses, trainer):
    importlib.reload(m)
from ifcnn import IFCNN, InputMasked
from losses import TargetLoss, ssim_score, psnr_score
from trainer import FedAvgTrainer
def build_model():
    # IFCNN with output zeroed outside the input-derived brain mask (T1>0)|(T2>0)
    return InputMasked(IFCNN())
print('IFCNN params:', sum(p.numel() for p in build_model().parameters()))
""")

code(r"""
# 2. Partition + copy needed case files Drive -> local disk (resumable)
with open(f'{DRIVE_ROOT}/partitions/brats_K3_v2.json') as f:
    part = json.load(f)
clients = part['clients']
train_cases = [cl['train'] for cl in clients]
val_cases   = sorted(c for cl in clients for c in cl['val'])
test_cases  = part['test_cases']
needed = sorted({c for l in train_cases for c in l} | set(val_cases) | set(test_cases))
print(f'clients train patients={[len(l) for l in train_cases]} | val={len(val_cases)} | test={len(test_cases)}')

t0 = time.time(); copied = 0
for cid in needed:
    src, dst = f'{DRIVE_ROOT}/brats/processed_v2/{cid}.npz', f'{LOCAL}/{cid}.npz'
    if not os.path.exists(src):
        raise FileNotFoundError(f'{src} missing - re-run notebook 13')
    if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
        continue
    shutil.copy(src, dst + '.tmp'); os.replace(dst + '.tmp', dst); copied += 1
print(f'Copied {copied} new files ({len(needed)} needed) in {time.time()-t0:.0f}s')
""")

code(r"""
# 3. Datasets: (t1, t2, flair) float32 tensors of shape (1,128,128); arrays kept in RAM as float16
from torch.utils.data import Dataset

def load_case(cid):
    with np.load(f'{LOCAL}/{cid}.npz') as d:
        return {k: d[k] for k in d.files}

class SliceDS(Dataset):
    def __init__(self, cases):
        parts = [load_case(c) for c in cases]
        self.t1 = np.concatenate([p['t1'] for p in parts])
        self.t2 = np.concatenate([p['t2'] for p in parts])
        self.fl = np.concatenate([p['flair'] for p in parts])
    def __len__(self):
        return len(self.t1)
    def __getitem__(self, i):
        f = lambda a: torch.from_numpy(a[i].astype(np.float32))[None]
        return f(self.t1), f(self.t2), f(self.fl)

t0 = time.time()
client_ds = [SliceDS(c) for c in train_cases]
pooled_ds = SliceDS([c for l in train_cases for c in l])
val_ds    = SliceDS(val_cases)
print(f'slices: clients={[len(d) for d in client_ds]} pooled={len(pooled_ds)} val={len(val_ds)} '
      f'| loaded in {time.time()-t0:.0f}s')
assert len(pooled_ds) == sum(len(d) for d in client_ds)
""")

code(r"""
# 4. Test-set metrics, identical definitions to notebook 13 cell 6 (+ brain-only PSNR/SSIM)
from skimage.metrics import structural_similarity as ssim_fn

@torch.no_grad()
def predict(model, d, bs=64):
    model.eval(); out = []
    for i in range(0, len(d['t1']), bs):
        a = torch.from_numpy(d['t1'][i:i+bs].astype(np.float32))[:, None].to(DEVICE)
        b = torch.from_numpy(d['t2'][i:i+bs].astype(np.float32))[:, None].to(DEVICE)
        out.append(model(a, b)[:, 0].float().cpu().numpy())
    return np.concatenate(out)

def case_metrics(p, d):
    fl = d['flair'].astype(np.float32); br = d['brain'] > 0
    maes, ssims, ssims_b, se_b, n_b = [], [], [], 0.0, 0
    for i in range(len(fl)):
        ssims.append(ssim_fn(fl[i], p[i], data_range=1.0))
        if br[i].any():
            maes.append(np.abs(p[i] - fl[i])[br[i]].mean())
            _, smap = ssim_fn(fl[i], p[i], data_range=1.0, full=True)
            ssims_b.append(smap[br[i]].mean())
            se_b += float(((p[i] - fl[i])[br[i]] ** 2).sum()); n_b += int(br[i].sum())
    mse = float(np.mean((p - fl) ** 2))
    return {'ssim': float(np.mean(ssims)), 'psnr': 10 * np.log10(1 / max(mse, 1e-12)),
            'mae_brain': float(np.mean(maes)), 'ssim_brain': float(np.mean(ssims_b)),
            'psnr_brain': 10 * np.log10(1 / max(se_b / max(n_b, 1), 1e-12))}

BASELINES = {
    'zero':        lambda t1, t2: np.zeros_like(t1),
    'T2':          lambda t1, t2: t2,
    'mean(T1,T2)': lambda t1, t2: 0.5 * (t1 + t2),
}
rows = []
for c in test_cases:
    d = load_case(c); t1, t2 = d['t1'].astype(np.float32), d['t2'].astype(np.float32)
    for name, f in BASELINES.items():
        rows.append({'case': c, 'method': name, **case_metrics(f(t1, t2), d)})
base_df = pd.DataFrame(rows)
base_sum = base_df.groupby('method').mean(numeric_only=True)
print(base_sum.round(4).to_string())

# Self-check: must reproduce notebook 13's numbers (same data, same metric code)
ref = pd.read_csv(f'{DRIVE_ROOT}/results/flair_trivial_baselines.csv', header=[0, 1], index_col=0)
for name in BASELINES:
    for m in ('ssim', 'psnr', 'mae_brain'):
        a, b = base_sum.loc[name, m], ref.loc[name, (m, 'mean')]
        assert abs(a - b) < 1e-3, f'metric mismatch {name}/{m}: {a:.4f} vs notebook-13 {b:.4f}'
print('Self-check vs notebook 13 baselines: PASSED')
""")

code(r"""
# 5. Run helpers (one (method, seed) run = train T rounds, save checkpoint, evaluate on test)
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

def run_paths(method, seed):
    tag = f'{method}_seed{seed}'
    return tag, f'{CKPT_DIR}/flair_{tag}.pt', f'{RUN_DIR}/{tag}_history.csv', f'{RUN_DIR}/{tag}_test.csv'

def run_done(method, seed):
    _, ck, hi, te = run_paths(method, seed)
    if not all(os.path.exists(p) for p in (ck, hi, te)):
        return False
    try:
        torch.load(ck, map_location='cpu'); pd.read_csv(hi); pd.read_csv(te)
        return True
    except Exception as e:
        print(f'  {method}/seed{seed}: unreadable output ({e}); will redo')
        return False

def make_trainer(method, seed, samples):
    set_seed(seed)
    ds = client_ds if method == 'fedavg' else [pooled_ds]
    per_client = samples if method == 'fedavg' else samples * len(client_ds)
    return FedAvgTrainer(
        build_model_fn=build_model, loss_fn_factory=lambda: TargetLoss(beta=1.0),
        client_datasets=ds, eval_dataset=val_ds, device=DEVICE, mode='standard',
        samples_per_local_epoch=per_client, lr=LR, batch_size=BATCH, seed=seed,
        supervised_target=True,
    )

def run_one(method, seed):
    tag, ck, hi, te = run_paths(method, seed)
    t0 = time.time()
    tr = make_trainer(method, seed, SAMPLES_PER_CLIENT)
    hist = tr.run(T=T_ROUNDS, E=E_LOCAL, ssim_score_fn=ssim_score, psnr_score_fn=psnr_score,
                  log_every=LOG_EVERY, tag=tag)
    model = build_model().to(DEVICE); model.load_state_dict(tr.global_state)
    rows = []
    for c in test_cases:
        d = load_case(c)
        rows.append({'case': c, 'method': method, 'seed': seed, **case_metrics(predict(model, d), d)})
    atomic_csv(pd.DataFrame(hist).assign(method=method, seed=seed), hi)
    atomic_csv(pd.DataFrame(rows), te)
    atomic_torch_save({'state_dict': tr.global_state, 'method': method, 'seed': seed, 'rounds': T_ROUNDS,
                       'samples_per_client': SAMPLES_PER_CLIENT, 'partition': 'brats_K3_v2'}, ck)  # last = done marker
    m = pd.DataFrame(rows).mean(numeric_only=True)
    print(f'  DONE {tag} in {(time.time()-t0)/60:.1f} min | test ssim={m.ssim:.4f} psnr={m.psnr:.2f} '
          f'mae_brain={m.mae_brain:.4f} ssim_brain={m.ssim_brain:.4f}')
print('OK')
""")

code(r"""
# 6. Sanity gate: short centralised run must beat mean(T1,T2) on validation before spending hours
if all(run_done(m, s) for m in METHODS for s in SEEDS):
    print('All runs already finished - gate skipped.')
else:
    t0 = time.time()
    gate = make_trainer('central', 0, SAMPLES_PER_CLIENT)
    gate.run(T=GATE_ROUNDS, E=1, ssim_score_fn=ssim_score, psnr_score_fn=psnr_score, log_every=GATE_ROUNDS, tag='gate')
    gm = build_model().to(DEVICE); gm.load_state_dict(gate.global_state)
    vm, vb = [], []
    for c in val_cases:
        d = load_case(c); p = predict(gm, d)
        vm.append(case_metrics(p, d)); vb.append(case_metrics(0.5 * (d['t1'].astype(np.float32) + d['t2'].astype(np.float32)), d))
    vm, vb = pd.DataFrame(vm).mean(), pd.DataFrame(vb).mean()
    print(f'Gate ({(time.time()-t0)/60:.1f} min) on VAL: model mae_brain={vm.mae_brain:.4f} ssim_brain={vm.ssim_brain:.4f} '
          f'| mean(T1,T2) mae_brain={vb.mae_brain:.4f} ssim_brain={vb.ssim_brain:.4f}')
    if vm.mae_brain >= vb.mae_brain:
        raise RuntimeError('Gate FAILED: model does not beat mean(T1,T2) after the short run. '
                           'Stop and paste this output to chat before running the long runs.')
    print('Gate PASSED')
    del gate, gm
""")

code(r"""
# 7. Main runs (resumable). Order: fedavg seed by seed, then central.
todo = [(m, s) for m in METHODS for s in SEEDS if not run_done(m, s)]
print(f'{len(todo)} run(s) to do:', todo)
for i, (m, s) in enumerate(todo):
    print(f'\n=== [{i+1}/{len(todo)}] {m} seed {s} ===')
    run_one(m, s)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
print('\nAll runs finished.')
""")

code(r"""
# 8. Summary: per-run means over the 54 test patients, then mean ± std over seeds; baselines alongside
te = pd.concat([pd.read_csv(run_paths(m, s)[3]) for m in METHODS for s in SEEDS])
hist = pd.concat([pd.read_csv(run_paths(m, s)[2]) for m in METHODS for s in SEEDS])
atomic_csv(hist, f'{DRIVE_ROOT}/results/flair_train_history.csv')

metrics = ['ssim', 'psnr', 'mae_brain', 'ssim_brain', 'psnr_brain']
per_run = te.groupby(['method', 'seed'])[metrics].mean()
model_sum = per_run.groupby('method').agg(['mean', 'std'])
base_rows = base_sum[metrics].copy()
base_rows.columns = pd.MultiIndex.from_product([metrics, ['mean']])
summary = pd.concat([model_sum, base_rows])
summary.columns = [f'{a}_{b}' for a, b in summary.columns]
summary.index.name = 'method'
summary.reset_index().to_csv(f'{DRIVE_ROOT}/results/flair_test_summary.csv', index=False)
print(f'Test set ({len(test_cases)} patients). Models: mean (std) over {len(SEEDS)} seeds of the per-run patient mean. '
      'Baselines: deterministic.')
print(summary.round(4).to_string())
print('\nPer-run means:')
print(per_run.round(4).to_string())
""")

code(r"""
# 9. Qualitative figure: FedAvg seed 42 on 3 test patients
ck = torch.load(run_paths('fedavg', SEEDS[0])[1], map_location=DEVICE)
model = build_model().to(DEVICE); model.load_state_dict(ck['state_dict'])
rng = np.random.default_rng(0)
n_show = min(3, len(test_cases))
fig, ax = plt.subplots(n_show, 5, figsize=(15, 3 * n_show), squeeze=False)
for r, c in enumerate(rng.choice(test_cases, size=n_show, replace=False)):
    d = load_case(c); i = len(d['t1']) // 2; p = predict(model, {k: d[k][i:i+1] for k in ('t1', 't2')})[0]
    fl = d['flair'][i].astype(np.float32)
    for j, (img, title, kw) in enumerate([
        (d['t1'][i], 'T1 (input)', dict(cmap='gray', vmin=0, vmax=1)),
        (d['t2'][i], 'T2 (input)', dict(cmap='gray', vmin=0, vmax=1)),
        (fl, 'FLAIR (target)', dict(cmap='gray', vmin=0, vmax=1)),
        (p, 'FedAvg prediction', dict(cmap='gray', vmin=0, vmax=1)),
        (np.abs(p - fl), '|error|', dict(cmap='magma', vmin=0, vmax=0.5)),
    ]):
        ax[r, j].imshow(np.asarray(img, dtype=np.float32), **kw); ax[r, j].axis('off')
        ax[r, j].set_title(f'{c[-3:]} {title}' if j == 0 else title, fontsize=9)
plt.tight_layout()
out = f'{DRIVE_ROOT}/figures/phase3_flair_examples.png'
plt.savefig(out, dpi=130, bbox_inches='tight'); plt.show()
print('Saved', out)
""")

md(r"""
## Done — paste back to chat
1. The **gate** line (cell 6).
2. The **summary table** and **per-run means** (cell 8).
3. Whether `phase3_flair_examples.png` looks sensible.

Then download `results/flair_test_summary.csv`, `results/flair_train_history.csv` and the PNG into the local repo.
""")

nb = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"},
                 "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"}},
    "nbformat": 4, "nbformat_minor": 5,
}
for c in nb["cells"]:
    c["source"] = c["source"].splitlines(keepends=True)
with open(os.path.join(REPO, "notebooks/14_fed_flair_train.ipynb"), "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", len(cells), "cells")
