"""Honest 5-fold VIPEEGNet + PRISM blend.
For each fold k the blend weight is chosen on the OTHER four folds only, then applied to fold k.
Confidence intervals resample patients (all rows of a patient together).
Writes the blended out-of-fold predictions in the archive schema to fair_comparison/<OUT>/ so
compare_models.ipynb can treat the blend as a model of its own.

Usage: python blend_5fold.py [PRISM_RUN] [OUT]     defaults: prism_wf5_tta  blend_vip_prism_wf5
"""
import os, sys, json, numpy as np, pandas as pd
P = os.path.expanduser(os.environ.get('HMS', '~/hms'))
RUN = sys.argv[1] if len(sys.argv) > 1 else 'prism_wf5_tta'
OUT = f'{P}/fair_comparison/' + (sys.argv[2] if len(sys.argv) > 2 else 'blend_vip_prism_wf5')
VIP = f'{P}/results/vipeegnet_groupkfold_v1/oof'
CLS = ['Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA', 'Other']
W = np.round(np.arange(0, 1.001, 0.05), 2)

def kld(t, p, eps=1e-7):
    t, p = np.clip(t, eps, 1), np.clip(p, eps, 1)
    return (t * np.log(t / p)).sum(1)

def keys(z):
    v = np.rint(z['yt'] * z['wt'][:, None]).astype(int)
    return np.array([f'{e}_' + '_'.join(map(str, r)) for e, r in zip(z['eeg_ids'], v)])

def mix(a, b, w, kind):
    if kind == 'arith': return (1 - w) * a + w * b
    q = np.exp((1 - w) * np.log(np.clip(a, 1e-7, 1)) + w * np.log(np.clip(b, 1e-7, 1)))
    return q / q.sum(1, keepdims=True)

pat = pd.read_csv(f'{P}/hms-data/train.csv', usecols=['eeg_id', 'patient_id']).drop_duplicates('eeg_id').set_index('eeg_id').patient_id
F = []
for k in range(5):
    zv = dict(np.load(f'{VIP}/fold{k}_oof.npz')); zp = np.load(f'{P}/fair_comparison/{RUN}/fold{k}_oof.npz')
    kv, kp = keys(zv), keys(zp)
    pos = pd.Series(np.arange(len(kp)), index=kp)
    assert pos.index.is_unique and set(kv) == set(kp), f'fold {k}: row sets differ'
    ip = pos[kv].values
    assert np.allclose(zv['yt'], zp['yt'][ip], atol=1e-5)
    F.append(dict(z=zv, t=zv['yt'], a=zv['yp'], b=zp['yp'][ip], pid=pat[zv['eeg_ids']].values))

res = {}
for kind in ('arith', 'geo'):
    ws, blend = [], []
    for k in range(5):
        other = [f for j, f in enumerate(F) if j != k]
        t = np.concatenate([f['t'] for f in other]); a = np.concatenate([f['a'] for f in other]); b = np.concatenate([f['b'] for f in other])
        w = W[np.argmin([kld(t, mix(a, b, w, kind)).mean() for w in W])]
        ws.append(float(w)); blend.append(mix(F[k]['a'], F[k]['b'], w, kind))
    res[kind] = (ws, blend)

t = np.concatenate([f['t'] for f in F]); a = np.concatenate([f['a'] for f in F]); b = np.concatenate([f['b'] for f in F])
pid = np.concatenate([f['pid'] for f in F])
kind = min(res, key=lambda k: kld(t, np.concatenate(res[k][1])).mean())
ws, blend = res[kind]; q = np.concatenate(blend)
kv_, kp_, kq = kld(t, a), kld(t, b), kld(t, q)

# patient bootstrap of the gain over VIPEEGNet
up, inv = np.unique(pid, return_inverse=True)
gs = np.bincount(inv, kv_ - kq); ns = np.bincount(inv)
rng = np.random.default_rng(0); bs = []
for _ in range(5000):
    s = rng.integers(0, len(up), len(up)); bs.append(gs[s].sum() / ns[s].sum())
lo, hi = np.percentile(bs, [2.5, 97.5])

print(f'{len(t):,} rows with >=10 votes, {len(up):,} patients | PRISM run: {RUN}')
print('=' * 70)
print(f'  VIPEEGNet alone                 {kv_.mean():.4f}')
print(f'  PRISM alone                     {kp_.mean():.4f}')
print(f'  blend ({kind}, leave-fold-out)   {kq.mean():.4f}   weights on PRISM per fold {ws}')
for o in res:
    print(f'     {o:<5} leave-fold-out        {kld(t, np.concatenate(res[o][1])).mean():.4f}')
print(f'  gain over VIPEEGNet             {kv_.mean() - kq.mean():.4f}  95% patient CI ({lo:.4f}, {hi:.4f})')
print('  per fold  VIP / blend:', [f'{kld(f["t"], f["a"]).mean():.4f}/{kld(f["t"], bl).mean():.4f}' for f, bl in zip(F, blend)])
print('=' * 70)

from sklearn.metrics import roc_auc_score
oh = (t.argmax(1)[:, None] == np.arange(6)).astype(int)
print(f'{"class":>8} | AUROC VIP | PRISM | blend')
for c in range(6):
    print(f'{CLS[c]:>8} |   {roc_auc_score(oh[:, c], a[:, c]):.3f}   | {roc_auc_score(oh[:, c], b[:, c]):.3f} | {roc_auc_score(oh[:, c], q[:, c]):.3f}')
nv = np.concatenate([f['z']['wt'] for f in F])
print('\nKLD by number of votes   VIP   PRISM  blend   rows')
for lo_, hi_ in ((10, 13), (13, 16), (16, 20), (20, 999)):
    m = (nv >= lo_) & (nv < hi_)
    print(f'  {lo_:>3}-{min(hi_, 99) - 1:<3}               {kv_[m].mean():.4f} {kp_[m].mean():.4f} {kq[m].mean():.4f} {m.sum():>6}')

os.makedirs(OUT, exist_ok=True)
for k, (f, bl) in enumerate(zip(F, blend)):
    z = dict(f['z']); z['yp'] = bl.astype(np.float32)
    if 'yp_raw_logits' in z: z['yp_raw_logits'] = np.log(np.clip(bl, 1e-8, 1)).astype(np.float32)
    np.savez(f'{OUT}/fold{k}_oof.npz', **z)
json.dump({'model': 'VIPEEGNet + PRISM blend', 'prism_run': RUN, 'kind': kind, 'weights_on_prism': ws,
           'weight_selection': 'leave-one-fold-out', 'oof_kld_hq': float(kq.mean()), 'oof_kld_hq_raw': float(kq.mean()),
           'vip_kld_hq': float(kv_.mean()), 'prism_kld_hq': float(kp_.mean()), 'gain_ci95_patient': [float(lo), float(hi)],
           'n_oof_hq': int(len(t))}, open(f'{OUT}/summary.json', 'w'), indent=1)
print(f'\nblend OOF written to {OUT}/')
