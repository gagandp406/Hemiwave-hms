"""Phase 0 — inference-only improvements for the archived VIPEEGNet models (no retraining).

For every VIPEEGNet validation row (20,183 rows, out-of-fold, same order as the archive) it predicts
with the archived stage-2 weights under several test-time views:
    base     : exactly the validation input VIPEEGNet was scored on (asserted bit-identical)
    flip     : left/right hemisphere swap (LL<->RL, LP<->RP)
    trev     : time reversal            (seen in stage-1 augmentation)
    neg      : polarity inversion       (seen in stage-1 augmentation)
    sm / sp  : window shifted -2.5 s / +2.5 s (training used +-5 s jitter)
Each view is saved in the shared .npz schema, then combinations and post-hoc calibration
(temperature + per-class logit bias, fitted leave-one-fold-out so it never sees the fold it is applied to)
are scored on the >=10-vote rows.

Usage (VIPEEGNet env):  python vip_tta_eval.py            ~15-30 min on the A6000
"""
import json, os, sys, itertools, time
import numpy as np
P = os.path.expanduser(os.environ.get('HMS_PROJECT_DIR', '~/hms'))
WEIGHTS = os.environ.get('VIP_WEIGHTS', f'{P}/results/vipeegnet_groupkfold_v1/weights')
OUT = os.environ.get('VIP_TTA_OUT', f'{P}/fair_comparison/vipeegnet_tta')
SHIFT = 2.5

# ---- reuse the exact VIPEEGNet notebook code (data, cache, folds, generator, model) ----
nb = json.load(open(f'{P}/VIPEEGNet_local.ipynb'))
SKIP = ('t_all = time.time()', 'FAIR_VIPEEG =', 'import seaborn', 'import shutil')
g = {'__name__': '__main__'}
for c in nb['cells']:
    if c['cell_type'] != 'code':
        continue
    src = ''.join(c['source'])
    if any(m in src for m in SKIP):
        continue
    if 'parameters' in c.get('metadata', {}).get('tags', []):
        src += f"\nPROJECT_DIR = {P!r}\nFOLD_MODE = 'groupkfold'\nSMOKE_TEST = False\n"
        src += os.environ.get('VIP_EXTRA_PARAMS', '')
    src = src.replace("model = build_model()\nprint(f'Total params", "pass\nif False: print(f'Total params") \
             .replace("del model; tf.keras.backend.clear_session()", "tf.keras.backend.clear_session()")
    exec(compile(src, 'VIPEEGNet_local.ipynb', 'exec'), g)
tf, train, folds, eegs, TARGET_COLS = g['tf'], g['train'], g['folds'], g['eegs'], g['TARGET_COLS']
SEG_LOOKUP, RSFREQ, TARGETS_RAW = g['SEG_LOOKUP'], g['RSFREQ'], g['TARGETS_RAW']
assert g['FAST_LOOKUP'], 'needs FAST_LOOKUP=True'

N = round(50 * RSFREQ)
def window_start(eid, raw):
    'Validation segment exactly as DataGenerator(mode="valid"): middle sub-window by eeg_sub_id.'
    sub_ids, offs = SEG_LOOKUP[(int(eid),) + tuple(int(v) for v in raw)]
    return offs[np.argsort(sub_ids, kind='stable')[len(offs) // 2]]

def make_input(eid, r_eeg):
    eeg = eegs[eid][:, round(r_eeg * RSFREQ):round((r_eeg + 50) * RSFREQ)]
    eeg = np.concatenate((eeg[0:8, :], eeg[-8:, :]), axis=0)
    if eeg.shape[1] < N:
        eeg = np.pad(eeg, ((0, 0), (0, N - eeg.shape[1])))
    e2 = eeg.copy()                                   # VIPEEGNet swap -> order LL, RL, LP, RP
    eeg[4:8], eeg[8:12], eeg[12:16] = e2[12:16], e2[4:8], e2[8:12]
    return ((np.clip(eeg, -1024, 1024) + 1024) / 2048 * 255).astype(np.float32)

FLIP = np.r_[4:8, 0:4, 12:16, 8:12]                 # LL<->RL, LP<->RP
VIEWS = {
    'base': lambda x: x,
    'flip': lambda x: x[:, FLIP, :],
    'trev': lambda x: x[:, :, ::-1],
    'neg':  lambda x: 255.0 - x,                      # -(eeg) in the [0,255] scaling
}

os.makedirs(OUT, exist_ok=True)
all_views = list(VIEWS) + ['sm', 'sp']
t0 = time.time()
for k in range(5):
    _, v = folds[k]
    df = train.iloc[v].reset_index(drop=True)
    raw = df[TARGETS_RAW].to_numpy().astype(np.int64)
    eids = df['eeg_id'].to_numpy()
    starts = np.array([window_start(e, r) for e, r in zip(eids, raw)], dtype=float)
    lens = np.array([eegs[e].shape[1] / RSFREQ for e in eids])

    tf.keras.backend.clear_session()
    m = g['build_model'](backbone_weights=None)
    m.load_weights(f'{WEIGHTS}/fold{k}_stage2.weights.h5')

    # bit-exact check against the original generator on the first rows
    gen = g['DataGenerator'](df.iloc[:64].reset_index(drop=True), batch_size=64, mode='valid', eegs=eegs, stage=2)
    ref = gen[0][0]
    mine = np.stack([make_input(e, s) for e, s in zip(eids[:64], starts[:64])])
    assert np.array_equal(ref, mine), 'make_input differs from VIPEEGNet validation input'

    preds = {name: [] for name in all_views}
    B = 64
    for i in range(0, len(df), B):
        sl = slice(i, i + B)
        xb = np.stack([make_input(e, s) for e, s in zip(eids[sl], starts[sl])])
        for name, fn in VIEWS.items():
            preds[name].append(m.predict_on_batch(np.ascontiguousarray(fn(xb))))
        for name, d in (('sm', -SHIFT), ('sp', SHIFT)):
            st = np.clip(starts[sl] + d, 0, np.maximum(lens[sl] - 50, 0))
            preds[name].append(m.predict_on_batch(np.stack([make_input(e, s) for e, s in zip(eids[sl], st)])))
    yt = df[TARGET_COLS].values.astype(np.float32)
    votes = df['total_votes_raw'].values.astype(np.float32)
    hq = votes >= 10
    for name in all_views:
        yp = np.concatenate([np.asarray(p, dtype=np.float32) for p in preds[name]])
        os.makedirs(f'{OUT}/{name}', exist_ok=True)
        for fn_, msk in ((f'fold{k}_oof_all.npz', np.ones(len(df), bool)), (f'fold{k}_oof.npz', hq)):
            np.savez(f'{OUT}/{name}/{fn_}', yt=yt[msk], yp=yp[msk], wt=votes[msk],
                     yp_raw_logits=np.log(np.clip(yp[msk], 1e-8, 1)).astype(np.float32),
                     eeg_ids=eids[msk], temperature=np.array([1.0], np.float32))
    print(f'fold {k}: {len(df):,} rows x {len(all_views)} views  ({(time.time() - t0) / 60:.1f} min)', flush=True)

# ---------------- scoring ----------------
from scipy.optimize import minimize
def kld(t, p, eps=1e-7):
    t = np.clip(t, eps, 1); p = np.clip(p, eps, 1)
    return np.sum(t * np.log(t / p), 1)
def load(name, k, sub='oof'):
    return np.load(f'{OUT}/{name}/fold{k}_{sub}.npz')
def combo(names, k, sub='oof', geo=True):
    ps = np.stack([load(n, k, sub)['yp'].astype(np.float64) for n in names])
    if geo:                                            # average log-probs, renormalise
        lp = np.log(np.clip(ps, 1e-8, 1)).mean(0); p = np.exp(lp - lp.max(1, keepdims=True))
    else:
        p = ps.mean(0)
    return p / p.sum(1, keepdims=True)

def score(names, geo=True):
    per = []; allk = []
    for k in range(5):
        yt = load('base', k)['yt']; p = combo(names, k, geo=geo); s = kld(yt, p)
        per.append(float(s.mean())); allk.append(s)
    return float(np.concatenate(allk).mean()), per

rows = []
cands = [('base',), ('flip',), ('trev',), ('neg',), ('sm',), ('sp',)]
extra = [v for v in all_views if v != 'base']
for r in range(1, len(extra) + 1):
    for c in itertools.combinations(extra, r):
        cands.append(('base',) + c)
for c in cands:
    pooled, per = score(c)
    rows.append((pooled, '+'.join(c), per))
rows.sort()
base_pooled = score(('base',))[0]
n_hq = sum(len(load('base', k)['yt']) for k in range(5))
print(f'\n{"=" * 70}\nVIPEEGNet test-time views on the {n_hq:,} rows with >=10 votes (geometric mean)')
print(f'{"views":<34} {"pooled":>8} {"delta":>8}   per fold')
for pooled, name, per in rows[:12] + [r for r in rows if r[1] in ('base', 'flip', 'trev', 'neg', 'sm', 'sp')]:
    print(f'{name:<34} {pooled:8.4f} {pooled - base_pooled:+8.4f}   {[round(x, 4) for x in per]}')

# Leave-one-fold-out calibration: fit temperature T and class bias b on 4 folds, apply to the 5th
def calib_fit(logp, yt):
    def f(theta):
        T = np.exp(np.clip(theta[0], -1.5, 1.5))           # T in [0.22, 4.5]
        z = logp / T + theta[1:]; z = z - z.max(1, keepdims=True)
        q = np.exp(z); q /= q.sum(1, keepdims=True); return kld(yt, q).mean()
    return minimize(f, np.zeros(7), method='Nelder-Mead',
                    options={'maxiter': 4000, 'xatol': 1e-5, 'fatol': 1e-7}).x
def calibrated(names):
    lp = {k: np.log(np.clip(combo(names, k), 1e-8, 1)) for k in range(5)}
    yt = {k: load('base', k)['yt'] for k in range(5)}
    out = []; thetas = []
    for k in range(5):
        tr = [j for j in range(5) if j != k]
        th = calib_fit(np.concatenate([lp[j] for j in tr]), np.concatenate([yt[j] for j in tr])); thetas.append(th)
        z = lp[k] / np.exp(np.clip(th[0], -1.5, 1.5)) + th[1:]; z -= z.max(1, keepdims=True); q = np.exp(z); q /= q.sum(1, keepdims=True)
        out.append(kld(yt[k], q))
    return float(np.concatenate(out).mean()), [float(o.mean()) for o in out], thetas

best = rows[0][1].split('+')
print(f'\nCalibration (temperature + class bias, fitted leave-one-fold-out):')
for names in (['base'], best):
    pooled, per, th = calibrated(names)
    print(f'  {"+".join(names):<32} {pooled:8.4f} {pooled - base_pooled:+8.4f}   {[round(x, 4) for x in per]}'
          f'   T={np.mean([np.exp(np.clip(t[0], -1.5, 1.5)) for t in th]):.3f}')
print('=' * 70)
json.dump({'base_pooled': base_pooled, 'ranking': [(p, n, per) for p, n, per in rows]},
          open(f'{OUT}/tta_summary.json', 'w'), indent=1)
print('Saved per-view OOF files to', OUT)
