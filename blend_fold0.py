"""Fold-0 check: how much does PRISM add on top of VIPEEGNet?
Usage:  python blend_fold0.py [run ...]     (default: the four screen4 runs, no-TTA and TTA)
Rows are matched on eeg_id + exact vote counts, so the two models are scored on identical rows.
"""
import os, sys, numpy as np, pandas as pd
P = os.path.expanduser(os.environ.get('HMS', '~/hms'))
VIP = f'{P}/results/vipeegnet_groupkfold_v1/oof/fold0_oof.npz'
runs = sys.argv[1:] or [f'prism_w_{t}{s}' for t in ('e12', 'acf', 'wide', 'flat') for s in ('', '_tta')]

def kld(t, p, eps=1e-7):
    t, p = np.clip(t, eps, 1), np.clip(p, eps, 1)
    return (t * np.log(t / p)).sum(1)

def table(f):
    z = np.load(f)
    v = np.rint(z['yt'] * z['wt'][:, None]).astype(int)
    key = [f'{e}_' + '_'.join(map(str, r)) for e, r in zip(z['eeg_ids'], v)]
    return pd.DataFrame({'key': key}), z['yt'], z['yp']

kv, yt_v, yp_v = table(VIP)
rng = np.random.default_rng(0)
print(f'VIPEEGNet fold 0: {len(kv)} rows, KLD {kld(yt_v, yp_v).mean():.4f}\n')
print(f'{"run":<20}{"n":>6}{"PRISM":>8}{"VIP":>8}{"best arith":>16}{"best geo":>16}{"gain (95% CI)":>24}')
for r in runs:
    f = f'{P}/fair_comparison/{r}/fold0_oof.npz'
    if not os.path.exists(f): print(f'{r:<20} missing'); continue
    kp, yt_p, yp_p = table(f)
    m = kv.reset_index().merge(kp.reset_index(), on='key', suffixes=('_v', '_p'))
    iv, ip = m.index_v.values, m.index_p.values
    t, a, b = yt_v[iv], yp_v[iv], yp_p[ip]
    assert np.allclose(t, yt_p[ip], atol=1e-5)
    base = kld(t, a)
    best = {}
    for kind in ('arith', 'geo'):
        res = []
        for w in np.round(np.arange(0, 1.01, 0.05), 2):
            if kind == 'arith': q = (1 - w) * a + w * b
            else:
                q = np.exp((1 - w) * np.log(np.clip(a, 1e-7, 1)) + w * np.log(np.clip(b, 1e-7, 1))); q /= q.sum(1, keepdims=True)
            res.append((kld(t, q).mean(), w, q))
        best[kind] = min(res, key=lambda x: x[0])
    q = best['geo'][2]; d = base - kld(t, q)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    print(f'{r:<20}{len(m):>6}{kld(t, b).mean():>8.4f}{base.mean():>8.4f}'
          f'{best["arith"][0]:>10.4f} w={best["arith"][1]:.2f}{best["geo"][0]:>10.4f} w={best["geo"][1]:.2f}'
          f'{d.mean():>10.4f} ({np.percentile(bs, 2.5):.4f}, {np.percentile(bs, 97.5):.4f})')
print('\nw = weight on PRISM. The weight is tuned on fold 0 itself, so treat the gain as optimistic;'
      '\nthe 5-fold run will tune it leave-one-fold-out.')
