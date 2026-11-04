"""Reuse archived results in future evaluations:
    import sys; sys.path.insert(0, '/home/nfs/jhb-dehzangi/ws04/gs1147/hms/results')
    from load_results import load_oof, kld, summary
    yt, yp, meta = load_oof('vipeegnet_groupkfold_v1')         # >=10-vote rows, all folds
    yt, yp, meta = load_oof('vipeegnet_groupkfold_v1', 'all')  # every validation row
"""
import os, glob, json, numpy as np, pandas as pd
RESULTS = os.path.expanduser(os.environ.get('HMS_RESULTS', '~/hms/results'))

def load_oof(tag, subset='hq'):
    suffix = '_oof.npz' if subset == 'hq' else '_oof_all.npz'
    files = sorted(glob.glob(f'{RESULTS}/{tag}/oof/fold*{suffix}'))
    assert files, f'no {suffix} files for {tag}'
    yt, yp, rows = [], [], []
    for f in files:
        d = np.load(f); k = int(os.path.basename(f)[4])
        yt.append(d['yt']); yp.append(d['yp'])
        rows.append(pd.DataFrame({'fold': k, 'eeg_id': d['eeg_ids'], 'votes': d['wt']}))
    return np.concatenate(yt), np.concatenate(yp), pd.concat(rows, ignore_index=True)

def kld(yt, yp, eps=1e-7):
    yt, yp = np.clip(yt, eps, 1), np.clip(yp, eps, 1)
    return np.sum(yt * np.log(yt / yp), axis=1)

def summary(tag):
    return json.load(open(f'{RESULTS}/{tag}/oof/summary.json'))

if __name__ == '__main__':
    yt, yp, meta = load_oof('vipeegnet_groupkfold_v1')
    print(len(yt), 'HQ rows | pooled KLD', kld(yt, yp).mean().round(4),
          '| per fold', meta.assign(k=kld(yt, yp)).groupby('fold')['k'].mean().round(4).tolist())
