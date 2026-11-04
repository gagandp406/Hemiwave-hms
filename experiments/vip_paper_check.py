"""Check the archived VIPEEGNet run against the numbers printed in the paper
(Sun et al., npj Digital Medicine 2025, 8:768). Read-only: changes nothing.

  python vip_paper_check.py            (any venv with numpy, pandas, scikit-learn)

1. Fold sizes vs paper Fig. 2 (patients per fold, all rows and >=10-vote rows)
2. KLD three ways: pooled, mean of folds (paper 0.225), mean of per-patient means (paper ablation 0.216)
3. How much the "best validation epoch" checkpointing helps (best vs last epoch, from the Keras logs)
4. Per-class AUROC / recall / precision vs paper (Fig. 3-4, >=10-vote subset)
"""
import os, glob, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix

P = os.path.expanduser(os.environ.get('HMS', '~/hms'))
A = f'{P}/results/vipeegnet_groupkfold_v1'
CLS = ['Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA', 'Other']
VOTES = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
kld = lambda t, p: (np.clip(t, 1e-7, 1) * np.log(np.clip(t, 1e-7, 1) / np.clip(p, 1e-7, 1))).sum(1)

df = pd.read_csv(f'{P}/hms-data/train.csv')
rows = df.drop_duplicates(['eeg_id'] + VOTES).reset_index(drop=True)
rows['n'] = rows[VOTES].sum(axis=1)

print('1. FOLD SIZES (patients)')
print('   paper Fig. 2   all rows: [389, 390, 391, 390, 390] | >=10 votes: [204, 203, 227, 209, 220]')
for name, f in (('ours (GroupKFold)', f'{A}/folds/patient_folds_groupkfold.csv'),
                ('old patient_folds.csv', f'{P}/fair_comparison/patient_folds.csv')):
    if not os.path.exists(f): print(f'   {name}: file not found'); continue
    m = rows.merge(pd.read_csv(f)[['patient_id', 'fold']], on='patient_id')
    a = m.groupby('fold').patient_id.nunique().tolist()
    h = m[m.n >= 10].groupby('fold').patient_id.nunique().tolist()
    r = m[m.n >= 10].groupby('fold').size().tolist()
    print(f'   {name:<22} all rows: {a} | >=10 votes: {h} | >=10-vote rows per fold: {r}')

print('\n2. KLD ON >=10-VOTE ROWS')
yt, yp, ids, fo = [], [], [], []
for f in sorted(glob.glob(f'{A}/oof/fold?_oof.npz')):
    z = np.load(f); yt.append(z['yt']); yp.append(z['yp']); ids.append(z['eeg_ids']); fo += [int(f[-9])] * len(z['yt'])
yt, yp, ids, fo = np.concatenate(yt), np.concatenate(yp), np.concatenate(ids), np.array(fo)
k = kld(yt, yp)
pid = df.groupby('eeg_id').patient_id.first().reindex(ids).values
print(f'   pooled over rows            {k.mean():.4f}')
print(f'   mean of 5 folds             {pd.Series(k).groupby(fo).mean().mean():.4f}   (paper: 0.225, 95% CI 0.211-0.238)')
print(f'   mean of per-patient means   {pd.Series(k).groupby(pid).mean().mean():.4f}   (paper ablation, full model: 0.216, n=1,063 patients)')
print(f'   patients: {len(np.unique(pid)):,}')

print('\n3. CHECKPOINT SELECTION (val_loss in the Keras logs; stage 2 val_loss = the fold KLD)')
gain = []
for s in (1, 2):
    for f in sorted(glob.glob(f'{A}/logs/fold?_stage{s}_log.csv')):
        L = pd.read_csv(f); b = L.val_loss.idxmin()
        print(f'   {os.path.basename(f):<22} best epoch {int(L.epoch[b]) + 1}/{len(L)}  best {L.val_loss[b]:.4f}  last {L.val_loss.iloc[-1]:.4f}')
        if s == 2: gain.append(L.val_loss.iloc[-1] - L.val_loss[b])
if gain: print(f'   stage 2: picking the best epoch instead of the last lowers the reported KLD by {np.mean(gain):.4f} on average')

print('\n4. PER CLASS (>=10-vote rows, majority vote as the label)')
ti, pi = yt.argmax(1), yp.argmax(1); cm = confusion_matrix(ti, pi, labels=range(6))
paper = {'AUROC': [0.972, 0.962, 0.972, 0.938, 0.949, 0.930], 'recall': [57.7, 80.3, 73.8, 36.8, 40.7, 88.2],
         'precision': [68.2, 77.5, 80.4, 55.6, 63.0, 79.6]}
print(f'   {"class":>8} | AUROC ours/paper | recall % ours/paper | precision % ours/paper')
for c in range(6):
    auc = roc_auc_score(ti == c, yp[:, c]); rc = cm[c, c] / cm[c].sum() * 100; pr = cm[c, c] / cm[:, c].sum() * 100
    print(f'   {CLS[c]:>8} |  {auc:.3f} / {paper["AUROC"][c]:.3f}  |   {rc:5.1f} / {paper["recall"][c]:5.1f}     |   {pr:5.1f} / {paper["precision"][c]:5.1f}')
