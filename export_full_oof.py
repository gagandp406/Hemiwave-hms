"""Predict ALL validation rows (not just >=10 votes) with the trained stage-2 models.
Run BEFORE archive_vipeegnet.sh.  Usage: python export_full_oof.py"""
import json, os, numpy as np
P = os.path.expanduser(os.environ.get('HMS_PROJECT_DIR', '~/hms'))
nb = json.load(open(f'{P}/VIPEEGNet_local.ipynb'))
SKIP = ('t_all = time.time()', 'FAIR_VIPEEG =', 'import seaborn', 'import shutil')
g = {'__name__': '__main__'}
for c in nb['cells']:
    if c['cell_type'] != 'code': continue
    src = ''.join(c['source'])
    if any(m in src for m in SKIP): continue
    if 'parameters' in c.get('metadata', {}).get('tags', []):
        src += f"\nPROJECT_DIR = {P!r}\nFOLD_MODE = 'groupkfold'\nSMOKE_TEST = False\n"
    src = src.replace('model = build_model()\nprint(f\'Total params', 'pass\nif False: print(f\'Total params') \
              .replace("del model; tf.keras.backend.clear_session()", "tf.keras.backend.clear_session()")
    exec(compile(src, 'nb', 'exec'), g)
tf, train, folds, TARGET_COLS = g['tf'], g['train'], g['folds'], g['TARGET_COLS']
out = f'{P}/fair_comparison/vipeegnet_groupkfold'
kl = lambda a, b: np.mean(np.sum(np.clip(a,1e-7,1)*np.log(np.clip(a,1e-7,1)/np.clip(b,1e-7,1)), 1))
for k in range(5):
    _, v = folds[k]
    df = train.iloc[v].reset_index(drop=True)
    tf.keras.backend.clear_session()
    m = g['build_model'](backbone_weights=None)
    m.load_weights(f'{P}/vipeegnet/output/fold{k}_stage2.weights.h5')
    gen = g['DataGenerator'](df, batch_size=64, mode='valid', eegs=g['eegs'], stage=2)
    yp = m.predict(gen, verbose=0).astype(np.float32)
    yt = df[TARGET_COLS].values.astype(np.float32)
    np.savez(f'{out}/fold{k}_oof_all.npz', yt=yt, yp=yp, wt=df['total_votes_raw'].values.astype(np.float32),
             yp_raw_logits=np.log(np.clip(yp, 1e-8, 1)), eeg_ids=df['eeg_id'].values,
             patient_ids=df['patient_id'].values, temperature=np.array([1.0], np.float32))
    hq = df['total_votes_raw'].values >= 10
    print(f'fold {k}: n_all={len(df)} KLD_all={kl(yt, yp):.4f} | HQ n={hq.sum()} KLD_HQ={kl(yt[hq], yp[hq]):.4f}')
print('Saved fold*_oof_all.npz to', out)
