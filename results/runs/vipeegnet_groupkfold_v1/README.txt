VIPEEGNet baseline — HMS Harmful Brain Activity Classification
Archived: 2026-10-16   |   Machine: jhb-dehzangi-ws04 (RTX A6000)
Recipe: Sunyuri VIPEEGNet (arXiv 2507.08874), unchanged. Seed 2024, mixed_float16.
Split:  GroupKFold(5) on dedup df  -> folds/patient_folds_groupkfold.csv
Eval subset: validation rows with >=10 votes (HQ), n=6350

Pooled OOF KLD (HQ): 0.2217
Mean of fold KLDs:   0.2221   (paper 0.225, 95% CI 0.211-0.238)
Per-fold: 0.2163, 0.2240, 0.2241, 0.2319, 0.2142

oof/fold{k}_oof.npz keys: yt (true probs), yp (pred probs), wt (vote counts),
  yp_raw_logits (=log yp), eeg_ids, spectrogram_ids, temperature
oof/fold{k}_oof_all.npz (if present): same, for ALL validation rows (any vote count)
weights/fold{k}_stage2.weights.h5 = final models; load with build_model() in code/VIPEEGNet_local.ipynb
Kaggle leaderboard: public ____  private ____   (paper 0.2233 / 0.2725)
