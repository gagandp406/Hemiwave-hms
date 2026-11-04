EfficientNetB0 (Chris Deotte starter LB 0.43) — HMS Harmful Brain Activity Classification
Archived: 2026-10-18 | Machine: jhb-dehzangi-ws04 (RTX A6000) | TF 2.15.1
Recipe: eeg_id table, 8 spectrograms -> 512x512x3, efficientnet pkg B0 AutoAugment ImageNet, Adam KLD, 4 epochs LR 1e-3,1e-3,1e-4,1e-5, BS 32, auto mixed precision, augment=False
Split: GroupKFold(5) on eeg_id table (Chris native) -> folds/
Native CV (model's own rows/metric): 0.5817
On VIPEEGNet rows with >=10 votes (n=6350): pooled 0.4648 | mean of folds 0.4630
On all VIPEEGNet rows: 0.6125
oof/fold{k}_oof.npz (>=10 votes) and fold{k}_oof_all.npz (all rows) share the VIPEEGNet schema/order -> load_results.load_oof
