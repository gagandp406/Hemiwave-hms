PRISM-EEG — HMS Harmful Brain Activity Classification
Archived: 2026-10-19 | Machine: jhb-dehzangi-ws04 (RTX A6000) | TF ?
Recipe: PRISM flat/sym/wave+acf/kl efficientnet_b0 12+5 ep
Split: VIPEEGNet patient split -> folds/
Native CV (model's own rows/metric): n/a
On VIPEEGNet rows with >=10 votes (n=6350): pooled 0.2744 | mean of folds 0.2749
On all VIPEEGNet rows: 0.6476
oof/fold{k}_oof.npz (>=10 votes) and fold{k}_oof_all.npz (all rows) share the VIPEEGNet schema/order -> load_results.load_oof
