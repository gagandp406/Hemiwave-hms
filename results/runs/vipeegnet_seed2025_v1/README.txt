VIPEEGNet — HMS Harmful Brain Activity Classification
Archived: 2026-10-25 | Machine: jhb-dehzangi-ws04 (RTX A6000) | TF 2.19.0
Recipe: Sunyuri VIPEEGNet — 2-stage, EffV2B3, SumToOne Conv1D. Native: no TTA, no temperature.
Split: GroupKFold(5) on dedup df -> folds/
Native CV (model's own rows/metric): n/a
On VIPEEGNet rows with >=10 votes (n=6350): pooled 0.2238 | mean of folds 0.2245
On all VIPEEGNet rows: 0.5807
oof/fold{k}_oof.npz (>=10 votes) and fold{k}_oof_all.npz (all rows) share the VIPEEGNet schema/order -> load_results.load_oof
