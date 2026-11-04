#!/usr/bin/env bash
# Freeze the VIPEEGNet baseline run into one read-only, checksummed folder.
set -euo pipefail
P="${HMS_PROJECT_DIR:-$HOME/hms}"
TAG="vipeegnet_groupkfold_v1"
A="$P/results/$TAG"
[ -e "$A" ] && { echo "$A already exists - refusing to overwrite"; exit 1; }
source "$P/.venv/bin/activate"
mkdir -p "$A"/{weights,oof,folds,logs,code,env}
cp "$P"/fair_comparison/vipeegnet_groupkfold/fold*_oof*.npz "$P"/fair_comparison/vipeegnet_groupkfold/summary.json "$A/oof/"
cp "$P"/fair_comparison/patient_folds_groupkfold.csv "$A/folds/"
cp "$P"/vipeegnet/output/fold*_stage*.weights.h5 "$A/weights/"
cp "$P"/vipeegnet/output/*_log.csv "$P"/vipeegnet/output/confusion.png "$A/logs/"
cp "$(ls -t "$P"/runs/full_*.ipynb | head -1)" "$A/logs/executed_full_run.ipynb"
cp "$(ls -t "$P"/runs/full_*.log   | head -1)" "$A/logs/full_run.log"
cp "$P"/VIPEEGNet_local.ipynb "$A/code/"; cp "$P"/VIPEEGNet_kaggle_inference.ipynb "$A/code/" 2>/dev/null || true
R="$(cd "$(dirname "$0")/.." && pwd)"; (cd "$R" && find setup hemiwave reference baselines ensemble evaluation tools -type f \( -name "*.sh" -o -name "*.py" -o -name "*.ipynb" -o -name "requirements*.txt" \) -exec cp --parents {} "$A/code/" \;) 2>/dev/null || true
pip freeze > "$A/env/pip_freeze.txt"
nvidia-smi > "$A/env/nvidia_smi.txt"
python -c "import tensorflow as tf, sys; print('python', sys.version); print('tf', tf.__version__)" > "$A/env/versions.txt" 2>/dev/null
sha256sum "$P/hms-data/train.csv" > "$A/env/train_csv.sha256"
python - "$A" <<'PY'
import json, sys, datetime
A = sys.argv[1]; s = json.load(open(f'{A}/oof/summary.json'))
txt = f"""VIPEEGNet baseline — HMS Harmful Brain Activity Classification
Archived: {datetime.date.today()}   |   Machine: jhb-dehzangi-ws04 (RTX A6000)
Recipe: Sunyuri VIPEEGNet (arXiv 2507.08874), unchanged. Seed 2024, mixed_float16.
Split:  {s['fair_splits']}  -> folds/patient_folds_groupkfold.csv
Eval subset: validation rows with >=10 votes (HQ), n={s['n_oof_hq']}

Pooled OOF KLD (HQ): {s['oof_kld_hq_native']:.4f}
Mean of fold KLDs:   {s['mean_fold_kld_hq']:.4f}   (paper 0.225, 95% CI 0.211-0.238)
Per-fold: {', '.join(f"{f['kld_hq_native']:.4f}" for f in s['per_fold'])}

oof/fold{{k}}_oof.npz keys: yt (true probs), yp (pred probs), wt (vote counts),
  yp_raw_logits (=log yp), eeg_ids, spectrogram_ids, temperature
oof/fold{{k}}_oof_all.npz (if present): same, for ALL validation rows (any vote count)
weights/fold{{k}}_stage2.weights.h5 = final models; load with build_model() in code/VIPEEGNet_local.ipynb
Kaggle leaderboard: public ____  private ____   (paper 0.2233 / 0.2725)
"""
open(f'{A}/README.txt', 'w').write(txt); print(txt)
PY
cd "$A" && find . -type f ! -name SHA256SUMS -exec sha256sum {} + | sort -k2 > SHA256SUMS
chmod -R a-w "$A"
tar czf "$P/results/${TAG}_small.tar.gz" -C "$P/results" --exclude="$TAG/weights" "$TAG"
echo; du -sh "$A" "$P/results/${TAG}_small.tar.gz"
echo "Archived to $A (read-only). Verify any time with: cd $A && sha256sum -c SHA256SUMS --quiet"
