#!/usr/bin/env bash
# Generic version of archive_vipeegnet.sh — freeze any model run into results/<TAG>/ (read-only, checksummed).
# Usage: ./archive_run.sh <TAG> <OOF_DIR> <WEIGHTS_DIR> <VENV> <RUN_PREFIX> <NOTEBOOK> [<SPLIT_CSV>]
# EfficientNetB0 comparison run:
#   ./archive_run.sh effnetb0_native_v1 ~/hms/fair_comparison/effnetb0_native \
#       ~/hms/effnet_b0/output/effnetb0_native ~/hms/.venv-tf215 effnet_native EffNetB0_local.ipynb ~/hms/fair_comparison/effnetb0_native/folds_native.csv
set -euo pipefail
P="${HMS_PROJECT_DIR:-$HOME/hms}"
TAG=$1; OOF=$2; WD=$3; VENV=$4; RUNP=$5; NB=$6
SPLIT=${7:-$P/results/vipeegnet_groupkfold_v1/folds/patient_folds_groupkfold.csv}
A="$P/results/$TAG"
[ -e "$A" ] && { echo "$A already exists - refusing to overwrite"; exit 1; }
source "$VENV/bin/activate"
mkdir -p "$A"/{weights,oof,folds,logs,code,env}
cp "$OOF"/*.npz "$OOF"/summary.json "$A/oof/"
cp "$SPLIT" "$A/folds/"
shopt -s nullglob; W=("$WD"/*.h5 "$WD"/*.pt); cp "${W[@]}" "$A/weights/"; shopt -u nullglob
cp "$WD"/*.png "$WD"/*_log.csv "$A/logs/" 2>/dev/null || true
cp "$(ls -t "$P"/runs/${RUNP}_*.ipynb | head -1)" "$A/logs/executed_run.ipynb"
cp "$(ls -t "$P"/runs/${RUNP}_*.log   | head -1)" "$A/logs/run.log"
cp "$P/$NB" "$A/code/"
HERE="$(cd "$(dirname "$0")" && pwd)"; cp "$HERE"/*.sh "$HERE"/*.py "$HERE"/requirements*.txt "$A/code/" 2>/dev/null || true
pip freeze > "$A/env/pip_freeze.txt"; nvidia-smi > "$A/env/nvidia_smi.txt"
sha256sum "$P/hms-data/train.csv" > "$A/env/train_csv.sha256"
python - "$A" "$TAG" <<'PY'
import json, sys, datetime
A, TAG = sys.argv[1:]; s = json.load(open(f'{A}/oof/summary.json'))
f = lambda k: f"{s[k]:.4f}" if isinstance(s.get(k), (int, float)) else 'n/a'
txt = f"""{s.get('model', TAG)} — HMS Harmful Brain Activity Classification
Archived: {datetime.date.today()} | Machine: jhb-dehzangi-ws04 (RTX A6000) | TF {s.get('tf_version', '?')}
Recipe: {s.get('recipe', '')}
Split: {s.get('fair_splits', '')} -> folds/
Native CV (model's own rows/metric): {f('native_cv_kaggle_kld')}
On VIPEEGNet rows with >=10 votes (n={s.get('n_oof_hq', 'n/a')}): pooled {f('oof_kld_hq_native')} | mean of folds {f('mean_fold_kld_hq')}
On all VIPEEGNet rows: {f('oof_kld_all')}
oof/fold{{k}}_oof.npz (>=10 votes) and fold{{k}}_oof_all.npz (all rows) share the VIPEEGNet schema/order -> load_results.load_oof
"""
open(f'{A}/README.txt', 'w').write(txt); print(txt)
PY
cd "$A" && find . -type f ! -name SHA256SUMS -exec sha256sum {} + | sort -k2 > SHA256SUMS
chmod -R a-w "$A"
tar czf "$P/results/${TAG}_small.tar.gz" -C "$P/results" --exclude="$TAG/weights" "$TAG"
du -sh "$A" "$P/results/${TAG}_small.tar.gz"; echo "Archived to $A (read-only)"
