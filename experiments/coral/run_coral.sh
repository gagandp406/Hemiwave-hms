#!/usr/bin/env bash
# CoRaL screening runner. Put this folder at ~/hms/coral and run inside tmux.
#   ./run_coral.sh logits     track A: committee heads on archived OOF logits (torch env, minutes)
#   ./run_coral.sh extract    dump VIPEEGNet features per fold (TF env, needs my_adapter.py)
#   ./run_coral.sh features   track B: heads + kNN vote memory on frozen features (torch env)
set -euo pipefail
export PYTHONUNBUFFERED=1
HMS=${HMS:-$HOME/hms}
HERE=$(cd "$(dirname "$0")" && pwd)
ARCH=${ARCH:-$HMS/results/vipeegnet_groupkfold_v1}
TRAIN_CSV=${TRAIN_CSV:?set TRAIN_CSV=/path/to/hms/train.csv}
OUT=${OUT:-$HMS/fair_comparison/coral_screen}
FEATS=${FEATS:-$HMS/fair_comparison/coral_feats}
mkdir -p "$OUT/logs"
ts=$(date +%Y%m%d_%H%M)

case "${1:-}" in
  logits)
    source "$HMS/.venv-torch/bin/activate"
    python "$HERE/screen_logits.py" --archive "$ARCH" --train-csv "$TRAIN_CSV" --out "$OUT" \
      ${ONLY:+--only "$ONLY"} 2>&1 | tee "$OUT/logs/logits_$ts.log" ;;
  extract)
    source "$HMS/.venv/bin/activate"
    python "$HERE/extract_vip_features.py" --adapter "$HERE/my_adapter.py" --out "$FEATS" \
      --archive "$ARCH" 2>&1 | tee "$OUT/logs/extract_$ts.log" ;;
  features)
    source "$HMS/.venv-torch/bin/activate"
    python "$HERE/screen_features.py" --feats "$FEATS/feats" --archive "$ARCH" --out "$OUT" \
      ${ONLY:+--only "$ONLY"} 2>&1 | tee "$OUT/logs/features_$ts.log" ;;
  *) echo "usage: $0 logits|extract|features"; exit 1 ;;
esac
