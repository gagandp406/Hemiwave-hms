#!/usr/bin/env bash
# Usage:  ./run_vipeegnet.sh smoke    (10-15 min pipeline check)
#         ./run_vipeegnet.sh full     (all 5 folds, several hours)
#         ./run_vipeegnet.sh seedsmoke   (10-15 min check of the seed path, fold 0, small data)
#         ./run_vipeegnet.sh seed 2025   (same recipe, another seed; writes *_seed2025 folders, never touches the original)
# Run inside tmux so it survives SSH / VS Code disconnects:  tmux new -s vip
set -euo pipefail
PROJECT_DIR="${HMS_PROJECT_DIR:-$HOME/hms}"
source "$PROJECT_DIR/.venv/bin/activate"
cd "$PROJECT_DIR"; mkdir -p runs
STAMP=$(date +%Y%m%d_%H%M)
case "${1:-}" in
  smoke) papermill VIPEEGNet_local.ipynb "runs/smoke_$STAMP.ipynb" -k hms-vipeegnet \
           -p PROJECT_DIR "$PROJECT_DIR" -p SMOKE_TEST True --log-output 2>&1 | tee "runs/smoke_$STAMP.log" ;;
  full)  papermill VIPEEGNet_local.ipynb "runs/full_$STAMP.ipynb" -k hms-vipeegnet \
           -p PROJECT_DIR "$PROJECT_DIR" ${FOLD_MODE:+-p FOLD_MODE $FOLD_MODE} --log-output 2>&1 | tee "runs/full_$STAMP.log" ;;
  seed)  S=${2:-2025}; [ "$S" = 2024 ] && { echo "2024 is the original seed - pick another"; exit 1; }
         papermill VIPEEGNet_local.ipynb "runs/full_seed${S}_$STAMP.ipynb" -k hms-vipeegnet \
           -p PROJECT_DIR "$PROJECT_DIR" -p SEED "$S" --log-output 2>&1 | tee "runs/full_seed${S}_$STAMP.log" ;;
  seedsmoke) papermill VIPEEGNet_local.ipynb "runs/smoke_seed2025_$STAMP.ipynb" -k hms-vipeegnet \
           -p PROJECT_DIR "$PROJECT_DIR" -p SEED 2025 -p SMOKE_TEST True --log-output 2>&1 | tee "runs/smoke_seed2025_$STAMP.log" ;;
  *) echo "usage: $0 smoke|full|seed [SEED]|seedsmoke"; exit 1 ;;
esac
