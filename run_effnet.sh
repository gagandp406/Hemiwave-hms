#!/usr/bin/env bash
# ./run_effnet.sh smoke    (~15 min, fold 0, 250 patients, native folds)
# ./run_effnet.sh native   Chris' own setup (his GroupKFold): native CV + predictions on the VIPEEGNet rows  <- main run
# ./run_effnet.sh shared   same recipe on VIPEEGNet's patient split (optional: measures the split effect)
set -euo pipefail
P="${HMS_PROJECT_DIR:-$HOME/hms}"
source "$P/.venv-tf215/bin/activate"
cd "$P"; mkdir -p runs; STAMP=$(date +%Y%m%d_%H%M)
NB=EffNetB0_local.ipynb; K="-k hms-effnet -p PROJECT_DIR $P"
case "${1:-}" in
  smoke)  papermill $NB runs/effnet_smoke_$STAMP.ipynb  $K -p SMOKE_TEST True -p FOLD_MODE native --log-output 2>&1 | tee runs/effnet_smoke_$STAMP.log ;;
  native) papermill $NB runs/effnet_native_$STAMP.ipynb $K -p FOLD_MODE native --log-output 2>&1 | tee runs/effnet_native_$STAMP.log ;;
  shared) papermill $NB runs/effnet_shared_$STAMP.ipynb $K -p FOLD_MODE csv    --log-output 2>&1 | tee runs/effnet_shared_$STAMP.log ;;
  *) echo "usage: $0 smoke|native|shared"; exit 1 ;;
esac
