#!/usr/bin/env bash
# ./run_prism.sh smoke        ~15 min   fold 0, 250 patients, 2+1 epochs
# ./run_prism.sh full         PRISM v1 (all new parts), 5 folds
# ./run_prism.sh screen       fold 0 only, ~25 min each: v1 recipe at 6 epochs, then v2 (global norm + masking), then v2 + B2
# ./run_prism.sh screen3      fold 0 only: v3 (maps + waveform morphology), then waveform stream alone
# ./run_prism.sh screen4      fold 0, waveform models, 12+5 epochs (~15 min each): longer / +learned ACF / wider / flat head
# ./run_prism.sh full5      best screen4 recipe (waveform + learned ACF, flat head), all 5 folds (~80 min)
# ./run_prism.sh ablations    three runs, each removing one new part: flat head / concat fusion / spectrum-only
set -euo pipefail
P="${HMS_PROJECT_DIR:-$HOME/hms}"; HERE="$(cd "$(dirname "$0")" && pwd)"
source "$P/.venv-torch/bin/activate"; cp -f "$HERE"/prism.py "$HERE"/prism_train.py "$HERE"/PRISM_EEG.ipynb "$P"/
cd "$P"; mkdir -p runs
run() { local T=$1; shift; local S=$(date +%Y%m%d_%H%M)
  papermill PRISM_EEG.ipynb runs/prism_${T}_$S.ipynb -k hms-torch -p PROJECT_DIR "$P" -p TAG $T "$@" --log-output 2>&1 | tee runs/prism_${T}_$S.log; }
case "${1:-}" in
  smoke)     run v1 -p SMOKE_TEST True ;;
  full)      run v4 ;;
  full5)     run wf5 -p REPR wave+acf -p HEAD flat ;;
  screen4)   run w_e12  -y "FOLDS: [0]" -p REPR wave
             run w_acf  -y "FOLDS: [0]" -p REPR wave+acf
             run w_wide -y "FOLDS: [0]" -p REPR wave -p WAVE_WIDTH 1.5
             run w_flat -y "FOLDS: [0]" -p REPR wave -p HEAD flat ;;
  screen3)   run s_v3 -y "FOLDS: [0]"
             run s_v3wave -y "FOLDS: [0]" -p REPR wave ;;
  screen)    run s_v1ep6 -y "FOLDS: [0]" -p NORM channel -p SPECAUG False
             run s_v2 -y "FOLDS: [0]"
             run s_v2b2 -y "FOLDS: [0]" -p BACKBONE efficientnet_b2 ;;
  ablations) run abl_flat -p HEAD flat; run abl_concat -p FUSION concat -p REPR rps; run abl_spec -p REPR spec ;;
  dm)        run v1dm -p LOSS kl+dm ;;
  *) echo "usage: $0 smoke|screen|screen3|screen4|full|full5|ablations|dm"; exit 1 ;;
esac
