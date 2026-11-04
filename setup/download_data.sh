#!/usr/bin/env bash
# Needs ~/.kaggle/kaggle.json (Kaggle > Settings > API > Create New Token) and accepted competition rules.
set -euo pipefail
PROJECT_DIR="${HMS_PROJECT_DIR:-$HOME/hms}"
source "$PROJECT_DIR/.venv/bin/activate"
chmod 600 ~/.kaggle/kaggle.json
cd "$PROJECT_DIR"
kaggle competitions download -c hms-harmful-brain-activity-classification -p "$PROJECT_DIR"
mkdir -p hms-data
# VIPEEGNet only needs train.csv, train_eegs/, test.csv, test_eegs/. Keep spectrograms if TCS-Net uses them:
#   set KEEP_SPEC=1 to extract everything.
if [ "${KEEP_SPEC:-0}" = "1" ]; then
  unzip -q -n hms-harmful-brain-activity-classification.zip -d hms-data
else
  unzip -q -n hms-harmful-brain-activity-classification.zip -d hms-data -x 'train_spectrograms/*' 'test_spectrograms/*'
fi
echo "train_eegs files: $(ls hms-data/train_eegs | wc -l)  (train.csv references 17,089 of them)"
wc -l hms-data/train.csv
