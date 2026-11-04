#!/usr/bin/env bash
# Copy the notebooks and Python modules into the project directory ($HMS_PROJECT_DIR, default ~/hms),
# where the run scripts and papermill expect them. Run again after editing any notebook or module.
set -euo pipefail
R="$(cd "$(dirname "$0")" && pwd)"; P="${HMS_PROJECT_DIR:-$HOME/hms}"
mkdir -p "$P/results"
cp -f "$R"/reference/*.ipynb "$R"/hemiwave/*.ipynb "$R"/hemiwave/*.py "$R"/baselines/effnetb0/*.ipynb \
      "$R"/evaluation/*.ipynb "$R"/tools/export_full_oof.py "$P"/
cp -f "$R"/evaluation/load_results.py "$P/results/"
echo "Copied notebooks and modules from $R to $P"
