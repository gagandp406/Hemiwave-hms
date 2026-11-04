#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="${HMS_PROJECT_DIR:-$HOME/hms}"
VENV="$PROJECT_DIR/.venv"
mkdir -p "$PROJECT_DIR"
echo "== GPU driver =="; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
if [ ! -x "$VENV/bin/pip" ]; then
  rm -rf "$VENV"
  if python3 -c "import ensurepip" 2>/dev/null; then
    python3 -m venv "$VENV"
  else
    echo "ensurepip not available -> creating venv without pip and bootstrapping pip"
    python3 -m venv --without-pip "$VENV"
    curl -sSL https://bootstrap.pypa.io/get-pip.py -o "$PROJECT_DIR/get-pip.py"
    "$VENV/bin/python" "$PROJECT_DIR/get-pip.py"
    rm -f "$PROJECT_DIR/get-pip.py"
  fi
fi
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel
python -m pip install -r "$HERE/requirements.txt"
python -m ipykernel install --user --name hms-vipeegnet --display-name "Python (hms-vipeegnet)"
cp -n "$HERE"/VIPEEGNet_local.ipynb "$HERE"/VIPEEGNet_kaggle_inference.ipynb "$PROJECT_DIR"/ || true
python - <<'PY'
import tensorflow as tf
g = tf.config.list_physical_devices('GPU')
print('TF', tf.__version__, '| GPUs:', g)
assert g, 'TensorFlow does not see the GPU'
with tf.device('/GPU:0'):
    print('matmul ok:', float(tf.reduce_sum(tf.matmul(tf.ones((256,256)), tf.ones((256,256))))))
PY
echo "Done. Activate with:  source $VENV/bin/activate"
