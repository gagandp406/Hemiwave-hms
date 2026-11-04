#!/usr/bin/env bash
# Second environment (TF 2.15 / Keras 2) for the EfficientNetB0 starter + its ImageNet weights. No sudo.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
P="${HMS_PROJECT_DIR:-$HOME/hms}"
VENV="$P/.venv-tf215"
if [ ! -x "$VENV/bin/pip" ]; then
  rm -rf "$VENV"
  python3 -m venv --without-pip "$VENV"
  curl -sSL https://bootstrap.pypa.io/get-pip.py -o "$P/get-pip.py"
  "$VENV/bin/python" "$P/get-pip.py"; rm -f "$P/get-pip.py"
fi
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel
python -m pip install -r "$HERE/requirements_effnet.txt"
python -m ipykernel install --user --name hms-effnet --display-name "Python (hms-effnet)"
cp -n "$HERE"/EffNetB0_local.ipynb "$HERE"/compare_models.ipynb "$P"/ 2>/dev/null || true

# ImageNet (AutoAugment) weights used by efficientnet 1.1.1 'imagenet' = what Chris loaded
mkdir -p "$P/pretrained"
W="$P/pretrained/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
if [ ! -s "$W" ]; then
  curl -fL -o "$W" https://github.com/Callidior/keras-applications/releases/download/efficientnet/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5 \
    || { rm -f "$W"; echo "!! weights download failed - see instructions (Kaggle dataset fallback)"; }
fi
python - "$W" <<'PY'
import sys, os, tensorflow as tf
g = tf.config.list_physical_devices('GPU'); print('TF', tf.__version__, '| GPUs:', g)
assert g, 'TensorFlow 2.15 does not see the GPU'
import efficientnet.tfkeras as efn
w = sys.argv[1]
if os.path.exists(w):
    m = efn.EfficientNetB0(include_top=False, weights=None, input_shape=None); m.load_weights(w)
    print('EfficientNetB0 ImageNet weights load OK:', os.path.getsize(w) // 1_000_000, 'MB')
else:
    print('Weights missing:', w)
with tf.device('/GPU:0'):
    print('matmul ok:', float(tf.reduce_sum(tf.matmul(tf.ones((256, 256)), tf.ones((256, 256))))))
PY
echo "Done. Activate with: source $VENV/bin/activate"
