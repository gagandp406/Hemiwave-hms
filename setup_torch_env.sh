#!/usr/bin/env bash
# PyTorch environment for PRISM-EEG (no sudo). CUDA 12.4 wheels work with driver 580.
set -euo pipefail
P="${HMS_PROJECT_DIR:-$HOME/hms}"; V="$P/.venv-torch"; HERE="$(cd "$(dirname "$0")" && pwd)"
if [ ! -x "$V/bin/pip" ]; then
  rm -rf "$V"; python3 -m venv --without-pip "$V"
  curl -sSL https://bootstrap.pypa.io/get-pip.py -o "$P/get-pip.py"; "$V/bin/python" "$P/get-pip.py"; rm -f "$P/get-pip.py"
fi
source "$V/bin/activate"
python -m pip install --upgrade pip wheel
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install "timm==1.0.11" "huggingface_hub==0.26.2" safetensors "numpy<2" pandas pyarrow scipy scikit-learn tqdm matplotlib seaborn ipykernel papermill
python -m ipykernel install --user --name hms-torch --display-name "Python (hms-torch)"
cp -f "$HERE"/prism.py "$HERE"/prism_train.py "$HERE"/PRISM_EEG.ipynb "$P"/
python - <<'PY'
import torch, timm
print('torch', torch.__version__, '| CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
m = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)   # downloads ImageNet weights once
print('timm', timm.__version__, '| efficientnet_b0 pretrained OK,', sum(p.numel() for p in m.parameters()) // 10**6, 'M params')
x = torch.randn(4, 18, 10000, device='cuda')
import sys, os; sys.path.insert(0, os.path.expanduser('~/hms')); from prism import PRISM
net = PRISM(pretrained=False).cuda().eval()
with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
    print('PRISM forward on GPU OK:', tuple(net(x).shape))
PY
echo "Done. Activate with: source $V/bin/activate"
