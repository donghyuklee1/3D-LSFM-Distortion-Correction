#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$PROJECT_DIR"
if [ ! -d ".venv" ]; then
  echo "[setup] creating .venv ..."
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --quiet --upgrade pip wheel
echo "[setup] installing pytorch ..."
python -m pip install --quiet torch torchvision
echo "[setup] installing project deps ..."
python -m pip install --quiet -r requirements.txt
python - <<'PY'
import torch
print(f"torch    : {torch.__version__}")
print(f"cuda?    : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device   : {torch.cuda.get_device_name(0)}")
    print(f"capabil. : {torch.cuda.get_device_capability(0)}")
PY
echo "[setup] done."
echo "Activate with:  source $PROJECT_DIR/.venv/bin/activate"
