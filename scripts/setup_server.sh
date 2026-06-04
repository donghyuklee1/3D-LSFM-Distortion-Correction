#!/usr/bin/env bash
# One-shot environment bootstrap on a fresh GPU box (Linux).
# Run from the project root:  bash scripts/setup_server.sh
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$PROJECT_DIR"

# 1) python venv
if [ ! -d ".venv" ]; then
  echo "[setup] creating .venv ..."
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --quiet --upgrade pip wheel

# 2) install pytorch (CUDA build picked automatically when nvcc is present)
echo "[setup] installing pytorch ..."
python -m pip install --quiet torch torchvision

# 3) project deps
echo "[setup] installing project deps ..."
python -m pip install --quiet -r requirements.txt

# 4) sanity: detect CUDA
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
