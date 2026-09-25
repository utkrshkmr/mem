#!/usr/bin/env bash
# Python environment (PLAN.md §5.2). Picks install path A or B from the driver version.
set -euo pipefail
cd "$(dirname "$0")/.."

VLLM_VERSION=0.30.0
CU12_TAG=cu129   # the CUDA 12 build published with the v0.30.0 GitHub release

[ -d .venv ] || uv venv .venv --python 3.11
source .venv/bin/activate

driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
driver_major=${driver%%.*}
if [ "$driver_major" -ge 580 ]; then
  echo "driver $driver: path A (default vLLM wheel, CUDA 13)"
  uv pip install "vllm==${VLLM_VERSION}"
else
  echo "driver $driver: path B (vLLM +${CU12_TAG} wheel)"
  # unsafe-best-match: let uv take non-torch packages from PyPI when the torch index has older copies
  uv pip install \
    "https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+${CU12_TAG}-cp38-abi3-manylinux_2_28_x86_64.whl" \
    --extra-index-url "https://download.pytorch.org/whl/${CU12_TAG}" \
    --index-strategy unsafe-best-match
fi

uv pip install "peft==0.21.0" accelerate "nvidia-ml-py>=13" pyyaml pandas pyarrow scipy statsmodels \
  matplotlib orjson pytest pytest-asyncio pytest-timeout
# transformers and huggingface_hub come from vLLM's pins
uv pip install -e . --no-deps

mkdir -p env
uv pip freeze > env/requirements.lock
python - <<'EOF'
import torch, vllm, transformers, peft
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("vllm", vllm.__version__, "transformers", transformers.__version__, "peft", peft.__version__)
EOF
