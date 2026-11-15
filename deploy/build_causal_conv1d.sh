#!/usr/bin/env bash
# Build the causal-conv1d wheel on a Thunder box (Base template has a CUDA driver but no nvcc).
# ~20 min on 6 vCPUs. Copies the wheel to deploy/wheels/ for reuse; pull it back to the Mac.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
TV=$($PY -c "import torch; print(torch.version.cuda.replace('.', '-'))")
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/k.deb
sudo dpkg -i /tmp/k.deb && sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "cuda-nvcc-$TV" "cuda-cudart-dev-$TV" "cuda-crt-$TV" \
  "cuda-cccl-$TV" "libcublas-dev-$TV" "libcusparse-dev-$TV" "libcusolver-dev-$TV"
export CUDA_HOME="/usr/local/cuda-${TV/-/.}" PATH="/usr/local/cuda-${TV/-/.}/bin:$PATH"
uv pip install --python $PY ninja packaging wheel setuptools
TORCH_CUDA_ARCH_LIST=8.6 CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS="$(nproc)" \
  uv pip install --python $PY --no-build-isolation causal-conv1d
mkdir -p deploy/wheels
cp "$(find ~/.cache/uv -name 'causal_conv1d-*.whl' | head -1)" deploy/wheels/
$PY -c "import causal_conv1d; print('causal_conv1d', causal_conv1d.__version__)"
