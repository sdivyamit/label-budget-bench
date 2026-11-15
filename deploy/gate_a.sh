#!/usr/bin/env bash
# Gate A on a fresh Thunder Compute A6000. From your Mac:
#   rsync -az --exclude .venv --exclude checkpoints label-budget-bench/ <host>:label-budget-bench/
#   ssh <host> 'cd label-budget-bench && bash deploy/gate_a.sh'
# Then pull results back with: bash queue/end_of_day.sh <host> --stop
set -euo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
[ -d .venv ] || uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt -r requirements-gpu.txt
# Qwen3.5 conv kernel (~7% faster decoder steps). PyPI has no wheel for this torch/CUDA, so use the
# one built on the first Gate A box (deploy/wheels/, ~20 min to rebuild with deploy/build_causal_conv1d.sh).
WHL=$(ls deploy/wheels/causal_conv1d-*.whl 2>/dev/null | head -1 || true)
if [ -n "$WHL" ]; then uv pip install --python .venv/bin/python "$WHL"
else echo "WARN: no causal-conv1d wheel; Qwen3.5 uses the torch fallback (run deploy/build_causal_conv1d.sh)"; fi
PY=.venv/bin/python
$PY -c "import torch; assert torch.cuda.is_available(), 'CUDA not visible to torch'; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0))"

echo "== 1/5 virtualization probe";       $PY tests/gpu_probe.py | tee logs_gate_a_probe.txt
echo "== 2/5 unit tests";                 $PY -m pytest -q tests/
echo "== 3/5 crash-resume (GPU-sized)";   PY=$PY bash tests/crash_resume.sh "--n 1000 --max-len 1024 --max-steps 60 --eval-limit 300"
echo "== 4/5 Gate A cells";               $PY queue/make_jobs.py >/dev/null
$PY queue/worker.py --families encoder,decoder,classical --stage gate_a --once
echo "== 5/5 GPU-hour projection";        $PY analysis/gpu_projection.py
echo "Gate A finished. Remaining check (needs API credit): one API model on 200 T2 test items -> labels/api.py run --set probe"
