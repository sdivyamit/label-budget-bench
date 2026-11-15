#!/usr/bin/env bash
# Open teacher (Qwen3.5-9B, vLLM) over all three tasks; resumable.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/.venv-teacher/bin:$PATH" CUDA_HOME=/usr/local/cuda-13.2 VLLM_USE_FLASHINFER_SAMPLER=0
.venv-teacher/bin/python train/teacher.py --task t2_response_safety --limit 50 || exit 1
for t in t2_response_safety t3_implicit_toxicity t1_rag_hallucination; do
  .venv-teacher/bin/python train/teacher.py --task $t || echo "FAILED $t"
done
echo TEACHER_ALL_DONE
