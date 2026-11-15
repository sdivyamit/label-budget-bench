"""Train one (family, model, task, N, seed, label source) cell and write preds + metrics.

Idempotent: a finished run (preds/runs/{run_id}/DONE) is skipped; an interrupted run resumes
from its last checkpoint.

  python train/run.py --family encoder --model modernbert_base --task t2_response_safety --n 1000 --seed 0
  python train/run.py --family decoder --model qwen35_2b --task t2_response_safety --n 1000 --seed 0 --labels open_teacher
  python train/run.py --family classical --model tfidf_lr --task t3_implicit_toxicity --n 500 --seed 1
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RunSpec, write_outputs  # noqa: E402


def execute(spec: RunSpec) -> int:
    if spec.done():
        print(f"[skip] {spec.run_id} already done")
        return 0
    print(f"[run] {spec.run_id}", flush=True)
    if spec.family == "encoder":
        import encoder as fam
    elif spec.family == "decoder":
        import decoder_lora as fam
    elif spec.family == "classical":
        import classical_lr as fam
    else:
        raise ValueError(spec.family)
    preds, info = fam.run(spec)
    write_outputs(spec, preds, info)
    if spec.ckpt.exists():
        shutil.rmtree(spec.ckpt)
    m = json.loads((spec.out / "metrics.json").read_text())
    print("  " + "  ".join(f"{s}: PR-AUC {m[s]['raw']['pr_auc']:.3f} maxF1 {m[s]['raw']['max_f1']:.3f} "
                           f"ECE {m[s]['raw']['ece']:.3f}" for s in ("val", "test", "shift")), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=["encoder", "decoder", "classical"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--labels", default="gold")
    ap.add_argument("--tag", default="")
    ap.add_argument("--hp", default="{}", help="JSON overrides, e.g. '{\"lr\": 2e-5}'")
    ap.add_argument("--max-len", type=int)
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--eval-limit", type=int)
    a = ap.parse_args()
    spec = RunSpec(a.family, a.model, a.task, a.n, a.seed, a.labels, json.loads(a.hp), a.tag,
                   a.max_len, a.max_steps, a.eval_limit)
    return execute(spec)


if __name__ == "__main__":
    sys.exit(main())
