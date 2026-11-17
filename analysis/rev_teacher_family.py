"""Open-teacher family check (revision 2h; review point 12).

The open teacher is Qwen3.5-9B, so a Qwen3.5-2B student might fit its labels more easily than a Gemma 4 E2B
student. For each task x eval set x N: teacher cost = gold minus teacher-label score (same seeds and items)
for each student, and the family difference = Qwen cost minus Gemma cost (negative: the Qwen student loses
less to its own family's teacher). The frontier teacher (GPT-6.1 Sol, neither family) is the control.
Seeds share D_N across models, so one seed draw is used for all four cells; items are shared too.

  python analysis/rev_teacher_family.py [--b 10000]
"""
from __future__ import annotations

import argparse
import zlib

import numpy as np
import pandas as pd

import rev_common as rc

TOP = {"t1_rag_hallucination": 8651, "t2_response_safety": 20000, "t3_implicit_toxicity": 5920}
STUDENTS = {"qwen": "qwen35_2b", "gemma": "gemma4_e2b"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int, default=10_000)
    a = ap.parse_args()
    rows = []
    for task in rc.TASKS:
        for es in ("test", "shift"):
            ch = rc.weight_chunks(task, es, a.b)
            for teacher in ("open_teacher", "frontier_teacher"):
                for n in sorted({1000, 5000, TOP[task]}):
                    cells = {(s, lab): rc.cell_seeds(task, "decoder", m, lab, n)
                             for s, m in STUDENTS.items() for lab in ("gold", teacher)}
                    seeds = sorted(set.intersection(*[set(c) for c in cells.values()]))
                    if not seeds:
                        continue
                    for metric in ("pr_auc", "f1_valthr"):
                        reps = {k: rc.item_reps(rc.per_seed([c[s] for s in seeds], es), task, es, a.b, metric, ch)
                                for k, c in cells.items()}
                        idx = rc.seed_draws(len(seeds), reps[("qwen", "gold")].shape[1] - 1,
                                            zlib.crc32(f"{task}/{es}/{teacher}/{n}/{metric}".encode()))
                        col = np.arange(idx.shape[0])[:, None]
                        v = {k: r[idx, col].mean(1) for k, r in reps.items()}
                        cost = {s: v[(s, "gold")] - v[(s, teacher)] for s in STUDENTS}
                        did = cost["qwen"] - cost["gemma"]
                        rows.append(dict(task=task, eval_set=es, teacher=teacher, n=n, metric=metric, k=len(seeds),
                                         cost_qwen=float(cost["qwen"][0]), cost_gemma=float(cost["gemma"][0]),
                                         family_diff=float(did[0]), lo=rc.ci(did)[1], hi=rc.ci(did)[2]))
    df = pd.DataFrame(rows)
    df.to_csv(rc.OUT / "teacher_family.csv", index=False, float_format="%.4f")
    rc.save_numbers("teacher_family", {
        f"{rc.SHORT[r.task]}/{r.eval_set}/{r.teacher}/n{r.n}/{r.metric}": dict(
            cost_qwen=rc.r4(r.cost_qwen), cost_gemma=rc.r4(r.cost_gemma), family_diff=rc.r4(r.family_diff),
            lo=rc.r4(r.lo), hi=rc.r4(r.hi), k=int(r.k)) for r in df.itertuples()})
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    print(df.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    np.seterr(all="ignore")
    raise SystemExit(main())
