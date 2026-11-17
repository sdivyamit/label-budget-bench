"""How many validation labels do the N=0 decoders need to choose their threshold? (review: re-analysis on random
subsets of V of 50-500 items; exploratory, CPU only.)

  python analysis/rev_val_subsets.py [--draws 1000]

For every zero-shot decoder (concise prompt) and every task, draw m in {50, 100, 200, 500} items without replacement
from the 1,000-item gold validation set V, select tau_m = argmax F1 on the subset, apply it unchanged to the
in-distribution test set and to the hosted-API shifted subsample, and report the mean F1, the 5th-95th percentile over
the draws and the share of draws that reach the stronger frontier reference (verdict F1 on identical items). The
columns m = 1000 (the full V used in the main text) and m = 0 (raw tau = 0.5, no labels) are the end points.

Writes results/revision/zeroshot_val_subsets.csv and the section "zs_val_subsets" of revision_numbers.json.
"""
from __future__ import annotations

import argparse
import zlib

import numpy as np
import pandas as pd

import rev_common as rc

DECODERS = ["gemma4_e2b", "qwen35_2b", "gemma4_e4b", "qwen35_4b"]
SUBSET_SIZES = (50, 100, 200, 500)
SETS = ("test", "shift_api")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=1000)
    a = ap.parse_args()
    rows, numbers = [], {}
    for task in rc.TASKS:
        for es in SETS:
            _, y = rc.hl.items(task, es)
            ref = rc.ref_model(task, es)
            q_ref = rc.M.f1_at(y, rc.api_scores(task, ref, es), 0.5)
            floor = rc.floor(task, es)
            for model in DECODERS:
                dirs = rc.cell_runs(task, "decoder", model, "none", 0, "zeroshot")
                r = rc.run(dirs[0])
                yv, pv = r["val"]
                s = r[es]
                full = rc.M.f1_at(y, s, r["t_val"])
                raw = rc.M.f1_at(y, s, 0.5)
                rng = np.random.default_rng(zlib.crc32(f"{task}/{es}/{model}".encode()))
                for m in SUBSET_SIZES:
                    f1s = []
                    for _ in range(a.draws):
                        idx = rng.choice(len(yv), m, replace=False)
                        if yv[idx].sum() in (0, m):
                            continue
                        t = rc.M.max_f1(yv[idx], pv[idx])[1]
                        f1s.append(rc.M.f1_at(y, s, t))
                    f1s = np.asarray(f1s)
                    row = dict(task=task, eval_set=es, model=model, m=m, draws=len(f1s), f1_mean=f1s.mean(),
                               f1_sd=f1s.std(ddof=1), f1_p05=np.percentile(f1s, 5), f1_p95=np.percentile(f1s, 95),
                               f1_full_v=full, f1_raw05=raw, ref=ref, q_ref=q_ref, floor=floor,
                               p_ge_ref=float((f1s >= q_ref).mean()), p_ge_floor=float((f1s >= floor).mean()))
                    rows.append(row)
                    numbers[f"{rc.SHORT[task]}/{es}/{model}/m{m}"] = {
                        k: (rc.r4(v) if isinstance(v, float) else v) for k, v in row.items()
                        if k in ("f1_mean", "f1_sd", "f1_p05", "f1_p95", "f1_full_v", "f1_raw05", "q_ref", "floor",
                                 "p_ge_ref", "p_ge_floor")}
    df = pd.DataFrame(rows)
    df.to_csv(rc.OUT / "zeroshot_val_subsets.csv", index=False, float_format="%.4f")
    rc.save_numbers("zs_val_subsets", numbers)
    print(df[df.eval_set == "test"][["task", "model", "m", "f1_mean", "f1_p05", "f1_p95", "f1_full_v", "f1_raw05",
                                     "q_ref", "p_ge_ref"]].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
