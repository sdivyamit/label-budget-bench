"""Constant-output runs: raw in-distribution test scores with SD < 0.01 (criterion stated in the paper, Section IV-E).

Writes results/tables/tableB_collapse.csv (one row per flagged run) and tableB_collapse_summary.csv
(per task x model: finished main-sweep runs, flagged runs, and the gold PR-AUC at N_max with and without them).

  python analysis/collapse.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RUNS, TAB = ROOT / "preds" / "runs", ROOT / "results" / "tables"
SD_MIN = 0.01


def main() -> int:
    rows = []
    for d in sorted(RUNS.iterdir()):
        if not (d / "DONE").exists() or not (d / "run.json").exists():
            continue
        r = json.loads((d / "run.json").read_text())
        if r.get("eval_limit") or r.get("max_steps") or r["family"] == "classical":
            continue
        base = f"{r['task']}__{r['model']}__{r['labels']}__n{r['n']}__s{r['seed']}"
        tag = d.name[len(base) + 2:]
        p = pd.read_parquet(d / "preds.parquet", columns=["split", "p"])
        sd = float(p[p.split == "test"].p.std())
        rows.append(dict(run_id=d.name, task=r["task"], model=r["model"], labels=r["labels"], n=r["n"],
                         seed=r["seed"], tag=tag, test_score_sd=sd, constant=sd < SD_MIN))
    df = pd.DataFrame(rows)
    TAB.mkdir(parents=True, exist_ok=True)
    df[df.constant].to_csv(TAB / "tableB_collapse.csv", index=False, float_format="%.5f")
    main_sweep = df[df.tag == ""]
    m = pd.read_parquet(ROOT / "results" / "metrics.parquet")
    m = m[(m.metric == "pr_auc") & (m.eval_set == "test") & (m.calib == "raw") & (m.tag.fillna("") == "")
          & (m.labels == "gold")]
    m = m[m.n == m.groupby("task").n.transform("max")]
    flagged = set(main_sweep[main_sweep.constant].run_id)
    summ = []
    for (task, model), g in main_sweep.groupby(["task", "model"]):
        mm = m[(m.task == task) & (m.model == model)]
        keep = mm[~mm.run_id.isin(flagged)]
        summ.append(dict(task=task, model=model, runs=len(g), constant=int(g.constant.sum()),
                         gold_top_pr_auc=mm.value.mean() if len(mm) else None,
                         gold_top_pr_auc_without=keep.value.mean() if len(keep) else None,
                         gold_top_k_without=len(keep)))
    s = pd.DataFrame(summ)
    s.to_csv(TAB / "tableB_collapse_summary.csv", index=False, float_format="%.4f")
    print(df[df.constant][["run_id", "test_score_sd"]].to_string(index=False))
    print(s[s.constant > 0].to_string(index=False))
    print(f"total: {int(main_sweep.constant.sum())} of {len(main_sweep)} main-sweep neural runs; "
          f"{int(df[df.tag != ''].constant.sum())} more in ablations and probes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
