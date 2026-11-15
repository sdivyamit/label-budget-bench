"""Build results/metrics.parquet: one row per run x eval set x calibration x metric (plan §7).

For every finished run in preds/runs/ (main sweep and tagged runs alike) this recomputes, from the
per-item predictions, everything the figures and tables need:
  * §6.5 metrics on val / test / shift for raw, temperature-scaled and isotonic scores;
  * RQ3 operating points chosen on val: target precision 0.90 (raw, temperature, isotonic) and
    target FPR 0.05 (raw, temperature, isotonic, split-conformal), realized on test and shift;
  * the same on `shift_api`, the shifted items the API models were run on (shift_api_concise).
API / Jev rows come from results/api_preds.parquet (run analysis/api_eval.py first).
Runs already collected are cached in results/collect_cache/ keyed by their DONE timestamp.

  python analysis/collect.py            # writes results/metrics.parquet
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval import calibrate, metrics, thresholds  # noqa: E402

RUNS, OUT, CACHE = ROOT / "preds" / "runs", ROOT / "results", ROOT / "results" / "collect_cache"
CALIBS = ("raw", "temperature", "isotonic")
TARGET_PRECISION, TARGET_FPR = 0.90, 0.05
KEYS = ["run_id", "task", "family", "model", "labels", "n", "seed", "tag"]


def score_rows(y_val, s_val, evals: dict[str, tuple]) -> list[dict]:
    """Metric and operating-point rows for one score source; evals maps set name -> (y, s)."""
    rows = []
    cals = {c: calibrate.fit("none" if c == "raw" else c, s_val, y_val) for c in CALIBS}
    for c, cal in cals.items():
        for name, (y, s) in {"val": (y_val, s_val), **evals}.items():
            for k, v in metrics.all_metrics(y, cal(s)).items():
                rows.append(dict(eval_set=name, calib=c, metric=k, value=float(v)))
    for c, cal in cals.items():
        d = thresholds.precision_target_drift(y_val, cal(s_val), {n: (y, cal(s)) for n, (y, s) in evals.items()},
                                              TARGET_PRECISION)
        for name in evals:
            for k, v in d[name].items():
                rows.append(dict(eval_set=name, calib=c, metric=f"at_prec90_{k}", value=float(v)))
    for method, per_set in thresholds.fpr_control(y_val, s_val, evals, TARGET_FPR).items():
        for name, rates in per_set.items():
            for k, v in rates.items():
                rows.append(dict(eval_set=name, calib=method, metric=f"at_fpr05_{k}", value=float(v)))
    return rows


def collect_run(d: Path) -> pd.DataFrame:
    stamp = (d / "DONE").read_text().strip()
    cache = CACHE / f"{d.name}.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
        if len(df) and df["done"].iloc[0] == stamp:
            return df
    run = json.loads((d / "run.json").read_text())
    pp = pd.read_parquet(d / "preds.parquet")
    val = pp[pp.split == "val"]
    evals = {s: (g.label.to_numpy(), g.p.to_numpy()) for s, g in pp.groupby("split") if s != "val"}
    # the API models' 2,000-item shifted subsample, for like-for-like comparison with them
    api_uids = set(json.loads((ROOT / "data" / "splits" / run["task"] / "subsets.json").read_text())["shift_api_concise"])
    g = pp[(pp.split == "shift") & pp.uid.isin(api_uids)]
    if len(g):
        evals["shift_api"] = (g.label.to_numpy(), g.p.to_numpy())
    df = pd.DataFrame(score_rows(val.label.to_numpy(), val.p.to_numpy(), evals))
    for k in KEYS:
        df[k] = run.get(k)
    df["run_id"] = d.name
    base = f"{run['task']}__{run['model']}__{run['labels']}__n{run['n']}__s{run['seed']}"
    df["tag"] = d.name[len(base) + 2:]
    df["done"] = stamp
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


def api_rows() -> pd.DataFrame:
    """API and Jev: the 'val' for operating points is the model's own test set (no val is labelled
    by APIs); rows are flagged family = 'api' and are used for references, not for RQ3 drift."""
    f = OUT / "api_preds.parquet"
    if not f.exists():
        return pd.DataFrame()
    ap = pd.read_parquet(f)
    out = []
    for (task, model, kind), g in ap.groupby(["task", "model", "kind"]):
        for st, h in g.groupby("set"):
            for k, v in metrics.all_metrics(h.label, h.p).items():
                out.append(dict(run_id=f"{task}__{model}__{kind}", task=task, family="api", model=model,
                                labels="none", n=0, seed=0, tag=kind, eval_set=st, calib="raw", metric=k,
                                value=float(v)))
    return pd.DataFrame(out)


def main() -> int:
    dirs = sorted(d for d in RUNS.iterdir() if (d / "DONE").exists() and (d / "preds.parquet").exists())
    with ProcessPoolExecutor() as ex:
        parts = list(ex.map(collect_run, dirs, chunksize=4))
    df = pd.concat(parts + [api_rows()], ignore_index=True).drop(columns=["done"], errors="ignore")
    df["n"] = df["n"].astype(int)
    df["seed"] = df["seed"].astype(int)
    df["tag"] = df["tag"].fillna("")
    OUT.mkdir(exist_ok=True)
    df.to_parquet(OUT / "metrics.parquet", index=False)
    main_sweep = df[(df.tag == "") & (df.family != "api")]
    print(f"{len(dirs)} runs ({main_sweep.run_id.nunique()} main-sweep), {len(df):,} rows -> results/metrics.parquet")
    print(main_sweep.groupby(["task", "family"]).run_id.nunique().to_string())
    return 0


if __name__ == "__main__":
    np.seterr(all="ignore")
    raise SystemExit(main())
