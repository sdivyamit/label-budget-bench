"""95% bootstrap CIs for every cell and the single source of numbers for the paper (plan §6.5, §7).

A cell is task x family x model x labels x N x tag; its value is the mean over seeds. The CI resamples
test items (10,000 resamples per plan §6.5; the same resamples for every cell of a task and eval set,
so analysis/compare.py gets paired differences from the same draws); the seed SD is reported next to
it. Hosted models and detectors are one run per cell. Eval sets: test, shift (all shifted items) and
shift_api (the 2,000-item shifted subsample the hosted models were run on). Metrics: PR-AUC, max-F1,
F1 at 0.5, ECE raw and after temperature scaling (fitted on the run's validation set); for cells with
two or more seeds also alpha_K and the flip rate of the 0.5 verdicts across seeds (no CI). Rows of a
verdict-only hosted LLM carry a note on PR-AUC (D31) and ECE (D16): computed, but not comparable.

Writes results/tables/cells_ci.csv and results/paper_numbers.json.

  python analysis/headline.py [--b 10000]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zlib
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")                     # one thread per worker process
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval import calibrate, fastboot as fb, stability  # noqa: E402

RUNS, OUT = ROOT / "preds" / "runs", ROOT / "results"
B = 10_000
SETS = ("test", "shift", "shift_api")
METRICS = ("pr_auc", "max_f1", "f1_at_05", "ece", "ece_temp")
CELL = ["task", "family", "model", "labels", "n", "tag"]
VERDICT_ONLY = {m for m, s in yaml.safe_load((ROOT / "configs" / "models.yaml").read_text())["api"].items()
                if s["tier"] != "hosted_score"}
NOT_COMPARABLE = {"pr_auc": "verdict-only, PR-AUC not comparable (D31)",
                  "ece": "verdict-only, calibration not comparable (D16)"}


@lru_cache(maxsize=None)
def api_uids(task: str) -> frozenset:
    return frozenset(json.loads((ROOT / "data" / "splits" / task / "subsets.json").read_text())["shift_api_concise"])


def run_index() -> pd.DataFrame:
    rows = []
    for d in sorted(RUNS.iterdir()):
        if not ((d / "DONE").exists() and (d / "preds.parquet").exists()):
            continue
        r = json.loads((d / "run.json").read_text())
        if r.get("eval_limit") or r.get("max_steps"):      # setup probes (crash / memory tests), not results
            continue
        base = f"{r['task']}__{r['model']}__{r['labels']}__n{r['n']}__s{r['seed']}"
        rows.append(dict(dir=str(d), task=r["task"], family=r["family"], model=r["model"], labels=r["labels"],
                         n=int(r["n"]), seed=int(r["seed"]), tag=d.name[len(base) + 2:]))
    return pd.DataFrame(rows)


@lru_cache(maxsize=None)
def items(task: str, eval_set: str) -> tuple[np.ndarray, np.ndarray]:
    """Canonical item order (sorted uid) and labels for one task x eval set."""
    s = pd.read_parquet(ROOT / "data" / "processed" / f"{task}.parquet", columns=["uid", "label", "split"])
    s = s[s.split == ("test" if eval_set == "test" else "shift")]
    if eval_set == "shift_api":
        s = s[s.uid.isin(api_uids(task))]
    s = s.sort_values("uid")
    return s.uid.to_numpy(), s.label.to_numpy().astype(np.float32)


def boot_seed(task: str, eval_set: str) -> int:
    return zlib.crc32(f"{task}/{eval_set}".encode())


def run_scores(d: str | Path, task: str) -> dict[str, tuple[np.ndarray, np.ndarray | None]]:
    """eval set -> (p, temperature-scaled p) aligned to items(task, set); sets the run does not fully cover are left out."""
    pp = pd.read_parquet(Path(d) / "preds.parquet")
    val = pp[pp.split == "val"]
    temp = calibrate.fit("temperature", val.p.to_numpy(), val.label.to_numpy()) if len(val) else None
    out = {}
    for st in SETS:
        g = pp[pp.split == ("test" if st == "test" else "shift")]
        if st == "shift_api":
            g = g[g.uid.isin(api_uids(task))]
        uid, _ = items(task, st)
        if len(g) != len(uid):
            if len(g):
                print(f"skipped {Path(d).name} on {st}: {len(g)} of {len(uid)} items", file=sys.stderr)
            continue
        out[st] = align(uid, g.uid.to_numpy(), g.p.to_numpy(), temp)
    return out


def align(uid, u, p, temp=None):
    idx = np.searchsorted(uid, u)
    if not np.array_equal(uid[np.minimum(idx, len(uid) - 1)], u) or len(set(idx)) != len(uid):
        raise ValueError("run items do not match the eval set")
    v = np.empty(len(uid), np.float32); v[idx] = p
    vt = None
    if temp is not None:
        vt = np.empty(len(uid), np.float32); vt[idx] = temp(p)
    return v, vt


def replicates(per_seed: list[tuple[np.ndarray, np.ndarray | None]], task: str, eval_set: str, b: int,
               metrics=METRICS) -> dict[str, np.ndarray]:
    """metric -> [k seeds x (b+1)] values; column 0 = the full eval set."""
    _, y = items(task, eval_set)
    parts = defaultdict(list)
    for w in fb.chunks(len(y), b, boot_seed(task, eval_set)):
        for m in metrics:
            vals = [(fb.ece if m.startswith("ece") else fb.METRICS[m])(w, y, vt if m == "ece_temp" else v)
                    for v, vt in per_seed if not (m == "ece_temp" and vt is None)]
            if vals:
                parts[m].append(np.vstack(vals))
    return {m: np.hstack(p) for m, p in parts.items()}


def cell_job(args) -> list[dict]:
    key, dirs, b = args
    task = key[0]
    by_set = defaultdict(list)
    for d in dirs:
        for st, sc in run_scores(d, task).items():
            by_set[st].append(sc)
    out = []
    for st, per_seed in by_set.items():
        for m, r in replicates(per_seed, task, st, b).items():
            out.append(dict(key=key, eval_set=st, metric=m, k=len(r),
                            sd=float(r[:, 0].std(ddof=1)) if len(r) > 1 else float("nan"), **fb.summary(r.mean(0))))
        if len(per_seed) > 1:                       # seed stability of the 0.5 verdicts (RQ2), point values only
            v = np.vstack([p >= 0.5 for p, _ in per_seed]).astype(int)
            pairs = [(v[i] != v[j]).mean() for i in range(len(v)) for j in range(i + 1, len(v))]
            # flip_rate grows with K (5 seeds up to N = 2k, 3 above); pair_disagree does not
            for m, val in (("alpha_k", stability.alpha_k(v)), ("flip_rate", stability.flip_rate(v)),
                           ("pair_disagree", float(np.mean(pairs)))):
                out.append(dict(key=key, eval_set=st, metric=m, k=len(per_seed), sd=float("nan"),
                                mean=float(val), lo=float("nan"), hi=float("nan")))
    return out


def api_job(args) -> list[dict]:
    (task, model, es), uid_p, b = args
    uid, _ = items(task, es)
    per_seed = [align(uid, uid_p[0], uid_p[1])]
    return [dict(key=(task, "api", model, "none", 0, "concise"), eval_set=es, metric=m, k=1, sd=float("nan"),
                 **fb.summary(r.mean(0))) for m, r in replicates(per_seed, task, es, b).items()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int, default=B)
    a = ap.parse_args()
    idx = run_index()
    jobs = [(k, g.dir.tolist(), a.b) for k, g in idx.groupby(CELL, sort=True)]
    api = []
    f = OUT / "api_preds.parquet"
    if f.exists():
        apr = pd.read_parquet(f)
        apr = apr[(apr.kind == "concise") & apr.set.isin(["test", "shift_api_concise"])]
        for (task, model, st), g in apr.groupby(["task", "model", "set"]):
            api.append(((task, model, "test" if st == "test" else "shift_api"), (g.uid.to_numpy(), g.p.to_numpy()), a.b))
    with ProcessPoolExecutor() as ex:
        rows = [r for part in ex.map(cell_job, jobs, chunksize=2) for r in part]
        rows += [r for part in ex.map(api_job, api) for r in part]

    df = pd.DataFrame([dict(zip(CELL, r.pop("key")), **r) for r in rows])
    df = df.sort_values(["task", "eval_set", "family", "model", "labels", "n", "tag", "metric"])
    verdict = (df.family == "api") & df.model.isin(VERDICT_ONLY)
    df["note"] = [NOT_COMPARABLE.get(m, "") if v else "" for m, v in zip(df.metric, verdict)]
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "tables" / "cells_ci.csv", index=False, float_format="%.4f")

    numbers = defaultdict(dict)
    for r in df.itertuples():
        name = f"{r.family}/{r.model}/{r.labels}/n{r.n}" + (f"/{r.tag}" if r.tag else "")
        numbers[f"{r.task}/{r.eval_set}"][f"{name}/{r.metric}"] = dict(
            mean=round(r.mean, 4), lo=None if np.isnan(r.lo) else round(r.lo, 4), hi=None if np.isnan(r.hi) else round(r.hi, 4),
            sd=None if np.isnan(r.sd) else round(r.sd, 4), k=int(r.k), **({"note": r.note} if r.note else {}))
    meta = dict(bootstrap_resamples=a.b, level=0.95, runs=int(len(idx)), cells=int(len(jobs)), hosted_cells=len(api),
                note="mean = seed mean on the full eval set; lo/hi = percentile CI over item resamples of the seed mean; "
                     "sd = seed SD; k = seeds")
    (OUT / "paper_numbers.json").write_text(json.dumps(dict(meta=meta, cells=numbers), indent=1, sort_keys=True))
    print(f"{len(idx)} runs, {len(jobs)} cells + {len(api)} hosted, {a.b:,} resamples -> {len(df):,} rows: "
          f"results/tables/cells_ci.csv, results/paper_numbers.json")
    return 0


if __name__ == "__main__":
    np.seterr(all="ignore")
    raise SystemExit(main())
