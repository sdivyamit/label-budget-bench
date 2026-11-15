"""Appendix H: paired bootstrap differences for configs/comparisons.yaml, Holm-corrected (plan §6.5).

Both sides of a comparison are scored on the same item resamples as analysis/headline.py (same seeds),
so the difference CI and two-sided p-value are paired. Adds a "comparisons" block to
results/paper_numbers.json (run headline.py first) and writes results/tables/comparisons.csv.

  python analysis/compare.py [--b 10000]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT))
import headline as hl  # noqa: E402
from eval.bootstrap import holm  # noqa: E402

OUT = ROOT / "results"


def side(spec: str, task: str, eval_set: str, idx: pd.DataFrame, api: pd.DataFrame):
    """Cell spec -> (resolved name, per-seed aligned scores)."""
    parts = spec.split("/")
    uid, _ = hl.items(task, eval_set)
    if parts[0] == "api":
        if eval_set == "shift":
            raise ValueError(f"{spec}: hosted models were scored on shift_api, not the full shifted set")
        g = api[(api.task == task) & (api.model == parts[1]) &
                (api.set == ("test" if eval_set == "test" else "shift_api_concise"))]
        return spec, [hl.align(uid, g.uid.to_numpy(), g.p.to_numpy())]
    family, model, labels, n = parts
    c = idx[(idx.task == task) & (idx.family == family) & (idx.model == model) & (idx.labels == labels) & (idx.tag == "")]
    n = int(c.n.max()) if n == "top" else int(n)
    c = c[c.n == n]
    if c.empty:
        raise ValueError(f"{task}: no finished runs for {spec}")
    per_seed = [s[eval_set] for d in c.dir for s in [hl.run_scores(d, task)] if eval_set in s]
    return f"{family}/{model}/{labels}/n{n}", per_seed


def clusters(task: str, eval_set: str) -> np.ndarray:
    """Cluster id per item in hl.items order: T1 source document, T2 prompt; T3 posts are their own cluster."""
    uid, _ = hl.items(task, eval_set)
    d = pd.read_parquet(ROOT / "data" / "processed" / f"{task}.parquet").set_index("uid").loc[uid]
    if task.startswith("t1"):
        key = d.source_id.astype(str)
    elif task.startswith("t2"):
        key = d.text_a.map(lambda s: " ".join(str(s).lower().split()))
    else:
        key = pd.Series(uid, index=d.index)
    return pd.factorize(key.to_numpy())[0]


def cluster_replicates(per_seed, task: str, eval_set: str, b: int, metric: str) -> np.ndarray:
    """[k seeds x (b+1)]: like hl.replicates, but each resample draws whole clusters with replacement."""
    _, y = hl.items(task, eval_set)
    g = clusters(task, eval_set)
    n_g = int(g.max()) + 1
    rng = np.random.default_rng(hl.boot_seed(task, eval_set) + 7)
    fn = hl.fb.ece if metric.startswith("ece") else hl.fb.METRICS[metric]
    parts = []
    for start in range(0, b + 1, 500):
        w = rng.multinomial(n_g, np.full(n_g, 1.0 / n_g), size=min(500, b + 1 - start)).astype(np.float32)[:, g]
        if start == 0:
            w[0] = 1.0
        parts.append(np.vstack([fn(w, y, vt if metric == "ece_temp" else v) for v, vt in per_seed]))
    return np.hstack(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int, default=hl.B)
    ap.add_argument("--cluster", action="store_true",
                    help="robustness check: resample T1 source documents and T2 prompts instead of items")
    a = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "configs" / "comparisons.yaml").read_text())
    idx = hl.run_index()
    api = pd.read_parquet(OUT / "api_preds.parquet")
    api = api[api.kind == "concise"]
    rows = []
    for c in cfg["comparisons"]:
        metric = c.get("metric", cfg["metric_default"])
        sides = {k: side(c[k], c["task"], c["eval_set"], idx, api) for k in ("a", "b")}
        if a.cluster:
            reps = {k: cluster_replicates(ps, c["task"], c["eval_set"], a.b, metric).mean(0)
                    for k, (_, ps) in sides.items()}
        else:
            reps = {k: hl.replicates(ps, c["task"], c["eval_set"], a.b, metrics=(metric,))[metric].mean(0)
                    for k, (_, ps) in sides.items()}
        d = reps["a"] - reps["b"]
        r = d[1:][~np.isnan(d[1:])]
        lo, hi = np.percentile(r, [2.5, 97.5])
        rows.append(dict(id=c["id"], task=c["task"], eval_set=c["eval_set"], metric=metric,
                         a=sides["a"][0], k_a=len(sides["a"][1]), value_a=float(reps["a"][0]),
                         b=sides["b"][0], k_b=len(sides["b"][1]), value_b=float(reps["b"][0]),
                         diff=float(d[0]), lo=float(lo), hi=float(hi),
                         p=float(min(1.0, 2 * min((r <= 0).mean(), (r >= 0).mean())))))
    df = pd.DataFrame(rows)
    df["p_holm"] = holm(df.p.to_numpy())
    name = "comparisons_cluster" if a.cluster else "comparisons"
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "tables" / f"{name}.csv", index=False, float_format="%.4f")
    f = OUT / "paper_numbers.json"
    numbers = json.loads(f.read_text()) if f.exists() else {}
    numbers[name] = dict(status=cfg.get("status"), bootstrap_resamples=a.b,
                                  rows={r["id"]: {k: (round(v, 4) if isinstance(v, float) else v)
                                                  for k, v in r.items() if k != "id"} for r in df.to_dict("records")})
    f.write_text(json.dumps(numbers, indent=1, sort_keys=True))
    print(df[["id", "eval_set", "metric", "a", "b", "diff", "lo", "hi", "p_holm"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    np.seterr(all="ignore")
    raise SystemExit(main())
