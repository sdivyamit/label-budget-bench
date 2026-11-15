"""Paper tables from results/tables/cells_ci.csv (run analysis/headline.py first) and the serving logs.

  Table 5   label sources at N = 5,000 (frontier-teacher sub-pool size) and at the top N, PR-AUC and ECE
  Table 8   serving: batched items/s, batch-1 latency p50/p95/p99, USD per million items on one A6000,
            next to the hosted models' measured USD per million items (concise prompt, test set)
  Table 9   off-the-shelf detectors next to fixed reference cells (no selection on test); a verdict-only
            hosted LLM's PR-AUC (D31) and ECE (D16) cells are replaced by a "not comparable" mark
  Table D   ablations (tags abl-*) next to the matching main-sweep cell (Appendix D)
Every cell is "mean [lo, hi]" with the 95% bootstrap CI; the serving block is added to paper_numbers.json.

  python analysis/tables.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
REF_ENCODER, REF_DECODER, REF_API = "modernbert_large", "qwen35_2b", "gpt61_sol"
VERDICT_ONLY = {m for m, s in yaml.safe_load((ROOT / "configs" / "models.yaml").read_text())["api"].items()
                if s["tier"] != "hosted_score"}
NOT_COMPARABLE = {"pr_auc": "verdict-only, PR-AUC not comparable (D31)",
                  "ece": "verdict-only, calibration not comparable (D16)"}
SOURCES = ["gold", "open_teacher", "frontier_teacher", "mix250_open_teacher", "mix1000_open_teacher",
           "mix250_frontier_teacher", "mix1000_frontier_teacher"]


def fmt(r) -> str:
    return f"{r['mean']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]"


def load_all_tags() -> pd.DataFrame:
    c = pd.read_csv(OUT / "tables" / "cells_ci.csv", keep_default_na=False, na_values=[""])
    c["tag"] = c["tag"].fillna("")
    return c


def top_n(c: pd.DataFrame) -> pd.Series:
    return c.groupby(["task", "family", "model", "labels"]).n.transform("max")


def table5(c: pd.DataFrame) -> pd.DataFrame:
    c = c[(c.family.isin(["encoder", "decoder", "classical"])) & c.labels.isin(SOURCES) &
          c.metric.isin(["pr_auc", "ece"]) & c.eval_set.isin(["test", "shift"])].copy()
    c["is_top"] = c.n == top_n(c.assign(labels="all"))
    rows = []
    for (task, fam, model, es, metric), g in c.groupby(["task", "family", "model", "eval_set", "metric"]):
        row = dict(task=task, family=fam, model=model, eval_set=es, metric=metric)
        for src in SOURCES:
            for where, h in (("n5000", g[(g.labels == src) & (g.n == 5000)]),
                             ("top", g[(g.labels == src) & g.is_top])):
                if len(h):
                    r = h.iloc[0]
                    row[f"{src}@{where}"] = f"{fmt(r)} k={r['k']}" if where == "n5000" else f"{fmt(r)} N={r['n']}"
        rows.append(row)
    return pd.DataFrame(rows)


def table8(c: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    logs = [f for f in sorted((ROOT / "logs").glob("serving_*.csv")) if "smoke" not in f.name]
    own = pd.concat([pd.read_csv(f) for f in logs], ignore_index=True) if logs else pd.DataFrame()
    am = pd.read_csv(OUT / "api_metrics.csv")
    am = am[(am.kind == "concise") & (am.set == "test")].copy()
    am["usd_per_million_items"] = (am.cost_usd / am.n * 1e6).round(2)
    api = am[["model", "task", "n", "usd_per_million_items"]].assign(family="api")
    t = pd.concat([own, api], ignore_index=True)
    cols = ["family", "model", "task", "items_per_s", "p50_ms", "p95_ms", "p99_ms", "usd_per_million_items",
            "mean_input_tokens", "gpu", "torch"]
    t = t[[k for k in cols if k in t]].sort_values(["task", "family", "model"])
    return t, dict(serving_logs=[f.name for f in logs], rows=t.where(t.notna(), None).to_dict("records"))


def table9(c: pd.DataFrame) -> pd.DataFrame:
    c = c[c.metric.isin(["pr_auc", "max_f1", "f1_at_05", "ece"])].copy()
    c["is_top"] = (c.family != "api") & (c.n == top_n(c))
    det = c[c.family == "detector"]
    ref = c[((c.family == "encoder") & (c.model == REF_ENCODER) & (c.labels == "gold") & c.is_top) |
            ((c.family == "decoder") & (c.model == REF_DECODER) & (c.labels == "gold") & c.is_top) |
            ((c.family == "api") & (c.model == REF_API))]
    rows = []
    for (task, fam, model, labels, n), g in pd.concat([det, ref]).groupby(["task", "family", "model", "labels", "n"]):
        verdict = fam == "api" and model in VERDICT_ONLY
        row = dict(task=task, family=fam, model=model, labels=labels, n=n,
                   note="verdict-only: compare on F1 at its verdict" if verdict else "")
        for r in g.itertuples():
            row[f"{r.eval_set}:{r.metric}"] = NOT_COMPARABLE.get(r.metric) if verdict and r.metric in NOT_COMPARABLE \
                else fmt(r._asdict())
        rows.append(row)
    return pd.DataFrame(rows)


def table_d(cc: pd.DataFrame) -> pd.DataFrame:
    """Each ablation cell next to the main-sweep cell with the same task, model, labels and N."""
    c = cc[cc.metric.isin(["pr_auc", "max_f1", "ece", "ece_temp"]) & cc.eval_set.isin(["test", "shift"])]
    key = ["task", "family", "model", "labels", "n", "eval_set", "metric"]
    abl = c[c.tag.str.startswith("abl-")]
    base = c[c.tag == ""][key + ["mean", "lo", "hi", "k"]]
    t = abl.merge(base, on=key, how="left", suffixes=("", "_main"))
    t["ablation"] = t.apply(lambda r: f"{r['mean']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]", axis=1)
    t["main"] = t.apply(lambda r: f"{r['mean_main']:.3f} [{r['lo_main']:.3f}, {r['hi_main']:.3f}]"
                        if pd.notna(r["mean_main"]) else "", axis=1)
    t["diff"] = (t["mean"] - t["mean_main"]).round(4)
    return t[["tag"] + key + ["k", "ablation", "k_main", "main", "diff"]].sort_values(["tag"] + key)


def main() -> int:
    c = load_all_tags()
    (OUT / "tables").mkdir(exist_ok=True)
    td = table_d(c); td.to_csv(OUT / "tables" / "tableD_ablations.csv", index=False)
    c = c[c.tag.isin(["", "concise"])]
    t5 = table5(c); t5.to_csv(OUT / "tables" / "table5_label_sources.csv", index=False)
    t8, serving = table8(c); t8.to_csv(OUT / "tables" / "table8_serving.csv", index=False)
    t9 = table9(c); t9.to_csv(OUT / "tables" / "table9_detectors.csv", index=False)
    f = OUT / "paper_numbers.json"
    numbers = json.loads(f.read_text()) if f.exists() else {}
    numbers["serving"] = serving
    f.write_text(json.dumps(numbers, indent=1, sort_keys=True))
    det = int((t9.family == "detector").sum()) if len(t9) else 0
    print(f"table5 {len(t5)} rows, table8 {len(t8)} rows ({len(serving['serving_logs'])} serving logs), "
          f"table9 {len(t9)} rows ({det} detector rows), tableD {len(td)} rows ({td.tag.nunique()} ablations)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
