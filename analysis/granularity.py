"""D12 probe: are API / Jev scores usable as probabilities, or verdict-only?

Reads the cached API answers on the 200-item probe set and reports, per model, how many distinct
scores occur and how much mass sits at exactly 0 or 1. Decision rule (D12): a model's scores are
reported as probabilities only if >= 10 distinct values and <= 80% of items at {0, 1}.

  python analysis/granularity.py --task t2_response_safety
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "labels"))
import api  # noqa: E402
from eval import metrics  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="t2_response_safety")
    a = ap.parse_args()
    probe = api.item_set(a.task, "probe")
    y = dict(zip(probe.uid, probe.label.astype(int)))
    print(f"{'model':22s} {'n':>4s} {'distinct':>8s} {'at0/1':>6s} {'PR-AUC':>7s} {'verdict F1':>10s}  usable as probability?")
    for m in api.MODELS["api"]:
        c = api.Cache(m, a.task, "concise")
        recs = [c.records[u] for u in probe.uid if u in c.records]
        if not recs:
            continue
        p = np.array([r["p_yes"] if r["p_yes"] is not None else (r["verdict"] if r["verdict"] is not None else 0) for r in recs], float)
        v = np.array([r["verdict"] if r["verdict"] is not None else 0 for r in recs])
        yy = np.array([y[r["uid"]] for r in recs])
        distinct = len(np.unique(np.round(p, 4)))
        at01 = float(np.isin(np.round(p, 6), [0.0, 1.0]).mean())
        ok = distinct >= 10 and at01 <= 0.8
        print(f"{m:22s} {len(recs):4d} {distinct:8d} {at01:6.2f} {metrics.pr_auc(yy, p):7.3f} "
              f"{metrics.f1_at(yy, v, 0.5):10.3f}  {'YES' if ok else 'no (verdict-only)'}")


if __name__ == "__main__":
    main()
