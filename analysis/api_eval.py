"""Per-item parquet + §6.5 metrics for every API / Jev result set that is complete.

Verdict-only models (D12) are scored on their 0/1 verdict (calibration metrics then describe a
hard classifier); unparseable answers count as "No" and are reported. An item the provider's
content filter refused to answer (D25) counts as the task's `refusal_label` (T2: "Yes", harmful);
the number refused and PR-AUC / max-F1 without those items are reported next to the main row.

  python analysis/api_eval.py            # writes results/api_preds.parquet, results/api_metrics.csv
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "labels"))
import api  # noqa: E402
from eval import metrics  # noqa: E402

SETS = ["probe", "test", "test_api_detailed", "shift_api_concise", "shift_api_detailed", "subpool"]


def main():
    preds, rows = [], []
    for task in api.TASK_NAMES:
        for m, s in api.MODELS["api"].items():
            for kind in ("concise", "detailed"):
                c = api.Cache(m, task, kind)
                if not c.records:
                    continue
                for st in SETS:
                    items = api.item_set(task, st)
                    if not items.uid.isin(set(c.records)).all():
                        continue
                    recs = [c.records[u] for u in items.uid]
                    fill = api.TASKS["tasks"][task].get("refusal_label")
                    refused = [bool(r.get("refused")) and fill is not None for r in recs]
                    verdict = [fill if rf else (r["verdict"] if r["verdict"] is not None else 0)
                               for r, rf in zip(recs, refused)]
                    p = [r["p_yes"] if (s["vendor"] == "typesafe" and r["p_yes"] is not None) else v
                         for r, v in zip(recs, verdict)]
                    df = pd.DataFrame({"task": task, "model": m, "kind": kind, "set": st, "uid": items.uid,
                                       "label": items.label.astype(int), "p": p, "verdict": verdict,
                                       "unparseable": [r["verdict"] is None and not rf for r, rf in zip(recs, refused)],
                                       "refused": refused})
                    preds.append(df)
                    row = metrics.all_metrics(df.label, df.p)
                    graded = s["tier"] == "hosted_score"
                    row.update(task=task, model=m, kind=kind, set=st,
                               scores="graded" if graded else "verdict-only, PR-AUC not comparable (D31)",
                               unparseable=int(df.unparseable.sum()),
                               refused=int(df.refused.sum()),
                               cost_usd=round(sum(r["cost_usd"] for r in recs), 4))
                    if df.refused.any():
                        kept = df[~df.refused]
                        excl = metrics.all_metrics(kept.label, kept.p)
                        row.update(pr_auc_excl_refused=excl["pr_auc"], max_f1_excl_refused=excl["max_f1"])
                    rows.append(row)
    if not rows:
        print("no complete API result sets yet")
        return
    (ROOT / "results").mkdir(exist_ok=True)
    pd.concat(preds).to_parquet(ROOT / "results" / "api_preds.parquet", index=False)
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "api_metrics.csv", index=False)
    cols = ["task", "model", "kind", "set", "n", "pr_auc", "max_f1", "f1_at_05", "ece", "brier", "bbs", "unparseable", "refused", "cost_usd"]
    print(out[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
