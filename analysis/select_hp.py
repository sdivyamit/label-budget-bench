"""Week 3 hyperparameter selection (§6.4): rank the `hp` stage runs by validation PR-AUC.

Prints one table per model and the pick; test/shift are shown for the record but never used.
Copy the picks into configs/train.yaml `hp`, then freeze the file (decisions.md).

  python analysis/select_hp.py
"""
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    jobs = [j for j in yaml.safe_load((ROOT / "queue" / "jobs.yaml").read_text()) if j["stage"] == "hp"]
    missing = 0
    by_model: dict[str, list] = {}
    for j in jobs:
        f = ROOT / "preds" / "runs" / j["id"] / "metrics.json"
        if not f.exists():
            missing += 1
            continue
        m = json.loads(f.read_text())
        by_model.setdefault((j["family"], j["model"]), []).append(
            (m["val"]["raw"]["pr_auc"], j["hp"], m["test"]["raw"]["pr_auc"], m["shift"]["raw"]["pr_auc"]))
    for (family, model), rows in by_model.items():
        rows.sort(key=lambda r: -r[0])
        print(f"\n{family}/{model}")
        print(f"  {'hp':28s} {'val PR-AUC':>10s} {'test':>7s} {'shift':>7s}")
        for v, hp, t, s in rows:
            print(f"  {json.dumps(hp):28s} {v:10.4f} {t:7.4f} {s:7.4f}")
        print(f"  pick: {json.dumps(rows[0][1])}  (margin over 2nd: {rows[0][0] - rows[1][0]:+.4f})" if len(rows) > 1
              else f"  only one run: {json.dumps(rows[0][1])}")
    print(f"\n{len(jobs) - missing}/{len(jobs)} hp runs finished")
    return 0 if not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
