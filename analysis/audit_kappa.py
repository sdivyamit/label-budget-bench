"""Score the label-alignment audit (plan §6.1): Cohen's kappa of each annotator vs. the shifted
set's own gold labels, and between annotators, on all rows (forced choice), with 95% bootstrap CIs.
Pass rule: kappa(annotator, gold) >= 0.6 for every annotator.

Reads `{task}_audit_final.csv` (columns uid, {annotator}_label). Gold comes from the processed
data, or from `gold.csv` for audits of rejected shifted sets that are no longer in it (D21, D22).

Usage: python analysis/audit_kappa.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load((ROOT / "configs" / "tasks.yaml").read_text())
AUDIT = ROOT / "docs" / "audit"
THRESHOLD = 0.6
AUDITS = {  # name: (task, folder, gold source)
    "t2_response_safety (WildGuardTest)": ("t2_response_safety", AUDIT, None),
    "t1_rag_hallucination (LLM-AggreFact, rejected)": ("t1_rag_hallucination", AUDIT / "t1_llm_aggrefact_rejected", "gold_unsupported"),
    "t3_implicit_toxicity (Implicit Hate, rejected)": ("t3_implicit_toxicity", AUDIT / "t3_implicit_hate_rejected", "gold_implicit_hate"),
}


def kappa_ci(a, b, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    idx = (rng.integers(0, len(a), len(a)) for _ in range(n_boot))
    return cohen_kappa_score(a, b), np.nanpercentile([cohen_kappa_score(a[i], b[i]) for i in idx], [2.5, 97.5])


def main() -> None:
    for name, (task, folder, gold_col) in AUDITS.items():
        f = folder / f"{task}_audit_final.csv"
        print(f"\n{name}")
        if not f.exists():
            print(f"  missing {f.relative_to(ROOT)}")
            continue
        a = pd.read_csv(f, dtype=str, keep_default_na=False)
        if gold_col:
            gold = pd.read_csv(folder / "gold.csv").set_index("uid")[gold_col]
        else:
            gold = pd.read_parquet(ROOT / "data" / "processed" / f"{task}.parquet").set_index("uid").label
        g = gold.loc[a.uid].astype(int).to_numpy()
        labels = {}
        for who in CFG["annotators"]:
            v = a[f"{who}_label"].str.strip()
            bad = v[~v.isin(["0", "1"])]
            if len(bad):
                raise ValueError(f"{f.name}: {who} labels must be 0 or 1; bad items {a.item[bad.index].tolist()[:5]}")
            labels[who] = v.astype(int).to_numpy()
        passed = True
        for who, lab in labels.items():
            k, (lo, hi) = kappa_ci(lab, g)
            passed &= k >= THRESHOLD
            print(f"  {who:8s} kappa vs gold {k:.2f} [{lo:.2f}, {hi:.2f}]  agreement {np.mean(lab == g):.0%}  "
                  f"{'PASS' if k >= THRESHOLD else 'BELOW 0.6'}")
        x, y = labels.values()
        k, (lo, hi) = kappa_ci(x, y)
        print(f"  between annotators kappa {k:.2f} [{lo:.2f}, {hi:.2f}] on {len(x)} items")
        print(f"  => {'ACCEPT' if passed else 'REJECT'} shift pair")


if __name__ == "__main__":
    main()
