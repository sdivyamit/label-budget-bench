"""Final-revision numbers (exploratory, CPU only): the cost of choosing thresholds on a validation set whose prevalence
differs from the test set (T1: 0.308 vs 0.205), and the serving-throughput ratios quoted in the text.

  python analysis/rev_final.py

1. For every gold-trained model at N_max (seed mean) and every zero-shot decoder (concise prompt), F1 on the in-distribution
   test set at the validation-chosen threshold, F1 at the oracle test threshold (max F1 over tau on the test set), and their
   difference. Written to results/revision/threshold_prevalence_gap.csv.
2. Encoder-to-decoder items/s ratios (best encoder over best decoder per task and the range over all encoder-decoder pairs),
   from results/tables/table8_serving.csv. Written to the section "final" of revision_numbers.json.
"""
from __future__ import annotations

import gzip
import json
import re

import numpy as np
import pandas as pd

import rev_common as rc

ENC = ["modernbert_base", "modernbert_large", "deberta_v3_large"]
DEC = ["gemma4_e2b", "qwen35_2b", "gemma4_e4b", "qwen35_4b"]


def _norm(x) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(x).lower()).split())


def overlap_denominators() -> dict:
    """T2 ID test items whose normalized prompt occurs (a) anywhere in the full BeaverTails-30k training split before
    pool sub-sampling, (b) in the 20,000-item pool (the basis of the H1c unseen-prompt analysis)."""
    raw = rc.ROOT / "data" / "raw" / "beavertails" / "train.jsonl.gz"
    full = {_norm(json.loads(l)["prompt"]) for l in gzip.open(raw, "rt")}
    p = pd.read_parquet(rc.ROOT / "data" / "processed" / "t2_response_safety.parquet")
    test, pool = p[p.split == "test"], p[p.split == "pool"]
    pool_p = {_norm(a) for a in pool.text_a}
    n = len(test)
    a = int(sum(_norm(x) in full for x in test.text_a))
    b = int(sum(_norm(x) in pool_p for x in test.text_a))
    return {"overlap/full_train/items": a, "overlap/full_train/total": n, "overlap/full_train/pct": rc.r4(100 * a / n),
            "overlap/pool/items": b, "overlap/pool/total": n, "overlap/pool/pct": rc.r4(100 * b / n)}


def main() -> int:
    rows = []
    for task in rc.TASKS:
        _, y = rc.hl.items(task, "test")
        nmax = {"t1_rag_hallucination": 8651, "t2_response_safety": 20000, "t3_implicit_toxicity": 5920}[task]
        cells = [(m, "encoder", "gold", nmax, "") for m in ENC + ["tfidf_lr"] if m != "tfidf_lr"] + \
                [(m, "decoder", "gold", nmax if m in ("gemma4_e2b", "qwen35_2b") else nmax, "") for m in DEC] + \
                [(m, "decoder", "none", 0, "zeroshot") for m in DEC]
        for model, fam, lab, n, tag in cells:
            dirs = rc.cell_runs(task, fam, model, lab, n, tag)
            if not dirs:
                continue
            f_val, f_orc, t_val = [], [], []
            for d in dirs:
                r = rc.run(d)
                s = r["test"]
                f_val.append(rc.M.f1_at(y, s, r["t_val"]))
                f_orc.append(rc.M.max_f1(y, s)[0])
            rows.append(dict(task=task, model=model, labels=lab, n=n, k=len(dirs), prev_val=float(rc.run(dirs[0])["val"][0].mean()),
                             prev_test=float(y.mean()), f1_valthr=np.mean(f_val), f1_oracle=np.mean(f_orc),
                             gap=np.mean(f_orc) - np.mean(f_val)))
    df = pd.DataFrame(rows)
    df.to_csv(rc.OUT / "threshold_prevalence_gap.csv", index=False, float_format="%.4f")
    print(df.round(3).to_string(index=False))

    numbers = {}
    for task, g in df.groupby("task"):
        sh = rc.SHORT[task]
        numbers[f"thrgap/{sh}/min"] = rc.r4(g.gap.min())
        numbers[f"thrgap/{sh}/max"] = rc.r4(g.gap.max())
        numbers[f"thrgap/{sh}/median"] = rc.r4(g.gap.median())
        numbers[f"thrgap/{sh}/prev_val"] = rc.r4(g.prev_val.iloc[0])
        numbers[f"thrgap/{sh}/prev_test"] = rc.r4(g.prev_test.iloc[0])
    t = pd.read_csv(rc.ROOT / "results" / "tables" / "table8_serving.csv")
    t = t[t.family != "api"]
    allr = []
    for task, g in t.groupby("task"):
        sh = rc.SHORT[task]
        e = g[g.model.isin(ENC[:2])].set_index("model").items_per_s
        d = g[g.model.isin(DEC)].set_index("model").items_per_s
        ratios = [a / b for a in e for b in d]
        numbers[f"serve/{sh}/ratio_min"] = rc.r4(min(ratios))
        numbers[f"serve/{sh}/ratio_max"] = rc.r4(max(ratios))
        allr += ratios
    numbers["serve/all/ratio_min"] = rc.r4(min(allr))
    numbers["serve/all/ratio_max"] = rc.r4(max(allr))
    numbers.update(overlap_denominators())
    rc.save_numbers("final", numbers)
    print(numbers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
