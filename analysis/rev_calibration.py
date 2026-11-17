"""Full calibration tables (revision 2e; review point 9).

From the per-run metrics already in results/metrics.parquet (main sweep, gold labels, N = 1,000 and N_max)
and, for the zero-shot decoders (not in that file), recomputed with the same code (analysis/collect.py):
ECE (10 equal-width bins), ECE (15 equal-mass bins), Brier, BBS, BS-Correct, BS-Error (multiclass scale
[0, 2]), each raw, temperature-scaled and isotonic (fitted on the run's validation set), on test, shift and
shift_api; seed mean and SD. Jev (hosted score classifier, no validation set): raw only.
Also the family ranges of BBS and BS-Error on T1 (the "more confident errors" claim restated on BBS) and
pairwise seed disagreement of the 0.5 verdicts from results/tables/cells_ci.csv.

  python analysis/rev_calibration.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import rev_common as rc
import collect  # noqa: E402  (analysis/collect.py: score_rows)

CAL = ["ece", "ece_mass15", "brier", "bbs", "bs_correct", "bs_error"]
TOP = {"t1_rag_hallucination": 8651, "t2_response_safety": 20000, "t3_implicit_toxicity": 5920}


def zeroshot_rows() -> pd.DataFrame:
    i = rc.index()
    z = i[i.tag.isin(["zeroshot", "zeroshot-detailed"])]
    out = []
    for r in z.itertuples():
        x = rc.run(r.dir)
        yv, sv = x["val"]
        evals = {es: (rc.hl.items(r.task, es)[1].astype(int), x[es]) for es in ("test", "shift", "shift_api") if es in x}
        for row in collect.score_rows(yv, sv, evals):
            if row["metric"] in CAL:
                out.append(dict(row, task=r.task, family=r.family, model=r.model, labels="none", n=0, seed=r.seed,
                                tag=r.tag))
    return pd.DataFrame(out)


def main() -> int:
    rc.OUT.mkdir(parents=True, exist_ok=True)
    m = pd.read_parquet(rc.ROOT / "results" / "metrics.parquet")
    trained = m[(m.tag == "") & (m.family != "api") & (m.labels == "gold") & m.metric.isin(CAL)
                & m.eval_set.isin(["test", "shift", "shift_api"])]
    trained = trained[(trained.n == 1000) | (trained.n == trained.task.map(TOP))]
    zs = zeroshot_rows()
    jev = m[(m.family == "api") & (m.model == "jev") & (m.tag == "concise") & m.metric.isin(CAL)
            & m.eval_set.isin(["test", "shift_api_concise"])]
    jev = jev.assign(eval_set=jev.eval_set.replace({"shift_api_concise": "shift_api"}), n=0, seed=0)
    allr = pd.concat([trained, zs, jev], ignore_index=True)
    allr["tag"] = allr.tag.fillna("")
    keys = ["task", "family", "model", "labels", "n", "tag", "eval_set", "calib"]
    t = allr.groupby(keys + ["metric"]).value.agg(["mean", "std", "count"]).reset_index()
    wide = t.pivot_table(index=keys, columns="metric", values=["mean", "std"]).round(4)
    wide.columns = [f"{c}" if a == "mean" else f"{c}_sd" for a, c in wide.columns]
    wide = wide.reset_index()
    wide.to_csv(rc.OUT / "calibration_full.csv", index=False)

    numbers = {}
    for r in t.itertuples():
        nm = "jev" if r.model == "jev" else (f"{r.model}/{r.tag}" if r.tag else f"{r.model}/n{r.n}")
        numbers[f"{rc.SHORT[r.task]}/{r.eval_set}/{nm}/{r.calib}/{r.metric}"] = dict(mean=rc.r4(r.mean), sd=rc.r4(r.std),
                                                                                    k=int(r.count))
    fam = t[(t.n == t.task.map(TOP)) & (t.calib == "raw") & t.metric.isin(["bbs", "bs_error", "ece"]) & (t.tag == "")]
    fam = fam[fam.family.isin(["encoder", "decoder"])]
    rng = fam.groupby(["task", "eval_set", "metric", "family"])["mean"].agg(["min", "max"]).reset_index()
    rng.to_csv(rc.OUT / "calibration_family_ranges.csv", index=False, float_format="%.4f")
    for r in rng.itertuples():
        numbers[f"family/{rc.SHORT[r.task]}/{r.eval_set}/{r.metric}/{r.family}"] = dict(min=rc.r4(r.min), max=rc.r4(r.max))
    sep = rng.pivot_table(index=["task", "eval_set", "metric"], columns="family", values=["min", "max"])
    sep["decoders_all_worse"] = sep[("min", "decoder")] > sep[("max", "encoder")]
    sep.to_csv(rc.OUT / "calibration_family_separation.csv", float_format="%.4f")

    c = pd.read_csv(rc.ROOT / "results" / "tables" / "cells_ci.csv")
    pdis = c[(c.metric == "pair_disagree") & (c.labels == "gold") & c.tag.isna() & c.eval_set.isin(["test", "shift"])]
    pdis = pdis[(pdis.n == 1000) | (pdis.n == pdis.task.map(TOP))][["task", "eval_set", "family", "model", "n", "k", "mean"]]
    pdis.to_csv(rc.OUT / "seed_disagreement.csv", index=False, float_format="%.4f")
    for r in pdis.itertuples():
        numbers[f"pair_disagree/{rc.SHORT[r.task]}/{r.eval_set}/{r.model}/n{r.n}"] = rc.r4(r.mean)
    rc.save_numbers("calibration", numbers)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 300)
    show = wide[(wide.calib == "raw") & (wide.eval_set.isin(["test", "shift"]))]
    print(show[["task", "model", "labels", "n", "tag", "eval_set", "ece", "ece_mass15", "brier", "bbs", "bs_correct", "bs_error"]].to_string(index=False))
    print(sep.to_string())
    print(pdis.pivot_table(index=["task", "model"], columns=["eval_set", "n"], values="mean").round(3).to_string())
    return 0


if __name__ == "__main__":
    np.seterr(all="ignore")
    raise SystemExit(main())
