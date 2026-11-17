"""Threshold drift restated, and two threshold methods that react to shift (revision 2d, 2f; points 5, 6, 10).

Scope: gold labels at N_max, every model, every seed (the cells of Table 7).

drift          realized FPR at a validation target of 0.05 for raw, temperature, isotonic and split-conformal
               thresholds: seed mean, median, SD and per-seed values, constant-output seeds (D32) flagged;
               the largest gap between the four methods within a run (they are one validation quantile).
exchangeable   the band that exchangeability allows: with n validation negatives, the split-conformal
               threshold flags a population fraction F ~ Beta(k, n + 1 - k), k = floor(alpha (n + 1)), and a
               test set of m negatives then shows Binomial(m, F) / m. Central 95% band per task x eval set.
t3_split       why T3 in distribution exceeds 0.05: the T3 test items are ToxiGen's annotated test split and
               validation comes from its train split. Same procedure inside the validation set (half
               calibrates, half is scored, 200 random halvings) against the test value; test negatives'
               human toxicity ratings and target-group mix against validation.
recal          target-negative recalibration: split-conformal threshold fitted on m in {50, 100, 200} labelled
               negatives drawn from the shifted set; realized FPR and recall on the rest of that set
               (200 draws per run).
prior          prior-shift adjustment for the precision target 0.90: temperature-scaled validation
               probabilities, shifted-set prevalence estimated by EM without labels (Saerens et al. 2002),
               posteriors adjusted to it, threshold = lowest score whose estimated precision is >= 0.90;
               also with the true shifted prevalence (prior_oracle), which separates estimation error from
               the failure of the label-shift assumption itself.

  python analysis/rev_drift.py
"""
from __future__ import annotations

import argparse
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import rev_common as rc
from eval import calibrate, thresholds  # noqa: E402
from eval.conformal import ConformalFPR  # noqa: E402

ALPHA, PREC = 0.05, 0.90
TOP = {"t1_rag_hallucination": 8651, "t2_response_safety": 20000, "t3_implicit_toxicity": 5920}
METHODS = ["raw", "temperature", "isotonic", "conformal"]


def top_runs() -> pd.DataFrame:
    i = rc.index()
    return i[(i.tag == "") & (i.labels == "gold") & (i.n == i.task.map(TOP))].sort_values(["task", "model", "seed"])


def labels(task: str, es: str) -> np.ndarray:
    return rc.hl.items(task, es)[1].astype(int)


def collapsed(d: str) -> bool:
    return float(np.std(rc.run(d)["test"])) < 0.01


def drift() -> tuple[pd.DataFrame, pd.DataFrame]:
    m = pd.read_parquet(rc.ROOT / "results" / "metrics.parquet")
    t = top_runs()
    m = m[m.run_id.isin([rc.Path(d).name for d in t.dir]) & m.eval_set.isin(["test", "shift"])]
    col = {rc.Path(d).name: collapsed(d) for d in t.dir}
    f = m[m.metric == "at_fpr05_fpr"].copy()
    f["collapsed"] = f.run_id.map(col)
    rows = []
    for (task, family, model, es, calib), g in f.groupby(["task", "family", "model", "eval_set", "calib"]):
        g = g.sort_values("seed")
        ok = g[~g.collapsed]
        rows.append(dict(task=task, family=family, model=model, eval_set=es, method=calib, k=len(g),
                         mean=g.value.mean(), median=g.value.median(), sd=g.value.std(ddof=1),
                         per_seed=";".join(f"{v:.3f}{'*' if c else ''}" for v, c in zip(g.value, g.collapsed)),
                         n_collapsed=int(g.collapsed.sum()), mean_without_collapsed=ok.value.mean() if len(ok) else np.nan))
    fpr = pd.DataFrame(rows)
    spread = f.pivot_table(index=["task", "model", "seed", "eval_set"], columns="calib", values="value")
    spread = (spread[METHODS].max(axis=1) - spread[METHODS].min(axis=1)).rename("max_gap_between_methods").reset_index()
    p = m[m.metric.isin(["at_prec90_precision", "at_prec90_recall", "at_prec90_flag_rate"])]
    prec = p.pivot_table(index=["task", "family", "model", "eval_set", "calib"], columns="metric", values="value",
                         aggfunc=["mean", "median", "std"]).round(4)
    prec.columns = [f"{a}_{b.replace('at_prec90_', '')}" for a, b in prec.columns]
    return fpr, spread, prec.reset_index()


def band(n_cal: int, m_test: int, alpha: float = ALPHA, draws: int = 200_000, seed: int = 0) -> tuple[float, float, float]:
    k = int(np.floor(alpha * (n_cal + 1)))
    rng = np.random.default_rng(seed)
    f = rng.beta(k, n_cal + 1 - k, draws)
    x = rng.binomial(m_test, f) / m_test
    return float(k / (n_cal + 1)), float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))


def exchangeable(fpr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task in rc.TASKS:
        yv = rc.run(top_runs()[top_runs().task == task].dir.iloc[0])["val"][0]
        n_cal = int((yv == 0).sum())
        for es in ("test", "shift"):
            m_test = int((labels(task, es) == 0).sum())
            mean, lo, hi = band(n_cal, m_test, seed=zlib.crc32(f"{task}/{es}".encode()))
            g = fpr[(fpr.task == task) & (fpr.eval_set == es) & (fpr.method == "conformal")]
            seeds = [float(v.rstrip("*")) for s in g.per_seed for v in s.split(";")]
            rows.append(dict(task=task, eval_set=es, n_val_neg=n_cal, n_test_neg=m_test, expected=mean, band_lo=lo,
                             band_hi=hi, models=len(g), models_above=int((g["mean"] > hi).sum()),
                             seeds=len(seeds), seeds_above=int(sum(v > hi for v in seeds)),
                             model_min=g["mean"].min(), model_max=g["mean"].max()))
    return pd.DataFrame(rows)


def t3_split(reps: int = 200) -> tuple[pd.DataFrame, dict]:
    task = "t3_implicit_toxicity"
    d = pd.read_parquet(rc.ROOT / "data" / "processed" / f"{task}.parquet")
    neg = d[d.label == 0]
    tv = 0.5 * (neg[neg.split == "test"].group.value_counts(normalize=True)
                .sub(neg[neg.split == "val"].group.value_counts(normalize=True), fill_value=0).abs().sum())
    info = dict(tox_neg_val=float(neg[neg.split == "val"].toxicity_human.mean()),
                tox_neg_test=float(neg[neg.split == "test"].toxicity_human.mean()),
                tox_neg_pool=float(neg[neg.split == "pool"].toxicity_human.mean()),
                group_tv_neg=float(tv), prev_val=float(d[d.split == "val"].label.mean()),
                prev_test=float(d[d.split == "test"].label.mean()))
    rows = []
    rng = np.random.default_rng(3)
    for r in top_runs()[top_runs().task == task].itertuples():
        x = rc.run(r.dir)
        yv, sv = x["val"]
        nv = sv[yv == 0]
        within = []
        for _ in range(reps):
            perm = rng.permutation(len(nv))
            cal, ev = nv[perm[: len(nv) // 2]], nv[perm[len(nv) // 2:]]
            cp = ConformalFPR().fit(cal, np.zeros(len(cal), int))
            within.append(cp.flag(ev, ALPHA).mean())
        cp = ConformalFPR().fit(sv, yv)
        yt = labels(task, "test")
        rows.append(dict(model=r.model, seed=r.seed, within_val_fpr=float(np.mean(within)),
                         test_fpr=float(cp.flag(x["test"][yt == 0], ALPHA).mean())))
    return pd.DataFrame(rows), info


def recal_job(args) -> list[dict]:
    d, task, family, model, seed, draws = args
    x = rc.run(d)
    yv, sv = x["val"]
    y, s = labels(task, "shift"), x["shift"]
    neg, pos = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    base = ConformalFPR().fit(sv, yv).flag(s, ALPHA)
    rows = [dict(task=task, family=family, model=model, seed=seed, m=0, fpr=float(base[neg].mean()),
                 fpr_p05=np.nan, fpr_p95=np.nan, recall=float(base[pos].mean()), p_over_target=np.nan)]
    rng = np.random.default_rng(zlib.crc32(d.encode()))
    for m in (50, 100, 200):
        if m >= len(neg):
            continue
        f, r = [], []
        for _ in range(draws):
            pick = rng.choice(neg, m, replace=False)
            rest = np.setdiff1d(neg, pick)
            flag = ConformalFPR().fit(s[pick], np.zeros(m, int)).flag(s, ALPHA)
            f.append(flag[rest].mean())
            r.append(flag[pos].mean())
        f = np.array(f)
        rows.append(dict(task=task, family=family, model=model, seed=seed, m=m, fpr=float(f.mean()),
                         fpr_p05=float(np.quantile(f, 0.05)), fpr_p95=float(np.quantile(f, 0.95)),
                         recall=float(np.mean(r)), p_over_target=float((f > ALPHA).mean())))
    return rows


def saerens(p: np.ndarray, prior_train: float, iters: int = 200, tol: float = 1e-8) -> tuple[float, np.ndarray]:
    """EM estimate of the new prevalence and the adjusted posteriors (Saerens, Latinne and Decaestecker 2002)."""
    pi = prior_train
    p = np.clip(p, 1e-6, 1 - 1e-6)
    for _ in range(iters):
        a = (pi / prior_train) * p
        b = ((1 - pi) / (1 - prior_train)) * (1 - p)
        q = a / (a + b)
        new = float(q.mean())
        if abs(new - pi) < tol:
            pi = new
            break
        pi = new
    return pi, q


def threshold_est_precision(s: np.ndarray, q: np.ndarray, target: float) -> float:
    """Lowest score t whose estimated precision mean(q | s >= t) is >= target (inf if none)."""
    order = np.argsort(-s, kind="stable")
    ss, qs = s[order], q[order]
    prec = np.cumsum(qs) / np.arange(1, len(qs) + 1)
    last = np.r_[ss[1:] != ss[:-1], True]
    ok = np.flatnonzero(last & (prec >= target))
    return float(ss[ok.max()]) if len(ok) else float("inf")


def prior_job(args) -> list[dict]:
    d, task, family, model, seed = args
    x = rc.run(d)
    yv, sv = x["val"]
    cal = calibrate.fit("temperature", sv, yv)
    pv = cal(sv)
    prior = float(yv.mean())
    t_val = thresholds.threshold_for_precision(yv, pv, PREC)
    rows = []
    for es in ("test", "shift"):
        y, p = labels(task, es), cal(x[es])
        pi_hat, q = saerens(p, prior)
        t_adj = threshold_est_precision(p, q, PREC)
        pi = float(y.mean())
        pc = np.clip(p, 1e-6, 1 - 1e-6)
        a_, b_ = (pi / prior) * pc, ((1 - pi) / (1 - prior)) * (1 - pc)
        t_oracle = threshold_est_precision(p, a_ / (a_ + b_), PREC)
        for method, t in (("validation", t_val), ("prior_adjusted", t_adj), ("prior_oracle", t_oracle)):
            r = thresholds.drift(y, p, t)
            rows.append(dict(task=task, family=family, model=model, seed=seed, eval_set=es, method=method,
                             prev_val=prior, prev_true=float(y.mean()), prev_est=pi_hat, precision=r["precision"],
                             recall=r["recall"], flag_rate=r["flag_rate"]))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=200)
    a = ap.parse_args()
    rc.OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 300)
    numbers = {}

    fpr, spread, prec = drift()
    fpr.to_csv(rc.OUT / "drift_fpr.csv", index=False, float_format="%.4f")
    spread.to_csv(rc.OUT / "drift_method_spread.csv", index=False, float_format="%.4f")
    prec.to_csv(rc.OUT / "drift_precision_target.csv", index=False, float_format="%.4f")
    for r in fpr.itertuples():
        numbers[f"fpr/{rc.SHORT[r.task]}/{r.eval_set}/{r.model}/{r.method}"] = dict(
            mean=rc.r4(r.mean), median=rc.r4(r.median), sd=rc.r4(r.sd), per_seed=r.per_seed,
            n_collapsed=r.n_collapsed, mean_without_collapsed=rc.r4(r.mean_without_collapsed))
    numbers["method_spread"] = {es: dict(median=rc.r4(g.max_gap_between_methods.median()),
                                         p90=rc.r4(g.max_gap_between_methods.quantile(0.9)),
                                         max=rc.r4(g.max_gap_between_methods.max()))
                                for es, g in spread.groupby("eval_set")}
    print(fpr[fpr.method == "conformal"][["task", "eval_set", "model", "mean", "median", "sd", "per_seed"]].to_string(index=False))
    print(numbers["method_spread"])

    ex = exchangeable(fpr)
    ex.to_csv(rc.OUT / "drift_exchangeability_band.csv", index=False, float_format="%.4f")
    for r in ex.itertuples():
        numbers[f"band/{rc.SHORT[r.task]}/{r.eval_set}"] = {k: (rc.r4(v) if isinstance(v, float) else v)
                                                           for k, v in r._asdict().items() if k not in ("Index", "task", "eval_set")}
    print(ex.to_string(index=False))

    t3, info = t3_split()
    t3.to_csv(rc.OUT / "drift_t3_within_val.csv", index=False, float_format="%.4f")
    numbers["t3_split"] = {**{k: rc.r4(v) for k, v in info.items()},
                           "within_val_fpr_mean": rc.r4(t3.within_val_fpr.mean()),
                           "within_val_fpr_range": f"{t3.within_val_fpr.min():.3f}-{t3.within_val_fpr.max():.3f}",
                           "test_fpr_mean": rc.r4(t3.test_fpr.mean()),
                           "test_fpr_range": f"{t3.test_fpr.min():.3f}-{t3.test_fpr.max():.3f}"}
    print(numbers["t3_split"])

    runs = top_runs()
    jobs = [(r.dir, r.task, r.family, r.model, int(r.seed)) for r in runs.itertuples()]
    with ProcessPoolExecutor() as ex_:
        recal = pd.DataFrame([x for part in ex_.map(recal_job, [j + (a.draws,) for j in jobs]) for x in part])
        prior = pd.DataFrame([x for part in ex_.map(prior_job, jobs) for x in part])
    recal.to_csv(rc.OUT / "recal_target_negatives.csv", index=False, float_format="%.4f")
    prior.to_csv(rc.OUT / "prior_shift_precision.csv", index=False, float_format="%.4f")
    rs = recal.groupby(["task", "model", "m"])[["fpr", "fpr_p05", "fpr_p95", "recall", "p_over_target"]].mean().reset_index()
    ps = prior.groupby(["task", "eval_set", "model", "method"])[["prev_val", "prev_true", "prev_est", "precision", "recall",
                                                                 "flag_rate"]].mean().reset_index()
    rs.to_csv(rc.OUT / "recal_target_negatives_summary.csv", index=False, float_format="%.4f")
    ps.to_csv(rc.OUT / "prior_shift_precision_summary.csv", index=False, float_format="%.4f")
    for r in rs.itertuples():
        numbers[f"recal/{rc.SHORT[r.task]}/{r.model}/m{r.m}"] = dict(
            fpr=rc.r4(r.fpr), fpr_p05=rc.r4(r.fpr_p05), fpr_p95=rc.r4(r.fpr_p95), recall=rc.r4(r.recall),
            p_over_target=rc.r4(r.p_over_target))
    for r in ps.itertuples():
        numbers[f"prior/{rc.SHORT[r.task]}/{r.eval_set}/{r.model}/{r.method}"] = dict(
            prev_val=rc.r4(r.prev_val), prev_true=rc.r4(r.prev_true), prev_est=rc.r4(r.prev_est),
            precision=rc.r4(r.precision), recall=rc.r4(r.recall), flag_rate=rc.r4(r.flag_rate))
    print(rs.round(3).to_string(index=False))
    print(ps.round(3).to_string(index=False))
    rc.save_numbers("drift", numbers)
    return 0


if __name__ == "__main__":
    np.seterr(all="ignore")
    raise SystemExit(main())
