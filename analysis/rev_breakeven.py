"""Break-even budgets with deployable thresholds and their uncertainty (revision 2a, 2c; review points 1, 4).

For every task x eval set (test, shift_api) x model with a gold-label curve, and three F1 variants
(f1_valthr = threshold chosen on the run's validation set, f1_at_05, max_f1 = the original oracle metric):
  * N*_mean and N*_var as in eval/breakeven.py (sd_ref = 0 until the hosted repeats of Phase 3a exist),
    against the better and the weaker frontier-tier hosted model; for decoders also with the zero-shot
    run (concise prompt) as the point N = 0 of the curve;
  * the two-level bootstrap distribution of N*_mean (seeds and items resampled, the reference rescored on
    the same items): probability the sweep reaches the reference at all, probability N* <= the point
    estimate, and a 90% interval over the budget grid;
  * per budget, the probability that the seed mean is at or above the reference and the paired difference CI;
  * the always-positive F1 floor of each eval set beside each reference.
Also: budget-honest runs (abl-honest, validation carved from D_n) against the main runs' f1_valthr.

  python analysis/rev_breakeven.py [--b 2000]
"""
from __future__ import annotations

import argparse
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import rev_common as rc

SETS = ("test", "shift_api")
INF = np.inf


def sd_seed(task, es, model, n, metric) -> int:
    return zlib.crc32(f"{task}/{es}/{model}/{n}/{metric}".encode())


def two_level_mean_sd(r: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    k, cols = r.shape
    idx = rc.seed_draws(k, cols - 1, seed)
    v = r[idx, np.arange(cols)[:, None]]
    return v.mean(1), (v.std(1, ddof=1) if k > 1 else np.zeros(cols))


def first_sustained(ns: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """ok [levels x reps] -> first N from which ok holds at every larger N (inf if never)."""
    suffix = np.flip(np.logical_and.accumulate(np.flip(ok, 0), 0), 0)
    has = suffix.any(0)
    first = np.argmax(suffix, 0)
    return np.where(has, ns[first].astype(float), INF)


def ref_sd_map() -> dict[tuple[str, str, str], float]:
    """Run-to-run SD of frontier hosted references from Phase 3a (0.0 if repeats not yet run)."""
    p = rc.OUT / "ref_repeat_sd.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    return {(r.task, r.eval_set, r.model): float(r.sd_ref) for r in df.itertuples()}


def model_job(args):
    task, es, family, model, b, sd_map = args
    ch = rc.weight_chunks(task, es, b)
    _, y = rc.hl.items(task, es)
    refs = {kind: rc.ref_model(task, es, weaker=(kind == "weaker")) for kind in ("better", "weaker")}
    i = rc.index()
    ns = sorted(i[(i.task == task) & (i.model == model) & (i.labels == "gold") & (i.tag == "")].n.unique())
    zs = rc.cell_runs(task, family, model, "none", 0, "zeroshot") if family == "decoder" else []
    curve_rows, nstar_rows = [], []
    for metric in rc.F1S:
        ref = {k: rc.item_reps([(rc.api_scores(task, m, es), 0.5)], task, es, b, metric, ch)[0] for k, m in refs.items()}
        mean, sd = {}, {}
        for n in ns:
            r = rc.item_reps(rc.per_seed(rc.cell_runs(task, family, model, "gold", n), es), task, es, b, metric, ch)
            mean[n], sd[n] = two_level_mean_sd(r, sd_seed(task, es, model, n, metric))
        if zs:
            r = rc.item_reps(rc.per_seed(zs, es), task, es, b, metric, ch)
            mean[0], sd[0] = r[0], np.zeros(r.shape[1])
        for n in sorted(mean):
            for kind, rv in ref.items():
                d = mean[n] - rv
                curve_rows.append(dict(task=task, eval_set=es, family=family, model=model, metric=metric, n=n,
                                       ref_kind=kind, ref=refs[kind], q_ref=float(rv[0]), q=float(mean[n][0]),
                                       q_lo=rc.ci(mean[n])[1], q_hi=rc.ci(mean[n])[2], seed_sd=float(sd[n][0]),
                                       diff=float(d[0]), diff_lo=rc.ci(d)[1], diff_hi=rc.ci(d)[2],
                                       p_ge_ref=float((d[1:] >= 0).mean())))
        for with_zs in ([False, True] if zs else [False]):
            grid = np.array([n for n in sorted(mean) if with_zs or n > 0])
            m_ = np.vstack([mean[n] for n in grid])
            s_ = np.vstack([sd[n] for n in grid])
            for kind, rv in ref.items():
                sd_ref = sd_map.get((task, es, refs[kind]), 0.0)
                star = first_sustained(grid, m_ >= rv[None, :])
                star_var0 = first_sustained(grid, (m_ - s_) >= rv[None, :])
                star_var = first_sustained(grid, (m_ - s_) >= (rv[None, :] + sd_ref))
                bs, point = star[1:], star[0]
                lo, hi = np.percentile(bs, [5, 95], method="lower"), np.percentile(bs, [5, 95], method="higher")
                nstar_rows.append(dict(
                    task=task, eval_set=es, family=family, model=model, metric=metric, ref_kind=kind, ref=refs[kind],
                    q_ref=float(rv[0]), sd_ref=sd_ref, floor=rc.floor(task, es), with_zeroshot=with_zs,
                    n_max=int(grid.max()), n_star_mean=point, n_star_var=star_var[0],
                    n_star_var_sdref0=star_var0[0], p_reached=float(np.isfinite(bs).mean()),
                    p_le_point=float((bs <= point).mean()) if np.isfinite(point) else float("nan"),
                    p_eq_point=float((bs == point).mean()), n_star_p05=lo[0], n_star_p95=hi[1],
                    q_at_nmax=float(mean[grid.max()][0]),
                    p_ge_ref_at_nmax=float((mean[grid.max()][1:] >= rv[1:]).mean())))
    return curve_rows, nstar_rows


def honest(b: int) -> pd.DataFrame:
    """Budget-honest deployable F1 (threshold chosen on 20% of D_n) against the main runs' f1_valthr and f1_at_05."""
    rows = []
    i = rc.index()
    h = i[i.tag == "abl-honest"]
    for (task, family, model, n), _ in h.groupby(["task", "family", "model", "n"]):
        for es in ("test", "shift_api"):
            ch = rc.weight_chunks(task, es, b)
            a = rc.cell_seeds(task, family, model, "gold", n, "abl-honest")
            m = rc.cell_seeds(task, family, model, "gold", n)
            seeds = sorted(set(a) & set(m))
            ra = rc.item_reps(rc.per_seed([a[k] for k in seeds], es), task, es, b, "f1_valthr", ch)
            for mm in ("f1_valthr", "f1_at_05"):
                rm = rc.item_reps(rc.per_seed([m[k] for k in seeds], es), task, es, b, mm, ch)
                idx = rc.seed_draws(len(seeds), ra.shape[1] - 1, sd_seed(task, es, model, n, "honest" + mm))
                col = np.arange(ra.shape[1])[:, None]
                d = ra[idx, col].mean(1) - rm[idx, col].mean(1)
                rows.append(dict(task=task, eval_set=es, family=family, model=model, n=n, k=len(seeds),
                                 against=mm, honest=float(ra[:, 0].mean()), main=float(rm[:, 0].mean()),
                                 diff=float(d[0]), lo=rc.ci(d)[1], hi=rc.ci(d)[2]))
    return pd.DataFrame(rows)


def fmt_n(x) -> str:
    return "not reached" if not np.isfinite(x) else str(int(x))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int, default=2000)
    a = ap.parse_args()
    sd_map = ref_sd_map()
    i = rc.index()
    g = i[(i.tag == "") & (i.labels == "gold")][["task", "family", "model"]].drop_duplicates()
    jobs = [(t, es, f, m, a.b, sd_map) for t, f, m in g.itertuples(index=False) for es in SETS]
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(model_job, jobs))
    curve = pd.DataFrame([r for c, _ in res for r in c])
    nstar = pd.DataFrame([r for _, s in res for r in s])
    rc.OUT.mkdir(parents=True, exist_ok=True)
    curve.to_csv(rc.OUT / "breakeven_curves.csv", index=False, float_format="%.4f")
    nstar.to_csv(rc.OUT / "breakeven.csv", index=False, float_format="%.4f")
    hon = honest(a.b)
    hon.to_csv(rc.OUT / "breakeven_honest.csv", index=False, float_format="%.4f")

    note = "two-level bootstrap (seeds x items); sd_ref from Phase 3a repeats" if sd_map else "two-level bootstrap (seeds x items); sd_ref = 0"
    numbers = {"meta": dict(bootstrap_resamples=a.b, note=note)}
    for r in nstar.itertuples():
        key = (f"{rc.SHORT[r.task]}/{r.eval_set}/{r.model}/{r.metric}/{r.ref_kind}"
               + ("/zs" if r.with_zeroshot else ""))
        numbers[key] = dict(q_ref=rc.r4(r.q_ref), sd_ref=rc.r4(r.sd_ref), floor=rc.r4(r.floor),
                            n_star_mean=fmt_n(r.n_star_mean), n_star_var=fmt_n(r.n_star_var),
                            n_star_var_sdref0=fmt_n(r.n_star_var_sdref0),
                            p_reached=rc.r4(r.p_reached), p_le_point=rc.r4(r.p_le_point),
                            n_star_p05=fmt_n(r.n_star_p05), n_star_p95=fmt_n(r.n_star_p95),
                            q_at_nmax=rc.r4(r.q_at_nmax), p_ge_ref_at_nmax=rc.r4(r.p_ge_ref_at_nmax))
    for r in hon.itertuples():
        numbers[f"honest/{rc.SHORT[r.task]}/{r.eval_set}/{r.model}/n{r.n}/vs_{r.against}"] = dict(
            honest=rc.r4(r.honest), main=rc.r4(r.main), diff=rc.r4(r.diff), lo=rc.r4(r.lo), hi=rc.r4(r.hi))
    rc.save_numbers("breakeven", numbers)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    view = nstar[(nstar.ref_kind == "better") & ~nstar.with_zeroshot].pivot_table(
        index=["task", "eval_set", "model"], columns="metric", values="n_star_mean", aggfunc="first")
    print(view.to_string())
    return 0


if __name__ == "__main__":
    np.seterr(all="ignore")
    raise SystemExit(main())
