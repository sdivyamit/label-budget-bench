"""Table V (RQ2, plan §5 V-B, §6.5): what teacher labels are worth against gold.

  (a) teacher quality: accuracy, precision, recall, F1 of each teacher's training labels against gold
      (train/common.py teacher_labels, so a refused item takes the task's refusal_label, D25) on the pool
      items it labelled, on the fixed 5k sub-pool (= the N = 5,000 training set of every seed, D1) and on
      the test / shifted items it labelled;
  (b) plateau gap: gold minus teacher-label PR-AUC at N = 5,000 and at the top N, with the paired 95% CI
      and two-sided p from the same item resamples as analysis/headline.py;
  (c) exchange rate: q(N) = a - b N^-c fitted by least squares to the gold seed-mean PR-AUC at the
      log-spaced N levels, per task x model (4+ levels); a teacher-labelled cell's gold-equivalent N is the
      fitted curve's inverse at its PR-AUC (infinite at or above the asymptote a). The CI refits the curve
      on the first --b-fit of the same item resamples (paired with the teacher cell). a_at_bound marks a
      curve still rising at the top N (a pinned at 1): its inverse is only trustworthy inside the sweep;
  (d) precision / recall / flag rate at 0.5 (raw scores) of teacher- vs gold-trained models at N = 5,000,
      seed means with the paired CI of the difference.
Main-sweep cells of the trained families only; hosted models never train on labels.

Writes results/tables/table5a_teacher_quality.csv, table5b_plateau_gap.csv, table5c_gold_fits.csv,
table5c_exchange_rate.csv, table5d_pr_skew.csv and a "label_source" block in results/paper_numbers.json
(run headline.py first).

  python analysis/label_source.py [--b 10000] [--b-fit 1000]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "train"))
import common  # noqa: E402
import headline as hl  # noqa: E402
from eval import fastboot as fb, metrics  # noqa: E402

OUT = ROOT / "results"
TEACHERS = ("open_teacher", "frontier_teacher")
SOURCES = TEACHERS + ("mix250_open_teacher", "mix1000_open_teacher", "mix250_frontier_teacher",
                      "mix1000_frontier_teacher")
FAMILIES = ("classical", "encoder", "decoder")
SETS = ("test", "shift")
METRIC = "pr_auc"
N_SUB = 5000
B_FIT = 1000
MIN_LEVELS = 4
BOUNDS = ([0.0, 0.0, 0.01], [1.0, np.inf, 3.0])


def attempted(task: str, teacher: str) -> set:
    if teacher == "open_teacher":
        path = ROOT / common.TRAIN_CFG["label_sources"]["open_teacher"]["path"].format(task=task)
        return set(pd.read_parquet(path, columns=["uid"]).uid)
    return {r["uid"] for r in common._frontier_records(task)}


def complete(task: str, teacher: str) -> bool:
    if teacher == "open_teacher":
        path = ROOT / common.TRAIN_CFG["label_sources"]["open_teacher"]["path"].format(task=task)
        return path.with_suffix(".done").exists()
    return (ROOT / "preds" / "api" / task / "frontier_teacher.done").exists()


def teacher_quality() -> pd.DataFrame:
    rows = []
    for task in common.TASKS_CFG:
        d = common.load_all(task)[["uid", "split", "label"]]
        ch = pd.read_parquet(ROOT / "data" / "splits" / task / "chains.parquet", columns=["uid", "in_subpool"])
        sub = set(ch[ch.in_subpool].uid)
        for teacher in TEACHERS:
            try:
                lab, tried, refused = (common.teacher_labels(task, teacher), attempted(task, teacher),
                                       common.refused_uids(task, teacher))
            except FileNotFoundError as e:
                rows.append(dict(task=task, teacher=teacher, scope="", note=f"teacher file missing: {e.filename}"))
                continue
            g = d[d.uid.isin(tried)].assign(teacher=lambda x: x.uid.map(lab))
            for scope, m in (("pool", g.split == "pool"), ("subpool", g.uid.isin(sub)),
                             ("test", g.split == "test"), ("shift", g.split == "shift")):
                h = g[m]
                if h.empty:
                    continue
                ok = h[h.teacher.notna()]
                r = metrics.confusion_rates(ok.label, ok.teacher.astype(int) == 1)
                rows.append(dict(task=task, teacher=teacher, scope=scope, n_items=len(h),
                                 n_unlabelled=int(h.teacher.isna().sum()), n_refused=int(h.uid.isin(refused).sum()),
                                 refusal_label=common.TASKS_CFG[task].get("refusal_label") if refused else None,
                                 gold_pos_rate=ok.label.mean(), teacher_pos_rate=r["flag_rate"],
                                 accuracy=r["accuracy"], precision=r["precision"], recall=r["recall"], f1=r["f1"],
                                 complete=complete(task, teacher), note=""))
    return pd.DataFrame(rows)


def sat(x, a, b, c):
    return a - b * np.power(x, -c)


def fit(n: np.ndarray, q: np.ndarray, p0=None) -> np.ndarray:
    """(a, b, c) of q = a - b (N/1000)^-c."""
    x = n / 1000.0
    if p0 is None:
        a0 = min(1.0, q.max() + 0.02)
        p0 = (a0, max(a0 - q.min(), 1e-3) * x.min() ** 0.5, 0.5)
    return curve_fit(sat, x, q, p0=p0, bounds=BOUNDS, maxfev=10000)[0]


def n_equiv(p, q):
    a, b, c = p
    q = np.asarray(q, float)
    with np.errstate(all="ignore"):
        return np.where(q < a, 1000.0 * (b / np.maximum(a - q, 1e-12)) ** (1.0 / c), np.inf)


def ci(r: np.ndarray) -> tuple[float, float]:
    r = r[~np.isnan(r)]
    if not len(r):
        return float("nan"), float("nan")
    lo, hi = np.quantile(r, [0.025, 0.975], method="inverted_cdf")
    return float(lo), float(hi)


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    """a - b on shared resamples (column 0 = full eval set): difference, percentile CI, two-sided p."""
    d = a - b
    r = d[1:][~np.isnan(d[1:])]
    lo, hi = np.percentile(r, [2.5, 97.5]) if len(r) else (np.nan, np.nan)
    p = float(min(1.0, 2 * min((r <= 0).mean(), (r >= 0).mean()))) if len(r) else float("nan")
    return dict(diff=float(d[0]), lo=float(lo), hi=float(hi), p=p)


def stacked(cells: dict, task: str, es: str, b: int, metric: str = METRIC) -> dict:
    """cell -> [b+1] seed-mean replicates; every cell scored on the same resamples in one pass."""
    flat, where = [], {}
    for key, per_seed in cells.items():
        if per_seed:
            where[key] = slice(len(flat), len(flat) + len(per_seed))
            flat += per_seed
    if not flat:
        return {}
    reps = hl.replicates(flat, task, es, b, metrics=(metric,))[metric]
    return {key: reps[s].mean(0) for key, s in where.items()}


def rates_at_half(cells: dict, task: str, es: str, b: int) -> dict:
    """cell -> {precision, recall, flag_rate: [b+1] seed-mean replicates} of the verdicts p >= 0.5."""
    _, y = hl.items(task, es)
    v = {k: np.vstack([p >= 0.5 for p, _ in ps]).astype(np.float32) for k, ps in cells.items() if ps}
    out = {k: defaultdict(list) for k in v}
    for w in fb.chunks(len(y), b, hl.boot_seed(task, es)):
        pos, tot = w @ y, w.sum(1)
        for k, pred in v.items():
            tp, pp = w @ (pred * y).T, w @ pred.T
            with np.errstate(all="ignore"):
                out[k]["precision"].append((tp / np.where(pp > 0, pp, np.nan)).mean(1))
            out[k]["recall"].append((tp / pos[:, None]).mean(1))
            out[k]["flag_rate"].append((pp / tot[:, None]).mean(1))
    return {k: {m: np.concatenate(r) for m, r in d.items()} for k, d in out.items()}


def model_job(args) -> dict:
    (task, family, model), dirs, b, b_fit = args
    scores = {key: [hl.run_scores(d, task) for d in ds] for key, ds in dirs.items()}
    gold_levels = sorted(n for src, n in scores if src == "gold")
    top = max(gold_levels) if gold_levels else None
    base = dict(task=task, family=family, model=model)
    gap, skew, fits, exch = [], [], [], []
    for es in SETS:
        per_seed = {key: [s[es] for s in ss if es in s] for key, ss in scores.items()}

        at = {key: ps for key, ps in per_seed.items() if key[1] in (N_SUB, top) and key[0] in ("gold",) + SOURCES}
        reps = stacked(at, task, es, b)
        for n_ref in sorted({N_SUB, top} - {None}):
            g = reps.get(("gold", n_ref))
            if g is None:
                continue
            for src in SOURCES:
                t = reps.get((src, n_ref))
                if t is not None:
                    gap.append(dict(base, eval_set=es, n=n_ref, labels=src, k_gold=len(per_seed[("gold", n_ref)]),
                                    gold=float(g[0]), k_teacher=len(per_seed[(src, n_ref)]), teacher=float(t[0]),
                                    **{("gap" if k == "diff" else k): v for k, v in paired(g, t).items()}))

        at5 = {key: ps for key, ps in per_seed.items() if key[1] == N_SUB and key[0] in ("gold",) + SOURCES}
        rr = rates_at_half(at5, task, es, b)
        if ("gold", N_SUB) in rr:
            g = rr[("gold", N_SUB)]
            for (src, _), t in sorted(rr.items(), key=lambda kv: kv[0][0] != "gold"):
                row = dict(base, eval_set=es, n=N_SUB, labels=src, k=len(at5[(src, N_SUB)]))
                for m in ("precision", "recall", "flag_rate"):
                    row[m] = float(t[m][0])
                    if src != "gold":
                        d = paired(t[m], g[m])
                        row.update({f"d_{m}": d["diff"], f"d_{m}_lo": d["lo"], f"d_{m}_hi": d["hi"]})
                skew.append(row)

        levels = [n for n in gold_levels if per_seed.get(("gold", n))]
        fr = dict(base, eval_set=es, metric=METRIC, n_levels=len(levels), levels=",".join(map(str, levels)))
        if len(levels) < MIN_LEVELS:
            fits.append(dict(fr, status=f"not fitted: {len(levels)} N levels"))
            continue
        cells = {("gold", n): per_seed[("gold", n)] for n in levels}
        cells.update({key: ps for key, ps in per_seed.items() if key[0] in SOURCES})
        rep = stacked(cells, task, es, b_fit)
        n_arr = np.array(levels, float)
        q_rep = np.vstack([rep[("gold", n)] for n in levels])            # levels x (b_fit+1)
        try:
            p_hat = fit(n_arr, q_rep[:, 0])
        except (RuntimeError, ValueError) as e:
            fits.append(dict(fr, status=f"fit failed: {e}"))
            continue
        resid = q_rep[:, 0] - sat(n_arr / 1000.0, *p_hat)
        ss_tot = ((q_rep[:, 0] - q_rep[:, 0].mean()) ** 2).sum()
        r2 = float(1 - (resid ** 2).sum() / ss_tot) if ss_tot > 0 else float("nan")
        params = np.full((q_rep.shape[1], 3), np.nan)
        params[0] = p_hat
        for j in range(1, q_rep.shape[1]):
            if np.isfinite(q_rep[:, j]).all():
                try:
                    params[j] = fit(n_arr, q_rep[:, j], p0=p_hat)
                except (RuntimeError, ValueError):
                    pass
        failed = float(np.isnan(params[1:, 0]).mean())
        a_lo, a_hi = ci(params[1:, 0])
        fits.append(dict(fr, a=p_hat[0], a_lo=a_lo, a_hi=a_hi, b=p_hat[1], c=p_hat[2], r2=r2,
                         rmse=float(np.sqrt((resid ** 2).mean())), max_abs_resid=float(np.abs(resid).max()),
                         a_at_bound=bool(p_hat[0] >= BOUNDS[1][0] - 1e-3), refit_failed_share=failed,
                         status="ok" if r2 >= 0.9 else "poor fit (R2 < 0.9)"))
        for (src, n), t in sorted(rep.items()):
            if src == "gold":
                continue
            eq = n_equiv(p_hat, t[0])[()]
            eq_r = np.array([n_equiv(params[j], t[j])[()] if np.isfinite(params[j, 0]) else np.nan
                             for j in range(1, len(t))])
            lo, hi = ci(eq_r)
            flag = ("at or above gold asymptote" if not np.isfinite(eq) else
                    "below sweep range (extrapolated)" if eq < min(levels) else
                    "beyond sweep range (extrapolated)" if eq > max(levels) else "")
            g = rep.get(("gold", n))
            exch.append(dict(base, eval_set=es, labels=src, n=n, k=len(per_seed[(src, n)]), q=float(t[0]),
                             gold_q_same_n=float(g[0]) if g is not None else float("nan"),
                             n_gold_equiv=float(eq), lo=lo, hi=hi, exchange=float(eq / n),
                             exchange_lo=lo / n, exchange_hi=hi / n,
                             share_above_asymptote=float(np.isinf(eq_r).mean()), flag=flag,
                             fit_status=fits[-1]["status"]))
    return dict(gap=gap, skew=skew, fits=fits, exch=exch)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int, default=hl.B)
    ap.add_argument("--b-fit", type=int, default=B_FIT)
    a = ap.parse_args()
    tq = teacher_quality()
    idx = hl.run_index()
    idx = idx[(idx.tag == "") & idx.family.isin(FAMILIES) & idx.labels.isin(("gold",) + SOURCES)]
    jobs = [(key, {(src, int(n)): h.dir.tolist() for (src, n), h in g.groupby(["labels", "n"])}, a.b, a.b_fit)
            for key, g in idx.groupby(["task", "family", "model"])]
    with ProcessPoolExecutor() as ex:
        parts = list(ex.map(model_job, jobs))
    tabs = {k: pd.DataFrame([r for p in parts for r in p[k]]) for k in ("gap", "skew", "fits", "exch")}
    q = tq[tq.scope == "subpool"].set_index(["task", "teacher"])[["precision", "recall", "teacher_pos_rate"]]
    sk = tabs["skew"]
    if len(sk):
        teacher = sk.labels.str.replace(r"^mix\d+_", "", regex=True)
        for col, src in (("teacher_label_precision", "precision"), ("teacher_label_recall", "recall"),
                         ("teacher_label_pos_rate", "teacher_pos_rate")):
            sk[col] = [q[src].get((t, s), np.nan) if l != "gold" else np.nan
                       for t, s, l in zip(sk.task, teacher, sk.labels)]
    tab = OUT / "tables"
    tab.mkdir(parents=True, exist_ok=True)
    outs = {"table5a_teacher_quality": tq, "table5b_plateau_gap": tabs["gap"], "table5c_gold_fits": tabs["fits"],
            "table5c_exchange_rate": tabs["exch"], "table5d_pr_skew": sk}
    for name, t in outs.items():
        t.to_csv(tab / f"{name}.csv", index=False, float_format="%.4f")
    f = OUT / "paper_numbers.json"
    numbers = json.loads(f.read_text()) if f.exists() else {}
    numbers["label_source"] = dict(
        metric=METRIC, bootstrap_resamples=a.b, fit_resamples=a.b_fit,
        note="gap = gold - teacher-label cell (seed means, paired CI over the same item resamples); exchange: "
             "gold-equivalent N from q(N) = a - b N^-c fitted to the gold seed-mean curve",
        **{name: json.loads(t.to_json(orient="records", double_precision=4)) for name, t in outs.items()})
    f.write_text(json.dumps(numbers, indent=1, sort_keys=True))
    for name, t in outs.items():
        print(f"{name}: {len(t)} rows -> results/tables/{name}.csv")
    fl = tabs["fits"]
    if len(fl):
        print(fl.status.value_counts().to_string())
    return 0


if __name__ == "__main__":
    np.seterr(all="ignore")
    raise SystemExit(main())
