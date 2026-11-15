"""Break-even label budgets (RQ1), following Pecher et al. (EMNLP 2025).

N*_mean: first N from which mean_seeds(q) >= q_ref holds for every larger N in the sweep.
N*_var:  same with mean(q) - sd(q) >= q_ref + sd_ref (Pecher's "average + variance" point:
         worst case of the small model vs best case of the reference). sd_ref is the SD of the
         reference over repeated runs; deterministic single-run references use sd_ref = 0 (D16).
The "holds for every larger N" clause implements Pecher's "the point after which ..." wording
and prevents a single noisy crossing from defining N*. Returns None if never reached.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def _summary(curve: pd.DataFrame) -> pd.DataFrame:
    """curve columns: n, seed, q  ->  n, mean, sd (sample SD, ddof=1)."""
    g = curve.groupby("n").q
    return pd.DataFrame({"mean": g.mean(), "sd": g.std(ddof=1).fillna(0.0)}).sort_index()


def _first_sustained(ns, ok) -> int | None:
    ok = np.asarray(ok, bool)
    for i in range(len(ns)):
        if ok[i:].all():
            return int(ns[i])
    return None


def n_star_mean(curve: pd.DataFrame, q_ref: float) -> int | None:
    s = _summary(curve)
    return _first_sustained(s.index.to_numpy(), s["mean"].to_numpy() >= q_ref)


def n_star_var(curve: pd.DataFrame, q_ref: float, sd_ref: float = 0.0) -> int | None:
    s = _summary(curve)
    return _first_sustained(s.index.to_numpy(), (s["mean"] - s["sd"]).to_numpy() >= q_ref + sd_ref)


def power_law(n, a, b, c):
    return a - b * np.power(n, -c)


def fit_curve(curve: pd.DataFrame) -> dict:
    """Fit q(N) = a - b N^-c to seed means; used for plots and clearly-labelled extrapolation only."""
    s = _summary(curve)
    n, q = s.index.to_numpy(float), s["mean"].to_numpy()
    try:
        (a, b, c), _ = curve_fit(power_law, n, q, p0=(q.max(), 1.0, 0.5),
                                 bounds=([0, 0, 0], [1.5, np.inf, 3]), maxfev=20000)
    except RuntimeError:
        return dict(a=np.nan, b=np.nan, c=np.nan)
    return dict(a=float(a), b=float(b), c=float(c))


def extrapolated_n(fit: dict, q_ref: float) -> float:
    a, b, c = fit["a"], fit["b"], fit["c"]
    if not np.isfinite(a) or q_ref >= a or b <= 0 or c <= 0:
        return float("inf")
    return float((b / (a - q_ref)) ** (1 / c))
