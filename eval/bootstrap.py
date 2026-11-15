"""Paired bootstrap CIs, exact McNemar, and Holm correction (§6.6)."""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.stats import binomtest


def bootstrap_ci(y, s, metric: Callable, b: int = 2000, seed: int = 0, level: float = 0.95) -> tuple[float, float, float]:
    y, s = np.asarray(y), np.asarray(s)
    rng = np.random.default_rng(seed)
    est = metric(y, s)
    reps = []
    for _ in range(b):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            reps.append(metric(y[i], s[i]))
    lo, hi = np.nanpercentile(reps, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(est), float(lo), float(hi)


def paired_bootstrap(y, s_a, s_b, metric: Callable, b: int = 2000, seed: int = 0) -> dict:
    """Difference metric(a) - metric(b) on the same resampled items; two-sided bootstrap p-value."""
    y, s_a, s_b = map(np.asarray, (y, s_a, s_b))
    rng = np.random.default_rng(seed)
    diff = metric(y, s_a) - metric(y, s_b)
    reps = []
    for _ in range(b):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            reps.append(metric(y[i], s_a[i]) - metric(y[i], s_b[i]))
    reps = np.asarray(reps)
    p = min(1.0, 2 * min((reps <= 0).mean(), (reps >= 0).mean()))
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return dict(diff=float(diff), lo=float(lo), hi=float(hi), p=float(p), b=len(reps))


def mcnemar(y, pred_a, pred_b) -> dict:
    """Exact McNemar test on discordant correct/incorrect pairs."""
    y, a, b = (np.asarray(x).astype(int) for x in (y, pred_a, pred_b))
    ca, cb = a == y, b == y
    n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return dict(a_only=n01, b_only=n10, p=float(p))


def holm(pvals) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj
