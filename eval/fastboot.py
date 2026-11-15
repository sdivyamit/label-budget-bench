"""Vectorised item bootstrap: the same B resamples (multinomial item counts) score many runs at once.

Every metric matches eval/metrics.py on unit weights (tested in tests/test_fastboot.py): PR-AUC is
sklearn's average precision (ties grouped), max-F1 runs over the same distinct-score thresholds,
F1 at 0.5 predicts s >= 0.5, ECE is top-label with 10 equal-width bins.
"""
from __future__ import annotations

import numpy as np


def resamples(n: int, b: int, seed: int) -> np.ndarray:
    """b bootstrap resamples of n items as item counts (b x n)."""
    return np.random.default_rng(seed).multinomial(n, np.full(n, 1.0 / n), size=b).astype(np.float32)


def weights(n: int, b: int, seed: int = 0) -> np.ndarray:
    """Row 0 = the original sample (all ones); rows 1..b = bootstrap resamples as item counts."""
    return np.vstack([np.ones((1, n), np.float32), resamples(n, b, seed)])


def chunks(n: int, b: int, seed: int, size: int = 1000):
    """weights() in blocks: yields the all-ones row first, then the resamples size rows at a time, so
    10,000 resamples never sit in memory together. Block i always uses seed + i (reproducible)."""
    yield np.ones((1, n), np.float32)
    for i, start in enumerate(range(0, b, size)):
        yield resamples(n, min(size, b - start), seed + 1 + i)


def _curve(w, y, s):
    order = np.argsort(-s, kind="stable")
    ys, ss = y[order], s[order]
    wo = w[:, order]
    tp = np.cumsum(wo * ys, axis=1)
    fp = np.cumsum(wo * (1 - ys), axis=1)
    ends = np.r_[np.flatnonzero(np.diff(ss) != 0), len(ss) - 1]      # last item of each tie group
    return tp[:, ends], fp[:, ends], tp[:, -1]


def pr_auc(w, y, s) -> np.ndarray:
    tp, fp, pos = _curve(w, y, s)
    prec = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1e-12), 1.0)
    rec = tp / np.maximum(pos[:, None], 1e-12)
    dr = np.diff(np.c_[np.zeros(len(rec)), rec], axis=1)
    out = (dr * prec).sum(1)
    neg = (w * (1 - y)).sum(1)
    return np.where((pos > 0) & (neg > 0), out, np.nan)


def max_f1(w, y, s) -> np.ndarray:
    tp, fp, pos = _curve(w, y, s)
    f1 = 2 * tp / np.maximum(2 * tp + fp + (pos[:, None] - tp), 1e-12)
    return f1.max(1)


def f1_at(w, y, s, t: float = 0.5) -> np.ndarray:
    pred = (s >= t).astype(np.float32)
    tp = w @ (y * pred)
    fp = w @ ((1 - y) * pred)
    fn = w @ (y * (1 - pred))
    return np.where(tp > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1e-12), 0.0)


def ece(w, y, p, n_bins: int = 10) -> np.ndarray:
    conf = np.maximum(p, 1 - p)
    correct = ((p >= 0.5).astype(int) == y).astype(np.float32)
    b = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    onehot = np.eye(n_bins, dtype=np.float32)[b]                       # n x bins
    gap = np.abs(w @ (onehot * correct[:, None]) - w @ (onehot * conf[:, None].astype(np.float32)))
    return gap.sum(1) / w.sum(1)


METRICS = {"pr_auc": pr_auc, "max_f1": max_f1, "f1_at_05": f1_at, "ece": ece}


def summary(reps: np.ndarray, level: float = 0.95) -> dict:
    """reps: [b+1] values (row 0 = original sample). Point estimate and percentile CI."""
    r = reps[1:][~np.isnan(reps[1:])]
    lo, hi = np.percentile(r, [(1 - level) / 2 * 100, (1 + level) / 2 * 100]) if len(r) else (np.nan, np.nan)
    return dict(mean=float(reps[0]), lo=float(lo), hi=float(hi))
