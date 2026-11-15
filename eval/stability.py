"""Seed stability of per-item verdicts: Krippendorff's alpha across K seeds (alpha_K) and flip rate."""
from __future__ import annotations

import numpy as np


def krippendorff_alpha_nominal(ratings: np.ndarray) -> float:
    """ratings: (n_raters, n_items) array of category codes; NaN = missing.

    Krippendorff (2011) nominal alpha = 1 - D_o / D_e from the coincidence matrix.
    """
    r = np.asarray(ratings, dtype=float)
    cats = np.unique(r[~np.isnan(r)])
    if len(cats) < 2:
        return 1.0
    idx = {c: i for i, c in enumerate(cats)}
    o = np.zeros((len(cats), len(cats)))
    for col in r.T:
        vals = col[~np.isnan(col)]
        m = len(vals)
        if m < 2:
            continue
        counts = np.zeros(len(cats))
        for v in vals:
            counts[idx[v]] += 1
        o += (np.outer(counts, counts) - np.diag(counts)) / (m - 1)
    n = o.sum()
    nc = o.sum(axis=1)
    d_o = n - np.trace(o)
    d_e = (n * n - (nc ** 2).sum()) / (n - 1)
    return float(1 - d_o / d_e) if d_e > 0 else 1.0


def alpha_k(verdicts_by_seed: np.ndarray) -> float:
    """alpha across seeds; verdicts_by_seed: (K, n_items) of 0/1."""
    return krippendorff_alpha_nominal(np.asarray(verdicts_by_seed, float))


def flip_rate(verdicts_by_seed: np.ndarray) -> float:
    """Share of items on which at least two seeds disagree."""
    v = np.asarray(verdicts_by_seed)
    return float((v.min(axis=0) != v.max(axis=0)).mean())
