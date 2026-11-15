"""Operating thresholds chosen on validation and their drift on test / shifted sets (RQ3)."""
from __future__ import annotations

import numpy as np

from eval import calibrate
from eval.conformal import ConformalFPR
from eval.metrics import confusion_rates


def threshold_for_precision(y, s, target: float) -> float:
    """Lowest threshold whose validation precision is >= target (max recall at that precision).

    Returns +inf if no threshold reaches the target (flag nothing).
    """
    y, s = np.asarray(y, int), np.asarray(s, float)
    order = np.argsort(-s, kind="stable")
    ys, ss = y[order], s[order]
    tp = np.cumsum(ys)
    prec = tp / np.arange(1, len(ys) + 1)
    last = np.r_[ss[1:] != ss[:-1], True]          # evaluate only at the end of tied blocks
    ok = np.flatnonzero(last & (prec >= target))
    return float(ss[ok.max()]) if len(ok) else float("inf")


def threshold_for_fpr(y, s, alpha: float) -> float:
    """Lowest threshold t with validation FPR(s >= t) <= alpha."""
    s = np.asarray(s, float)
    neg = np.sort(s[np.asarray(y, int) == 0])
    k = int(np.floor(alpha * len(neg)))            # at most k validation negatives may be flagged
    cand = np.unique(s)
    n_flagged = len(neg) - np.searchsorted(neg, cand, side="left")
    ok = cand[n_flagged <= k]
    return float(ok.min()) if len(ok) else float("inf")


def drift(y, s, t: float) -> dict:
    """Realized operating point at a fixed threshold."""
    return confusion_rates(y, np.asarray(s, float) >= t)


def precision_target_drift(y_val, s_val, evals: dict, target: float) -> dict:
    """Pick t on val for precision >= target; report realized rates on every eval set."""
    t = threshold_for_precision(y_val, s_val, target)
    return {"threshold": t, **{name: drift(y, s, t) for name, (y, s) in evals.items()}}


def fpr_control(y_val, s_val, evals: dict, alpha: float) -> dict:
    """Four ways to hit FPR <= alpha (ConformalGuard comparison): raw val threshold,
    temperature-scaled val threshold, isotonic val threshold, split-conformal on val negatives."""
    out = {}
    t = threshold_for_fpr(y_val, s_val, alpha)
    out["raw"] = {n: drift(y, s, t) for n, (y, s) in evals.items()}
    for name in ("temperature", "isotonic"):
        cal = calibrate.fit(name, s_val, y_val)
        tc = threshold_for_fpr(y_val, cal(s_val), alpha)
        out[name] = {n: drift(y, cal(s), tc) for n, (y, s) in evals.items()}
    cp = ConformalFPR().fit(s_val, y_val)
    out["conformal"] = {n: confusion_rates(y, cp.flag(s, alpha)) for n, (y, s) in evals.items()}
    return out
