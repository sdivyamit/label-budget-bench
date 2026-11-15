"""Discrimination and calibration metrics for binary classifiers (§6.5).

Conventions (D15):
  * y in {0,1}; s / p = score or probability of the positive class.
  * Calibration is top-label (Guo et al. 2017): confidence = max(p, 1-p), correct = (p >= 0.5) == y.
  * Brier-family metrics use the multiclass form of Fonseca et al. (ACL 2026), range [0, 2], so BS,
    BBS, BS-Correct and BS-Error share one scale. Binary (p - y)^2 Brier = BS / 2.
  * ECE: primary 10 equal-width bins (as in the BBS paper), robustness 15 equal-mass bins.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def _arr(y, p):
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    if y.shape != p.shape:
        raise ValueError("y and p must have the same shape")
    return y, p


# ------------------------------------------------------------------ discrimination

def pr_auc(y, s) -> float:
    y, s = _arr(y, s)
    return float(average_precision_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def roc_auc(y, s) -> float:
    y, s = _arr(y, s)
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def f1_at(y, s, t: float) -> float:
    y, s = _arr(y, s)
    pred = s >= t
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def max_f1(y, s) -> tuple[float, float]:
    """Best positive-class F1 over all thresholds and the threshold achieving it (predict s >= t)."""
    y, s = _arr(y, s)
    prec, rec, thr = precision_recall_curve(y, s)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)[:-1]
    i = int(np.argmax(f1))
    return float(f1[i]), float(thr[i])


def recall_at_precision(y, s, target: float) -> float:
    y, s = _arr(y, s)
    prec, rec, _ = precision_recall_curve(y, s)
    ok = prec >= target
    return float(rec[ok].max()) if ok.any() else 0.0


def confusion_rates(y, pred) -> dict:
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=bool)
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    return dict(precision=tp / (tp + fp) if tp + fp else float("nan"),
                recall=tp / (tp + fn) if tp + fn else float("nan"),
                fpr=fp / (fp + tn) if fp + tn else float("nan"),
                flag_rate=(tp + fp) / len(y) if len(y) else float("nan"),
                f1=2 * tp / (2 * tp + fp + fn) if tp else 0.0,
                accuracy=(tp + tn) / len(y) if len(y) else float("nan"))


# ------------------------------------------------------------------ calibration

def _top_label(y, p):
    conf = np.maximum(p, 1 - p)
    correct = ((p >= 0.5).astype(int) == y)
    return conf, correct


def _bins(conf: np.ndarray, n_bins: int, scheme: str) -> np.ndarray:
    if scheme == "width":
        return np.minimum((conf * n_bins).astype(int), n_bins - 1)
    if scheme == "mass":
        order = np.argsort(conf, kind="stable")
        b = np.empty(len(conf), dtype=int)
        b[order] = np.arange(len(conf)) * n_bins // max(len(conf), 1)
        return b
    raise ValueError(scheme)


def ece(y, p, n_bins: int = 10, scheme: str = "width") -> float:
    y, p = _arr(y, p)
    conf, correct = _top_label(y, p)
    b = _bins(conf, n_bins, scheme)
    tot = 0.0
    for k in np.unique(b):
        m = b == k
        tot += m.sum() * abs(correct[m].mean() - conf[m].mean())
    return float(tot / len(y))


def brier_mc(y, p) -> np.ndarray:
    """Per-item multiclass Brier for K=2: (p1-o1)^2 + (p0-o0)^2 = 2 (p - y)^2."""
    y, p = _arr(y, p)
    return 2.0 * (p - y) ** 2


def brier(y, p) -> float:
    return float(brier_mc(y, p).mean())


def bs_correct(y, p) -> float:
    y, p = _arr(y, p)
    _, correct = _top_label(y, p)
    return float(brier_mc(y, p)[correct].mean()) if correct.any() else float("nan")


def bs_error(y, p) -> float:
    y, p = _arr(y, p)
    _, correct = _top_label(y, p)
    return float(brier_mc(y, p)[~correct].mean()) if (~correct).any() else float("nan")


def bbs(y, p, n_bins: int = 10, n_rep: int = 100, seed: int = 0) -> float:
    """Balanced Brier Score, Algorithm 1 of Fonseca et al. (ACL 2026).

    Per equal-width confidence bin, undersample the larger of {correct, incorrect} to the size of
    the smaller, pool the balanced bins, and take the multiclass Brier. The paper draws one random
    undersample; we average `n_rep` draws to remove sampling noise (D15). NaN if no bin holds both
    a correct and an incorrect prediction.
    """
    y, p = _arr(y, p)
    conf, correct = _top_label(y, p)
    b = _bins(conf, n_bins, "width")
    per = brier_mc(y, p)
    groups = []
    for k in np.unique(b):
        c = np.flatnonzero((b == k) & correct)
        i = np.flatnonzero((b == k) & ~correct)
        km = min(len(c), len(i))
        if km:
            groups.append((c, i, km))
    if not groups:
        return float("nan")
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_rep):
        idx = np.concatenate([np.concatenate([rng.choice(c, km, replace=False),
                                              rng.choice(i, km, replace=False)])
                              for c, i, km in groups])
        vals.append(per[idx].mean())
    return float(np.mean(vals))


def nll(y, p, eps: float = 1e-6) -> float:
    y, p = _arr(y, p)
    p = np.clip(p, eps, 1 - eps)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def all_metrics(y, p) -> dict:
    """The per-run metric row written to results/metrics.parquet."""
    f1, thr = max_f1(y, p)
    return dict(pr_auc=pr_auc(y, p), roc_auc=roc_auc(y, p), max_f1=f1, max_f1_threshold=thr,
                f1_at_05=f1_at(y, p, 0.5), ece=ece(y, p), ece_mass15=ece(y, p, 15, "mass"),
                brier=brier(y, p), bbs=bbs(y, p), bs_correct=bs_correct(y, p), bs_error=bs_error(y, p),
                nll=nll(y, p), n=len(np.asarray(y)), prevalence=float(np.mean(y)),
                n_unique_scores=int(len(np.unique(np.round(np.asarray(p, float), 6)))))
