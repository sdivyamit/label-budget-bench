"""Split-conformal false-positive control (ConformalGuard, arXiv 2610.14365, Eq. 1).

Calibration uses negatives only (C0). For a fixed scorer s,
    p(x) = (1 + #{i in C0 : s_i >= s(x)}) / (n + 1),
and flagging when p(x) <= alpha gives Pr(flag | y = 0) <= alpha under exchangeability.
Ties are handled conservatively (>=), so coarse score plateaus are never split.
"""
from __future__ import annotations

import numpy as np


class ConformalFPR:
    def fit(self, s_val, y_val):
        neg = np.sort(np.asarray(s_val, float)[np.asarray(y_val, int) == 0])
        if not len(neg):
            raise ValueError("no negatives in the calibration set")
        self.neg = neg
        return self

    def pvalues(self, s) -> np.ndarray:
        s = np.asarray(s, float)
        n_ge = len(self.neg) - np.searchsorted(self.neg, s, side="left")
        return (1.0 + n_ge) / (len(self.neg) + 1.0)

    def flag(self, s, alpha: float) -> np.ndarray:
        return self.pvalues(s) <= alpha
