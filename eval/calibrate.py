"""Post-hoc calibrators fitted on the validation split and applied to test / shifted sets."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

EPS = 1e-6


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


class Identity:
    name = "none"

    def fit(self, p, y):
        return self

    def __call__(self, p):
        return np.asarray(p, dtype=float)


class Temperature:
    """p' = sigmoid(logit(p) / T), T fitted by validation NLL (Guo et al. 2017)."""
    name = "temperature"

    def fit(self, p, y):
        z, y = logit(p), np.asarray(y, dtype=float)

        def loss(log_t):
            q = np.clip(sigmoid(z / np.exp(log_t)), EPS, 1 - EPS)
            return -(y * np.log(q) + (1 - y) * np.log(1 - q)).mean()

        self.T = float(np.exp(minimize_scalar(loss, bounds=(-5, 5), method="bounded").x))
        return self

    def __call__(self, p):
        return sigmoid(logit(p) / self.T)


class Platt:
    """p' = sigmoid(a * logit(p) + b)."""
    name = "platt"

    def fit(self, p, y):
        self.lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p).reshape(-1, 1), y)
        return self

    def __call__(self, p):
        return self.lr.predict_proba(logit(p).reshape(-1, 1))[:, 1]


class Isotonic:
    name = "isotonic"

    def fit(self, p, y):
        self.iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p, y)
        return self

    def __call__(self, p):
        return self.iso.predict(np.asarray(p, dtype=float))


CALIBRATORS = {c.name: c for c in (Identity, Temperature, Platt, Isotonic)}


def fit(name: str, p_val, y_val):
    return CALIBRATORS[name]().fit(np.asarray(p_val, float), np.asarray(y_val, int))
