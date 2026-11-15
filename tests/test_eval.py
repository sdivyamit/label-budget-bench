"""Evaluation code on 200-item toy sets with known answers (Week 2 deliverable)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval import bootstrap, breakeven, calibrate, metrics, stability, thresholds  # noqa: E402
from eval.conformal import ConformalFPR  # noqa: E402


def toy(n=200, seed=0, sharp=1.0):
    """Labels drawn from a known probability, so p_true is perfectly calibrated."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.02, 0.98, n)
    y = (rng.random(n) < p).astype(int)
    over = calibrate.sigmoid(calibrate.logit(p) * sharp)   # sharp > 1 -> overconfident
    return y, p, over


def test_max_f1_and_threshold():
    y = np.array([0, 0, 1, 1, 1, 0])
    s = np.array([0.1, 0.2, 0.7, 0.8, 0.9, 0.75])
    f1, t = metrics.max_f1(y, s)
    assert f1 == pytest.approx(6 / 7) and t == pytest.approx(0.7)
    assert metrics.f1_at(y, s, 0.7) == pytest.approx(2 * 3 / (2 * 3 + 1))


def test_brier_multiclass_scale():
    y = np.array([1, 0])
    p = np.array([0.8, 0.3])
    assert metrics.brier(y, p) == pytest.approx(np.mean([2 * 0.2 ** 2, 2 * 0.3 ** 2]))


def test_ece_calibrated_vs_overconfident():
    y, p, over = toy(n=20000, sharp=4.0)
    assert metrics.ece(y, p) < 0.02
    assert metrics.ece(y, over) > 0.08
    assert metrics.ece(y, over, 15, "mass") > 0.08


def test_bbs_by_hand_and_edge_cases():
    # bin 0.9-1.0 holds 3 correct (conf 0.95) and 1 error (conf 0.95); km = 1
    y = np.array([1, 1, 1, 0])
    p = np.array([0.95, 0.95, 0.95, 0.95])
    expected = np.mean([2 * 0.05 ** 2, 2 * 0.95 ** 2])
    assert metrics.bbs(y, p) == pytest.approx(expected)
    assert metrics.brier(y, p) < metrics.bbs(y, p)           # errors are diluted in plain Brier
    assert np.isnan(metrics.bbs(np.array([1, 0]), np.array([0.9, 0.1])))   # no errors at all
    assert metrics.bs_error(y, p) == pytest.approx(2 * 0.95 ** 2)
    assert metrics.bs_correct(y, p) == pytest.approx(2 * 0.05 ** 2)


def test_bbs_penalizes_overconfidence_more_than_brier():
    y, p, over = toy(n=2000, sharp=4.0)
    assert metrics.bbs(y, over) - metrics.bbs(y, p) > metrics.brier(y, over) - metrics.brier(y, p)


def test_temperature_and_isotonic_fix_overconfidence():
    y, _, over = toy(n=4000, sharp=3.0, seed=1)
    yv, ov, yt, ot = y[:2000], over[:2000], y[2000:], over[2000:]
    t = calibrate.fit("temperature", ov, yv)
    assert t.T == pytest.approx(3.0, rel=0.25)
    assert metrics.ece(yt, t(ot)) < metrics.ece(yt, ot) / 2
    iso = calibrate.fit("isotonic", ov, yv)
    q = iso(np.sort(ot))
    assert np.all(np.diff(q) >= -1e-12)
    pl = calibrate.fit("platt", ov, yv)
    assert metrics.nll(yt, pl(ot)) < metrics.nll(yt, ot)


def test_conformal_fpr_guarantee():
    rng = np.random.default_rng(3)
    alpha, fprs = 0.05, []
    for _ in range(300):
        neg_cal = rng.normal(0, 1, 200)
        neg_test = rng.normal(0, 1, 200)
        cp = ConformalFPR().fit(neg_cal, np.zeros(200))
        fprs.append(cp.flag(neg_test, alpha).mean())
    assert np.mean(fprs) <= alpha + 0.005
    cp = ConformalFPR().fit(np.array([0.1, 0.2, 0.2, 0.9]), np.zeros(4))
    assert cp.pvalues(np.array([0.2]))[0] == pytest.approx((1 + 3) / 5)   # ties count as >=


def test_conformal_ties_on_verdict_only_scores():
    cp = ConformalFPR().fit(np.array([0.0] * 95 + [1.0] * 5), np.zeros(100))
    assert cp.pvalues(np.array([1.0]))[0] == pytest.approx(6 / 101)
    assert not cp.flag(np.array([1.0]), 0.05)[0]                         # cannot split the plateau


def test_threshold_for_precision_and_fpr():
    y = np.array([1, 1, 0, 1, 0, 0, 0, 0, 0, 0])
    s = np.array([.9, .8, .7, .6, .5, .4, .3, .2, .1, .05])
    assert thresholds.threshold_for_precision(y, s, 1.0) == pytest.approx(0.8)
    assert thresholds.threshold_for_precision(y, s, 0.75) == pytest.approx(0.6)
    assert thresholds.threshold_for_precision(y, s, 1.01) == float("inf")
    t = thresholds.threshold_for_fpr(y, s, 1 / 7)                       # 7 negatives, allow 1
    assert t == pytest.approx(0.6)                                       # flags only the 0.7 negative
    assert metrics.confusion_rates(y, s >= t)["fpr"] == pytest.approx(1 / 7)


def test_fpr_control_runs_all_methods():
    y, _, over = toy(n=2000, sharp=2.0, seed=5)
    out = thresholds.fpr_control(y[:1000], over[:1000], {"test": (y[1000:], over[1000:])}, 0.05)
    assert set(out) == {"raw", "temperature", "isotonic", "conformal"}
    for m in out.values():
        assert m["test"]["fpr"] < 0.12


def test_krippendorff_alpha():
    assert stability.krippendorff_alpha_nominal(np.array([[0, 0, 1, 0], [0, 0, 1, 1]])) == pytest.approx(1 - 14 / 30)
    same = np.tile(np.array([0, 1, 1, 0, 1]), (5, 1))
    assert stability.alpha_k(same) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    assert abs(stability.alpha_k(rng.integers(0, 2, (5, 2000)))) < 0.05
    assert stability.flip_rate(np.array([[0, 1, 1], [0, 1, 0]])) == pytest.approx(1 / 3)


def test_breakeven_pecher_definitions():
    rows = []
    means = {100: .50, 250: .62, 500: .71, 1000: .69, 2000: .74, 5000: .78}
    for n, m in means.items():
        for k, d in enumerate([-.02, -.01, 0, .01, .02]):
            rows.append(dict(n=n, seed=k, q=m + d))
    curve = pd.DataFrame(rows)
    assert breakeven.n_star_mean(curve, 0.70) == 2000          # 500 crosses, 1000 dips: not sustained
    assert breakeven.n_star_mean(curve, 0.65) == 500
    sd = np.std([-.02, -.01, 0, .01, .02], ddof=1)
    assert breakeven.n_star_var(curve, 0.70) == 2000            # 0.74 - sd >= 0.70
    assert breakeven.n_star_var(curve, 0.72, sd_ref=sd) == 5000
    assert breakeven.n_star_mean(curve, 0.99) is None
    fit = breakeven.fit_curve(curve)
    assert fit["a"] > 0.74


def test_bootstrap_and_tests():
    y, p, over = toy(n=200, seed=7)
    est, lo, hi = bootstrap.bootstrap_ci(y, p, metrics.pr_auc, b=300)
    assert lo <= est <= hi
    same = bootstrap.paired_bootstrap(y, p, p, metrics.pr_auc, b=200)
    assert same["diff"] == 0 and same["p"] == 1.0
    noise = np.random.default_rng(1).random(200)
    diff = bootstrap.paired_bootstrap(y, p, noise, metrics.pr_auc, b=300)
    assert diff["diff"] > 0 and diff["p"] < 0.01
    assert bootstrap.mcnemar(y, p >= .5, p >= .5)["p"] == 1.0
    assert np.allclose(bootstrap.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_all_metrics_row():
    y, p, _ = toy()
    row = metrics.all_metrics(y, p)
    assert {"pr_auc", "max_f1", "ece", "bbs", "bs_error", "n_unique_scores"} <= set(row)
