import numpy as np
import pytest

from eval import fastboot as fb
from eval import metrics as m


@pytest.mark.parametrize("ties", [False, True])
def test_matches_metrics_on_original_and_resample(ties):
    rng = np.random.default_rng(1)
    n = 800
    y = (rng.random(n) < 0.3).astype(int)
    s = np.clip(rng.normal(0.3 + 0.4 * y, 0.25), 0, 1)
    if ties:
        s = np.round(s, 1)
    w = fb.weights(n, 2, seed=0)
    i = np.repeat(np.arange(n), w[1].astype(int))
    for row, (yy, ss) in ((0, (y, s)), (1, (y[i], s[i]))):
        assert fb.pr_auc(w, y, s)[row] == pytest.approx(m.pr_auc(yy, ss), abs=1e-9)
        assert fb.max_f1(w, y, s)[row] == pytest.approx(m.max_f1(yy, ss)[0], abs=1e-9)
        assert fb.f1_at(w, y, s)[row] == pytest.approx(m.f1_at(yy, ss, 0.5), abs=1e-9)
        assert fb.ece(w, y, s)[row] == pytest.approx(m.ece(yy, ss), abs=1e-5)
