import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recal import fit_bins, fit_isotonic, nested_eval  # noqa: E402

FITS = [fit_isotonic, fit_bins]


def _data(distort, n=40_000, seed=0):
    rng = np.random.default_rng(seed)
    mu = np.exp(rng.normal(0, 0.6, n))
    y = rng.gamma(2.0, mu / 2.0) * (rng.random(n) < 0.5) * 2
    g = rng.integers(0, 4000, n)
    return mu, y, g, distort(mu)


def test_distortion_corrected():
    mu, y, g, pred = _data(lambda m: m ** 0.6 * 1.3)
    for f in FITS:
        b, r = nested_eval(pred, y, g, f)
        assert r > b + 0.002
        m = f(pred, y)
        grid = np.linspace(pred.min(), pred.max(), 500)
        assert (np.diff(m(grid)) >= -1e-12).all()
        assert abs(m(pred).sum() / pred.sum() - 1) < 1e-9


def test_calibrated_not_harmed():
    mu, y, g, pred = _data(lambda m: m)
    for f in FITS:
        b, r = nested_eval(pred, y, g, f)
        assert r > b - 0.0005


def test_out_of_range_clipped():
    mu, y, g, pred = _data(lambda m: m)
    for f in FITS:
        m = f(pred, y)
        hi, lo = m(np.array([pred.max()]))[0], m(np.array([pred.min()]))[0]
        assert m(np.array([pred.max() * 100]))[0] == hi
        assert m(np.array([pred.min() / 100]))[0] == lo
