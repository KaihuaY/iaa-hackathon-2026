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


# ---- covariate calibration ----
from recal import fit_covariate, nested_eval_cov  # noqa: E402


def _cov_data(factor, n=60_000, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.lognormal(0, 0.7, n)
    pred = np.exp(rng.normal(0, 0.5, n))
    mean = pred * factor(x)
    y = rng.gamma(2.0, mean / 2.0) * (rng.random(n) < 0.5) * 2
    return x, pred, y, rng.integers(0, 6000, n)


def _tilt(x):
    return 1 + 0.3 * (np.argsort(np.argsort(x)) / len(x) - 0.5)


def test_cov_shift_corrected_monotone():
    x, pred, y, g = _cov_data(_tilt)
    for mono in (True, False):
        f = lambda xx, pp, yy, m=mono: fit_covariate(xx, pp, yy, monotone=m)
        b, r = nested_eval_cov(x, pred, y, g, f)
        assert r > b + 0.0005
    gf = fit_covariate(x, pred, y)
    assert (np.diff(gf(np.sort(x))) >= -1e-12).all()


def test_cov_calibrated_not_harmed():
    x, pred, y, g = _cov_data(lambda x: np.ones_like(x))
    b, r = nested_eval_cov(x, pred, y, g, fit_covariate)
    assert abs(r - b) < 0.0005


def test_cov_level_and_bounds():
    x, pred, y, _ = _cov_data(_tilt)
    for mono in (True, False):
        gf = fit_covariate(x, pred, y, monotone=mono)
        v = gf(x)
        assert abs((v * pred).sum() / pred.sum() - 1) < 1e-9
        assert v.min() >= 0.7 - 1e-9 and v.max() <= 1.4 + 1e-9
    y2 = pred * np.where(x > np.median(x), 3.0, 0.2)          # extreme: bounds must bind
    v = fit_covariate(x, pred, y2)(x)
    assert v.min() >= 0.7 - 1e-9 and v.max() <= 1.4 + 1e-9
    assert abs((v * pred).sum() / pred.sum() - 1) < 1e-9
