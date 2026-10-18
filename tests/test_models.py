import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from models import GLMBoost, LGBMTweedie  # noqa: E402
from models_adv import CatBoostTweedie  # noqa: E402


def _data(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    expo = rng.uniform(0.2, 1, n)
    freq = rng.poisson(0.1 * expo * np.exp(0.5 * X["a"] - 0.3 * X["b"]))
    y = freq * rng.gamma(2, 1000, n)
    return X, y, np.ones(n), expo


def test_monotone_increasing():
    X, y, w, expo = _data()
    m = LGBMTweedie({"learning_rate": 0.1, "min_data_in_leaf": 200, "monotone": {"a": 1}}).fit(X, y, w, expo, n_iter=100)
    assert m.monotone == {"a": 1} and "monotone" not in m.params
    grid = pd.DataFrame({"a": np.linspace(-3, 3, 200), "b": 0.0})
    p = m.predict(grid, np.ones(len(grid)))
    assert np.all(np.diff(p) >= -1e-9)


def test_unconstrained_fits():
    X, y, w, expo = _data()
    m = LGBMTweedie({"learning_rate": 0.1, "min_data_in_leaf": 200}).fit(X, y, w, expo, n_iter=50)
    p = m.predict(X, expo)
    assert m.monotone == {} and np.all(np.isfinite(p)) and np.all(p > 0)


_CB = {"learning_rate": 0.1, "depth": 3, "min_data_in_leaf": 200, "thread_count": 2}


def test_catboost_margin_equivalence():
    X, y, w, expo = _data(5000)
    base = CatBoostTweedie(dict(_CB)).fit(X, y, w, expo, n_iter=30)
    p0 = base.predict(X, expo)
    flat = base.base + np.log(expo)
    m = CatBoostTweedie(dict(_CB)).fit(X, y, w, expo, n_iter=30, margin=flat)
    assert np.allclose(m.predict(X, expo, margin=flat), p0)
    other = flat + 0.5 * X["a"].to_numpy()
    m2 = CatBoostTweedie(dict(_CB)).fit(X, y, w, expo, n_iter=30, margin=other)
    p2 = m2.predict(X, expo, margin=other)
    assert not np.allclose(p2, p0) and np.all(p2 > 0)


def test_glmboost_catboost():
    X, y, w, expo = _data(5000)
    m = GLMBoost({"boost": "catboost", "glm": {"mode": "smooth", "alpha": 1e-3}, **_CB}).fit(X, y, w, expo, n_iter=30)
    p = m.predict(X, expo)
    assert isinstance(m.gbm, CatBoostTweedie) and "boost" not in m.gbm.params
    assert np.all(np.isfinite(p)) and np.all(p > 0)
    assert abs(p.sum() / y.sum() - 1) < 0.05


# ---- FreqSev freq_base (smooth Poisson GLM base for the frequency stage) ----
from models import FreqSev, GLMTweedie  # noqa: E402

_FS = {"freq": {"learning_rate": 0.1, "min_data_in_leaf": 200, "num_leaves": 7},
       "sev": {"min_data_in_leaf": 50, "n_iter": 20}}


def _fs_data(n, seed):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.normal(size=n)})
    expo = rng.uniform(0.2, 1, n)
    cnt = rng.poisson(0.15 * expo * np.exp(0.5 * X["a"] - 0.4 * X["b"] + 0.3 * X["c"])).astype(float)
    y = cnt * rng.gamma(2, 1000, n)
    return X, y, cnt, np.ones(n), expo


def _pdev(c, mu):
    mu = np.maximum(mu, 1e-12)
    return 2 * np.sum(np.where(c > 0, c * np.log(np.where(c > 0, c, 1) / mu), 0) - (c - mu))


def test_freqsev_no_base_unchanged():
    X, y, c, w, e = _fs_data(8000, 1)
    m = FreqSev(_FS).fit(X, y, w, e, n_iter=40, counts=c)
    assert m.freq_glm is None
    manual = m.scale * m.freq.predict(X, e) * np.exp(
        m.sev.predict(X, raw_score=True, num_iteration=m.sev_rounds) + m.sev_base)
    assert np.array_equal(m.predict(X, e), manual)


def test_freqsev_freq_base():
    X, y, c, w, e = _fs_data(20000, 2)
    Xv, yv, cv, wv, ev = _fs_data(40000, 3)
    fb = {"mode": "smooth", "alpha": 1e-4}
    res = {}
    for name, extra in (("plain", {}), ("base", {"freq_base": fb})):
        m = FreqSev({**_FS, **extra}).fit(X, y, w, e, Xv, yv, wv, ev, counts=c, counts_val=cv)
        mg = np.log(m.freq_glm.predict(Xv, ev)) if name == "base" else None
        res[name] = (m, m.freq.predict(Xv, ev, margin=mg))
    assert res["base"][0].freq_glm.power == 1.0
    assert _pdev(cv, res["base"][1]) <= _pdev(cv, res["plain"][1]) * 1.0005
    p = res["base"][0].predict(Xv, ev)
    assert np.all(np.isfinite(p)) and np.all(p > 0)
    assert abs(res["base"][0].predict(X, e).sum() / y.sum() - 1) < 0.03
    assert abs(res["base"][1].sum() / cv.sum() - 1) < 0.03


def test_glm_poisson_power():
    rng = np.random.default_rng(5)
    n = 20000
    X = pd.DataFrame({"a": rng.normal(size=n)})
    e = rng.uniform(0.2, 1, n)
    c = rng.poisson(0.2 * e * np.exp(0.4 * X["a"])).astype(float)
    g = GLMTweedie(mode="smooth", power=1.0).fit(X, c, np.ones(n), e)
    assert g.model.family.power == 1.0
    assert abs(g.predict(X, e).sum() / c.sum() - 1) < 0.01
