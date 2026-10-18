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
