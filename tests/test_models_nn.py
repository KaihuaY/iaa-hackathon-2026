import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from models import GLMTweedie, make  # noqa: E402
from models_nn import NNBoost  # noqa: E402

GLM = {"mode": "smooth", "alpha": 1e-3}
PARAMS = dict(glm=GLM, epochs=5, lr=3e-3, batch=1024, threads=2)


def _sim(n, seed):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"x1": rng.uniform(-1, 1, n).astype(np.float32), "x2": rng.uniform(-1, 1, n).astype(np.float32),
                      "x3": rng.normal(size=n).astype(np.float32),
                      "g": pd.Categorical(rng.choice(list("abc"), n))})
    X.loc[rng.random(n) < 0.05, "x3"] = np.nan
    expo = rng.uniform(0.5, 1.0, n)
    mu = expo * np.exp(-0.5 + 0.5 * X.x1 + 0.5 * X.x2 + 1.2 * X.x1 * X.x2)
    cnt = rng.poisson(mu)
    y = np.where(cnt > 0, rng.gamma(np.maximum(cnt, 1) * 2.0, 0.5), 0.0)
    return X, y, np.ones(n), expo


def _dev(y, mu, p=1.5):
    mu = np.maximum(mu, 1e-9)
    return np.mean(2 * (y ** (2 - p) / ((1 - p) * (2 - p)) - y * mu ** (1 - p) / (1 - p) + mu ** (2 - p) / (2 - p)))


@pytest.fixture(scope="module")
def fitted():
    X, y, w, e = _sim(20000, 1)
    Xv, yv, wv, ev = _sim(10000, 2)
    m = NNBoost(dict(PARAMS)).fit(X, y, w, e, Xv, yv, wv, ev)
    return m, (X, y, w, e), (Xv, yv, wv, ev)


def test_reduces_validation_deviance(fitted):
    m, tr, (Xv, yv, wv, ev) = fitted
    glm = GLMTweedie(**GLM).fit(*tr)
    assert _dev(yv, m.predict(Xv, ev)) < _dev(yv, glm.predict(Xv, ev))


def test_calibration_and_clamp(fitted):
    m, (X, y, w, e), _ = fitted
    p = m.predict(X, e)
    assert np.isfinite(p).all() and (p > 0).all()
    assert abs(p.sum() / y.sum() - 1) < 0.03
    r = np.log(p / m.scale / m.glm.predict(X, e))
    assert np.abs(r).max() <= 3 + 1e-6


def test_unseen_category_and_nan(fitted):
    m, _, (Xv, yv, wv, ev) = fitted
    Z = Xv.head(50).copy()
    Z["g"] = pd.Categorical(["zzz"] * 50)
    assert np.isfinite(m.predict(Z, ev[:50])).all()


def test_n_iter_and_make(capsys):
    X, y, w, e = _sim(3000, 3)
    m = make("nnboost", dict(PARAMS))
    assert isinstance(m, NNBoost)
    m.fit(X, y, w, e, n_iter=2)
    assert m.best_iter == 2
    assert capsys.readouterr().out.count("epoch ") == 2
