import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from largeloss import model_loading  # noqa: E402

P = dict(min_data_in_leaf=200, rounds=60, learning_rate=0.1, lambda_l2=10.0, num_leaves=7)


def synth(n=6000, slope=0.0, seed=0):
    r = np.random.default_rng(seed)
    X = r.normal(size=(n + 500, 3))
    expo = r.uniform(0.5, 1.0, n + 500)
    capped = r.gamma(2.0, 50.0, n + 500) * expo
    p_ex = 0.05 * np.exp(slope * X[:, 0])
    excess = (r.random(n + 500) < p_ex) * r.gamma(2.0, 400.0, n + 500)
    return X[:n], X[n:], capped[:n] + excess[:n], capped[:n], expo[:n], np.arange(n) // 3


def run(slope, **kw):
    X, Xt, yt, yc, e, g = synth(slope=slope)
    return X, Xt, model_loading(X, Xt, yt, yc, e, np.ones(len(X)), g, 1.2, {**P, **kw})


def test_independent_excess_is_flat():
    X, Xt, (lr, lt, d) = run(0.0)
    assert lr.std() / lr.mean() < 0.15


def test_rising_excess_increases_loading():
    X, Xt, (lr, lt, d) = run(1.2, shrink=1.0)
    hi, lo = X[:, 0] > 1, X[:, 0] < -1
    assert lr[hi].mean() > lr[lo].mean() * 1.2
    assert d["oof_corr_excess"] > 0


def test_mean_equals_flat():
    X, Xt, yt, yc, e, g = synth(slope=1.0)
    lr, lt, d = model_loading(X, Xt, yt, yc, e, np.ones(len(X)), g, 1.2, P)
    assert abs(np.sum(e * lr) / np.sum(e) - 1.2) < 1e-9


def test_shrink_zero_and_clip():
    X, Xt, (lr, lt, d) = run(1.2, shrink=0.0)
    assert np.allclose(lr, 1.2) and np.allclose(lt, 1.2)
    X, Xt, (lr, lt, d) = run(2.0, shrink=3.0, clip=1.1)
    assert lr.max() <= 1.2 * 1.1 * 1.05 and lr.min() >= 1.2 / 1.1 * 0.95 and d["clipped_share"] > 0


def test_shapes_and_capped_pred():
    X, Xt, yt, yc, e, g = synth()
    cp = (yc + 1.0, np.full(len(Xt), 100.0))
    lr, lt, d = model_loading(X, Xt, yt, yc, e, np.ones(len(X)), g, 1.2, {**P, "capped_pred": cp})
    assert len(lr) == len(X) and len(lt) == len(Xt) and np.isfinite(lr).all() and np.isfinite(lt).all()


def test_num_accepts_pandas_categorical():
    import pandas as pd
    from largeloss import _num
    X = pd.DataFrame({"a": pd.Categorical(["x", "y", "x"]), "b": [1.0, np.nan, 3.0], "c": [True, False, True]})
    A = _num(X)
    assert A.shape == (3, 3) and A[0, 0] == 0.0 and A[1, 0] == 1.0 and np.isnan(A[1, 1]) and A[0, 2] == 1.0
