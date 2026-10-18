"""Covariate-shift importance weights (synthetic, fast)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from shift import density_ratio_weights  # noqa: E402

P = {"min_data_in_leaf": 400, "learning_rate": 0.01, "num_threads": 2}   # small synthetic data: keep the classifier shallow


def _frame(n, rng):
    return pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})


def _shifted_target(n, rng):                    # x1 > 0 for ~80% of rows
    x1 = np.abs(rng.normal(size=n)) * np.where(rng.random(n) < 0.8, 1, -1)
    return pd.DataFrame({"x1": x1, "x2": rng.normal(size=n)})


def test_same_distribution_gives_unit_weights():
    rng = np.random.default_rng(0)
    w, d = density_ratio_weights(_frame(4000, rng), _frame(4000, rng), params=P, verbose=False)
    assert abs(w.mean() - 1) < 1e-9
    assert np.abs(w - 1).max() < 0.5
    assert d["ess_ratio"] > 0.9


def test_shifted_target_upweights_and_clips():
    rng = np.random.default_rng(1)
    S, T = _frame(5000, rng), _shifted_target(5000, rng)
    w, d = density_ratio_weights(S, T, clip=(0.3, 4.0), params=P, verbose=False)
    assert abs(w.mean() - 1) < 1e-9
    pos = (S.x1 > 0).values
    assert w[pos].mean() > 1 > w[~pos].mean()
    assert d["auc"] > 0.55 and "x1" in d["top_features"]
    # clip applies before renormalisation, so ratio of extremes is bounded by clip ratio
    assert w.max() / w.min() <= 4.0 / 0.3 + 1e-6


def test_categoricals_and_constant_columns():
    rng = np.random.default_rng(2)
    S, T = _frame(3000, rng), _shifted_target(3000, rng)
    S["c"] = pd.Categorical(rng.choice(["a", "b", "c"], len(S)))
    T["c"] = pd.Categorical(rng.choice(["a", "b", "d"], len(T)))
    S["k"] = 1.0
    T["k"] = 1.0
    w, d = density_ratio_weights(S, T, params=P, verbose=False)
    assert len(w) == len(S) and np.isfinite(w).all() and abs(w.mean() - 1) < 1e-9
    assert "k" not in d["top_features"]


def test_pipeline_hook_noop_without_key(monkeypatch):
    import pipeline
    import shift

    def boom(*a, **k):
        raise AssertionError("density_ratio_weights must not be called")

    monkeypatch.setattr(shift, "density_ratio_weights", boom)
    got = []

    class Fake:
        def fit(self, X, y, w, *a, **k):
            got.append(np.asarray(w).copy())

        def predict(self, X, e):
            return np.zeros(len(X))

    monkeypatch.setattr(pipeline, "make", lambda kind, params: Fake())
    X, e = pd.DataFrame({"x": [1.0, 2.0]}), np.ones(2)
    for cfg in ({"model": "lgbm"}, {"model": "lgbm", "target": {}}):
        pipeline._fit_predict(cfg, X, np.ones(2), np.array([1.0, 3.0]), e, X, e, n_iter=5)
    assert all((g == [1.0, 3.0]).all() for g in got)


def test_pipeline_hook_multiplies_weights(monkeypatch):
    import pipeline
    import shift

    seen = {}
    monkeypatch.setattr(shift, "density_ratio_weights",
                        lambda Xs, Xt, clip=None, **k: (seen.update(clip=clip) or (np.array([2.0, 0.5]), {})))
    got = []

    class Fake:
        def fit(self, X, y, w, *a, **k):
            got.append(np.asarray(w).copy())

        def predict(self, X, e):
            return np.zeros(len(X))

    monkeypatch.setattr(pipeline, "make", lambda kind, params: Fake())
    X, e = pd.DataFrame({"x": [1.0, 2.0]}), np.ones(2)
    cfg = {"model": "lgbm", "target": {"shift_weights": {"clip": [0.5, 2.0]}}}
    pipeline._fit_predict(cfg, X, np.ones(2), np.array([1.0, 3.0]), e, X, e, n_iter=5)
    assert (got[0] == [2.0, 1.5]).all() and seen["clip"] == (0.5, 2.0)
