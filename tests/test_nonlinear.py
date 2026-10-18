import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features import ENGINEERED, model_matrix  # noqa: E402
from metrics import d2_tweedie  # noqa: E402
from models import GLMTweedie, make  # noqa: E402


def _power_law(n, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.lognormal(3.0, 0.8, n)
    expo = rng.uniform(0.5, 1.0, n)
    mu = expo * 300.0 * (x / x.mean()) ** 0.7
    cnt = rng.poisson(mu / 1000.0)
    y = np.array([rng.gamma(2.0, 500.0, k).sum() for k in cnt])
    return pd.DataFrame({"x": x, "z": rng.normal(size=n)}), y, expo


def test_smooth_recovers_power_law():
    X, y, e = _power_law(60_000)
    n = 40_000
    kw = dict(X=X.iloc[:n], y=y[:n], w=e[:n], expo=e[:n])
    sm = GLMTweedie(mode="smooth").fit(**kw)
    bd = GLMTweedie(mode="bands").fit(**kw)
    Xt, yt, et = X.iloc[n:], y[n:], e[n:]
    # elasticity d ln(mu) / d ln(x), averaged over x between the 5th and 95th percentile
    lo, hi = Xt.x.quantile([0.05, 0.95])
    Xm = Xt[(Xt.x > lo) & (Xt.x < hi)]
    up = Xm.assign(x=Xm.x * 1.01)
    el = np.log(sm.predict(up, np.ones(len(Xm))) / sm.predict(Xm, np.ones(len(Xm)))) / np.log(1.01)
    assert abs(el.mean() - 0.7) < 0.15, el.mean()
    assert d2_tweedie(yt, sm.predict(Xt, et)) > d2_tweedie(yt, bd.predict(Xt, et))


def test_predict_same_columns_unseen_level():
    rng = np.random.default_rng(1)
    n = 3000
    X = pd.DataFrame({"a": rng.lognormal(0, 1, n), "b": rng.normal(size=n),
                      "c": pd.Categorical(rng.choice(["u", "v", "w"], n))})
    X.loc[:50, "a"] = np.nan
    y = rng.gamma(2.0, 100.0, n) * (rng.random(n) < 0.3)
    e = np.ones(n)
    m = make("glm", {"mode": "smooth", "interactions": [("a", "b"), ("a", "c")]}).fit(X, y, e, e)
    cols = list(m.cols)
    assert any(c.startswith("a*c=") for c in cols) and "a*b" in cols and "a__na" in cols
    Xn = X.iloc[:20].copy()
    Xn["c"] = pd.Categorical(["zzz"] * 20)
    p = m.predict(Xn, np.ones(20))
    assert len(p) == 20 and np.isfinite(p).all()
    assert list(m._design(Xn, fit=False)[cols].columns) == cols


def test_smooth_bounded_extrapolation():
    rng = np.random.default_rng(5)
    n = 20_000
    x = np.exp(rng.uniform(0, np.log(100), n))                      # x in [1, 100]
    X = pd.DataFrame({"x": x, "z": rng.normal(size=n), "k": pd.Categorical(rng.choice(["p", "q"], n))})
    e = rng.uniform(0.5, 1.0, n)
    y = rng.poisson(e * 0.2 * x ** 0.5) * 100.0
    m = GLMTweedie(mode="smooth", interactions=[("x", "z"), ("x", "k")]).fit(X, y, e, e)

    def pred(v):
        Xp = X.iloc[:200].copy()
        Xp["x"] = v
        return m.predict(Xp, np.ones(200))

    assert np.isfinite(pred(1e4)).all() and np.isfinite(pred(0.01)).all()
    hi, lo = pred(1e4) / pred(100.0), pred(0.01) / pred(1.0)
    assert ((hi >= 0.2) & (hi <= 5)).all(), (hi.min(), hi.max())
    assert ((lo >= 0.2) & (lo <= 5)).all(), (lo.min(), lo.max())


def _frame(n=30, seed=2):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({
        "driver_age": rng.integers(20, 70, n).astype(float), "years_licensed": rng.integers(1, 30, n).astype(float),
        "ncb_years": rng.integers(0, 9, n).astype(float), "annual_mileage": rng.uniform(3000, 20000, n),
        "engine_size_cc": rng.uniform(900, 3000, n), "vehicle_value_group": rng.integers(1, 20, n).astype(float),
        "vehicle_age": rng.integers(0, 15, n).astype(float), "licence_points": rng.integers(0, 6, n).astype(float),
        "customer_tenure": rng.uniform(0, 20, n), "customer_age": rng.integers(18, 80, n).astype(float),
        "prior_claims_5y": rng.integers(0, 3, n).astype(float), "pol_claims": rng.integers(0, 3, n).astype(float),
        "pol_expo": rng.uniform(0, 4, n), "year": 2023, "exposure": 1.0, "row_id": np.arange(n)})
    d.loc[0, "driver_age"] = np.nan
    return d


def test_engineered_columns():
    tr, te = _frame(), _frame(seed=3)
    X0, Xt0, _ = model_matrix(tr, te, {"history": True})
    X1, Xt1, _ = model_matrix(tr, te, {"history": True, "engineered": False})
    assert list(X0.columns) == list(X1.columns) and list(Xt0.columns) == list(Xt1.columns)
    X, Xt, _ = model_matrix(tr, te, {"history": True, "engineered": True})
    expected = ["age_at_licence", "ncb_ratio", "mileage_per_cc", "value_per_age", "points_per_year",
                "tenure_ratio", "prior_rate", "hist_rate"]      # no buildings columns in this frame
    new = [c for c in X.columns if c not in X0.columns]
    assert new == expected and set(expected) <= set(ENGINEERED)
    assert list(X.columns[: len(X0.columns)]) == list(X0.columns) and list(Xt.columns) == list(X.columns)
    assert all(X[c].dtype == np.float32 for c in new)
    assert np.isnan(X.loc[0, "age_at_licence"]) and not np.isnan(X.loc[1, "age_at_licence"])
    assert np.isclose(X.loc[1, "ncb_ratio"], tr.ncb_years[1] / (tr.years_licensed[1] + 1), rtol=1e-5)
