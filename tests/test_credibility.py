import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from credibility import apply_segment_layer, buhlmann_adjust, buhlmann_terms, fit_segment_layer  # noqa: E402
from metrics import d2_tweedie  # noqa: E402


def test_segment_layer():
    rng = np.random.default_rng(0)
    n = 50_000
    expo = rng.uniform(0.5, 1.0, n)
    mu = expo * 500.0
    seg = rng.random(n) < 0.05
    noise = rng.random(n) < 0.3
    true = mu * np.where(seg, 1.5, 1.0)
    cnt = rng.poisson(true / 1000.0)
    shape = 2.0
    y = np.array([rng.gamma(shape, 1000.0 / shape, c).sum() if c else 0.0 for c in cnt])
    D = sp.csr_matrix(np.column_stack([seg, noise]).astype(float))
    m = fit_segment_layer(y, mu, D, 1e-2)
    mult = np.exp(m.coef_)
    assert abs(mult[0] - 1.5) < 0.15, mult
    assert abs(mult[1] - 1.0) < 0.05, mult
    assert apply_segment_layer(m, mu, D).shape == (n,)


def _frailty(n_pol=20_000, years=4, seed=1):
    rng = np.random.default_rng(seed)
    theta = rng.gamma(2.0, 0.5, n_pol)  # mean-1 frailty
    pid = np.repeat(np.arange(n_pol), years)
    year = np.tile(np.arange(2021, 2021 + years), n_pol)
    th = np.repeat(theta, years)
    cnt = rng.poisson(0.3 * th)
    df = pd.DataFrame({"policy_id": pid.astype(str), "year": year, "exposure": 1.0,
                       "claim_count": cnt.astype(float), "claim_cost": cnt * 1000.0})
    return df


def test_buhlmann():
    df = _frailty()
    mu = np.full(len(df), 300.0)
    np.testing.assert_allclose(buhlmann_adjust(mu, np.ones_like(mu), np.ones_like(mu), mu, 1e9), mu, rtol=1e-7)
    test = df[df.year == 2024].copy()
    train = df[df.year < 2024].reset_index(drop=True)
    test["year"] = 2025  # next year after the history
    _, te, _ = buhlmann_terms(train, test, np.full(len(train), 300.0), train["claim_cost"].to_numpy(), 1.0)
    y = test["claim_cost"].to_numpy()
    m = np.full(len(test), 300.0)
    adj = buhlmann_adjust(m, te["E"], te["A_c"], te["P"], 8.0)
    assert d2_tweedie(y, adj) > d2_tweedie(y, m)


def test_no_future_information():
    df = _frailty(n_pol=50, years=3, seed=3)
    df["claim_cost"] = np.arange(len(df), dtype=float) + 1.0
    df["claim_count"] = 1.0
    oof = np.arange(len(df), dtype=float) + 5.0
    test = df[df.year == 2023].copy()
    test["year"] = 2025
    train = df[df.year < 2023].reset_index(drop=True)
    o = oof[df.year.to_numpy() < 2023]
    yv = train["claim_cost"].to_numpy() * 2.0  # y_eval passed explicitly
    tt, _, _ = buhlmann_terms(train, test, o, yv, 1.0, cap_q=1.0)
    y21, y22 = (train.year == 2021).to_numpy(), (train.year == 2022).to_numpy()
    assert (tt["E"][y21] == 0).all() and (tt["A_c"][y21] == 0).all() and (tt["P"][y21] == 0).all()
    d21 = train[y21].set_index("policy_id")
    p21 = pd.Series(o[y21], index=d21.index)
    r22 = train[y22]
    np.testing.assert_allclose(tt["E"][y22], d21.loc[r22.policy_id, "exposure"].to_numpy())
    np.testing.assert_allclose(tt["A_c"][y22], d21.loc[r22.policy_id, "claim_cost"].to_numpy() * 2.0)
    np.testing.assert_allclose(tt["A_n"][y22], d21.loc[r22.policy_id, "claim_count"].to_numpy())
    np.testing.assert_allclose(tt["P"][y22], p21.loc[r22.policy_id].to_numpy())
