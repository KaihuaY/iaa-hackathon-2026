"""Guardrails: data fixes, leakage, target construction, metric and submission format."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import LEAKY_COLS, LINES, clean_claims, clean_policies, load  # noqa: E402
from features import build_base, model_matrix  # noqa: E402
from metrics import d2_tweedie, tweedie_deviance  # noqa: E402
from target import prepare_target  # noqa: E402


@pytest.fixture(scope="session")
def base():
    return build_base()


def test_metric_matches_sklearn():
    from sklearn.metrics import d2_tweedie_score
    rng = np.random.default_rng(0)
    y = rng.gamma(0.3, 500, 5000) * (rng.random(5000) < 0.1)
    mu = rng.uniform(10, 200, 5000)
    assert d2_tweedie(y, mu) == pytest.approx(d2_tweedie_score(y, mu, power=1.5), abs=1e-9)
    assert tweedie_deviance([0.0], [1.0])[0] > 0


@pytest.mark.parametrize("line", LINES)
def test_cleaning(line):
    for name in (f"policies_{line}", f"test_{line}"):
        d = clean_policies(load(name), line)
        days = (d.period_end - d.period_start).dt.days + 1
        assert ((d.exposure - days / 365).abs() <= 0.02).all(), "exposure not consistent with dates"
        assert (d.customer_id.str.len() == 8).all()
        if "sum_insured" in d:
            assert d.sum_insured.max() < 5e6
        if "construction_type" in d:
            assert set(d.construction_type.dropna()) <= {"brick", "stone", "timber", "non_standard"}


def test_claims_deduplicated():
    c = clean_claims(load("claims"))
    assert not c.drop(columns="claim_id").duplicated().any()


def test_customer_join_complete(base):
    cust = set(load("customers").customer_id)
    for l in LINES:
        assert base[f"test_{l}"].customer_id.isin(cust).all()


@pytest.mark.parametrize("line", LINES)
def test_no_leakage_in_features(base, line):
    X, Xt, _ = model_matrix(base[f"train_{line}"], base[f"test_{line}"], {"history": True})
    for col in LEAKY_COLS + ["claim_count", "claim_cost", "row_id", "policy_id", "written_premium"]:
        assert col not in X.columns, col
    assert list(X.columns) == list(Xt.columns)


def test_history_uses_only_prior_years(base):
    tr = base["train_motor"]
    first = tr[tr.year == 2021]
    assert (first.pol_claims == 0).all() and (first.pol_expo == 0).all(), "2021 rows cannot have history"
    # a policy's own same-year claims must not appear in its history
    t = tr[tr.year == 2022]
    prior = tr[tr.year == 2021].groupby("policy_id").claim_count.sum()
    assert np.allclose(t.pol_claims.values, t.policy_id.map(prior).fillna(0).values)


def test_history_window_is_stable_across_years(base):
    """History features must mean the same thing in every year (bounded look-back), incl. the 2025 test set."""
    from features import LOOKBACK
    for l in LINES:
        for d in (base[f"train_{l}"], base[f"test_{l}"]):
            assert d.pol_expo.max() <= LOOKBACK + 0.05
    tr, te = base["train_motor"], base["test_motor"]
    m24, m25 = tr.loc[tr.year == 2024, "pol_expo"].mean(), te.pol_expo.mean()
    assert abs(m25 / m24 - 1) < 0.1, (m24, m25)


def test_claims_fall_inside_row_period(base):
    c = base["claims"].dropna(subset=["row_id"])
    rows = pd.concat([base[f"train_{l}"][["row_id", "period_start", "period_end"]] for l in LINES])
    m = c.merge(rows, on="row_id")
    assert ((m.loss_date >= m.period_start) & (m.loss_date <= m.period_end)).all()


@pytest.mark.parametrize("line", LINES)
def test_target_development(base, line):
    cfg = {"develop": True, "cap_q": 0.995, "trend": "premium", "weight_power": 1.0}
    t, info = prepare_target(base[f"train_{line}"], base[f"test_{line}"], base["claims"], line, cfg)
    assert (t.ibnr >= 0).all() and (t.y_train >= 0).all() and (t.y_eval >= 0).all()
    assert info["loading"] >= 1.0
    assert t.completeness.between(0, 1).all()
    assert (t.w > 0).all() and (t.w <= 1).all()


def test_submission_validation():
    from submit import validate
    s = load("sample_submission").assign(prediction=1.0)
    assert len(validate(s)) == 626_639
    with pytest.raises(AssertionError):
        validate(s.assign(prediction=-1.0))
    with pytest.raises(AssertionError):
        validate(s.iloc[:-1])


@pytest.mark.parametrize("kind", ["lgbm", "glm"])
def test_models_are_calibrated(kind):
    """Total predicted cost must match total actual on training data (catches offset/start-level bugs)."""
    from models import make
    rng = np.random.default_rng(1)
    n = 20000
    X = pd.DataFrame({"a": rng.normal(size=n), "b": pd.Categorical(rng.choice(list("xyz"), n))})
    expo = rng.uniform(0.1, 1.0, n)
    mu = 500 * expo * np.exp(0.3 * X.a)
    y = rng.poisson(mu / 4000) * rng.gamma(2, 2000, n)
    params = {"learning_rate": 0.1} if kind == "lgbm" else None
    m = make(kind, params).fit(X, y, np.ones(n), expo, n_iter=50) if kind == "lgbm" else make(kind).fit(X, y, np.ones(n), expo)
    ratio = m.predict(X, expo).sum() / y.sum()
    assert 0.95 < ratio < 1.05, ratio
