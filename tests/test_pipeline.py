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
