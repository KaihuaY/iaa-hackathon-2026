"""Fast synthetic tests for the new-market readiness tool (src/market.py)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import market  # noqa: E402

N = 200
LINES = ("motor", "buildings", "contents")


def _policies(line, test=False, seed=0):
    rng = np.random.default_rng(seed)
    cols = market.EXPECTED_SCHEMA[("test_" if test else "policies_") + line]
    year = 2025 if test else 2022
    df = pd.DataFrame(index=range(N))
    for c in cols:
        k = market.expected_kind(c)
        df[c] = ({"str": "x", "num": 1.0, "bool": False, "date": pd.Timestamp("2022-01-01")})[k]
    df["row_id"] = [f"{line[0]}{'t' if test else 'p'}{i}" for i in range(N)]
    df["policy_id"] = [f"P{line[0]}{i}" for i in range(N)]
    df["customer_id"] = [f"{i % 50:08d}" for i in range(N)]
    df["address_id"] = [f"A{i % 50}" for i in range(N)]
    df["period_start"] = pd.Timestamp(f"{year}-01-01")
    df["period_end"] = pd.Timestamp(f"{year}-12-31")
    df["exposure"] = 1.0
    df["channel"] = rng.choice(["direct", "broker"], N)
    for c in ("written_premium", "voluntary_excess", "compulsory_excess"):
        df[c] = rng.uniform(100, 500, N)
    if "sum_insured" in df:
        df["sum_insured"] = rng.uniform(100, 400, N)
    if "construction_type" in df:
        df["construction_type"] = rng.choice(["brick", "stone"], N)
    if "driver_age" in df:
        df["driver_age"] = rng.integers(20, 70, N)
        df["years_licensed"] = 2
    if "renewal_decision" in df:
        df["renewal_decision"] = "renewed"
    return df


def _claims():
    rows = []
    for i in range(40):
        rows.append(dict(claim_id=f"C{i}", policy_id=f"Pmotor{i}" if False else f"Pm{i}", line="motor", peril="own_damage",
                         loss_date=pd.Timestamp("2022-03-01") + pd.Timedelta(days=i), report_date=pd.Timestamp("2022-03-05") + pd.Timedelta(days=i),
                         status="closed" if i % 4 else "open", paid_to_date=100.0 + i, incurred_to_date=120.0 + i,
                         recoveries=0.0, repairer_id="R1", third_party_id="T1"))
    return pd.DataFrame(rows)


def _write_market(d, plant=True):
    d.mkdir(parents=True, exist_ok=True)
    for line in LINES:
        for test in (False, True):
            df = _policies(line, test)
            if plant and line == "motor" and not test:
                df = df.rename(columns={"driver_age": "main_driver_age"})
                df.loc[0, "exposure"] = -0.5
                df.loc[1:3, "channel"] = "Direct"
            if plant and line == "buildings" and not test:
                df = df.drop(columns=["mortgage"])
            df.to_parquet(d / f"{'test_' if test else 'policies_'}{line}.parquet")
    claims = _claims()
    if plant:
        dup = claims.iloc[[0]].copy()
        dup["claim_id"] = "C_dup"
        claims = pd.concat([claims, dup], ignore_index=True)
        claims.loc[5, "report_date"] = claims.loc[5, "loss_date"] - pd.Timedelta(days=3)
    claims.to_parquet(d / "claims.parquet")
    pd.DataFrame({"customer_id": [f"{i:08d}" for i in range(50)], "surname": "s", "date_of_birth": pd.Timestamp("1980-01-01"),
                  "occupation": "o", "marital_status": "single", "household_size": 2.0, "credit_band": "A",
                  "prior_claims_5y": 0, "licence_points": 0, "years_at_address": 3.0, "customer_since_year": 2015,
                  "email_domain": "e", "marketing_consent": True, "document_preference": "email",
                  "previous_insurer": "p", "campaign_code": "c", "branch_id": 1, "marketing_segment": "m"}
                 ).to_parquet(d / "customers.parquet")


@pytest.fixture(scope="module")
def planted(tmp_path_factory):
    d = tmp_path_factory.mktemp("new_market")
    _write_market(d)
    return d


@pytest.fixture(scope="module")
def clean(tmp_path_factory):
    d = tmp_path_factory.mktemp("ref_market")
    _write_market(d, plant=False)
    return d


def test_schema_diff_catches_planted(planted):
    s = market.schema_diff(planted)
    m = s[(s.table == "policies_motor")]
    assert ((m.status == "missing_column") & (m.column == "driver_age")).any()
    assert ((m.status == "new_column") & (m.column == "main_driver_age")).any()
    assert "main_driver_age" in m[m.column == "driver_age"].detail.iloc[0]  # rename hint
    b = s[(s.table == "policies_buildings") & (s.status == "missing_column")]
    assert list(b.column) == ["mortgage"]
    assert set(s[s.status == "missing_table"].table) == {"addresses", "weather", "sample_submission"}


def test_schema_dtype_changed(tmp_path):
    df = _policies("contents")
    df["exposure"] = df["exposure"].astype(str)
    df.to_parquet(tmp_path / "policies_contents.parquet")
    s = market.schema_diff(tmp_path)
    assert ((s.table == "policies_contents") & (s.status == "dtype_changed") & (s.column == "exposure")).any()


def test_clean_market_has_no_column_diffs(clean):
    s = market.schema_diff(clean)
    assert not s.status.isin(["missing_column", "new_column", "dtype_changed"]).any()


def test_dq_register_catches_planted(planted):
    dq = market.dq_register(planted)
    has = lambda txt, **kw: ((dq.issue.str.contains(txt, regex=False)) & (dq.rows > 0)
                             & np.logical_and.reduce([dq[k] == v for k, v in kw.items()] or [True])).any()
    assert has("duplicate claims", table="claims")
    assert has("negative exposure", table="policies_motor")
    assert has("inconsistent casing", table="policies_motor", column="channel")
    assert has("report_date < loss_date", table="claims")


def test_clean_market_has_no_serious_dq(clean):
    dq = market.dq_register(clean)
    bad = dq[(dq.rows > 0) & dq.issue.str.contains("duplicate|negative|casing|<", regex=True)]
    assert bad.empty, bad


def test_other_functions_run(planted, clean):
    assert not market.portfolio_summary(planted).empty
    dev = market.claims_development(planted)
    assert set(dev.loss_year) >= {"all", 2022}
    dr = market.drift_report(planted, clean)
    assert "channel" in set(dr.column)
    assert dr[dr.line == "contents"].psi.abs().max() < 1e-6  # identical data
    empty = market.drift_report(planted, planted / "nope")
    assert empty.empty and "note" in empty.attrs


def test_readiness_report_end_to_end(planted, clean):
    md = market.readiness_report(planted, clean)
    for h in ("## 1. Schema differences", "## 2. Data-quality register", "## 3. Portfolio",
              "## 4. Claims development", "## 5. Drift vs Part 1", "## 6. Adaptation checklist"):
        assert h in md
    assert "`driver_age` missing in `policies_motor`" in md
    assert "evidence:" in md
    assert "Adaptation checklist" in market.readiness_report(planted)  # no reference


def test_partial_directory_does_not_crash(tmp_path):
    _policies("motor").to_parquet(tmp_path / "policies_motor.parquet")  # claims etc. absent
    md = market.readiness_report(tmp_path, None)
    assert "Adaptation checklist" in md and "missing_table" in md
    empty = tmp_path / "empty"
    empty.mkdir()
    assert "Adaptation checklist" in market.readiness_report(empty, tmp_path)
    claims_only = tmp_path / "c"
    claims_only.mkdir()
    _claims().to_parquet(claims_only / "claims.parquet")
    assert "Adaptation checklist" in market.readiness_report(claims_only, None)
