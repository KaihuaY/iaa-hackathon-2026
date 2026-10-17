"""MarketSpec: values are read at call time, so use_spec() switches behaviour without re-import. Synthetic, fast."""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import backtest  # noqa: E402
import credibility  # noqa: E402
import data  # noqa: E402
import features  # noqa: E402
import market_spec as ms  # noqa: E402
import submit  # noqa: E402
import target  # noqa: E402


@pytest.fixture(autouse=True)
def restore_spec():
    prev = ms.get_spec()
    yield
    ms.use_spec(prev)


def test_part1_defaults():
    s = ms.PART1
    assert s.name == "part1" and s.lines == ("motor", "buildings", "contents")
    assert s.train_years == (2021, 2022, 2023, 2024) and s.test_year == 2025
    assert s.all_years == [2021, 2022, 2023, 2024, 2025]
    assert s.cache_path == ROOT / "experiments" / "cache"
    assert replace(s, name="part2").cache_path == ROOT / "experiments" / "cache" / "part2"
    assert s.expected_rows == 626_639 and s.valuation_ts == pd.Timestamp("2024-12-31")


def test_use_spec_switches_data_target_features():
    ms.use_spec("part1")
    assert data.VALUATION_DATE == pd.Timestamp("2024-12-31") and features.YEARS == [2021, 2022, 2023, 2024, 2025]
    assert target.DAY0 == pd.Timestamp("2021-01-01") and target.MAX_LAG == 730
    m2 = replace(ms.PART1, name="m2", train_years=(2019, 2020, 2021), test_year=2022, valuation_date="2021-12-31",
                 max_lag=365, leaky_cols=("x",), si_unit_error={}, curve_years=(2019, 2020), mature_years=(2019, 2020))
    ms.register(m2)
    ms.use_spec("m2")
    assert data.VALUATION_DATE == pd.Timestamp("2021-12-31") and data.LEAKY_COLS == ["x"] and data.SI_UNIT_ERROR == {}
    assert features.YEARS == [2019, 2020, 2021, 2022] and features.CACHE == ROOT / "experiments" / "cache" / "m2"
    assert target.DAY0 == pd.Timestamp("2019-01-01") and target.MAX_LAG == 365 and target.MATURE_YEARS == (2019, 2020)
    assert set(target.trend_index(None, None, "none")) == {2019, 2020, 2021}
    daily = target._daily(np.arange(366, dtype=float))
    assert len(daily) == (pd.Timestamp("2021-12-31") - pd.Timestamp("2019-01-01")).days + 1
    ms.use_spec(ms.PART1)
    assert features.CACHE == ROOT / "experiments" / "cache" and target.DAY0 == pd.Timestamp("2021-01-01")


def test_unknown_market_raises():
    with pytest.raises(KeyError):
        ms.use_spec("nope")


def test_single_line_spec_limits_consumers(monkeypatch):
    ms.use_spec(replace(ms.PART1, name="motor_only", lines=("motor",)))
    assert data.LINES == ("motor",)
    seen = []
    class _Base(dict):
        def __getitem__(self, k):
            seen.append(k)
            raise StopIteration
    monkeypatch.setattr(backtest, "build_base", lambda: _Base())
    with pytest.raises(StopIteration):
        backtest.level_backtest({})
    assert seen == ["train_motor"]                    # first line touched is motor and nothing else is requested
    # build_all loads only the spec's lines
    loaded = []

    def fake_load(name, *a, **k):
        loaded.append(name)
        raise StopIteration
    monkeypatch.setattr(data, "load", fake_load)
    with pytest.raises(StopIteration):
        data.build_all()
    assert loaded == ["claims"]


def test_clean_policies_width_and_exposure_flags():
    df = pd.DataFrame({"customer_id": ["123", "12345678"], "exposure": [200.0, 1.0],
                       "period_start": pd.to_datetime(["2022-01-01"] * 2), "period_end": pd.to_datetime(["2022-12-31"] * 2)})
    ms.use_spec("part1")
    out = data.clean_policies(df, "motor")
    assert out.customer_id.tolist() == ["00000123", "12345678"] and abs(out.exposure[0] - 1.0) < 1e-9
    ms.use_spec(replace(ms.PART1, name="raw", customer_id_width=None, exposure_days_check=False))
    out = data.clean_policies(df, "motor")
    assert out.customer_id.tolist() == ["123", "12345678"] and out.exposure[0] == 200.0


def test_expected_rows_none_accepts_any_count(monkeypatch):
    sample = pd.DataFrame({"row_id": ["a", "b", "c"], "prediction": 0.0})
    monkeypatch.setattr(data, "load", lambda name, *a, **k: sample)
    sub = pd.DataFrame({"row_id": ["c", "a", "b"], "prediction": [3.0, 1.0, 2.0]})
    ms.use_spec("part1")
    with pytest.raises(AssertionError):
        submit.validate(sub)                                  # Part 1 expects 626,639 rows
    ms.use_spec(replace(ms.PART1, name="anyrows", expected_rows=None))
    out = submit.validate(sub)
    assert out.row_id.tolist() == ["a", "b", "c"] and out.prediction.tolist() == [1.0, 2.0, 3.0]
    ms.use_spec(replace(ms.PART1, name="n3", expected_rows=3))
    assert len(submit.validate(sub)) == 3


def test_competition_slug_follows_spec():
    ms.use_spec(replace(ms.PART1, name="p2", competition="some-other-slug"))
    assert submit.COMP == "some-other-slug"


def test_segment_defs_skip_missing_columns():
    tr = pd.DataFrame({"pol_claims": [0, 1, 2]})
    assert set(credibility._segment_defs("motor", tr, None)) == {"pol_claims=0", "pol_claims=1", "pol_claims>=2"}
    assert credibility._segment_defs("buildings", pd.DataFrame({"x": [1]}), None) == {}


def test_immature_years_drive_completeness_scaling():
    n = 4
    df = pd.DataFrame({"policy_id": list("aabb"), "year": [2022, 2023, 2022, 2023], "exposure": 1.0, "claim_count": 1.0})
    test = df.assign(year=2024)
    oof = np.ones(n)
    y = np.ones(n)
    ms.use_spec("part1")
    _, _, sev_p1 = credibility.buhlmann_terms(df, test, oof, y, 0.5)         # no 2024 rows in train: factor unused
    ms.use_spec(replace(ms.PART1, name="imm", immature_years=(2023,)))
    _, _, sev_imm = credibility.buhlmann_terms(df, test, oof, y, 0.5)
    assert sev_imm != sev_p1                                                 # 2023 counts now scaled by 1/0.5


# ---------------------------------------------------------------- phase B: aliases, missing columns, env selection
def test_column_aliases_rename_on_load(tmp_path):
    pd.DataFrame({"row_id": [1, 2], "age_driver": [30, 40], "other": [1.0, 2.0]}).to_parquet(tmp_path / "policies_motor.parquet")
    ms.use_spec(replace(ms.PART1, name="alias_t", column_aliases={"age_driver": "driver_age"}))
    df = data.load("policies_motor", tmp_path)
    assert "driver_age" in df.columns and "age_driver" not in df.columns and df["row_id"].tolist() == ["1", "2"]
    ms.use_spec("part1")                                   # no alias in Part 1: columns untouched
    assert "age_driver" in data.load("policies_motor", tmp_path).columns


def test_clean_policies_tolerates_missing_columns():
    df = pd.DataFrame({"row_id": ["1", "2"], "customer_id": ["12", "34"],
                       "period_start": pd.to_datetime(["2021-01-01"] * 2), "period_end": pd.to_datetime(["2021-12-31"] * 2),
                       "exposure": [1.0, 1.0], "years_licensed": [5, 6]})          # no driver_age
    out = data.clean_policies(df, "motor", [], "motor")
    assert len(out) == 2 and out["customer_id"].str.len().eq(8).all()
    out2 = data.clean_policies(df.drop(columns=["customer_id", "years_licensed"]), "motor")   # no customer_id either
    assert len(out2) == 2


def test_env_var_selects_spec_registered_later(monkeypatch):
    monkeypatch.setenv("IAA_MARKET", "rehearsal")
    monkeypatch.setattr(ms, "_ACTIVE", None)
    assert ms.get_spec().name == "rehearsal"               # REHEARSAL is registered after the module-level code
    assert ms.get_spec().column_aliases == {"age_driver": "driver_age"}
    monkeypatch.setattr(ms, "_ACTIVE", None)
    monkeypatch.setenv("IAA_MARKET", "nope")
    with pytest.raises(KeyError):
        ms.get_spec()
