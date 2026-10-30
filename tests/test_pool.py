"""Pooled market (Review 10): schema mapping of scripts/make_pooled_market.py, the PART2POOL spec, the market-weight hook
and the emerging filter of scripts/score_market.py. Synthetic frames only, fast."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mp = _load("make_pooled_market")
sm = _load("score_market")


def test_age_bands_match_emerging_labels():
    ages = pd.Series([17, 24, 25, 34, 35, 49, 50, 64, 65, 91])
    assert mp.band_age(ages).tolist() == ["17-24", "17-24", "25-34", "25-34", "35-49", "35-49", "50-64", "50-64",
                                          "65+", "65+"]
    assert mp.band_age(pd.Series([np.nan, 30.0])).tolist()[1] == "25-34"
    assert mp.band_age(pd.Series([np.nan, 30.0])).isna().tolist() == [True, False]


def test_construction_vocabulary_is_shared():
    dev = pd.Series(["Brick", "brick", "BRICK", " stone", "STONE", "W", "w", "T", "t", "non_standard", "NON_STANDARD", None])
    emg = pd.Series(["BRK", "STN", "WOD", "MIX"])
    d, e = mp.canon_construction(dev), mp.canon_construction(emg)
    assert d.tolist()[:-1] == ["brick"] * 3 + ["stone"] * 2 + ["timber"] * 4 + ["non_standard"] * 2
    assert d.isna().tolist()[-1]
    assert e.tolist() == ["brick", "stone", "timber", "non_standard"]
    with pytest.raises(ValueError):
        mp.canon_construction(pd.Series(["thatch"]))


def test_prefix_ids_pads_customer_and_keeps_missing():
    df = pd.DataFrame({"row_id": ["M1"], "policy_id": ["M9"], "customer_id": ["5180324"], "address_id": ["A1"],
                       "repairer_id": [None], "other": ["x"]})
    out = mp.prefix_ids(df)
    assert out.loc[0, "customer_id"] == "D05180324" and len(out.loc[0, "customer_id"]) == 9
    assert out.loc[0, "row_id"] == "DM1" and out.loc[0, "policy_id"] == "DM9" and out.loc[0, "address_id"] == "DA1"
    assert out.loc[0, "repairer_id"] is None and out.loc[0, "other"] == "x"


def _dev_motor():
    return pd.DataFrame({"row_id": ["M0", "M1"], "policy_id": ["M5", "M6"], "customer_id": ["123", "12345678"],
                         "address_id": ["A1", "A2"], "driver_age": [20, 70], "years_licensed": [10, 40],
                         "annual_mileage": [1.0, 2.0], "telematics_opt_in": [False, True], "telematics_score": [np.nan, 1.0],
                         "written_premium": [100.0, 200.0]})


def _emg_motor():
    return pd.DataFrame({"row_id": ["M0"], "policy_id": ["M5"], "customer_id": ["00000001"], "address_id": ["A1"],
                         "driver_age_band": ["35-49"], "years_licensed": [5], "written_premium": [150.0]})


def test_developed_policies_mapped_to_emerging_schema():
    log = {}
    d = mp.developed_policies(_dev_motor(), "motor", premium_scale=1.5, log=log)
    assert set(d.columns) == set(_emg_motor().columns)
    assert d.driver_age_band.tolist() == ["17-24", "65+"]
    assert d.years_licensed.tolist() == [3, 40]                  # licence repair before banding (20 - 17 = 3)
    assert log["motor: years_licensed repaired"] == 1
    assert d.written_premium.tolist() == [150.0, 300.0]
    assert d.row_id.tolist() == ["DM0", "DM1"] and d.customer_id.tolist() == ["D00000123", "D12345678"]


def test_stack_dtypes_equal_and_market_column():
    e = _emg_motor()
    d = mp.developed_policies(_dev_motor(), "motor")
    s = mp.stack(e, d, market=True)
    assert s.market.tolist() == ["emerging", "developed", "developed"]
    assert list(s.columns[:-1]) == list(e.columns)
    for c in e.columns:
        assert e[c].dtype == d[c].dtype == s[c].dtype, c
    assert s.row_id.is_unique                                   # M0 (emerging) vs DM0 (developed)
    with pytest.raises(ValueError):
        mp.stack(e, d.drop(columns="written_premium"), market=True)


def test_developed_customers_get_gender_and_drop_dob():
    c = pd.DataFrame({"customer_id": ["42"], "date_of_birth": ["1980-01-01"], "credit_band": ["A"], "x": [1]})
    out = mp.developed_customers(c)
    assert "date_of_birth" not in out and "credit_band" not in out
    assert out.gender.isna().all() and out.customer_id.tolist() == ["D00000042"]


def test_pool_spec_registered_and_consistent():
    import market_spec as ms
    import market_spec_pool as msp
    s = ms.REGISTRY["part2pool"]
    assert s is msp.PART2POOL
    assert s.customer_id_width is None and s.expected_rows is None and s.holdout_year == 2024
    assert s.suspect_claims == ("DCL30973963",)
    for k in ("brk", "stn", "wod", "mix", "brick", "t", "w"):
        assert s.construction_map[k] in {"brick", "stone", "timber", "non_standard"}
    assert "annual_mileage" not in s.monotone and "driver_age" not in str(s.segments)
    assert s.segments["motor"]["num"] == "years_licensed"


def test_market_weight_hook_default_off():
    import pipeline
    tr = pd.DataFrame({"market": ["emerging", "developed", "developed"]})
    assert pipeline._market_weights({}, tr) is None
    assert pipeline._market_weights({"market_weight_developed": 0.3}, tr.drop(columns="market")) is None
    w, wv = pipeline._market_weights({"market_weight_developed": 0.3, "market_weight_developed_val": 0.0}, tr)
    assert w.tolist() == [1.0, 0.3, 0.3] and wv.tolist() == [1.0, 0.0, 0.0]


def test_score_market_filters_emerging_rows():
    rid = np.array(["M1", "M2", "DM1", "DM2"], dtype=object)
    assert sm.emerging_mask(rid).tolist() == [True, True, False, False]
    y = np.array([0.0, 100.0, 0.0, 1e6])
    oof = np.array([40.0, 60.0, 5e5, 5e5])                     # developed rows would dominate if not filtered
    L = {"row_id": rid, "y_eval": y, "oof": oof, "test_row_id": np.array(["M9", "M8"], dtype=object),
         "test_pred": np.array([10.0, 30.0]), "holdout_idx": np.array([1, 3]), "holdout_pred": np.array([80.0, 1.0])}
    tr = pd.DataFrame({"exposure": 1.0, "year": [2023, 2024, 2023, 2024], "new_business": [True, False, True, False]},
                      index=rid)
    te = pd.DataFrame({"exposure": [1.0, 1.0], "year": 2025, "new_business": [True, False]}, index=["M9", "M8"])
    r = sm.score_line(L, tr, te, "motor")
    from metrics import d2_tweedie
    assert r["rows"] == 2
    assert r["cv"] == pytest.approx(d2_tweedie(y[:2], oof[:2]))
    assert r["calib"] == pytest.approx(1.0)
    assert r["holdout_calib"] == pytest.approx(0.8)              # only the emerging holdout row (y 100, pred 80)
    assert r["nb_ratio_test"] == pytest.approx(10 / 30)
    assert r["test_rows"] == 2 and r["test_mean"] == pytest.approx(20.0)
    ref = pd.Series([0.0, 200.0, 0.0, 0.0], index=rid)
    assert sm.score_line(L, tr, te, "motor", ref)["calib_ref"] == pytest.approx(0.5)
