import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

import office_premium as op  # noqa: E402


def _table(fixed=0.0, var=0.0, direct=0.0, comp=np.nan, broker=0.0, tax=0.0, cat=0.0, profit=0.0):
    rows = [("fixed_expense_per_policy", fixed, "CU"), ("variable_expense", var, "share of premium"),
            ("commission_direct", direct, "share of premium"),
            ("commission_comparison_site", comp, "share of premium"),
            ("commission_broker", broker, "share of premium"), ("premium_tax", tax, "share of premium"),
            ("cat_reinsurance", cat, "share of property risk premium"),
            ("target_profit_margin", profit, "share of premium")]
    return pd.DataFrame({"item": [r[0] for r in rows], "mkt": [r[1] for r in rows], "unit": [r[2] for r in rows]})


def _policies():
    return pd.DataFrame({
        "row_id": ["M1", "B1", "C1"], "line": ["motor", "buildings", "contents"],
        "channel": ["broker", "direct", "broker"], "exposure": [1.0, 0.5, 0.25],
        "written_premium": [500.0, 300.0, 100.0], "sum_insured": [np.nan, 250000.0, 30000.0],
        "cover_type": ["comprehensive", None, None]})


RP = np.array([400.0, 120.0, 30.0])


def _build(**kw):
    return op.build(_policies(), RP, op.load_expenses(market="mkt", table=_table(**kw)))


def test_all_zero_gives_risk_premium():
    b = _build()
    np.testing.assert_allclose(b["office"], RP)
    np.testing.assert_allclose(b["office_tax_on_top"], RP)
    np.testing.assert_allclose(b["denominator"], 1.0)


def test_tax_only():
    b = _build(tax=0.08)
    np.testing.assert_allclose(b["office"], RP / 0.92)
    np.testing.assert_allclose(b["office_tax_on_top"], RP * 1.08)
    np.testing.assert_allclose(b["part_premium_tax"], 0.08 * RP / 0.92)


def test_cat_property_only():
    b = _build(cat=0.12)
    assert b.loc[0, "cat"] == 0.0
    np.testing.assert_allclose(b.loc[1:, "cat"], 0.12 * RP[1:])
    np.testing.assert_allclose(b["office"], RP * np.array([1.0, 1.12, 1.12]))


def test_commission_by_channel():
    b = _build(direct=0.0, broker=0.20)
    np.testing.assert_allclose(b["commission_rate"], [0.20, 0.0, 0.20])
    np.testing.assert_allclose(b["office"], RP / np.array([0.8, 1.0, 0.8]))


def test_comparison_site_nan_taken_as_zero_and_flagged():
    pol = _policies()
    pol.loc[0, "channel"] = "comparison_site"
    b = op.build(pol, RP, op.load_expenses(market="mkt", table=_table(broker=0.2)))
    assert b.loc[0, "commission_rate"] == 0.0 and bool(b.loc[0, "commission_assumed_zero"])
    assert not b.loc[1:, "commission_assumed_zero"].any()


def test_fixed_not_prorated_and_prorated_alternative():
    b = _build(fixed=85.0)
    np.testing.assert_allclose(b["office"], RP + 85.0)
    np.testing.assert_allclose(b["office_fixed_prorated"], RP + 85.0 * np.array([1.0, 0.5, 0.25]))
    # annual equivalent: RP/exposure + one full fixed expense
    np.testing.assert_allclose(b["office_pa"], RP / np.array([1.0, 0.5, 0.25]) + 85.0)


def test_full_emerging_style_algebra_and_components_sum():
    kw = dict(fixed=85.0, var=0.15, direct=0.0, broker=0.20, tax=0.08, cat=0.12, profit=0.08)
    b = _build(**kw)
    d = np.array([1 - 0.15 - 0.20 - 0.08 - 0.08, 1 - 0.15 - 0.08 - 0.08, 1 - 0.15 - 0.20 - 0.08 - 0.08])
    base = RP * np.array([1.0, 1.12, 1.12]) + 85.0
    np.testing.assert_allclose(b["denominator"], d)
    np.testing.assert_allclose(b["office"], base / d)
    np.testing.assert_allclose(b["office_tax_on_top"], base / (d + 0.08) * 1.08)
    assert (b["office_tax_on_top"] < b["office"]).all()
    parts = b["cost_base"] + b[[c for c in b.columns if c.startswith("part_")]].sum(axis=1)
    np.testing.assert_allclose(parts, b["office"])


def test_real_expenses_table_parses_both_markets():
    path = ROOT / "data" / "part2" / "emerging" / "expenses.parquet"
    if not path.exists():
        pytest.skip("data not present")
    for m in ("emerging", "developed"):
        e = op.load_expenses(path, m)
        assert e["fixed"] > 0 and 0 < e["cat"] < 1 and set(e["commission"]) >= {"direct", "broker"}
    assert math.isnan(op.load_expenses(path, "emerging")["commission"]["comparison_site"])


def test_missing_prediction_raises():
    with pytest.raises(ValueError):
        op.build(_policies(), [1.0, np.nan, 2.0], op.load_expenses(market="mkt", table=_table()))
