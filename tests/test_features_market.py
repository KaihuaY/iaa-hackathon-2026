"""Review 11: hist_nan_2021 in a pooled market marks the first observed year of EACH market as unobservable history."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]

import market_spec as ms  # noqa: E402
from features import model_matrix  # noqa: E402


def _frames(market=True):
    tr = pd.DataFrame({"year": [2021, 2022, 2023, 2024, 2023, 2024], "pol_claims": [0, 1, 2, 3, 0, 1],
                       "cust_expo": [0.0, 1, 2, 2, 0, 1], "lines_held": [1, 1, 2, 2, 1, 1], "x": [1.0, 2, 3, 4, 5, 6]})
    if market:
        tr["market"] = ["developed"] * 4 + ["emerging"] * 2
    te = tr.head(2).assign(year=2025)
    return tr, te


def test_first_year_of_each_market_is_nan():
    prev = ms.get_spec()
    ms.use_spec("part1")
    try:
        tr, te = _frames()
        X, Xt, _ = model_matrix(tr, te, {"hist_nan_2021": True})
        nan = X.pol_claims.isna().tolist()
        assert nan == [True, False, False, False, True, False]
        assert X.cust_expo.isna().tolist() == nan and not X.lines_held.isna().any() and not X.x.isna().any()
        assert not Xt.pol_claims.isna().any()                     # test rows keep their history
        X0, _, _ = model_matrix(tr, te, {})                        # option off: unchanged
        assert not X0.pol_claims.isna().any()
        tr1, te1 = _frames(market=False)                           # no market column: spec first year only (Part 1 rule)
        X1, _, _ = model_matrix(tr1, te1, {"hist_nan_2021": True})
        assert X1.pol_claims.isna().tolist() == [True] + [False] * 5
    finally:
        ms.use_spec(prev)
