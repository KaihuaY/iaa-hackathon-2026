"""Review 13: the written-premium features are cfg-gated, additive and off by default."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]

import market_spec as ms  # noqa: E402
from features import model_matrix  # noqa: E402


def _frames():
    tr = pd.DataFrame({"year": [2023, 2023, 2024, 2024], "market": ["developed", "emerging", "emerging", "emerging"],
                       "exposure": [1.0, 0.5, 1.0, 0.25], "written_premium": [100.0, 200.0, 300.0, 100.0],
                       "x": [1.0, 2, 3, 4]})
    te = pd.DataFrame({"year": [2025, 2025], "market": ["emerging", "emerging"], "exposure": [1.0, 0.5],
                       "written_premium": [400.0, 200.0], "x": [5.0, 6]})
    return tr, te


def test_premium_gate_default_off_and_values():
    prev = ms.get_spec()
    ms.use_spec("part1")
    try:
        tr, te = _frames()
        X0, Xt0, _ = model_matrix(tr, te, {})
        assert "written_premium" not in X0 and not any(c.startswith("log_premium") for c in X0)
        X, Xt, _ = model_matrix(tr, te, {"premium": True, "premium_log": True, "premium_rate": True, "premium_rel": True})
        assert "written_premium" in X
        np.testing.assert_allclose(X.log_premium, np.log(tr.written_premium), rtol=1e-6)
        np.testing.assert_allclose(X.log_premium_rate, np.log(tr.written_premium / tr.exposure), rtol=1e-6)
        # (market, year) means: developed 2023 = 100, emerging 2023 = 200, emerging 2024 = 200, emerging 2025 = 300
        np.testing.assert_allclose(X.log_premium_rel, np.log([1.0, 1.0, 1.5, 0.5]), rtol=1e-6)
        np.testing.assert_allclose(Xt.log_premium_rel, np.log([4 / 3, 2 / 3]), rtol=1e-6)
        assert list(X.columns) == list(Xt.columns)
        for c in X0.columns:                                         # additive: existing columns unchanged
            pd.testing.assert_series_equal(X[c], X0[c])
        Xn, _, _ = model_matrix(tr.drop(columns="market"), te.drop(columns="market"), {"premium_rel": True})
        np.testing.assert_allclose(Xn.log_premium_rel, np.log(tr.written_premium.values / [150, 150, 200, 200]), rtol=1e-6)
    finally:
        ms.use_spec(prev)
