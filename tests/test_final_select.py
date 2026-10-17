import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import final_select as fs  # noqa: E402


def fake(seed, cv):
    r = np.random.default_rng(seed)
    lines = {}
    for i, l in enumerate(["motor", "buildings"]):
        lines[l] = {"y_eval": np.random.default_rng(0).gamma(1, 1, 100) * (np.arange(100) % 3 == 0),
                    "oof": r.uniform(0.2, 1.5, 100), "test_row_id": np.arange(50) + 1000 * i,
                    "test_pred": r.uniform(0.5, 2, 50)}
    return {"lines": lines, "cv": cv, "cv_nested": cv}


A, B = fake(1, 0.10), fake(2, 0.12)


def test_equal_average_and_factors():
    w = fs.weights([A, B], "equal")
    df, d2 = fs.blend([A, B], w, {"buildings": 2.0})
    m = df[df.row_id < 1000].prediction.to_numpy()
    b = df[df.row_id >= 1000].prediction.to_numpy()
    assert len(df) == 100
    np.testing.assert_allclose(m, (A["lines"]["motor"]["test_pred"] + B["lines"]["motor"]["test_pred"]) / 2)
    np.testing.assert_allclose(b, (A["lines"]["buildings"]["test_pred"] + B["lines"]["buildings"]["test_pred"]))  # x2 /2
    assert set(d2) == {"motor", "buildings", "ALL"} and np.isfinite(d2["ALL"][0])


def test_cv_weights_sum_to_one():
    w = fs.weights([A, B], "cv")
    assert w.sum() == pytest.approx(1) and w[1] > w[0]
