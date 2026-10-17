"""Regression gate for the market-spec refactor: Part 1 targets and model matrices must stay bit-identical.

tests/golden_part1.json was computed on the pre-refactor code for EXPERIMENTS["L20_mono_lowreg"]."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

from market_spec import use_spec  # noqa: E402
from configs import EXPERIMENTS  # noqa: E402
from features import build_base, model_matrix  # noqa: E402
from target import prepare_target  # noqa: E402

pytestmark = pytest.mark.part1_data

GOLDEN = json.loads((ROOT / "tests" / "golden_part1.json").read_text())


def h(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, float)).round(6).tobytes()).hexdigest()[:16]


@pytest.fixture(scope="module")
def base():
    use_spec("part1")
    return build_base()


@pytest.mark.parametrize("line", ["motor", "buildings", "contents"])
def test_golden_part1(base, line):
    use_spec("part1")
    cfg = EXPERIMENTS["L20_mono_lowreg"]
    train, test = base[f"train_{line}"], base[f"test_{line}"]
    tgt, info = prepare_target(train, test, base["claims"], line, cfg["target"])
    fc = dict(cfg["features"])
    fc["drop"] = list(fc.get("drop", []))
    X, Xt, _ = model_matrix(train, test, fc)
    got = {
        "y_train": h(tgt.y_train), "y_eval": h(tgt.y_eval), "w": h(tgt.w), "ibnr": h(tgt.ibnr),
        "loading_rows": h(info["loading_rows"]), "loading": round(float(info["loading"]), 6),
        "trend": {str(y): round(float(v), 6) for y, v in info["trend"].items()},
        "cap": round(float(info["cap"]), 3), "X_cols": list(X.columns),
        "X_numeric_hash": h(X.select_dtypes("number").fillna(-1).to_numpy()),
        "Xt_numeric_hash": h(Xt.select_dtypes("number").fillna(-1).to_numpy()),
        "n_train": len(train), "n_test": len(test),
    }
    assert got == GOLDEN[line]
