import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from segment_probe import segment_mask  # noqa: E402


@pytest.fixture
def te():
    n = 100
    return pd.DataFrame({
        "si": np.arange(1, n + 1, dtype=float),
        "wind": (np.arange(n) * 7 % 100).astype(float),
        "cat": ["a", "b", "c", "d"] * 25,
        "period_start": pd.date_range("2025-01-01", periods=n, freq="15D"),
    })


def test_equals(te):
    assert segment_mask(te, "cat=a").sum() == 25


def test_quartile(te):
    assert segment_mask(te, "si_q=top").sum() == 25
    assert segment_mask(te, "si_q=bottom").sum() == 25


def test_decile(te):
    m = segment_mask(te, "si_d=top")
    assert m.sum() == 10 and te.si[m].min() == 91
    assert segment_mask(te, "si_d=bottom").sum() == 10


def test_percent(te):
    assert segment_mask(te, "si_p=10").sum() == 10
    assert segment_mask(te, "si_p=30").sum() == 30


def test_in(te):
    assert segment_mask(te, "cat in a|b|c").sum() == 75


def test_compound(te):
    m = segment_mask(te, "si_q=top & wind_q=top")
    a, b = segment_mask(te, "si_q=top"), segment_mask(te, "wind_q=top")
    assert (m == (a & b)).all() and 0 < m.sum() < 25
    m3 = segment_mask(te, "si_q=top & cat in a|b")
    assert (m3 == (a & te.cat.isin(["a", "b"]).values)).all()


def test_month(te):
    m = segment_mask(te, "period_start_month in 1|2|3")
    assert m.sum() == te.period_start.dt.month.isin([1, 2, 3]).sum() > 0


def test_guard_conditions(te):
    assert segment_mask(te, "cat=zzz").sum() == 0
    assert segment_mask(te, "cat in a|b|c|d").all()
