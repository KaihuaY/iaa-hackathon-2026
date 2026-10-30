import numpy as np
import pandas as pd

import pipeline
import season


def _d(*s):
    return pd.to_datetime(list(s))


def test_month_days():
    d = season.month_days(_d("2021-01-01"), _d("2021-12-31"), (2021,))
    assert d[0].tolist() == [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    d = season.month_days(_d("2021-12-15"), _d("2022-01-10"), (2021, 2022))
    assert d[0, 11] == 17 and d[0, 0] == 10 and d.sum() == 27


def test_seasonal_weight():
    ps, pe = _d("2021-01-01", "2021-12-01"), _d("2021-12-31", "2021-12-31")
    assert np.allclose(season.seasonal_weight(ps, pe, np.ones(12)), 1.0)
    idx = np.ones(12); idx[11] = 2.0
    assert np.allclose(season.seasonal_weight(ps[1:], pe[1:], idx), 2.0)
    # index normalised over uniform full-year exposure -> full-year weight ~1
    rng = np.random.default_rng(0)
    raw = rng.uniform(0.5, 2, 12)
    d = season.month_days(ps[:1], pe[:1], (2021,))[0]
    idx = raw / ((raw * d).sum() / d.sum())
    assert abs(season.seasonal_weight(ps[:1], pe[:1], idx)[0] - 1) < 1e-3


def test_monthly_index_storm():
    rng = np.random.default_rng(1)
    rows = pd.DataFrame({"period_start": _d(*["2021-01-01"] * 2000), "period_end": _d(*["2022-12-31"] * 2000)})
    months = np.repeat(np.arange(1, 13), 1000)
    n = np.where(np.isin(months, [12, 1, 2]), 3000, 1000)
    dates = []
    for y in (2021, 2022):
        for m in range(1, 13):
            k = 300 if m in (12, 1, 2) else 100
            dates += [pd.Timestamp(y, m, 15)] * k
    claims = pd.DataFrame({"line": "buildings", "peril": "storm", "loss_date": dates,
                           "net_incurred": rng.uniform(100, 200, len(dates))})
    idx = season.monthly_index(claims, rows, "buildings")
    assert idx[0] > 1.3 and idx[6] < 1


def test_offset_exposure(monkeypatch):
    tr = pd.DataFrame({"exposure": [1.0, 0.5], "period_start": _d("2021-01-01", "2021-12-01"), "period_end": _d("2021-12-31", "2021-12-31")})
    te = tr.copy()
    e_tr, e_te = pipeline._offset_exposure(tr, te, {})
    assert np.array_equal(e_tr, tr.exposure.values) and np.array_equal(e_te, te.exposure.values)
    idx = np.ones(12); idx[11] = 2.0
    monkeypatch.setattr(season, "get_index", lambda line: idx)
    e_tr, _ = pipeline._offset_exposure(tr, te, {"season_offset": 1.0}, line="buildings")
    w = season.seasonal_weight(tr.period_start, tr.period_end, idx)
    assert np.allclose(e_tr, tr.exposure.values * w) and abs(e_tr[1] - 1.0) < 1e-9
