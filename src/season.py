"""Seasonal exposure weights: a monthly claim-cost index (per exposure day) applied through the exposure offset."""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

_DAY = np.timedelta64(1, "D")


def month_days(ps, pe, years):
    """Days each row covers in each calendar month (columns 0..11), summed over `years`; period_end inclusive."""
    ps = np.asarray(pd.to_datetime(ps), dtype="datetime64[D]")
    pe = np.asarray(pd.to_datetime(pe), dtype="datetime64[D]") + 1
    out = np.zeros((len(ps), 12))
    for y in years:
        for m in range(12):
            a = np.datetime64(f"{y}-{m + 1:02d}-01")
            b = np.datetime64(f"{y + 1}-01-01") if m == 11 else np.datetime64(f"{y}-{m + 2:02d}-01")
            d = (np.minimum(pe, b) - np.maximum(ps, a)) / _DAY
            out[:, m] += np.clip(d, 0, None)
    return out


def monthly_index(claims, rows, line, years=(2021, 2022), cap_q=0.99):
    """Monthly cost index (12,) per exposure day for one line: sum over perils of claim count x capped mean severity,
    divided by exposure days in the month, normalised to exposure-day-weighted mean 1."""
    c = claims[(claims["line"] == line) & (claims["net_incurred"] > 0)]
    c = c[c["loss_date"].dt.year.isin(years)]
    cost = np.zeros(12)
    for _, g in c.groupby("peril"):
        sev = g["net_incurred"].clip(upper=g["net_incurred"].quantile(cap_q)).mean()
        cnt = np.bincount(g["loss_date"].dt.month.values - 1, minlength=12)
        cost += cnt * sev
    days = month_days(rows["period_start"], rows["period_end"], years).sum(axis=0)
    idx = cost / np.where(days > 0, days, np.nan)
    idx = np.nan_to_num(idx, nan=0.0)
    return idx / ((idx * days).sum() / days.sum())


def seasonal_weight(ps, pe, index):
    """Per-row exposure-weighted mean of `index` over the days the row covers (all years 2020..2026); 0 days -> 1."""
    d = month_days(ps, pe, range(2020, 2027))
    tot = d.sum(axis=1)
    return np.where(tot > 0, (d @ np.asarray(index, float)) / np.where(tot > 0, tot, 1), 1.0)


@lru_cache(maxsize=None)
def get_index(line):
    """Monthly index for `line` from the cached claims / train rows, years 2021-2022 only (2023 holdout stays clean)."""
    root = Path(__file__).resolve().parents[1] / "experiments" / "cache"
    claims = pd.read_parquet(root / "claims.parquet")
    rows = pd.read_parquet(root / f"train_{line}.parquet", columns=["period_start", "period_end"])
    return monthly_index(claims, rows, line, years=(2021, 2022))
