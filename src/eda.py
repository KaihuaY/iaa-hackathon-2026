"""Aggregation helpers for the EDA notebook."""
import numpy as np
import pandas as pd

SKIP = {"row_id", "policy_id", "customer_id", "address_id", "period_start", "period_end", "exposure",
        "claim_count", "claim_cost", "year", "compulsory_excess", "customer_since_year", "year_built"}


def summary(df: pd.DataFrame, by) -> pd.DataFrame:
    """Exposure, claims, frequency, severity, pure premium and loss ratio by group."""
    g = df.groupby(by, observed=True).agg(rows=("row_id", "size"), exposure=("exposure", "sum"),
                                          claims=("claim_count", "sum"), cost=("claim_cost", "sum"),
                                          premium=("written_premium", "sum"))
    g["frequency"] = g.claims / g.exposure
    g["severity"] = g.cost / g.claims
    g["pure_premium"] = g.cost / g.exposure
    g["loss_ratio"] = g.cost / g.premium
    return g


def bin_feature(s: pd.Series, n_bins: int = 10, top: int = 15) -> pd.Series:
    """Ordered string levels for a one-way: quantile bins for numerics, top-N for categoricals."""
    if s.dtype == bool or s.dtype == object or s.nunique() <= 12:
        out = s.astype(object).where(s.notna(), "missing").astype(str)
        if out.nunique() > top:
            keep = out.value_counts().index[:top]
            out = out.where(out.isin(keep), "other")
        return out
    q = pd.qcut(s, n_bins, duplicates="drop")
    labels = q.cat.categories
    fmt = lambda x: f"{x:,.0f}" if abs(x) >= 10 else f"{x:.2f}"
    names = {iv: f"{fmt(iv.left)}–{fmt(iv.right)}" for iv in labels}
    out = q.map(names).astype(object).where(s.notna(), "missing")
    order = [names[iv] for iv in labels] + (["missing"] if s.isna().any() else [])
    return pd.Categorical(out, categories=order, ordered=True)


def oneway(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """Frequency and pure-premium relativities (to the overall level) by feature level."""
    levels = bin_feature(df[feature])
    t = summary(df.assign(_lvl=levels), "_lvl")
    base = df.claim_count.sum() / df.exposure.sum(), df.claim_cost.sum() / df.exposure.sum()
    t["freq_rel"] = t.frequency / base[0]
    t["pp_rel"] = t.pure_premium / base[1]
    if not isinstance(levels, pd.Categorical):
        t = t.sort_index(key=lambda i: i.map(_sort_key))
    return t.reset_index().rename(columns={"_lvl": "level"})


def _sort_key(v):
    try:
        return (0, float(v))
    except ValueError:
        return (1 if v not in ("missing", "other") else 2, v)


def features(df: pd.DataFrame, exclude=()) -> list[str]:
    return [c for c in df.columns if c not in SKIP and c not in exclude and df[c].nunique() > 1]


def lag_cdf(claims: pd.DataFrame, years=(2021, 2022), max_days: int = 730) -> pd.DataFrame:
    """Share of claims reported within d days of loss, from mature loss years, per line."""
    c = claims[claims.loss_date.dt.year.isin(years)]
    lag = (c.report_date - c.loss_date).dt.days
    days = np.arange(0, max_days + 1)
    return pd.DataFrame({line: [(lag[c.line == line] <= d).mean() for d in days]
                         for line in sorted(c.line.unique())}, index=days)


def psi(a: pd.Series, b: pd.Series) -> float:
    """Population stability index of b relative to a, on a's binning."""
    if a.dtype == bool or a.dtype == object or a.nunique() <= 12:
        pa, pb = a.astype(str).value_counts(normalize=True), b.astype(str).value_counts(normalize=True)
    else:
        edges = np.unique(np.nanquantile(a, np.linspace(0, 1, 11)))
        edges[0], edges[-1] = -np.inf, np.inf
        pa = pd.cut(a, edges).value_counts(normalize=True)
        pb = pd.cut(b, edges).value_counts(normalize=True)
    idx = pa.index.union(pb.index)
    pa, pb = pa.reindex(idx).fillna(1e-4).clip(lower=1e-4), pb.reindex(idx).fillna(1e-4).clip(lower=1e-4)
    return float(((pb - pa) * np.log(pb / pa)).sum())
