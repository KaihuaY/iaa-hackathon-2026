"""Level-frequency stability screen: does a feature's level ranking by claim frequency persist over time?

For each feature (categorical levels, or 8 quantile bins for numerics) we compute the claim frequency
(claims / exposure) of every level in the earlier training years (all but the last two) and in the last mature
year (`spec.last_mature_year`), and report
  stability = exposure-weighted correlation between the two sets of level frequencies,
  spread    = exposure-weighted sd of level frequencies (all years) relative to the overall frequency.
A feature with low stability AND low spread is noise-like (no persistent signal and little differentiation).

CAVEAT: weather variables (hail_days, rainfall_mm, ...) vary by year by design; their level frequencies
shuffle between years because the weather does, not because the feature is noise. Do not drop them on this
screen alone.
"""
import numpy as np
import pandas as pd

NOISE_STAB, NOISE_SPREAD, WATCH_STAB = 0.2, 0.1, 0.5


def _wcorr(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    w = w / w.sum()
    mx, my = (w * x).sum(), (w * y).sum()
    vx, vy = (w * (x - mx) ** 2).sum(), (w * (y - my) ** 2).sum()
    if vx <= 1e-18 or vy <= 1e-18:
        return float("nan")
    return float((w * (x - mx) * (y - my)).sum() / np.sqrt(vx * vy))


def _keys(s: pd.Series, is_cat: bool) -> pd.Series:
    if is_cat:
        return s.astype(object)
    s = pd.to_numeric(s, errors="coerce")
    if s.nunique() <= 8:
        return s
    return pd.qcut(s, 8, duplicates="drop").astype(object)


def stability_table(train_df: pd.DataFrame, X: pd.DataFrame, cats: list, spec, min_rows: int = 300) -> pd.DataFrame:
    """Columns: feature, kind, n_levels, stability, spread; sorted by stability (descending, NaN last).
    `train_df` needs year, claim_count, exposure and the same row order as X. Levels with fewer than
    `min_rows` rows in either period are ignored."""
    yrs = sorted(spec.train_years)
    early_years = yrs[:-2]
    late_year = spec.last_mature_year
    year = train_df["year"].to_numpy()
    early, late = np.isin(year, early_years), year == late_year
    cc, ex = train_df["claim_count"].to_numpy(float), train_df["exposure"].to_numpy(float)
    overall = cc.sum() / ex.sum()
    rows = []
    catset = set(cats)
    for c in X.columns:
        k = _keys(X[c], c in catset).reset_index(drop=True)
        d = pd.DataFrame({"k": k, "n_e": early.astype(int), "n_l": late.astype(int), "cc": cc, "ex": ex,
                          "cc_e": cc * early, "ex_e": ex * early, "cc_l": cc * late, "ex_l": ex * late}).dropna(subset=["k"])
        g = d.groupby("k", observed=True)[["n_e", "n_l", "cc", "ex", "cc_e", "ex_e", "cc_l", "ex_l"]].sum()
        ok = g[(g.n_e >= min_rows) & (g.n_l >= min_rows) & (g.ex_e > 0) & (g.ex_l > 0)]
        stab = float("nan")
        if len(ok) >= 3:
            stab = _wcorr((ok.cc_e / ok.ex_e).to_numpy(), (ok.cc_l / ok.ex_l).to_numpy(),
                          np.minimum(ok.ex_e, ok.ex_l).to_numpy())
        allg = g[(g.n_e + g.n_l >= min_rows) & (g.ex > 0)]
        spread = float("nan")
        if len(allg) >= 2 and overall > 0:
            f, w = (allg.cc / allg.ex).to_numpy(), allg.ex.to_numpy()
            m = np.average(f, weights=w)
            spread = float(np.sqrt(np.average((f - m) ** 2, weights=w)) / overall)
        rows.append({"feature": c, "kind": "cat" if c in catset else "num", "n_levels": len(ok),
                     "stability": stab, "spread": spread})
    out = pd.DataFrame(rows)
    return out.sort_values("stability", ascending=False, na_position="last", ignore_index=True)


def suggest(tab: pd.DataFrame) -> tuple[list, list]:
    """(noise, watch): noise = stability < 0.2 and spread < 0.1; watch = stability 0.2-0.5."""
    s = tab[tab.stability.notna()]
    noise = s[(s.stability < NOISE_STAB) & (s.spread < NOISE_SPREAD)].feature.tolist()
    watch = s[(s.stability >= NOISE_STAB) & (s.stability < WATCH_STAB)].feature.tolist()
    return noise, watch
