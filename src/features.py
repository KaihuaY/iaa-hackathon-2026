"""Model matrices per line: enriched rating factors + experience rating (strictly prior years)."""
from pathlib import Path

import numpy as np
import pandas as pd

from data import LINES, LEAKY_COLS, build_all, enrich, row_type

CACHE = Path(__file__).resolve().parents[1] / "experiments" / "cache"
YEARS = [2021, 2022, 2023, 2024, 2025]
WEATHER = ["hail_days", "wind_exposure_index", "freezing_days", "dry_summer_index", "wildfire_index",
           "rainfall_mm", "mean_temperature_c"]
DROP = ["row_id", "policy_id", "customer_id", "address_id", "period_start", "period_end", "claim_count",
        "claim_cost", "year", "compulsory_excess", "customer_since_year", "year_built", "exposure"] + LEAKY_COLS


LOOKBACK = 2  # years. A fixed window keeps history features comparable across years; cumulative totals grew
               # every year, so the model learned "more history = later year" and extrapolated on 2025 (G5, 17 Oct).


def _prior(rows: pd.DataFrame, key: str, prefix: str) -> pd.DataFrame:
    """Exposure and claim count for `key` over the LOOKBACK years strictly before each row's year."""
    agg = rows.groupby([key, "year"]).agg(e=("exposure", "sum"), n=("claim_count", "sum")).unstack("year")
    agg = agg.reindex(columns=pd.MultiIndex.from_product([["e", "n"], YEARS]), fill_value=0).fillna(0)
    out = []
    for m in ["e", "n"]:
        win = sum(agg[m].shift(k, axis=1).fillna(0) for k in range(1, LOOKBACK + 1))   # years Y-1 .. Y-LOOKBACK
        out.append(win.stack().rename(f"{prefix}_{'expo' if m == 'e' else 'claims'}"))
    return pd.concat(out, axis=1).rename_axis([key, "year"]).reset_index()


def _prior_cols(rows: pd.DataFrame, key: str, prefix: str, cols: list) -> pd.DataFrame:
    """Sum of arbitrary per-row value columns for `key` over the LOOKBACK years before each row's year."""
    agg = rows.groupby([key, "year"])[cols].sum().unstack("year")
    agg = agg.reindex(columns=pd.MultiIndex.from_product([cols, YEARS]), fill_value=0).fillna(0)
    out = []
    for c in cols:
        win = sum(agg[c].shift(k, axis=1).fillna(0) for k in range(1, LOOKBACK + 1))
        out.append(win.stack().rename(f"hx_{prefix}_{c}"))
    return pd.concat(out, axis=1).rename_axis([key, "year"]).reset_index()


def rich_history(train_all: pd.DataFrame, test_all: pd.DataFrame, claims: pd.DataFrame) -> pd.DataFrame:
    """Prior claim COST (capped at the line's 99th pct) and prior claim counts BY PERIL, for policy and customer.
    Columns are prefixed hx_ and only used when a config sets features['rich_history']."""
    c = claims.dropna(subset=["row_id"]).copy()
    cap = c.groupby("line").net_incurred.transform(lambda s: s.quantile(0.99))
    c["cost"] = c.net_incurred.clip(upper=cap)
    per_row = c.pivot_table(index="row_id", columns="peril", values="claim_id", aggfunc="count", fill_value=0)
    per_row.columns = [f"n_{p}" for p in per_row.columns]
    per_row["cost"] = c.groupby("row_id").cost.sum()
    vals = list(per_row.columns)
    rows = pd.concat([train_all[["row_id", "line", "policy_id", "customer_id", "year"]],
                      test_all[["row_id", "line", "policy_id", "customer_id", "year"]]], ignore_index=True)
    rows = rows.merge(per_row, left_on="row_id", right_index=True, how="left")
    rows[vals] = rows[vals].fillna(0)
    feats = rows[["row_id", "policy_id", "customer_id", "year"]].copy()
    feats = feats.merge(_prior_cols(rows, "policy_id", "pol", vals), on=["policy_id", "year"], how="left")
    feats = feats.merge(_prior_cols(rows, "customer_id", "cust", vals), on=["customer_id", "year"], how="left")
    hx = [x for x in feats.columns if x.startswith("hx_")]
    feats[hx] = feats[hx].fillna(0).astype("float32")
    return feats.drop(columns=["policy_id", "customer_id", "year"])


def experience_features(train_all: pd.DataFrame, test_all: pd.DataFrame) -> pd.DataFrame:
    """Per (line,row) prior-claims features: same policy, same customer (own line and all lines)."""
    rows = pd.concat([train_all, test_all.assign(claim_count=np.nan)], ignore_index=True)
    rows["claim_count"] = rows.claim_count.fillna(0)               # test years never feed earlier years anyway
    feats = rows[["row_id", "line", "policy_id", "customer_id", "year"]].copy()
    feats = feats.merge(_prior(rows, "policy_id", "pol"), on=["policy_id", "year"], how="left")
    feats = feats.merge(_prior(rows, "customer_id", "cust"), on=["customer_id", "year"], how="left")
    for l in LINES:
        part = _prior(rows[rows.line == l], "customer_id", f"cust_{l}")
        feats = feats.merge(part, on=["customer_id", "year"], how="left")
    # lines held by the customer in the same year (known at pricing time)
    held = rows.groupby(["customer_id", "year"]).line.nunique().rename("lines_held").reset_index()
    feats = feats.merge(held, on=["customer_id", "year"], how="left")
    num = [c for c in feats.columns if c.endswith(("_expo", "_claims"))]
    feats[num] = feats[num].fillna(0)
    for p in ["pol", "cust"] + [f"cust_{l}" for l in LINES]:
        feats[f"{p}_rate"] = feats[f"{p}_claims"] / (feats[f"{p}_expo"] + 0.5)
    return feats.drop(columns=["policy_id", "customer_id", "year"])


def build_base(refresh: bool = False) -> dict:
    """Enriched + experience features for train and test, per line (cached as parquet)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    paths = {f"{k}_{l}": CACHE / f"{k}_{l}.parquet" for k in ("train", "test") for l in LINES}
    if not refresh and all(p.exists() for p in paths.values()):
        out = {k: pd.read_parquet(p) for k, p in paths.items()}
        out["claims"] = pd.read_parquet(CACHE / "claims.parquet")
        return out
    log = []
    D = build_all(log)
    tr = {l: enrich(D[f"train_{l}"], D["customers"], D["addresses"], D["weather"]) for l in LINES}
    te = {l: enrich(D[f"test_{l}"], D["customers"], D["addresses"], D["weather"]) for l in LINES}
    train_all = pd.concat([tr[l].assign(line=l) for l in LINES], ignore_index=True)
    test_all = pd.concat([te[l].assign(line=l) for l in LINES], ignore_index=True)
    exp = experience_features(train_all[["row_id", "line", "policy_id", "customer_id", "year", "exposure", "claim_count"]],
                              test_all[["row_id", "line", "policy_id", "customer_id", "year", "exposure"]])
    exp = exp.merge(rich_history(train_all, test_all, D["claims"]), on="row_id", how="left")
    out = {}
    for l in LINES:
        for k, d in (("train", tr[l]), ("test", te[l])):
            d = d.merge(exp[exp.line == l].drop(columns="line"), on="row_id", how="left")
            d["row_type"] = row_type(d)
            d["start_month"] = d.period_start.dt.month
            d.to_parquet(paths[f"{k}_{l}"])
            out[f"{k}_{l}"] = d
    D["claims"].to_parquet(CACHE / "claims.parquet")
    out["claims"] = D["claims"]
    return out


def model_matrix(train: pd.DataFrame, test: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """Feature frames (same columns, shared category levels) for one line."""
    drop = set(DROP)
    if not cfg.get("history", True):
        drop |= {c for c in train.columns if c.startswith(("pol_", "cust_")) or c == "lines_held"}
    if not cfg.get("rich_history", False):
        drop |= {c for c in train.columns if c.startswith("hx_")}
    if not cfg.get("premium", False):
        drop.add("written_premium")
    drop |= set(cfg.get("drop", []))
    cols = [c for c in train.columns if c not in drop and c in test.columns]
    X, Xt = train[cols].copy(), test[cols].copy()
    if cfg.get("hist_nan_2021", False):
        # Data starts in 2021, so those rows' history is unobservable, not "no claims" (else history proxies the year).
        hist = [c for c in cols if c.startswith(("pol_", "cust_"))]
        X.loc[train.year.values == 2021, hist] = np.nan
    if cfg.get("weather_pca"):
        from sklearn.decomposition import PCA
        w = [c for c in WEATHER if c in X]
        mu, sd = X[w].mean(), X[w].std()
        pca = PCA(n_components=cfg["weather_pca"]).fit(((X[w] - mu) / sd).fillna(0))
        for d in (X, Xt):
            comp = pca.transform(((d[w] - mu) / sd).fillna(0))
            for i in range(comp.shape[1]):
                d[f"weather_pc{i + 1}"] = comp[:, i]
            d.drop(columns=w, inplace=True)
    for d in (X, Xt):                                     # halve memory: float64 -> float32
        f64 = d.select_dtypes("float64").columns
        d[f64] = d[f64].astype("float32")
    cats = []
    for c in X.columns:
        if X[c].dtype == object or X[c].dtype == bool or str(X[c].dtype) == "category":
            levels = sorted(set(X[c].dropna().astype(str)) | set(Xt[c].dropna().astype(str)))
            X[c] = pd.Categorical(X[c].astype(str).where(X[c].notna()), categories=levels)
            Xt[c] = pd.Categorical(Xt[c].astype(str).where(Xt[c].notna()), categories=levels)
            cats.append(c)
    return X, Xt, cats
