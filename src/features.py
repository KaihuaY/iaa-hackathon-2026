"""Model matrices per line: enriched rating factors + experience rating (strictly prior years)."""
from pathlib import Path

import numpy as np
import pandas as pd

from data import build_all, enrich, row_type
from market_spec import get_spec, warn_once

WEATHER = ["hail_days", "wind_exposure_index", "freezing_days", "dry_summer_index", "wildfire_index",
           "rainfall_mm", "mean_temperature_c"]
_DROP_BASE = ["row_id", "policy_id", "customer_id", "address_id", "period_start", "period_end", "claim_count",
              "claim_cost", "year", "compulsory_excess", "customer_since_year", "year_built", "exposure"]


def __getattr__(name):
    """CACHE / YEARS / DROP follow the active MarketSpec at access time (PEP 562)."""
    s = get_spec()
    if name == "CACHE":
        return s.cache_path
    if name == "YEARS":
        return s.all_years
    if name == "DROP":
        return _DROP_BASE + list(s.leaky_cols)
    raise AttributeError(f"module 'features' has no attribute {name!r}")


LOOKBACK = 2  # years. A fixed window keeps history features comparable across years; cumulative totals grew
               # every year, so the model learned "more history = later year" and extrapolated on 2025 (G5, 17 Oct).


def _prior(rows: pd.DataFrame, key: str, prefix: str) -> pd.DataFrame:
    """Exposure and claim count for `key` over the LOOKBACK years strictly before each row's year."""
    agg = rows.groupby([key, "year"]).agg(e=("exposure", "sum"), n=("claim_count", "sum")).unstack("year")
    agg = agg.reindex(columns=pd.MultiIndex.from_product([["e", "n"], get_spec().all_years]), fill_value=0).fillna(0)
    out = []
    for m in ["e", "n"]:
        win = sum(agg[m].shift(k, axis=1).fillna(0) for k in range(1, LOOKBACK + 1))   # years Y-1 .. Y-LOOKBACK
        out.append(win.stack().rename(f"{prefix}_{'expo' if m == 'e' else 'claims'}"))
    return pd.concat(out, axis=1).rename_axis([key, "year"]).reset_index()


def _prior_cols(rows: pd.DataFrame, key: str, prefix: str, cols: list) -> pd.DataFrame:
    """Sum of arbitrary per-row value columns for `key` over the LOOKBACK years before each row's year."""
    agg = rows.groupby([key, "year"])[cols].sum().unstack("year")
    agg = agg.reindex(columns=pd.MultiIndex.from_product([cols, get_spec().all_years]), fill_value=0).fillna(0)
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
    if "peril" in c.columns:
        per_row = c.pivot_table(index="row_id", columns="peril", values="claim_id", aggfunc="count", fill_value=0)
        per_row.columns = [f"n_{p}" for p in per_row.columns]
    else:
        warn_once("rich_history: claims have no peril column; per-peril counts skipped")
        per_row = pd.DataFrame(index=c.row_id.unique())
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
    for l in get_spec().lines:
        part = _prior(rows[rows.line == l], "customer_id", f"cust_{l}")
        feats = feats.merge(part, on=["customer_id", "year"], how="left")
    # lines held by the customer in the same year (known at pricing time)
    held = rows.groupby(["customer_id", "year"]).line.nunique().rename("lines_held").reset_index()
    feats = feats.merge(held, on=["customer_id", "year"], how="left")
    num = [c for c in feats.columns if c.endswith(("_expo", "_claims"))]
    feats[num] = feats[num].fillna(0)
    for p in ["pol", "cust"] + [f"cust_{l}" for l in get_spec().lines]:
        feats[f"{p}_rate"] = feats[f"{p}_claims"] / (feats[f"{p}_expo"] + 0.5)
    return feats.drop(columns=["policy_id", "customer_id", "year"])


def build_base(refresh: bool = False) -> dict:
    """Enriched + experience features for train and test, per line (cached as parquet)."""
    spec = get_spec()
    LINES, CACHE = spec.lines, spec.cache_path
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


ENGINEERED = ["age_at_licence", "ncb_ratio", "mileage_per_cc", "value_per_age", "points_per_year",
              "si_per_bedroom", "si_per_age", "river_x_flood", "elev_x_flood",
              "tenure_ratio", "prior_rate", "hist_rate"]


def engineered_features(df: pd.DataFrame) -> dict:
    """Ratios/differences of raw columns (float32, NaN if any input is NaN); only those whose inputs exist."""
    f = lambda c: pd.to_numeric(df[c], errors="coerce").astype("float64")
    has = lambda *cs: all(c in df.columns for c in cs)
    out = {}
    if has("driver_age", "years_licensed"):
        out["age_at_licence"] = f("driver_age") - f("years_licensed")
    if has("ncb_years", "years_licensed"):
        out["ncb_ratio"] = f("ncb_years") / (f("years_licensed") + 1)
    if has("annual_mileage", "engine_size_cc"):
        out["mileage_per_cc"] = f("annual_mileage") / (f("engine_size_cc") + 1)
    if has("vehicle_value_group", "vehicle_age"):
        out["value_per_age"] = f("vehicle_value_group") / (f("vehicle_age") + 1)
    if has("licence_points", "years_licensed"):
        out["points_per_year"] = f("licence_points") / (f("years_licensed") + 1)
    if has("sum_insured", "bedrooms"):
        out["si_per_bedroom"] = f("sum_insured") / (f("bedrooms") + 1)
    if has("sum_insured", "property_age"):
        out["si_per_age"] = f("sum_insured") / (f("property_age") + 1)
    if has("flood_zone", "distance_to_river_km"):
        out["river_x_flood"] = f("flood_zone") * (1 / (f("distance_to_river_km") + 0.1))
    if has("flood_zone", "elevation_m"):
        out["elev_x_flood"] = f("flood_zone") / (f("elevation_m") + 1)
    if has("customer_tenure", "customer_age"):
        out["tenure_ratio"] = f("customer_tenure") / (f("customer_age") - 17).clip(lower=1)
    if has("prior_claims_5y"):
        out["prior_rate"] = f("prior_claims_5y") / 5
    if has("pol_claims", "pol_expo"):
        out["hist_rate"] = f("pol_claims") / (f("pol_expo") + 0.5)
    return {k: v.astype("float32") for k, v in out.items()}


def model_matrix(train: pd.DataFrame, test: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """Feature frames (same columns, shared category levels) for one line."""
    spec = get_spec()
    drop = set(_DROP_BASE) | set(spec.leaky_cols)
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
        X.loc[train.year.values == spec.first_year, hist] = np.nan
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
    if cfg.get("engineered", False):
        for d, raw in ((X, train), (Xt, test)):
            for k, v in engineered_features(raw).items():
                if k not in d.columns:
                    d[k] = v.values
    cats = []
    for c in X.columns:
        if X[c].dtype == object or X[c].dtype == bool or str(X[c].dtype) == "category":
            levels = sorted(set(X[c].dropna().astype(str)) | set(Xt[c].dropna().astype(str)))
            X[c] = pd.Categorical(X[c].astype(str).where(X[c].notna()), categories=levels)
            Xt[c] = pd.Categorical(Xt[c].astype(str).where(Xt[c].notna()), categories=levels)
            cats.append(c)
    return X, Xt, cats
