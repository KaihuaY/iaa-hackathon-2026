"""Data loading, cleaning and target construction for the IAA hackathon (Part 1).

Every cleaning rule here is a finding from EDA (see notebooks/01_eda); each one is
logged so its impact can be reported, not applied silently.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from market_spec import get_spec

ID_COLS = ["row_id", "policy_id", "customer_id", "address_id", "claim_id"]
DATE_COLS = ["period_start", "period_end", "loss_date", "report_date", "date_of_birth"]


def __getattr__(name):
    """Market-dependent constants, resolved from the active MarketSpec at access time (PEP 562).
    LEAKY_COLS: columns that must never be model features (train-only, or recorded after the outcome).
    SI_UNIT_ERROR: sum insured above these is taken as entered in units instead of thousands (x1000 error)."""
    s = get_spec()
    table = {"DATA_DIR": lambda: s.data_path, "LINES": lambda: s.lines, "VALUATION_DATE": lambda: s.valuation_ts,
             "LEAKY_COLS": lambda: list(s.leaky_cols), "SI_UNIT_ERROR": lambda: s.si_unit_error,
             "CONSTRUCTION_MAP": lambda: s.construction_map}
    if name in table:
        return table[name]()
    raise AttributeError(f"module 'data' has no attribute {name!r}")


def load(name: str, data_dir: Path | None = None) -> pd.DataFrame:
    """Read one parquet table, keep ids as text and parse date columns."""
    data_dir = Path(data_dir) if data_dir is not None else get_spec().data_path
    df = pd.read_parquet(data_dir / f"{name}.parquet")
    for c in df.columns:
        if c in ID_COLS:
            df[c] = df[c].astype(str)
        elif c in DATE_COLS:
            df[c] = pd.to_datetime(df[c])
    return df


# ---------------------------------------------------------------- cleaning

def clean_policies(df: pd.DataFrame, line: str, log: list | None = None, label: str | None = None) -> pd.DataFrame:
    """Fix the data-quality issues found in policy / test tables. Appends (issue, rows) to log."""
    df = df.copy()
    spec = get_spec()
    label = label or line
    note = (lambda issue, n: log.append({"line": label, "issue": issue, "rows": int(n)})) if log is not None else (lambda *a: None)

    width = spec.customer_id_width
    if width is not None:
        short = df["customer_id"].str.len() < width
        note("customer_id lost leading zeros", short.sum())
        df["customer_id"] = df["customer_id"].str.zfill(width)

    if spec.exposure_days_check:
        days = (df["period_end"] - df["period_start"]).dt.days + 1
        expo_from_dates = days / 365
        bad = (df["exposure"] - expo_from_dates).abs() > spec.exposure_days_tol
        note("exposure recorded in days, not years", bad.sum())
        df.loc[bad, "exposure"] = expo_from_dates[bad]

    if "sum_insured" in df and line in spec.si_unit_error:
        big = df["sum_insured"] > spec.si_unit_error[line]
        note("sum_insured x1000 unit error", big.sum())
        df.loc[big, "sum_insured"] = df.loc[big, "sum_insured"] / 1000

    if "construction_type" in df:
        raw = df["construction_type"].str.strip().str.lower()
        note("construction_type inconsistent labels", (raw != df["construction_type"]).sum() + raw.isin(["t", "w"]).sum())
        df["construction_type"] = raw.map(spec.construction_map)

    if "smoke_alarms" in df:
        df["smoke_alarms"] = df["smoke_alarms"].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})

    if "years_licensed" in df:
        imp = df["years_licensed"] > df["driver_age"] - spec.min_licence_age
        note("years_licensed impossible for driver_age", imp.sum())
        df["licence_inconsistent"] = imp
        df.loc[imp, "years_licensed"] = (df.loc[imp, "driver_age"] - spec.min_licence_age).clip(lower=0)

    return df


def clean_claims(claims: pd.DataFrame, log: list | None = None) -> pd.DataFrame:
    """Drop duplicate claims (same claim re-keyed under a new claim_id)."""
    key = [c for c in claims.columns if c != "claim_id"]
    dup = claims.duplicated(key, keep="first")
    if log is not None:
        log.append({"line": "claims", "issue": "duplicate claims under new claim_id", "rows": int(dup.sum())})
    return claims[~dup].copy()


def net_claims(claims: pd.DataFrame) -> pd.DataFrame:
    """Claims cost as defined by the competition: incurred net of recoveries, repudiated excluded."""
    c = claims[~claims["status"].isin(get_spec().excluded_claim_statuses)].copy()
    c["net_incurred"] = (c["incurred_to_date"] - c["recoveries"]).clip(lower=0)
    return c


# ---------------------------------------------------------------- target

def attach_claims(policies: pd.DataFrame, claims: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign each claim to the policy row whose period contains its loss_date.

    Returns (policies with claim_count / claim_cost columns, claims with matched row_id).
    """
    p = policies[["row_id", "policy_id", "period_start", "period_end"]]
    m = claims.merge(p, on="policy_id", how="inner")
    m = m[(m["loss_date"] >= m["period_start"]) & (m["loss_date"] <= m["period_end"])]
    m = m.drop_duplicates("claim_id")
    agg = m.groupby("row_id").agg(claim_count=("claim_id", "size"), claim_cost=("net_incurred", "sum"))
    out = policies.merge(agg, on="row_id", how="left")
    out[["claim_count", "claim_cost"]] = out[["claim_count", "claim_cost"]].fillna(0)
    claims_out = claims.merge(m[["claim_id", "row_id"]], on="claim_id", how="left")
    return out, claims_out


def row_type(df: pd.DataFrame) -> pd.Series:
    """Why a row covers less than a full year. 2024 periods end on 30-Dec (365 days in a leap year)."""
    s, e = df["period_start"], df["period_end"]
    jan1 = (s.dt.month == 1) & (s.dt.day == 1)
    year_end = e >= pd.to_datetime(s.dt.year.astype(str) + "-12-30")
    k = df.groupby(["policy_id", s.dt.year])["row_id"].transform("size")
    return pd.Series(np.select([k > 1, jan1 & year_end, ~jan1 & year_end, jan1 & ~year_end],
                               ["split by mid-term change", "full year", "starts mid-year", "ends early"],
                               "starts mid-year & ends early"), index=df.index)


def enrich(df: pd.DataFrame, customers: pd.DataFrame, addresses: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Join customer, address and same-year weather attributes onto policy rows."""
    out = df.copy()
    out["year"] = out["period_start"].dt.year
    cu = customers.drop(columns=["surname"])
    out = out.merge(cu, on="customer_id", how="left")
    out["customer_age"] = ((out["period_start"] - out["date_of_birth"]).dt.days / 365.25).round()
    out["customer_tenure"] = out["year"] - out["customer_since_year"]
    out = out.drop(columns=["date_of_birth"])
    out = out.merge(addresses, on="address_id", how="left")
    out["property_age"] = out["year"] - out["year_built"]
    out = out.merge(weather, on=["address_id", "year"], how="left")
    return out


def build_all(log: list | None = None) -> dict:
    """Cleaned train (with raw, undeveloped target), test, claims and lookup tables."""
    claims = net_claims(clean_claims(load("claims"), log))
    out = {"claims": claims, "customers": load("customers"), "addresses": load("addresses"),
           "weather": load("weather")}
    matched = []
    for line in get_spec().lines:
        tr = clean_policies(load(f"policies_{line}"), line, log)
        te = clean_policies(load(f"test_{line}"), line, log, label=f"{line} (test)")
        tr, cm = attach_claims(tr, claims[claims["line"] == line])
        out[f"train_{line}"], out[f"test_{line}"] = tr, te
        matched.append(cm)
    out["claims"] = pd.concat(matched, ignore_index=True)
    return out
