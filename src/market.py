"""New-market readiness: compare a freshly released dataset with the Part 1 (developed market) data.

Everything here is read-only and runs BEFORE any modelling. Every function tolerates missing
tables / columns: problems are reported as rows, not raised.
"""
from __future__ import annotations

import difflib
from pathlib import Path

import numpy as np
import pandas as pd

import data as D  # src/ is on sys.path (same convention as the other modules)
import eda

LINES = D.LINES

# ------------------------------------------------------------------ expected Part 1 schema
# Hard-coded from docs/kaggle_pages.md ("data-description"): works without Part 1 data present.
_POLICY_COMMON = [
    "row_id", "policy_id", "customer_id", "address_id", "period_start", "period_end", "exposure"]
_POLICY_TAIL = [
    "voluntary_excess", "compulsory_excess", "channel", "payment_method", "missed_payments_12m",
    "new_business", "renewal_count", "days_quote_to_inception", "discount_applied", "written_premium",
    "customer_service_contacts"]
_LINE_COLS = {
    "motor": ["driver_age", "years_licensed", "ncb_years", "vehicle_value_group", "vehicle_age", "make_model",
              "engine_size_cc", "body_type", "transmission", "seats", "vehicle_colour", "fuel_type",
              "driver_assist", "cover_type", "annual_mileage", "overnight_parking", "vehicle_use",
              "named_young_driver", "telematics_opt_in", "telematics_score"],
    "buildings": ["policyholder_age", "sum_insured", "construction_type", "occupancy", "days_unoccupied",
                  "smoke_alarms", "accidental_damage_cover", "mortgage"],
    "contents": ["policyholder_age", "sum_insured", "tenure", "security_level", "high_value_items",
                 "accidental_damage_cover", "works_from_home", "lodgers", "bicycle_cover"],
}

EXPECTED_SCHEMA: dict[str, list[str]] = {
    "customers": ["customer_id", "surname", "date_of_birth", "occupation", "marital_status", "household_size",
                  "credit_band", "prior_claims_5y", "licence_points", "years_at_address", "customer_since_year",
                  "email_domain", "marketing_consent", "document_preference", "previous_insurer",
                  "campaign_code", "branch_id", "marketing_segment"],
    "addresses": ["address_id", "region", "urban_density", "flood_zone", "distance_to_river_km", "elevation_m",
                  "soil_type", "year_built", "property_type", "roof_type", "basement", "trees_within_5m",
                  "listed_building", "bedrooms", "energy_rating", "solar_panels"],
    "weather": ["address_id", "year", "hail_days", "wind_exposure_index", "freezing_days", "dry_summer_index",
                "wildfire_index", "rainfall_mm", "mean_temperature_c"],
    "claims": ["claim_id", "policy_id", "line", "peril", "loss_date", "report_date", "status", "paid_to_date",
               "incurred_to_date", "recoveries", "repairer_id", "third_party_id"],
    "sample_submission": ["row_id", "prediction"],
}
for _l in LINES:
    _cols = _POLICY_COMMON + _LINE_COLS[_l] + _POLICY_TAIL
    EXPECTED_SCHEMA[f"policies_{_l}"] = _cols + ["renewal_decision"]
    EXPECTED_SCHEMA[f"test_{_l}"] = list(_cols)

# Expected value kind per column name (str / num / bool / date), hard-coded from the same page.
_BOOL = {"basement", "trees_within_5m", "listed_building", "solar_panels", "marketing_consent", "smoke_alarms",
         "accidental_damage_cover", "mortgage", "new_business", "driver_assist", "named_young_driver",
         "telematics_opt_in", "works_from_home", "lodgers", "bicycle_cover"}
_DATE = {"date_of_birth", "period_start", "period_end", "loss_date", "report_date"}
_STR = {"row_id", "policy_id", "customer_id", "address_id", "claim_id", "surname", "occupation", "marital_status",
        "credit_band", "email_domain", "document_preference", "previous_insurer", "campaign_code",
        "marketing_segment", "region", "urban_density", "soil_type", "property_type", "roof_type",
        "energy_rating", "construction_type", "occupancy", "channel", "payment_method", "renewal_decision",
        "make_model", "body_type", "transmission", "vehicle_colour", "fuel_type", "cover_type",
        "overnight_parking", "vehicle_use", "tenure", "security_level", "line", "peril", "status",
        "repairer_id", "third_party_id"}


def expected_kind(col: str) -> str:
    return "bool" if col in _BOOL else "date" if col in _DATE else "str" if col in _STR else "num"


def _arrow_kind(t) -> str:
    import pyarrow as pa
    if pa.types.is_boolean(t):
        return "bool"
    if pa.types.is_date(t) or pa.types.is_timestamp(t):
        return "date"
    if pa.types.is_string(t) or pa.types.is_large_string(t) or pa.types.is_dictionary(t):
        return "str"
    if pa.types.is_integer(t) or pa.types.is_floating(t) or pa.types.is_decimal(t):
        return "num"
    return str(t)


# ------------------------------------------------------------------ io helpers
def _path(data_dir, name) -> Path:
    return Path(data_dir) / f"{name}.parquet"


def _read(data_dir, name) -> pd.DataFrame | None:
    """Read a table tolerantly: ids as text, dates parsed (bad values become NaT). None if absent/unreadable."""
    p = _path(data_dir, name)
    if data_dir is None or not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
    except Exception:
        return None
    for c in df.columns:
        if c in D.ID_COLS:
            df[c] = df[c].astype(str)
        elif c in D.DATE_COLS:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def _line_of(table: str) -> str:
    for l in LINES:
        if table.endswith(f"_{l}"):
            return l
    return "claims" if table == "claims" else ""


def _has_data(data_dir) -> bool:
    return data_dir is not None and Path(data_dir).is_dir() and any(Path(data_dir).glob("*.parquet"))


def _clean_policy(df, line, log=None, label=None):
    """clean_policies, falling back to the raw frame if the new data does not fit the Part 1 rules."""
    try:
        return D.clean_policies(df, line, log, label)
    except Exception as e:  # noqa: BLE001 - report, never crash
        if log is not None:
            log.append({"line": label or line, "issue": f"cleaning skipped ({type(e).__name__}: {e})", "rows": 0})
        return df


def _clean_claims_net(claims, log=None):
    try:
        return D.net_claims(D.clean_claims(claims, log))
    except Exception as e:  # noqa: BLE001
        if log is not None:
            log.append({"line": "claims", "issue": f"cleaning skipped ({type(e).__name__}: {e})", "rows": 0})
        return None


# ------------------------------------------------------------------ 1. schema
SCHEMA_COLS = ["table", "status", "column", "detail"]


def schema_diff(data_dir) -> pd.DataFrame:
    import pyarrow.parquet as pq
    rows = []
    d = Path(data_dir)
    for table, exp in EXPECTED_SCHEMA.items():
        p = _path(d, table)
        if not p.exists():
            rows.append((table, "missing_table", "", "file not found"))
            continue
        try:
            sch = pq.read_schema(p)
        except Exception as e:  # noqa: BLE001
            rows.append((table, "missing_table", "", f"unreadable: {e}"))
            continue
        got = dict(zip(sch.names, sch.types))
        miss = [c for c in exp if c not in got]
        new = [c for c in got if c not in exp and not c.startswith("__index_level")]
        for c in miss:
            hint = difflib.get_close_matches(c, new, n=1, cutoff=0.6)
            rows.append((table, "missing_column", c, f"possible rename to '{hint[0]}'" if hint else "expected, not present"))
        for c in new:
            hint = difflib.get_close_matches(c, miss, n=1, cutoff=0.6)
            rows.append((table, "new_column", c, f"{got[c]}" + (f"; possible rename of '{hint[0]}'" if hint else "")))
        for c in exp:
            if c in got:
                k = _arrow_kind(got[c])
                if k != expected_kind(c):
                    rows.append((table, "dtype_changed", c, f"expected {expected_kind(c)}, found {got[c]}"))
    for p in sorted(d.glob("*.parquet")):
        if p.stem not in EXPECTED_SCHEMA:
            rows.append((p.stem, "new_table", "", "table not in Part 1"))
    return pd.DataFrame(rows, columns=SCHEMA_COLS)


# ------------------------------------------------------------------ 2. data quality
DQ_COLS = ["table", "line", "kind", "issue", "column", "rows", "share"]
_ID_KEY = {"customers": ["customer_id"], "addresses": ["address_id"], "weather": ["address_id", "year"],
           "claims": ["claim_id"], "sample_submission": ["row_id"]}
for _l in LINES:
    _ID_KEY[f"policies_{_l}"] = ["row_id"]
    _ID_KEY[f"test_{_l}"] = ["row_id"]
_NONNEG = ["exposure", "driver_age", "policyholder_age", "years_licensed", "ncb_years", "vehicle_age",
           "sum_insured", "written_premium", "paid_to_date", "incurred_to_date", "recoveries",
           "annual_mileage", "engine_size_cc", "days_unoccupied", "household_size", "years_at_address",
           "voluntary_excess", "compulsory_excess", "high_value_items", "bedrooms"]
_NO_CASE = set(D.ID_COLS) | {"surname", "make_model", "email_domain"}


def _dq_row(table, kind, issue, column, rows, n):
    return {"table": table, "line": _line_of(table), "kind": kind, "issue": issue, "column": column,
            "rows": int(rows), "share": (rows / n) if n else np.nan}


def _generic_checks(table, df) -> list[dict]:
    out, n = [], len(df)
    if n == 0:
        return out
    key = [c for c in _ID_KEY.get(table, []) if c in df.columns]
    if key and len(key) == len(_ID_KEY.get(table, [])):
        out.append(_dq_row(table, "generic", "duplicate ids", "+".join(key), df.duplicated(key).sum(), n))
    miss = df.isna().sum()
    for c, k in miss[miss / n > 0.01].items():
        out.append(_dq_row(table, "generic", "missing rate > 1%", c, k, n))
    for c in _NONNEG:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]):
            out.append(_dq_row(table, "generic", f"negative {c}", c, (df[c] < 0).sum(), n))
    if "sum_insured" in df.columns and pd.api.types.is_numeric_dtype(df["sum_insured"]):
        out.append(_dq_row(table, "generic", "sum_insured > 1e7", "sum_insured", (df["sum_insured"] > 1e7).sum(), n))
    if "exposure" in df.columns and pd.api.types.is_numeric_dtype(df["exposure"]):
        out.append(_dq_row(table, "generic", "exposure > 1.01", "exposure", (df["exposure"] > 1.01).sum(), n))
    for c in df.columns:
        s = df[c]
        if c in _NO_CASE or not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
            continue
        if s.nunique(dropna=True) > 300:
            continue
        vc = s.value_counts()
        norm = pd.Series(vc.index.astype(str), index=vc.index).str.strip().str.lower()
        grp = norm.groupby(norm).transform("size")
        bad = vc[grp.values > 1]
        if len(bad):
            out.append(_dq_row(table, "generic", "inconsistent casing/whitespace", c, bad.sum(), n))
            out[-1]["issue"] += " (" + ", ".join(repr(x) for x in list(bad.index)[:6]) + ")"
    if {"period_start", "period_end"} <= set(df.columns):
        out.append(_dq_row(table, "generic", "period_end < period_start", "period_end",
                           (df["period_end"] < df["period_start"]).sum(), n))
    if {"loss_date", "report_date"} <= set(df.columns):
        out.append(_dq_row(table, "generic", "report_date < loss_date", "report_date",
                           (df["report_date"] < df["loss_date"]).sum(), n))
    return out


def dq_register(data_dir) -> pd.DataFrame:
    rows, log = [], []
    sizes, frames = {}, {}
    for table in EXPECTED_SCHEMA:
        df = _read(data_dir, table)
        if df is None:
            continue
        sizes[table] = len(df)
        frames[table] = df
        rows += _generic_checks(table, df)
    # Part 1 cleaning rules, applied to the new data
    for line in LINES:
        for pref, lab in (("policies", line), ("test", f"{line} (test)")):
            t = f"{pref}_{line}"
            if t not in frames:
                continue
            need = {"customer_id", "period_start", "period_end", "exposure"}
            if not need <= set(frames[t].columns):
                log.append({"line": lab, "issue": f"cleaning skipped (missing {sorted(need - set(frames[t].columns))})", "rows": 0})
                continue
            _clean_policy(frames[t], line, log, lab)
    if "claims" in frames:
        if {"policy_id", "line"} <= set(frames["claims"].columns):
            for line in LINES:
                pt = f"policies_{line}"
                if pt in frames and "policy_id" in frames[pt].columns:
                    c = frames["claims"]
                    c = c[c["line"] == line]
                    bad = ~c["policy_id"].isin(set(frames[pt]["policy_id"]))
                    rows.append(_dq_row("claims", "generic", "claims not matching any policy", f"policy_id ({line})", bad.sum(), len(c)))
        try:
            D.clean_claims(frames["claims"], log)
        except Exception as e:  # noqa: BLE001
            log.append({"line": "claims", "issue": f"cleaning skipped ({type(e).__name__}: {e})", "rows": 0})
    for r in log:
        lab = r["line"]
        t = "claims" if lab == "claims" else (f"test_{lab.split()[0]}" if lab.endswith("(test)") else f"policies_{lab}")
        rows.append(_dq_row(t, "cleaning", r["issue"], "", r["rows"], sizes.get(t, 0)))
        rows[-1]["line"] = lab
    return pd.DataFrame(rows, columns=DQ_COLS)


# ------------------------------------------------------------------ 3. portfolio
PORT_COLS = ["set", "line", "year", "rows", "exposure", "claims", "cost", "premium", "frequency", "severity",
             "pure_premium", "loss_ratio"]


def portfolio_summary(data_dir) -> pd.DataFrame:
    out = []
    claims = _read(data_dir, "claims")
    net = _clean_claims_net(claims) if claims is not None else None
    for line in LINES:
        tr = _read(data_dir, f"policies_{line}")
        if tr is not None and {"row_id", "period_start", "exposure"} <= set(tr.columns):
            tr = _clean_policy(tr, line)
            if "written_premium" not in tr:
                tr["written_premium"] = np.nan
            try:
                if net is None or "line" not in net.columns:
                    raise ValueError("no claims")
                tr, _ = D.attach_claims(tr, net[net["line"] == line])
            except Exception:  # noqa: BLE001
                tr["claim_count"], tr["claim_cost"] = np.nan, np.nan
            s = eda.summary(tr.assign(year=tr["period_start"].dt.year), "year").reset_index()
            s.insert(0, "line", line)
            s.insert(0, "set", "train")
            out.append(s.rename(columns={"cost": "cost"}))
        te = _read(data_dir, f"test_{line}")
        if te is not None and {"row_id", "period_start", "exposure"} <= set(te.columns):
            te = te.assign(year=te["period_start"].dt.year)
            g = te.groupby("year").agg(rows=("row_id", "size"), exposure=("exposure", "sum")).reset_index()
            g.insert(0, "line", line)
            g.insert(0, "set", "test")
            out.append(g)
    if not out:
        return pd.DataFrame(columns=PORT_COLS)
    return pd.concat(out, ignore_index=True).reindex(columns=PORT_COLS)


# ------------------------------------------------------------------ 4. claims development
DEV_COLS = ["line", "loss_year", "n_claims", "open_share", "lag_p50", "lag_p90", "lag_p99", "lag_max"]


def claims_development(data_dir) -> pd.DataFrame:
    c = _read(data_dir, "claims")
    if c is None or not {"loss_date", "report_date", "line"} <= set(c.columns):
        return pd.DataFrame(columns=DEV_COLS)
    c = c.assign(lag=(c["report_date"] - c["loss_date"]).dt.days, loss_year=c["loss_date"].dt.year)
    c["is_open"] = (c["status"] == "open") if "status" in c.columns else np.nan
    rows = []
    for line, g in c.groupby("line"):
        for yr, h in [("all", g)] + list(g.groupby("loss_year")):
            q = h["lag"].quantile([.5, .9, .99])
            rows.append({"line": line, "loss_year": yr, "n_claims": len(h), "open_share": h["is_open"].mean(),
                         "lag_p50": q.iloc[0], "lag_p90": q.iloc[1], "lag_p99": q.iloc[2], "lag_max": h["lag"].max()})
    return pd.DataFrame(rows, columns=DEV_COLS)


# ------------------------------------------------------------------ 5. drift
DRIFT_COLS = ["line", "column", "psi", "flag", "unseen_share", "new_levels"]
_DRIFT_SKIP = set(D.ID_COLS) | set(D.DATE_COLS) | set(D.LEAKY_COLS) | {"surname"}


def _is_cat(s) -> bool:
    return s.dtype == bool or s.dtype == object or pd.api.types.is_string_dtype(s) or s.nunique() <= 12


def _drift_pair(label, new, ref, rows):
    for c in [c for c in ref.columns if c in new.columns and c not in _DRIFT_SKIP]:
        a, b = ref[c], new[c]
        psi, unseen, newlv = np.nan, np.nan, ""
        try:
            if pd.api.types.is_numeric_dtype(a) != pd.api.types.is_numeric_dtype(b):
                raise TypeError("dtype mismatch")
            if _is_cat(a) or (a.notna().sum() and b.notna().sum() and a.nunique() > 1):
                psi = eda.psi(a, b)
        except Exception:  # noqa: BLE001
            pass
        if _is_cat(a) or _is_cat(b):
            seen = set(a.dropna().astype(str).unique())
            bs = b.dropna().astype(str)
            miss = ~bs.isin(seen)
            unseen = float(miss.mean()) if len(bs) else np.nan
            lv = bs[miss].value_counts()
            newlv = ", ".join(map(str, lv.index[:5]))
        flag = "" if np.isnan(psi) else "shift" if psi > 0.25 else "watch" if psi >= 0.1 else "ok"
        if np.isnan(psi) and unseen and unseen > 0:
            flag = "shift"
        rows.append({"line": label, "column": c, "psi": psi, "flag": flag, "unseen_share": unseen, "new_levels": newlv})


def drift_report(new_dir, ref_dir) -> pd.DataFrame:
    if not _has_data(ref_dir):
        r = pd.DataFrame(columns=DRIFT_COLS)
        r.attrs["note"] = "no reference (Part 1) data available: drift not computed"
        return r
    rows = []
    for line in LINES:
        n, r = _read(new_dir, f"policies_{line}"), _read(ref_dir, f"policies_{line}")
        if n is None or r is None:
            continue
        _drift_pair(line, _clean_policy(n, line), _clean_policy(r, line), rows)
        del n, r
    for t in ("customers", "addresses", "weather"):
        n, r = _read(new_dir, t), _read(ref_dir, t)
        if n is not None and r is not None:
            _drift_pair(t, n, r, rows)
    out = pd.DataFrame(rows, columns=DRIFT_COLS)
    if out.empty:
        out.attrs["note"] = "no shared tables between new and reference data"
    return out


# ------------------------------------------------------------------ report
def _fmt(v):
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return ""
        return f"{v:,.4f}" if abs(v) < 10 else f"{v:,.1f}"
    return str(v)


def _md(df: pd.DataFrame, max_rows: int = 60) -> str:
    if df is None or df.empty:
        return "_none_\n"
    d = df.head(max_rows)
    lines = ["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]
    for _, r in d.iterrows():
        lines.append("| " + " | ".join(_fmt(v).replace("|", "/") for v in r) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_({len(df) - max_rows} more rows not shown)_")
    return "\n".join(lines) + "\n"


def _level_compare(new_p, ref_p) -> pd.DataFrame:
    def tot(p):
        t = p[(p["set"] == "train")].groupby("line")[["exposure", "claims", "cost"]].sum(min_count=1)
        t["frequency"], t["severity"] = t.claims / t.exposure, t.cost / t.claims
        t["pure_premium"] = t.cost / t.exposure
        return t
    if new_p.empty or ref_p.empty:
        return pd.DataFrame()
    a, b = tot(new_p), tot(ref_p)
    j = a.join(b, lsuffix="_new", rsuffix="_ref", how="inner")
    for m in ("frequency", "severity", "pure_premium"):
        j[f"{m}_ratio"] = j[f"{m}_new"] / j[f"{m}_ref"]
    return j[[f"{m}_{s}" for m in ("frequency", "severity", "pure_premium") for s in ("new", "ref", "ratio")]].reset_index()


def _checklist(schema, dq, port, dev, dev_ref, drift, level) -> list[tuple[str, str]]:
    items = []
    for _, r in schema.iterrows():
        ev = f"schema_diff: {r.table} / {r.column or '-'} / {r.status} / {r.detail}"
        if r.status == "missing_table":
            items.append((f"table `{r.table}` missing: pipeline stage using it must be skipped or rebuilt", ev))
        elif r.status == "missing_column":
            leaky = " (leakage column: no impact on features)" if r.column in D.LEAKY_COLS else ""
            items.append((f"column `{r.column}` missing in `{r.table}`: feature list must drop it{leaky}", ev))
        elif r.status == "new_column":
            items.append((f"new column `{r.column}` in `{r.table}`: screen as a candidate feature and check for leakage", ev))
        elif r.status == "dtype_changed":
            items.append((f"`{r.column}` in `{r.table}` changed type: add a cast/parsing rule", ev))
        elif r.status == "new_table":
            items.append((f"new table `{r.table}`: inspect and decide whether to join it", ev))
    if not dq.empty:
        for _, r in dq[dq.rows > 0].iterrows():
            ev = f"dq_register: {r.table} / {r.issue} / {r.column or '-'} / rows={r.rows} share={r.share:.2%}"
            if r.kind == "cleaning":
                items.append((f"Part 1 fix still needed ({r.issue}) in `{r.table}`: {r.rows:,} rows, keep/re-validate the rule", ev))
            elif "missing rate" in r.issue:
                items.append((f"`{r.column}` in `{r.table}` is {r.share:.1%} missing: define imputation / missing-level handling", ev))
            elif "casing" in r.issue:
                items.append((f"`{r.column}` in `{r.table}` has inconsistent labels: add a normalisation map", ev))
            elif "claims not matching" in r.issue:
                items.append((f"{r.rows:,} claims do not match any policy ({r.column}): decide exclusion and check id format", ev))
            elif "duplicate" in r.issue:
                items.append((f"duplicate ids in `{r.table}` ({r.column}): de-duplicate before joins", ev))
            else:
                items.append((f"{r.issue} in `{r.table}`: add a validation/cleaning rule", ev))
    if not drift.empty:
        for _, r in drift[drift.flag == "shift"].iterrows():
            ps = "n/a" if pd.isna(r.psi) else f"{r.psi:.2f}"
            items.append((f"`{r.column}` ({r.line}) has shifted (PSI {ps}): re-check relativity, consider re-binning or dropping",
                          f"drift_report: {r.line} / {r.column} / psi={ps}"))
        for _, r in drift[(drift.unseen_share > 0) & drift.unseen_share.notna()].iterrows():
            items.append((f"new level(s) {r.new_levels} in `{r.column}` ({r.line}), {r.unseen_share:.1%} of rows: category handling (map to other / grouped level)",
                          f"drift_report: {r.line} / {r.column} / unseen_share={r.unseen_share:.3f}"))
        w = drift[drift.flag == "watch"]
        if len(w):
            items.append((f"{len(w)} column(s) on watch (PSI 0.1-0.25): " + ", ".join(f"{a}.{b}" for a, b in zip(w.line, w.column)),
                          "drift_report: flag=watch"))
    if not dev.empty:
        for line, g in dev[dev.loss_year != "all"].groupby("line"):
            g = g.sort_values("loss_year")
            last, prev = g.iloc[-1], g.iloc[:-1]
            ev = f"claims_development: {line} / loss_year={last.loss_year} / open_share={last.open_share:.3f} n={last.n_claims}"
            if len(prev) >= 1:
                imm = (last.open_share > 1.5 * prev.open_share.mean() + 0.02) or (last.n_claims < 0.9 * prev.n_claims.median())
                if imm:
                    items.append((f"{line}: latest loss year {last.loss_year} looks immature (open share/claim count): develop it before using as a target", ev))
                else:
                    items.append((f"{line}: no visible immaturity in {last.loss_year}: skip development / completeness adjustment, verify with the data owner", ev))
        if not dev_ref.empty:
            a = dev[dev.loss_year == "all"].set_index("line").lag_p99
            b = dev_ref[dev_ref.loss_year == "all"].set_index("line").lag_p99
            for line in a.index.intersection(b.index):
                if a[line] > 1.25 * b[line]:
                    items.append((f"{line}: claims report-lag tail longer (p99 {a[line]:.0f}d vs {b[line]:.0f}d): re-estimate the completeness curve",
                                  f"claims_development: {line} lag_p99 new={a[line]:.0f} ref={b[line]:.0f}"))
                elif a[line] < 0.75 * b[line]:
                    items.append((f"{line}: claims report-lag shorter (p99 {a[line]:.0f}d vs {b[line]:.0f}d): completeness curve may be flatter",
                                  f"claims_development: {line} lag_p99 new={a[line]:.0f} ref={b[line]:.0f}"))
    if not level.empty:
        for _, r in level.iterrows():
            for m in ("frequency", "severity", "pure_premium"):
                x = r[f"{m}_ratio"]
                if pd.notna(x) and (x > 1.25 or x < 0.8):
                    items.append((f"{r.line}: {m} is {x:.2f}x Part 1: recalibrate the base level (do not carry Part 1 intercept)",
                                  f"portfolio_summary: {r.line} {m} new={r[f'{m}_new']:.4g} ref={r[f'{m}_ref']:.4g}"))
    if not port.empty:
        yrs = sorted(port[port.set == "train"].year.dropna().unique())
        if yrs and yrs != [2021, 2022, 2023, 2024]:
            items.append((f"training years are {[int(y) for y in yrs]} (Part 1: 2021-2024): revisit backtest folds and year trend",
                          "portfolio_summary: train years"))
    return items


def readiness_parts(new_dir, ref_dir=None) -> dict:
    ref_ok = ref_dir is not None and _has_data(ref_dir)
    p = {"new_dir": Path(new_dir), "ref_dir": ref_dir if ref_ok else None}
    p["schema"] = schema_diff(new_dir)
    p["dq"] = dq_register(new_dir)
    p["port"] = portfolio_summary(new_dir)
    p["dev"] = claims_development(new_dir)
    p["drift"] = drift_report(new_dir, ref_dir) if ref_ok else pd.DataFrame(columns=DRIFT_COLS)
    p["dev_ref"] = claims_development(ref_dir) if ref_ok else pd.DataFrame(columns=DEV_COLS)
    p["level"] = _level_compare(p["port"], portfolio_summary(ref_dir)) if ref_ok else pd.DataFrame()
    p["items"] = _checklist(p["schema"], p["dq"], p["port"], p["dev"], p["dev_ref"], p["drift"], p["level"])
    return p


def readiness_report(new_dir, ref_dir=None, parts: dict | None = None) -> str:
    p = parts or readiness_parts(new_dir, ref_dir)
    schema, dq, port, dev, drift, level = p["schema"], p["dq"], p["port"], p["dev"], p["drift"], p["level"]
    ref_ok = p["ref_dir"] is not None
    nl = chr(10)
    out = ["# New market readiness report" + nl,
           f"Dataset: `{p['new_dir']}`  " + nl + f"Reference (Part 1): `{p['ref_dir'] if ref_ok else 'not available'}`" + nl]
    out.append("## 1. Schema differences" + nl)
    out.append(_md(schema, 200) if len(schema) else "_No schema differences vs Part 1._" + nl)
    out.append("## 2. Data-quality register" + nl)
    out.append("Rows with `kind=cleaning` are Part 1 fixes (`src/data.py`) applied to the new data; "
               "`generic` are new checks. Zero-row checks are omitted." + nl)
    out.append(_md(dq[dq.rows > 0] if not dq.empty else dq, 120))
    out.append("## 3. Portfolio and target overview" + nl)
    out.append(_md(port, 80))
    if not level.empty:
        out.append(nl + "### Level versus Part 1 (train, all years)" + nl)
        out.append(_md(level))
    out.append("## 4. Claims development" + nl)
    out.append(_md(dev, 80))
    out.append("## 5. Drift vs Part 1" + nl)
    if not ref_ok:
        out.append("_No reference data supplied: drift not computed._" + nl)
    elif drift.empty:
        out.append(f"_{drift.attrs.get('note', 'no comparable columns')}_" + nl)
    else:
        d = drift.sort_values("psi", ascending=False)
        out.append(f"Max PSI {drift.psi.max():.4f}; shift (>0.25): {(drift.flag == 'shift').sum()}, "
                   f"watch (0.1-0.25): {(drift.flag == 'watch').sum()}, columns compared: {len(drift)}." + nl)
        out.append(_md(d.head(40)))
    out.append("## 6. Adaptation checklist" + nl)
    if not p["items"]:
        out.append("_No items: the dataset matches Part 1 on all checks._" + nl)
    for i, (act, ev) in enumerate(p["items"], 1):
        out.append(f"{i}. [ ] {act}  " + nl + f"   evidence: `{ev}`")
    return nl.join(out) + nl


def summary_lines(p: dict) -> list[str]:
    """Ten-line console summary from readiness_parts()."""
    sch, dq, dr = p["schema"], p["dq"], p["drift"]
    cl = dq[(dq.kind == "cleaning") & (dq.rows > 0)].sort_values("rows", ascending=False)
    lines = [f"New market report for {p['new_dir']}",
             "schema: " + (", ".join(f"{k}={v}" for k, v in sch.status.value_counts().items()) or "identical to Part 1"),
             f"dq: {int((dq.rows > 0).sum())} issues with rows > 0, {len(cl)} Part 1 cleaning rules triggered"]
    for _, r in cl.head(4).iterrows():
        lines.append(f"  {r.line}: {r.issue}: {r.rows:,} ({r.share:.2%})")
    if p["ref_dir"] is not None and not dr.empty:
        lines.append(f"drift: max PSI {dr.psi.max():.4f}; shift={int((dr.flag == 'shift').sum())}, watch={int((dr.flag == 'watch').sum())}; "
                     f"unseen levels in {int((dr.unseen_share > 0).sum())} columns")
    lines.append(f"checklist items: {len(p['items'])}")
    return lines[:10]
