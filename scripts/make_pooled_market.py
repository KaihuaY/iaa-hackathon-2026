"""Pooled Part 2 market: developed (2021-24) + emerging (2023-24) history in the emerging schema, emerging test only.

  python -P -E scripts/make_pooled_market.py [--emerging data/part2/emerging] [--developed data/part2/developed]
                                             [--dst data/part2/pooled] [--premium-scale auto|none]

Mapping (developed half -> emerging schema; emerging half unchanged apart from the construction vocabulary):
- every developed id gets the prefix "D" (row_id, policy_id, customer_id, address_id, claim_id, repairer_id,
  third_party_id); developed customer_id is zero-padded to 8 characters first (Part 1 defect), so it is 9 characters;
- driver_age -> driver_age_band, policyholder_age -> policyholder_age_band (17-24, 25-34, 35-49, 50-64, 65+);
  the Part 1 licence repair (years_licensed <= driver_age - 17) is applied BEFORE the age is banded;
- annual_mileage, telematics_opt_in, telematics_score dropped (motor); date_of_birth, credit_band dropped (customers);
  gender added as missing (customers);
- construction_type: one canonical vocabulary in both halves, brick / stone / timber / non_standard
  (BRK->brick, STN->stone, WOD->timber, MIX->non_standard; Brick/BRICK/brick, W/w/T/t, stone/STONE, NON_STANDARD ...);
- written_premium (developed only): multiplied by one constant per line (--premium-scale auto) so that the mean
  full-year premium of 2023-24 equals the emerging one. written_premium is not a model feature; it drives the
  premium-index trend (target.trend_index), which would otherwise mix the two markets' price levels with time;
- a `market` column ("developed" / "emerging") on every policies_* / test_* table (a categorical model feature);
- test_* and sample_submission: emerging only. expenses / reference_policies: emerging copies.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LINES = ("motor", "buildings", "contents")
PREFIX = "D"
ID_COLS = ("row_id", "policy_id", "customer_id", "address_id", "claim_id", "repairer_id", "third_party_id")
AGE_EDGES = (25, 35, 50, 65)
AGE_LABELS = ("17-24", "25-34", "35-49", "50-64", "65+")
AGE_BAND = {"driver_age": "driver_age_band", "policyholder_age": "policyholder_age_band"}
DROP_MOTOR = ("annual_mileage", "telematics_opt_in", "telematics_score")
DROP_CUSTOMERS = ("date_of_birth", "credit_band")
CONSTRUCTION = {"brick": "brick", "brk": "brick", "stone": "stone", "stn": "stone", "timber": "timber", "t": "timber",
                "w": "timber", "wod": "timber", "non_standard": "non_standard", "mix": "non_standard"}
MIN_LICENCE_AGE = 17
PREMIUM_YEARS = (2023, 2024)


# ------------------------------------------------------------------ mapping helpers (pure; unit-tested)
def band_age(age: pd.Series) -> pd.Series:
    """Integer age -> emerging band label (object dtype, NaN stays NaN)."""
    b = pd.cut(age.astype(float), [-np.inf, *AGE_EDGES, np.inf], right=False, labels=list(AGE_LABELS))
    return b.astype(object).where(b.notna(), None)


def canon_construction(s: pd.Series) -> pd.Series:
    """Any construction label of either market -> brick / stone / timber / non_standard. Unknown labels raise."""
    raw = s.astype(object).where(s.isna(), s.astype(str).str.strip().str.lower())
    out = raw.map(CONSTRUCTION)
    bad = sorted(set(raw[out.isna() & raw.notna()]))
    if bad:
        raise ValueError(f"unmapped construction_type labels: {bad}")
    return out.astype(object).where(out.notna(), None)


def prefix_ids(df: pd.DataFrame, prefix: str = PREFIX) -> pd.DataFrame:
    """Prefix every id column present; customer_id is zero-padded to 8 first. Missing ids stay missing."""
    df = df.copy()
    for c in ID_COLS:
        if c not in df:
            continue
        v = df[c]
        s = v.astype(str)
        if c == "customer_id":
            s = s.str.zfill(8)
        df[c] = (prefix + s).where(v.notna(), None).astype(object)
    return df


def pad_customer(df: pd.DataFrame) -> pd.DataFrame:
    if "customer_id" in df:
        df = df.copy()
        df["customer_id"] = df["customer_id"].astype(str).str.zfill(8)
    return df


def developed_policies(df: pd.DataFrame, line: str, premium_scale: float = 1.0, log: dict | None = None) -> pd.DataFrame:
    """One developed policies_/test_ table mapped to the emerging schema (market column NOT added here)."""
    df = df.copy()
    if "years_licensed" in df and "driver_age" in df:             # Part 1 cleaning rule, needs the exact age
        cap = (df["driver_age"] - MIN_LICENCE_AGE).clip(lower=0)
        bad = df["years_licensed"] > cap
        if log is not None:
            log[f"{line}: years_licensed repaired"] = log.get(f"{line}: years_licensed repaired", 0) + int(bad.sum())
        df["years_licensed"] = df["years_licensed"].where(~bad, cap).astype(df["years_licensed"].dtype)
    for a, b in AGE_BAND.items():
        if a in df:
            df[b] = band_age(df[a])
            df = df.drop(columns=a)
    df = df.drop(columns=[c for c in DROP_MOTOR if c in df])
    if "construction_type" in df:
        df["construction_type"] = canon_construction(df["construction_type"])
    if premium_scale != 1.0 and "written_premium" in df:
        df["written_premium"] = (df["written_premium"] * premium_scale).round(2)
    return prefix_ids(df)


def emerging_policies(df: pd.DataFrame) -> pd.DataFrame:
    df = pad_customer(df)
    if "construction_type" in df:
        df["construction_type"] = canon_construction(df["construction_type"])
    return df


def developed_customers(df: pd.DataFrame, gender_dtype=object) -> pd.DataFrame:
    df = df.drop(columns=[c for c in DROP_CUSTOMERS if c in df]).copy()
    df["gender"] = pd.Series([None] * len(df), index=df.index, dtype=gender_dtype)
    return prefix_ids(df)


def stack(emerging: pd.DataFrame, developed: pd.DataFrame, market: bool) -> pd.DataFrame:
    """Developed rows reindexed to the emerging columns (same set required) and appended; dtypes must agree."""
    if set(developed.columns) != set(emerging.columns):
        raise ValueError(f"column sets differ: only emerging {sorted(set(emerging) - set(developed))}, "
                         f"only developed {sorted(set(developed) - set(emerging))}")
    developed = developed[list(emerging.columns)]
    for c in emerging.columns:
        de, dd = emerging[c].dtype, developed[c].dtype
        if de != dd:
            # all-missing columns (e.g. developed gender) take the emerging dtype; anything else is an error
            if developed[c].isna().all():
                developed = developed.assign(**{c: developed[c].astype(de)})
            elif emerging[c].isna().all():
                emerging = emerging.assign(**{c: emerging[c].astype(dd)})
            else:
                raise TypeError(f"dtype mismatch on {c}: emerging {de}, developed {dd}")
    if market:
        emerging = emerging.assign(market="emerging")
        developed = developed.assign(market="developed")
    return pd.concat([emerging, developed], ignore_index=True)


def premium_scales(E: dict, Dv: dict) -> dict:
    """Per line: emerging / developed mean full-year written premium over PREMIUM_YEARS (training tables)."""
    out = {}
    for line in LINES:
        m = []
        for t in (E[f"policies_{line}"], Dv[f"policies_{line}"]):
            y = pd.to_datetime(t["period_start"]).dt.year
            f = t[(t["exposure"] > 0.99) & y.isin(PREMIUM_YEARS)]
            m.append(float(f["written_premium"].mean()))
        out[line] = m[0] / m[1]
    return out


# ------------------------------------------------------------------ build + verify
def build(src_e: Path, src_d: Path, premium: str = "auto") -> tuple[dict, dict]:
    rd = lambda d, n: pd.read_parquet(d / f"{n}.parquet")
    names = [f"{k}_{l}" for k in ("policies", "test") for l in LINES] + ["claims", "customers", "addresses", "weather"]
    E = {n: rd(src_e, n) for n in names}
    Dv = {n: rd(src_d, n) for n in names if not n.startswith("test_")}
    log = {}
    scales = premium_scales(E, Dv) if premium == "auto" else {l: 1.0 for l in LINES}
    log["premium_scale"] = scales
    out = {}
    for line in LINES:
        out[f"policies_{line}"] = stack(emerging_policies(E[f"policies_{line}"]),
                                        developed_policies(Dv[f"policies_{line}"], line, scales[line], log), market=True)
        out[f"test_{line}"] = emerging_policies(E[f"test_{line}"]).assign(market="emerging")
    out["claims"] = stack(E["claims"], prefix_ids(Dv["claims"]), market=False)
    out["customers"] = stack(pad_customer(E["customers"]),
                             developed_customers(Dv["customers"], E["customers"]["gender"].dtype), market=False)
    out["addresses"] = stack(E["addresses"], prefix_ids(Dv["addresses"]), market=False)
    out["weather"] = stack(E["weather"], prefix_ids(Dv["weather"]), market=False)
    out["sample_submission"] = rd(src_e, "sample_submission")
    log["inputs"] = {"emerging": {n: len(d) for n, d in E.items()}, "developed": {n: len(d) for n, d in Dv.items()}}
    return out, log


def _attach_rate(pol: pd.DataFrame, claims: pd.DataFrame) -> tuple[int, int]:
    """(claims of this line, claims whose loss_date falls inside a period of their policy)."""
    p = pol[["row_id", "policy_id", "period_start", "period_end"]].copy()
    c = claims[["claim_id", "policy_id", "loss_date"]].copy()
    for d, cols in ((p, ["period_start", "period_end"]), (c, ["loss_date"])):
        for k in cols:
            d[k] = pd.to_datetime(d[k])
    m = c.merge(p, on="policy_id", how="inner")
    m = m[(m.loss_date >= m.period_start) & (m.loss_date <= m.period_end)]
    return len(c), int(m.claim_id.nunique())


def verify(out: dict, log: dict, src_e: Path, src_d: Path) -> dict:
    inp, chk = log["inputs"], {}
    rd = lambda d, n: pd.read_parquet(d / f"{n}.parquet")

    def ok(name, cond, detail=""):
        chk[name] = {"ok": bool(cond), "detail": detail}
        if not cond:
            raise AssertionError(f"verification failed: {name} {detail}")

    for n in ["claims", "customers", "addresses", "weather"] + [f"policies_{l}" for l in LINES]:
        exp = inp["emerging"][n] + inp["developed"][n]
        ok(f"rows {n}", len(out[n]) == exp, f"{len(out[n])} = {inp['emerging'][n]} + {inp['developed'][n]}")
    for l in LINES:
        ok(f"rows test_{l}", len(out[f"test_{l}"]) == inp["emerging"][f"test_{l}"], str(len(out[f"test_{l}"])))
    rows = pd.concat([out[f"{k}_{l}"]["row_id"] for k in ("policies", "test") for l in LINES], ignore_index=True)
    ok("row_id unique (all policy+test tables)", rows.is_unique, str(len(rows)))
    ok("claim_id unique", out["claims"].claim_id.is_unique)
    ok("customer_id unique", out["customers"].customer_id.is_unique)
    ok("address_id unique", out["addresses"].address_id.is_unique)
    ok("weather (address_id, year) unique", not out["weather"].duplicated(["address_id", "year"]).any())
    for l in LINES:
        p = out[f"policies_{l}"]
        pe, pdv = set(p.policy_id[p.market == "emerging"]), set(p.policy_id[p.market == "developed"])
        ok(f"{l}: no policy_id shared across markets", not (pe & pdv))
        ok(f"{l}: developed ids prefixed", p.loc[p.market == "developed", "row_id"].str.startswith(PREFIX).all()
           and p.loc[p.market == "developed", "customer_id"].str.len().eq(9).all())
        ok(f"{l}: emerging ids unprefixed", not p.loc[p.market == "emerging", "row_id"].str.startswith(PREFIX).any())
        ok(f"{l}: test is emerging only", (out[f"test_{l}"].market == "emerging").all())
    test_ids = set(pd.concat([out[f"test_{l}"].row_id for l in LINES]))
    ss = set(out["sample_submission"].row_id)
    ok("test row_ids == sample_submission row_ids", test_ids == ss, f"{len(test_ids)} vs {len(ss)}")
    ok("sample_submission rows 209,063", len(out["sample_submission"]) == 209_063)
    cust = set(out["customers"].customer_id)
    addr = set(out["addresses"].address_id)
    for l in LINES:
        for k in ("policies", "test"):
            t = out[f"{k}_{l}"]
            ok(f"{k}_{l}: customers join", t.customer_id.isin(cust).mean() == 1.0, f"{t.customer_id.isin(cust).mean():.6f}")
            ok(f"{k}_{l}: addresses join", t.address_id.isin(addr).mean() == 1.0, f"{t.address_id.isin(addr).mean():.6f}")
        if "construction_type" in out[f"policies_{l}"]:
            v = set(out[f"policies_{l}"].construction_type.dropna()) | set(out[f"test_{l}"].construction_type.dropna())
            ok(f"{l}: construction vocabulary", v <= {"brick", "stone", "timber", "non_standard"}, str(sorted(v)))
    # claims still attach to their own market's policies, exactly as in the source data
    for l in LINES:
        p, c = out[f"policies_{l}"], out["claims"][out["claims"].line == l]
        for mk, src in (("developed", src_d), ("emerging", src_e)):
            pm = p[p.market == mk]
            cm = c[c.claim_id.str.startswith(PREFIX)] if mk == "developed" else c[~c.claim_id.str.startswith(PREFIX)]
            sc = rd(src, "claims")
            n0, a0 = _attach_rate(rd(src, f"policies_{l}"), sc[sc.line == l])
            n1, a1 = _attach_rate(pm, cm)
            ok(f"{l}/{mk}: claims attach as in source", (n0, a0) == (n1, a1), f"{a1}/{n1} (source {a0}/{n0})")
        both = c.merge(p[["policy_id", "market"]].drop_duplicates(), on="policy_id", how="inner")
        cross = (both.claim_id.str.startswith(PREFIX) != (both.market == "developed")).sum()
        ok(f"{l}: claims join only their own market's policies", cross == 0, str(int(cross)))
    for l in LINES:                                           # dtype equality between halves
        p = out[f"policies_{l}"]
        for c in p.columns:
            kinds = {pd.api.types.infer_dtype(p.loc[p.market == m, c], skipna=True) for m in ("emerging", "developed")}
            kinds.discard("empty")
            ok(f"{l}: {c} same kind in both halves", len(kinds) <= 1 or kinds <= {"integer", "floating"}, str(kinds))
    return chk


def write(out: dict, dst: Path, src_e: Path, log: dict, chk: dict) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for n, d in out.items():
        d.to_parquet(dst / f"{n}.parquet", index=False)
        print(f"  {n:20s} {d.shape}", flush=True)
    for n in ("expenses", "reference_policies"):
        if (src_e / f"{n}.parquet").exists():
            shutil.copy2(src_e / f"{n}.parquet", dst / f"{n}.parquet")
    (dst / "pooled_manifest.json").write_text(json.dumps({"log": log, "checks": chk}, indent=1, default=str), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emerging", default=str(ROOT / "data" / "part2" / "emerging"))
    ap.add_argument("--developed", default=str(ROOT / "data" / "part2" / "developed"))
    ap.add_argument("--dst", default=str(ROOT / "data" / "part2" / "pooled"))
    ap.add_argument("--premium-scale", choices=["auto", "none"], default="auto")
    a = ap.parse_args()
    src_e, src_d, dst = Path(a.emerging), Path(a.developed), Path(a.dst)
    out, log = build(src_e, src_d, a.premium_scale)
    chk = verify(out, log, src_e, src_d)
    print(f"  {sum(v['ok'] for v in chk.values())}/{len(chk)} checks passed", flush=True)
    print("  premium scale:", {k: round(v, 4) for k, v in log["premium_scale"].items()},
          "| licence repairs:", {k: v for k, v in log.items() if "repaired" in k}, flush=True)
    write(out, dst, src_e, log, chk)


if __name__ == "__main__":
    main()
