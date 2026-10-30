"""Part 2 feature hunt (agent H1, 27-28 Oct): structure in the EMERGING data that the pooled model may be missing.

  $env:IAA_MARKET="part2pool"; python -P -E scripts/part2_feature_hunt.py <section> [--ref R2_FINAL] [--out DIR]
      sections: inventory scan nbint ids shift cross target stage2 forward overall factors | all
  $env:IAA_MARKET="part2";     python -P -E scripts/part2_feature_hunt.py wxshift
  $env:IAA_MARKET="part2";     python -P -E scripts/part2_feature_hunt.py screen <CONFIG> [<CONFIG> ...] [--seeds 11 12 13]

Reference model: experiments/results/part2pool/R2_P34_hn.pkl (pooled P34, all round-2 fixes). Its OOF on the emerging
rows (row_id without the "D" prefix) is the "expected" E. Level is settled by public probes, so every a/e below is
SHAPE: E is rescaled within each (year x new_business) cell so that the cell's a/e is 1 (`E_cell`), and a level's
a/e = sum(y_eval) / sum(E_cell). z = log(a/e) / sqrt((1 + cv_sev^2) / claims) (compound-Poisson approximation).
chi2/df = sum(z^2) / (levels - 1): a/e homogeneity across a feature's levels (random features give 0.4-2.2).
year_corr = claim-weighted correlation of level log a/e between 2023 and 2024 (stability; random features within +-0.4).
lift_bp = out-of-fold D2 gain (x1e4) of credibility-weighted level factors on top of E_cell (see residual_lift).
Output: printed tables, CSV copies in --out (default: experiments/logs/part2_feature_hunt/).
Write-up: docs/part2_feature_hunt.md.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]
LINES = ("motor", "buildings", "contents")
RAW = ROOT / "data/part2/emerging"
POOL_CACHE = ROOT / "experiments/cache/part2pool"
REF = ROOT / "experiments/results/part2pool/R2_P34_hn.pkl"
OUT = ROOT / "experiments/logs/part2_feature_hunt"
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 400)


def d2(y, p, power=1.5):
    from metrics import d2_tweedie
    return d2_tweedie(np.asarray(y, float), np.asarray(p, float))


def save(df: pd.DataFrame, name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"{name}.csv", index=False)


# ------------------------------------------------------------------------------------------------- data
_CACHE: dict = {}


def load_line(line: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Emerging history rows of the pooled cache with y, oof, E_cell, claims; and the emerging test rows."""
    if line in _CACHE:
        return _CACHE[line]
    tr = pd.read_parquet(POOL_CACHE / f"train_{line}.parquet", filters=[("market", "==", "emerging")])
    te = pd.read_parquet(POOL_CACHE / f"test_{line}.parquet")
    R = _ref()["lines"][line]
    rid = pd.Series(np.asarray(R["row_id"], dtype=object)).astype(str)
    em = ~rid.str.startswith("D").to_numpy()
    o = pd.DataFrame({"row_id": rid[em].values, "y": np.asarray(R["y_eval"], float)[em],
                      "oof": np.asarray(R["oof"], float)[em]})
    tr = tr.merge(o, on="row_id", how="inner")
    tr["claims"] = tr.claim_count.fillna(0).astype(float)
    cell = tr.year.astype(str) + np.where(tr.new_business.astype(bool), "_nb", "_ren")
    f = tr.groupby(cell).y.transform("sum") / tr.groupby(cell).oof.transform("sum")
    tr["E_cell"] = tr.oof * f
    tr["cell"] = cell
    tp = pd.DataFrame({"row_id": np.asarray(R["test_row_id"], dtype=object).astype(str),
                       "test_pred": np.asarray(R["test_pred"], float)})
    te = te.merge(tp, on="row_id", how="left")
    _CACHE[line] = (tr, te)
    return tr, te


_REF: dict = {}


def _ref():
    if not _REF:
        with open(REF, "rb") as fh:
            _REF.update(pickle.load(fh))
    return _REF


def cv_sev(tr: pd.DataFrame) -> float:
    s = tr.loc[tr.claims > 0]
    sev = s.y / s.claims
    return float(sev.std() / sev.mean())


def bands(s: pd.Series, q: int = 10) -> pd.Series:
    """Level labels: categoricals as they are (NaN -> 'NA'); numerics with <= 12 values as values, else deciles."""
    if s.dtype == object or str(s.dtype) in ("category", "bool", "string"):
        return s.astype(str).where(s.notna(), "NA")
    v = pd.to_numeric(s, errors="coerce")
    if v.nunique() <= 12:
        return v.map(lambda x: "NA" if pd.isna(x) else f"{x:g}")
    if (v == 0).mean() > 0.3:                               # zero-inflated (history counters): 0 + quartiles of v > 0
        nz = pd.qcut(v.where(v > 0), 4, duplicates="drop").astype(str)
        return pd.Series(np.where(v.isna(), "NA", np.where(v == 0, "0", nz)), index=s.index)
    try:
        b = pd.qcut(v, q, duplicates="drop")
    except ValueError:
        return v.astype(str)
    lab = b.astype(str).where(b.notna(), "NA")
    return lab


def ae_table(tr: pd.DataFrame, col, cvs: float, q: int = 10, label: str | None = None) -> pd.DataFrame:
    lv = bands(tr[col], q) if isinstance(col, str) else col
    g = pd.DataFrame({"lv": lv.values, "y": tr.y.values, "E": tr.E_cell.values, "n": tr.claims.values,
                      "rows": 1, "y23": np.where(tr.year == 2023, tr.y, 0), "E23": np.where(tr.year == 2023, tr.E_cell, 0),
                      "n23": np.where(tr.year == 2023, tr.claims, 0)}).groupby("lv").sum()
    g["ae"] = g.y / g.E
    g["ae_2023"] = g.y23 / g.E23
    g["ae_2024"] = (g.y - g.y23) / (g.E - g.E23)
    g["claims"] = g.n
    g["z"] = np.log(g.ae.clip(1e-6)) / np.sqrt((1 + cvs ** 2) / g.n.clip(lower=1))
    g["score"] = g.n * np.abs(np.log(g.ae.clip(1e-6)))
    g = g.reset_index().rename(columns={"lv": "level"})
    g.insert(0, "feature", label or (col if isinstance(col, str) else "custom"))
    return g[["feature", "level", "rows", "claims", "ae", "ae_2023", "ae_2024", "z", "score"]]


def residual_lift(tr: pd.DataFrame, cols, seed: int = 7, extra=None, tau: float = 0.15) -> float:
    """D2 gain x1e4 (OOF over 5 policy-grouped folds, emerging rows, uncapped y) of credibility-weighted level factors.
    Per fold: a/e of each level on the other folds (E_cell base), shrunk as 1 + Z (a/e - 1), Z = n / (n + K),
    K = (1 + cv_sev^2) / tau^2 claims (tau = prior SD of a true relativity), applied to the held-out fold.
    Levels = bands() of one column (several columns: the cross of their bands); `extra` = a ready level Series."""
    from sklearn.model_selection import GroupKFold
    if extra is not None:
        lv = pd.Series(np.asarray(extra), index=tr.index).astype(str)
    else:
        lv = bands(tr[cols[0]])
        for c in cols[1:]:
            lv = lv + "|" + bands(tr[c])
    lv = lv.values
    y, E, n = tr.y.values, tr.E_cell.values, tr.claims.values
    K = (1 + cv_sev(tr) ** 2) / tau ** 2
    p = E.copy()
    for a, b in GroupKFold(5).split(tr, groups=tr.policy_id.values):
        g = pd.DataFrame({"lv": lv[a], "y": y[a], "E": E[a], "n": n[a]}).groupby("lv").sum()
        g["f"] = 1 + g.n / (g.n + K) * (g.y / g.E * (E[a].sum() / y[a].sum()) - 1)
        p[b] = E[b] * pd.Series(lv[b]).map(g.f).fillna(1.0).values
    return (d2(y, p) - d2(y, E)) * 1e4


def feature_stats(tr: pd.DataFrame, col, cvs: float, label: str | None = None, lift: bool = True) -> dict:
    """chi2/df of level z-scores (a/e homogeneity; ~1 under no signal), claim-weighted correlation of level log a/e
    2023 vs 2024 (stability), max |z|, and the credibility lift."""
    t = ae_table(tr, col, cvs, label=label)
    t = t[t.claims > 0]
    L = len(t)
    chi = float((t.z ** 2).sum() / max(L - 1, 1))
    ok = t[(t.ae_2023 > 0) & (t.ae_2024 > 0) & np.isfinite(t.ae_2023) & np.isfinite(t.ae_2024)]
    if len(ok) >= 3:
        a, b, w = np.log(ok.ae_2023), np.log(ok.ae_2024), ok.claims
        ma, mb = np.average(a, weights=w), np.average(b, weights=w)
        yc = float(np.average((a - ma) * (b - mb), weights=w) / np.sqrt(np.average((a - ma) ** 2, weights=w) *
                                                                        np.average((b - mb) ** 2, weights=w)))
    else:
        yc = np.nan
    out = {"feature": label or col, "levels": L, "claims": float(t.claims.sum()), "chi2_df": chi, "year_corr": yc,
           "max_abs_z": float(t.z.abs().max()),
           "flagged": int((((t.ae < 0.8) | (t.ae > 1.25)) & (t.claims >= 100)).sum())}
    if lift:
        out["lift_bp"] = residual_lift(tr, [], extra=bands(tr[col]) if isinstance(col, str) else col)
    return out


def noise_controls(tr: pd.DataFrame, k: int = 9) -> dict:
    """Null distribution of residual_lift: random 10- and 40-level features."""
    rng = np.random.default_rng(0)
    out = [residual_lift(tr, [], extra=rng.integers(0, 10 if i % 2 else 40, len(tr))) for i in range(k)]
    return {"mean": float(np.mean(out)), "max": float(np.max(out)), "sd": float(np.std(out))}


def model_columns(line: str) -> list:
    """Exact columns R2_P34_hn feeds the model for this line (model_matrix on a sample, pooled spec)."""
    os.environ.setdefault("IAA_MARKET", "part2pool")
    from configs_pool import EXPERIMENTS as PE
    from features import model_matrix
    cfg = PE["R2_P34_hn"]
    fcfg = dict(cfg.get("features", {}))
    tr, te = load_line(line)
    X, _, _ = model_matrix(tr.head(3000).drop(columns=["y", "oof", "claims", "E_cell", "cell"]),
                           te.head(3000).drop(columns=["test_pred"]), fcfg)
    return list(X.columns)


# ------------------------------------------------------------------------------------------------- 1. inventory
UNUSED_CANDIDATES = ["days_quote_to_inception", "discount_applied", "campaign_code", "branch_id", "channel",
                     "payment_method", "previous_insurer", "customer_since_year", "email_domain", "marketing_segment",
                     "marketing_consent", "document_preference", "gender", "lines_held", "make_model", "compulsory_excess",
                     "year_built", "written_premium", "customer_service_contacts", "renewal_count", "occupation"]


def inventory():
    raw_cols = {}
    for f in sorted(RAW.glob("*.parquet")):
        raw_cols[f.stem] = list(pd.read_parquet(f).columns)
    rows = []
    for line in LINES:
        used = set(model_columns(line))
        tr, te = load_line(line)
        tables = [f"policies_{line}", "customers", "addresses", "weather"]
        for t in tables:
            for c in raw_cols[t]:
                in_cache = c in tr.columns
                in_test = c in te.columns
                rows.append({"line": line, "table": t, "column": c, "in_cache": in_cache, "in_test": in_test,
                             "used": c in used})
        for c in sorted(used - {r["column"] for r in rows if r["line"] == line}):
            rows.append({"line": line, "table": "engineered", "column": c, "in_cache": c in tr.columns,
                         "in_test": c in te.columns, "used": True})
        cached_unused = [c for c in tr.columns if c not in used and c not in ("y", "oof", "claims", "E_cell", "cell")]
        print(f"\n[{line}] model columns ({len(used)}): {sorted(used)}")
        print(f"[{line}] cache columns NOT fed to the model ({len(cached_unused)}): {cached_unused}")
    inv = pd.DataFrame(rows)
    save(inv, "inventory")
    un = inv[~inv.used]
    print("\nRaw columns not used, per line:")
    for line in LINES:
        print(f"  {line}: {un[un.line == line].column.tolist()}")
    print(f"\nclaims table columns: {raw_cols['claims']}")
    # univariate a/e of the unused candidates on emerging OOF rows
    out, stats = [], []
    for line in LINES:
        tr, _ = load_line(line)
        cvs = cv_sev(tr)
        used = set(model_columns(line))
        for c in UNUSED_CANDIDATES:
            if c not in tr.columns:
                continue
            t = ae_table(tr, c, cvs)
            t.insert(0, "line", line)
            t["used"] = c in used
            out.append(t)
            stats.append({"line": line, "used": c in used, **feature_stats(tr, c, cvs)})
        nc = noise_controls(tr)
        print(f"[{line}] credibility-lift null (random 10/40-level features): mean {nc['mean']:.1f} sd {nc['sd']:.1f} "
              f"max {nc['max']:.1f} bp")
    A = pd.concat(out, ignore_index=True)
    save(A, "ae_unused")
    L = pd.DataFrame(stats).sort_values(["line", "chi2_df"], ascending=[True, False])
    save(L, "stats_unused")
    print("\nCandidate columns: chi2/df of level a/e (null ~1, 95 % < ~2.2), 2023~2024 level correlation, lift (bp):")
    print(L.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    fl = A[((A.ae < 0.8) | (A.ae > 1.25)) & (A.claims >= 100)].sort_values("score", ascending=False)
    print("\nFlagged levels (a/e outside 0.8-1.25 on >= 100 claims):")
    print(fl.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


# ------------------------------------------------------------------------------------------------- 2. residual scan
def scan():
    out, lifts = [], []
    for line in LINES:
        tr, _ = load_line(line)
        cvs = cv_sev(tr)
        print(f"\n[{line}] emerging rows {len(tr):,}, claims {int(tr.claims.sum()):,}, cv_sev {cvs:.2f}, "
              f"OOF D2 {d2(tr.y, tr.oof):.4f}, E_cell D2 {d2(tr.y, tr.E_cell):.4f}, raw a/e {tr.y.sum() / tr.oof.sum():.3f}")
        g = tr.groupby("cell")
        print(pd.DataFrame({"rows": g.size(), "claims": g.claims.sum(), "ae_raw": g.y.sum() / g.oof.sum()}).to_string())
        used = [c for c in model_columns(line) if c in tr.columns]
        extra = {"expo_band": pd.cut(tr.exposure, [0, .25, .5, .75, .95, 1.1]).astype(str),
                 "nb_x_year": tr.cell,
                 "start_month": tr.period_start.dt.month.astype(str),
                 "nb_x_month": tr.new_business.astype(str) + "_" + tr.period_start.dt.month.astype(str)}
        for c in used + list(extra):
            col = extra.get(c, c)
            t = ae_table(tr, col, cvs, label=c)
            t.insert(0, "line", line)
            out.append(t)
            lifts.append({"line": line, **feature_stats(tr, col, cvs, label=c)})
    A = pd.concat(out, ignore_index=True)
    save(A, "ae_used")
    L = pd.DataFrame(lifts).sort_values(["line", "chi2_df"], ascending=[True, False])
    save(L, "stats_used")
    for line in LINES:
        print(f"\n[{line}] used features by chi2/df of level a/e (null ~1):")
        print(L[L.line == line].head(25).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    big = A[(A.claims >= 100)].copy()
    big = big[(big.ae < 0.8) | (big.ae > 1.25)].sort_values("score", ascending=False)
    print("\nUsed-feature levels with a/e outside 0.8-1.25 on >= 100 claims (ranked by claims x |log a/e|):")
    print(big.head(60).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    top = A[A.claims >= 100].sort_values("score", ascending=False)
    print("\nTop 40 levels by claims x |log a/e| (any a/e, >= 100 claims):")
    print(top.head(40).to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def nb_interaction():
    """Does a feature's relativity differ between new business and renewals (emerging OOF, E_cell base)?
    Per level: a/e on NB rows / a/e on renewal rows; chi2/df of the difference z (null ~1)."""
    rows = []
    for line in LINES:
        tr, _ = load_line(line)
        cvs = cv_sev(tr)
        nb = tr.new_business.astype(bool).values
        for c in [c for c in model_columns(line) if c in tr.columns and c not in ("new_business", "market")]:
            lv = bands(tr[c])
            g = pd.DataFrame({"lv": lv.values, "nb": nb, "y": tr.y.values, "E": tr.E_cell.values,
                              "n": tr.claims.values}).groupby(["lv", "nb"]).sum().unstack("nb")
            g = g.dropna()
            if len(g) < 2 or ("y", True) not in g.columns:
                continue
            a_nb, a_r = g[("y", True)] / g[("E", True)], g[("y", False)] / g[("E", False)]
            n_nb, n_r = g[("n", True)], g[("n", False)]
            ok = (n_nb >= 20) & (n_r >= 20) & (a_nb > 0) & (a_r > 0)
            if ok.sum() < 2:
                continue
            d = np.log(a_nb[ok] / a_r[ok])
            se = np.sqrt((1 + cvs ** 2) * (1 / n_nb[ok] + 1 / n_r[ok]))
            z = (d - np.average(d, weights=1 / se ** 2)) / se
            worst = z.abs().idxmax()
            rows.append({"line": line, "feature": c, "levels": int(ok.sum()), "nb_claims": float(n_nb[ok].sum()),
                         "chi2_df": float((z ** 2).sum() / (ok.sum() - 1)), "worst_level": worst,
                         "nb_ae": float(a_nb[worst]), "ren_ae": float(a_r[worst]), "nb_claims_lvl": float(n_nb[worst]),
                         "z": float(z[worst])})
    R = pd.DataFrame(rows).sort_values("chi2_df", ascending=False)
    save(R, "nb_interaction")
    print("NB x feature interaction (relativity on NB rows vs renewal rows), top 30 by chi2/df (null ~1):")
    print(R.head(30).to_string(index=False, float_format=lambda x: f"{x:.3f}"))


# ------------------------------------------------------------------------------------------------- 3. ids
def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(r"\D", "", regex=True), errors="coerce")


def ids():
    from scipy.stats import spearmanr
    cl = pd.read_parquet(RAW / "claims.parquet")
    cu = pd.read_parquet(RAW / "customers.parquet")
    for line in LINES:
        h = pd.read_parquet(RAW / f"policies_{line}.parquet")
        t = pd.read_parquet(RAW / f"test_{line}.parquet")
        for d in (h, t):
            d["ps"] = pd.to_datetime(d.period_start)
        print(f"\n[{line}] history {len(h):,} rows, test {len(t):,}")
        for c in ("row_id", "policy_id", "customer_id", "address_id"):
            hn, tn = _num(h[c]), _num(t[c])
            print(f"  {c:12s} history {hn.min():>10,.0f}-{hn.max():>10,.0f}  test {tn.min():>10,.0f}-{tn.max():>10,.0f}  "
                  f"prefix {sorted(set(h[c].astype(str).str[0]))} / {sorted(set(t[c].astype(str).str[0]))}  "
                  f"unique {h[c].nunique():,} / {t[c].nunique():,}")
        hn = _num(h.row_id)
        for yr in (2023, 2024):
            m = h.ps.dt.year == yr
            r1 = spearmanr(hn[m], h.ps[m].astype("int64")).statistic
            r2 = spearmanr(hn[m], h.new_business[m].astype(int)).statistic
            print(f"  row_id {yr}: range {hn[m].min():,.0f}-{hn[m].max():,.0f}, spearman vs period_start {r1:+.3f}, "
                  f"vs new_business {r2:+.3f}")
        tn = _num(t.row_id)
        print(f"  test row_id spearman vs period_start {spearmanr(tn, t.ps.astype('int64')).statistic:+.3f}, "
              f"vs new_business {spearmanr(tn, t.new_business.astype(int)).statistic:+.3f}")
        tr, _ = load_line(line)
        cvs = cv_sev(tr)
        tr = tr.assign(rid=_num(tr.row_id), pid=_num(tr.policy_id), cid=_num(tr.customer_id), aid=_num(tr.address_id))
        tr["rid_rank"] = tr.groupby("year").rid.rank(pct=True)
        for c in ("rid_rank", "pid", "cid", "aid"):
            st = feature_stats(tr, c, cvs, lift=False)
            print(f"  a/e by {c:8s} decile: chi2/df {st['chi2_df']:.2f}, year_corr {st['year_corr']:+.2f}, "
                  f"max|z| {st['max_abs_z']:.2f}")
        hp = set(h.policy_id)
        t["seen"] = t.policy_id.isin(hp)
        h24 = set(h.loc[h.ps.dt.year == 2024, "policy_id"])
        t["seen24"] = t.policy_id.isin(h24)
        print("  test: policy_id seen in history x new_business (rows):")
        print("   " + pd.crosstab(t.seen, t.new_business).to_string().replace("\n", "\n   "))
        print(f"  test renewals (NB False) whose policy_id is in 2024 history: {t.loc[~t.new_business, 'seen24'].mean():.3f}; "
              f"in any history year: {t.loc[~t.new_business, 'seen'].mean():.3f}")
        t["cust_seen"] = t.customer_id.isin(set(h.customer_id))
        print(f"  test NB rows whose customer_id holds a {line} policy in history: "
              f"{t.loc[t.new_business, 'cust_seen'].mean():.3f}")
        hh = h.sort_values("ps").groupby("policy_id").tail(1).set_index("policy_id")
        k = t[t.seen & ~t.new_business]
        if len(k):
            dif = k.renewal_count.values - hh.loc[k.policy_id, "renewal_count"].values
            print(f"  carried renewals: renewal_count(2025) - renewal_count(last history) = "
                  f"{pd.Series(dif).value_counts().head(5).to_dict()}")
        u = t[~t.seen & ~t.new_business]
        if len(u):
            print(f"  renewals NOT in history: {len(u):,} rows, renewal_count median {u.renewal_count.median():.0f}, "
                  f"share with renewal_count >= 2: {(u.renewal_count >= 2).mean():.3f}")
        cols = [c for c in ("driver_age_band", "policyholder_age_band") if c in t.columns]
        for c in cols:
            ct = pd.crosstab(t[c], [t.new_business, t.seen], normalize="columns").round(3)
            print(f"  test {c} share by (new_business, seen):\n   " + ct.to_string().replace("\n", "\n   "))
        print(f"  mean days_quote_to_inception by (NB, seen): "
              f"{t.groupby(['new_business', 'seen']).days_quote_to_inception.mean().round(1).to_dict()}")
    cl["cn"] = _num(cl.claim_id)
    cl["ld"] = pd.to_datetime(cl.loss_date)
    print(f"\nclaim_id: spearman vs loss_date {spearmanr(cl.cn, cl.ld.astype('int64')).statistic:+.3f}; "
          f"prefix {sorted(set(cl.claim_id.str[:2]))}")
    print(f"customers: {len(cu):,}; customer_since_year range {cu.customer_since_year.min()}-{cu.customer_since_year.max()}")


# ------------------------------------------------------------------------------------------------- 4. shift
def psi(a: pd.Series, b: pd.Series) -> float:
    pa = a.value_counts(normalize=True)
    pb = b.value_counts(normalize=True)
    idx = pa.index.union(pb.index)
    pa, pb = pa.reindex(idx).fillna(0) + 1e-4, pb.reindex(idx).fillna(0) + 1e-4
    return float(((pa - pb) * np.log(pa / pb)).sum())


def _bands_ref(ref: pd.Series, *others):
    """Band several series with the reference's deciles (categoricals unchanged)."""
    if ref.dtype == object or str(ref.dtype) in ("category", "bool", "string"):
        return [x.astype(str).where(x.notna(), "NA") for x in (ref, *others)]
    r = pd.to_numeric(ref, errors="coerce")
    if r.nunique() <= 12:
        return [pd.to_numeric(x, errors="coerce").map(lambda v: "NA" if pd.isna(v) else f"{v:g}") for x in (ref, *others)]
    e = np.unique(np.nanquantile(r, np.linspace(0, 1, 11)))
    e[0], e[-1] = -np.inf, np.inf
    return [pd.cut(pd.to_numeric(x, errors="coerce"), e).astype(str) for x in (ref, *others)]


def shift():
    rows = []
    for line in LINES:
        tr, te = load_line(line)
        h24 = tr[tr.year == 2024]
        cols = [c for c in model_columns(line) if c in te.columns and c not in ("market",)]
        cols += [c for c in ("make_model", "previous_insurer", "campaign_code", "branch_id", "email_domain",
                             "customer_since_year", "written_premium") if c in te.columns]
        nbh, nbt = h24.new_business.astype(bool), te.new_business.astype(bool)
        for c in cols:
            b24, bte, bnb, bren, btnb = _bands_ref(h24[c], te[c], h24.loc[nbh, c], h24.loc[~nbh, c], te.loc[nbt, c])
            rows.append({"line": line, "feature": c, "psi_test_vs_2024": psi(bte, b24),
                         "psi_testNB_vs_histNB": psi(btnb, bnb), "psi_testNB_vs_histREN": psi(btnb, bren),
                         "psi_histNB_vs_histREN": psi(bnb, bren)})
    P = pd.DataFrame(rows)
    P["testNB_closer_to"] = np.where(P.psi_testNB_vs_histNB < P.psi_testNB_vs_histREN, "NB", "REN")
    save(P, "shift_psi")
    print("PSI (>0.1 moderate, >0.25 large); NB = new business rows, history = 2024 emerging rows:")
    print(P.sort_values("psi_test_vs_2024", ascending=False).head(45).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nShare of features where test NB is closer to history NB than to history renewals:")
    print(P.groupby("line").testNB_closer_to.apply(lambda s: (s == "NB").mean()).round(3).to_string())
    big = P[P.psi_histNB_vs_histREN > 0.05]
    print("\nFeatures that separate history NB from renewals (PSI > 0.05): test NB closer to ...")
    print(big[["line", "feature", "psi_histNB_vs_histREN", "psi_testNB_vs_histNB", "psi_testNB_vs_histREN",
               "testNB_closer_to"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    for line in LINES:
        tr, te = load_line(line)
        h = tr[tr.year == 2024]
        hnb, tnb = h.new_business.astype(bool), te.new_business.astype(bool)
        print(f"\n[{line}] means history-2024-NB / history-2024-REN / test-NB / test-REN")
        for c in ("days_quote_to_inception", "discount_applied", "exposure", "customer_tenure", "renewal_count",
                  "lines_held", "pol_expo", "cust_expo", "cust_claims", "written_premium"):
            if c in te.columns:
                v = [pd.to_numeric(d[c], errors="coerce").mean() for d in (h[hnb], h[~hnb], te[tnb], te[~tnb])]
                print(f"   {c:24s} " + " / ".join(f"{x:8.3f}" for x in v))
        for c in ("channel", "payment_method", "campaign_code", "previous_insurer"):
            if c in te.columns:
                ct = pd.concat({k: d[c].astype(str).value_counts(normalize=True) for k, d in
                                (("h_nb", h[hnb]), ("h_ren", h[~hnb]), ("t_nb", te[tnb]), ("t_ren", te[~tnb]))},
                               axis=1).fillna(0)
                print(f"   {c}: top levels\n     " + ct.sort_values("t_nb", ascending=False).head(6).round(3)
                      .to_string().replace("\n", "\n     "))


# ------------------------------------------------------------------------------------------------- 5. cross-table
def cross():
    cl = pd.read_parquet(POOL_CACHE / "claims.parquet")
    cl = cl[~cl.claim_id.astype(str).str.startswith("D")]
    cu = pd.read_parquet(RAW / "customers.parquet")
    hist = {l: load_line(l)[0] for l in LINES}
    test = {l: load_line(l)[1] for l in LINES}
    allh = pd.concat([d[["row_id", "customer_id", "address_id", "policy_id", "year", "claims", "y", "E_cell"]]
                      .assign(line=l) for l, d in hist.items()])
    allt = pd.concat([d[["row_id", "customer_id", "address_id", "policy_id"]].assign(line=l, year=2025)
                      for l, d in test.items()])
    for nm, d in (("history", allh), ("test", allt)):
        g = d.groupby(["customer_id", "year"])
        print(f"{nm}: customer-years {g.ngroups:,}; lines per customer-year "
              f"{g.line.nunique().value_counts(normalize=True).sort_index().round(3).to_dict()}; "
              f"customer-year-lines with >1 policy {(d.groupby(['customer_id', 'year', 'line']).size() > 1).mean():.4f}")
    cy = allh.groupby(["customer_id", "year", "line"]).claims.sum().unstack("line")
    for a in LINES:
        for b in LINES:
            if a >= b:
                continue
            m = cy[[a, b]].dropna()
            if len(m) > 100:
                r = m.loc[m[a] > 0, b].gt(0).mean() / max(m.loc[m[a] == 0, b].gt(0).mean(), 1e-9)
                print(f"  same year: P({b} claim | {a} claim) / P({b} claim | no {a} claim) = {r:.2f} "
                      f"on {len(m):,} customer-years ({int((m[a] > 0).sum())} with {a} claims)")
    for l, tr in hist.items():
        cvs = cv_sev(tr)
        for o in LINES:
            c = f"cust_{o}_claims"
            if c in tr.columns:
                m = tr[tr.year == 2024]
                t = ae_table(m, (m[c] > 0).map({True: "prior_claim", False: "none"}), cvs, label=c)
                x = t.set_index("level")
                if "prior_claim" in x.index:
                    print(f"  [{l}] 2024 rows with a 2023 {o} claim: rows {int(x.loc['prior_claim', 'rows'])}, "
                          f"claims {x.loc['prior_claim', 'claims']:.0f}, a/e {x.loc['prior_claim', 'ae']:.3f} "
                          f"(none: {x.loc['none', 'ae']:.3f})")
    both = pd.concat([allh[["address_id", "customer_id", "year", "line"]], allt[["address_id", "customer_id", "year", "line"]]])
    nc = both.groupby(["address_id", "year"]).customer_id.nunique().rename("addr_customers")
    yrs = nc.index.get_level_values(1)
    print(f"\naddress-years with >1 customer: history {(nc[yrs < 2025] > 1).mean():.4f}, "
          f"test {(nc[yrs == 2025] > 1).mean():.4f}; max {nc.max()}")
    na = both.groupby("address_id").customer_id.nunique()
    print(f"addresses with >1 customer over all years: {(na > 1).sum():,} of {len(na):,}")
    for l, tr in hist.items():
        cvs = cv_sev(tr)
        m = tr.merge(nc.reset_index(), on=["address_id", "year"], how="left")
        m["addr_multi"] = np.where(m.addr_customers > 1, "shared", "single")
        st = ae_table(m, m.addr_multi, cvs, label="addr_shared").set_index("level")
        print(f"  [{l}] shared-address rows {int(st.rows.get('shared', 0))}, claims {st.claims.get('shared', 0):.0f}, "
              f"a/e {st.ae.get('shared', np.nan):.3f}")
    tp = cl.third_party_id.dropna().astype(str)
    print(f"\nthird_party_id: {len(tp):,} non-null, {tp.nunique():,} unique, prefixes {sorted(set(tp.str[:2]))[:5]}, "
          f"matching customer_id {tp.isin(set(cu.customer_id.astype(str))).sum()}, "
          f"rows sharing a third_party_id with another claim {tp.duplicated(keep=False).sum()}")
    rp = cl.repairer_id.dropna().astype(str)
    print(f"repairer_id: {len(rp):,} non-null, {rp.nunique()} unique; by line "
          f"{cl.dropna(subset=['repairer_id']).groupby('line').repairer_id.nunique().to_dict()}")
    rg = cl.dropna(subset=["repairer_id", "row_id"]).merge(
        pd.concat([d[["row_id", "region"]] for d in hist.values()]), on="row_id", how="left")
    pur = rg.groupby("repairer_id").region.agg(lambda s: s.value_counts(normalize=True).iloc[0] if s.notna().any() else np.nan)
    print(f"repairer region purity (share of its claims in its top region): median {pur.median():.2f}, "
          f"claims per repairer median {rg.groupby('repairer_id').size().median():.0f}")
    s = rg.groupby("repairer_id").net_incurred.agg(["count", "mean"])
    s20 = s[s["count"] >= 20]
    print(f"repairer mean severity: CV across {len(s20)} repairers with >= 20 claims {s20['mean'].std() / s20['mean'].mean():.2f}")
    from features import WEATHER
    for l, te in test.items():
        miss = te[WEATHER].isna().mean().round(4).to_dict()
        tr = hist[l]
        print(f"  [{l}] test weather missing share: {miss}")
        print(f"  [{l}] means 2023/2024/2025: hail_days {tr[tr.year == 2023].hail_days.mean():.1f} / "
              f"{tr[tr.year == 2024].hail_days.mean():.1f} / {te.hail_days.mean():.1f}; "
              f"freezing_days {tr[tr.year == 2023].freezing_days.mean():.1f} / {tr[tr.year == 2024].freezing_days.mean():.1f} / "
              f"{te.freezing_days.mean():.1f}; rainfall {tr[tr.year == 2023].rainfall_mm.mean():.0f} / "
              f"{tr[tr.year == 2024].rainfall_mm.mean():.0f} / {te.rainfall_mm.mean():.0f}; wind "
              f"{tr[tr.year == 2023].wind_exposure_index.mean():.2f} / {tr[tr.year == 2024].wind_exposure_index.mean():.2f} / "
              f"{te.wind_exposure_index.mean():.2f}")
    w = pd.read_parquet(RAW / "weather.parquet")
    print(f"weather table years {sorted(w.year.unique())}, rows per year {w.year.value_counts().sort_index().to_dict()}")
    ta = set(allt.address_id)
    print(f"test addresses with a 2025 weather row: {len(ta & set(w.loc[w.year == 2025, 'address_id'])) / len(ta):.4f}")
    for c in ("hail_days", "freezing_days", "rainfall_mm"):
        pv = w.pivot_table(index="address_id", columns="year", values=c)
        print(f"{c}: correlation across years per address {pv.corr().round(2).values.tolist()}")


# ------------------------------------------------------------------------------------------------- 6. target
def target():
    """Development of emerging rows by start quarter (emerging-only PART2 target), and OOF a/e by start month."""
    from market_spec import use_spec
    use_spec("part2")
    from target import prepare_target
    cache = ROOT / "experiments/cache/part2"
    claims = pd.read_parquet(cache / "claims.parquet")
    for line in LINES:
        tr = pd.read_parquet(cache / f"train_{line}.parquet")
        te = pd.read_parquet(cache / f"test_{line}.parquet")
        tg, info = prepare_target(tr, te, claims, line, {"develop": True, "cap_q": 0.99, "trend": "premium"})
        d = tr[["row_id", "year", "new_business", "exposure", "claim_count", "period_start"]].merge(tg, on="row_id")
        d["q"] = d.period_start.dt.quarter
        g = d.groupby(["year", "new_business", "q"]).agg(rows=("row_id", "size"), expo=("exposure", "sum"),
                                                        claims=("claim_count", "sum"), compl=("completeness", "mean"),
                                                        ibnr=("ibnr", "sum"), ydev=("y_dev", "sum"))
        g["freq"] = g.claims / g.expo
        g["ibnr_share"] = g.ibnr / g.ydev
        print(f"\n[{line}] development by (year, NB, start quarter):")
        print(g.round(3).to_string())
        f = g.freq.unstack("year")
        cpl = g.compl.unstack("year")
        print("  reported freq 2024 / 2023 by (NB, quarter) vs 2024 completeness:")
        print("  " + pd.DataFrame({"ratio": f[2024] / f[2023], "compl_2024": cpl[2024]}).round(3).to_string()
              .replace("\n", "\n  "))
    use_spec("part2pool")
    from season import month_days
    for line in LINES:
        tr, te = load_line(line)
        cvs = cv_sev(tr)
        t = ae_table(tr, tr.new_business.astype(str) + "_m" + tr.period_start.dt.month.astype(str).str.zfill(2), cvs,
                     label="nb_month")
        print(f"\n[{line}] OOF a/e (E_cell) by NB x start month:")
        print(t[t.claims >= 10].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        nb = tr[tr.new_business.astype(bool)]
        raw = nb.groupby(nb.period_start.dt.month).apply(lambda x: x.y.sum() / x.oof.sum(), include_groups=False)
        print(f"  raw OOF a/e NB rows by start month: {raw.round(3).to_dict()}")
        tnb = te[te.new_business.astype(bool)]
        print(f"  test NB start-month shares: {tnb.period_start.dt.month.value_counts(normalize=True).sort_index().round(3).to_dict()}")
        print(f"  test NB start day-of-month: min {tnb.period_start.dt.day.min()}, median {tnb.period_start.dt.day.median()}; "
              f"exposure mean {tnb.exposure.mean():.3f}")
        print(f"  row_type history NB / test NB: {nb.row_type.value_counts(normalize=True).round(3).to_dict()} / "
              f"{tnb.row_type.value_counts(normalize=True).round(3).to_dict()}")
        c = pd.read_parquet(POOL_CACHE / "claims.parquet")
        c = c[(c.line == line) & ~c.claim_id.astype(str).str.startswith("D") & c.row_id.notna()]
        ld = pd.to_datetime(c.loss_date)
        c = c[ld.dt.year == 2023]
        days = month_days(tr.period_start, tr.period_end, [2023]).sum(axis=0)
        cnt = np.bincount(pd.to_datetime(c.loss_date).dt.month.values - 1, minlength=12)
        print(f"  [{line}] 2023 claim frequency index by loss month (exposure-day weighted mean 1): "
              f"{np.round(cnt / days / (cnt.sum() / days.sum()), 2).tolist()}")


# ------------------------------------------------------------------------------------------------- 7a. stage 2 on pooled OOF
STAGE2 = {"motor": ["gender", "fuel_type"], "buildings": ["trees_within_5m"],
          "contents": ["gender", "works_from_home"]}


def _folds(groups: np.ndarray, seed: int, k: int = 5):
    u = pd.unique(groups)
    rng = np.random.default_rng(seed)
    f = pd.Series(rng.integers(0, k, len(u)), index=u)
    fid = f.loc[groups].values
    for i in range(k):
        yield np.where(fid != i)[0], np.where(fid == i)[0]


def _glm_stage2(tr: pd.DataFrame, cols, alpha: float, seed: int, base: str = "E_cell") -> np.ndarray:
    """OOF prediction of a ridge Tweedie (1.5) GLM on one-hot `cols` with offset log(base), fitted on y capped at the
    99.5 % quantile of claim rows and grossed up to the uncapped total (level kept)."""
    from glum import GeneralizedLinearRegressor, TweedieDistribution
    M = pd.get_dummies(pd.concat([bands(tr[c]).rename(c) for c in cols], axis=1), dtype=float).values
    y = tr.y.values
    yfit = np.minimum(y, np.quantile(y[tr.claims.values > 0], 0.995))
    yfit = yfit * y.sum() / yfit.sum()
    off = np.log(tr[base].values.clip(1e-9))
    p = np.zeros(len(tr))
    for a, b in _folds(tr.policy_id.values, seed):
        m = GeneralizedLinearRegressor(family=TweedieDistribution(1.5), link="log", alpha=alpha, l1_ratio=0.0,
                                       fit_intercept=True)
        m.fit(M[a], yfit[a], offset=off[a])
        p[b] = m.predict(M[b], offset=off[b])
        cb = tr.cell.values[b]                    # level is settled: rescale per (year x NB) cell to the base total
        sb = pd.Series(p[b]).groupby(cb).transform("sum").values
        eb = pd.Series(tr[base].values[b]).groupby(cb).transform("sum").values
        p[b] *= eb / sb
    return p


def stage2():
    """Emerging-specific relativities on top of the pooled OOF (E_cell base), honest 5-fold policy-grouped OOF,
    3 fold shuffles. Null: the same GLM on random features with the same number of levels."""
    rng = np.random.default_rng(3)
    rows = []
    for line, cols in STAGE2.items():
        tr, _ = load_line(line)
        base = d2(tr.y, tr.E_cell)
        for alpha in (1.0, 3.0, 10.0, 30.0):
            g = [(d2(tr.y, _glm_stage2(tr, cols, alpha, s)) - base) * 1e4 for s in (1, 2, 3)]
            nl = []
            for s in (1, 2, 3):
                fake = tr.copy()
                for c in cols:
                    k = bands(tr[c]).nunique()
                    fake[c] = rng.integers(0, k, len(tr)).astype(str)
                nl.append((d2(tr.y, _glm_stage2(fake, cols, alpha, s)) - base) * 1e4)
            rows.append({"line": line, "features": "+".join(cols), "alpha": alpha, "gain_bp": np.mean(g),
                         "gain_sd": np.std(g), "null_bp": np.mean(nl), "null_sd": np.std(nl)})
            print(f"  {line:9s} {'+'.join(cols):28s} alpha {alpha:5.1f}: gain {np.mean(g):6.1f} bp "
                  f"(folds {', '.join(f'{x:.1f}' for x in g)}) | random-feature null {np.mean(nl):6.1f} "
                  f"({', '.join(f'{x:.1f}' for x in nl)})", flush=True)
    R = pd.DataFrame(rows)
    save(R, "stage2")
    # fitted relativities on all emerging rows (alpha 3 and 10) and their effect on the 2025 test mix
    from glum import GeneralizedLinearRegressor, TweedieDistribution
    for a_, line, cols in [(a_, l, c) for a_ in (3.0, 10.0) for l, c in STAGE2.items()]:
        tr, te = load_line(line)
        X = pd.concat([bands(tr[c]).rename(c) for c in cols], axis=1)
        M = pd.get_dummies(X, dtype=float)
        y = tr.y.values
        yfit = np.minimum(y, np.quantile(y[tr.claims.values > 0], 0.995))
        yfit = yfit * y.sum() / yfit.sum()
        m = GeneralizedLinearRegressor(family=TweedieDistribution(1.5), link="log", alpha=a_, l1_ratio=0.0)
        m.fit(M.values, yfit, offset=np.log(tr.E_cell.values))
        rel = pd.Series(np.exp(m.coef_), index=M.columns)
        Mt = pd.get_dummies(pd.concat([bands(te[c]).rename(c) for c in cols], axis=1), dtype=float).reindex(
            columns=M.columns, fill_value=0.0)
        ft = np.exp(Mt.values @ m.coef_)
        f_nb = ft[te.new_business.astype(bool).values].mean() / ft.mean()
        print(f"  [{line}] alpha-{a_:g} relativities (before level renormalisation): {rel.round(3).to_dict()}; "
              f"test factor range {ft.min() / ft.mean():.3f}-{ft.max() / ft.mean():.3f}, NB/all mean {f_nb:.4f}")


def _fit_apply(trA: pd.DataFrame, trB: pd.DataFrame, cols, alpha: float) -> np.ndarray:
    """Ridge Tweedie GLM on one-hot cols with offset log(E_cell), fitted on A, applied to B, B level kept per cell."""
    from glum import GeneralizedLinearRegressor, TweedieDistribution
    XA = pd.concat([bands(trA[c]).rename(c) for c in cols], axis=1)
    MA = pd.get_dummies(XA, dtype=float)
    MB = pd.get_dummies(pd.concat([bands(trB[c]).rename(c) for c in cols], axis=1), dtype=float).reindex(
        columns=MA.columns, fill_value=0.0)
    y = trA.y.values
    yfit = np.minimum(y, np.quantile(y[trA.claims.values > 0], 0.995))
    yfit = yfit * y.sum() / yfit.sum()
    m = GeneralizedLinearRegressor(family=TweedieDistribution(1.5), link="log", alpha=alpha, l1_ratio=0.0)
    m.fit(MA.values, yfit, offset=np.log(trA.E_cell.values))
    p = m.predict(MB.values, offset=np.log(trB.E_cell.values))
    cb = trB.cell.values
    return p * pd.Series(trB.E_cell.values).groupby(cb).transform("sum").values / \
        pd.Series(p).groupby(cb).transform("sum").values


def forward():
    """Year-split check of stage 2: (a) the STAGE2 lists fitted on 2023 -> scored on 2024 and 2024 -> 2023;
    (b) selection-honest: per line, the top-k used features by chi2/df on 2023 ALONE, fitted on 2023, scored on 2024
    (and the reverse). Gains in bp of D2 on the scored year, vs E_cell."""
    rows = []
    for line in LINES:
        tr, _ = load_line(line)
        cvs = cv_sev(tr)
        A, B = tr[tr.year == 2023].reset_index(drop=True), tr[tr.year == 2024].reset_index(drop=True)
        cand = [c for c in model_columns(line) if c in tr.columns and c not in ("market", "new_business")
                and not c.startswith(("pol_", "cust_"))]
        for fit, sc, nm in ((A, B, "2023->2024"), (B, A, "2024->2023")):
            base = d2(sc.y, sc.E_cell)
            st = pd.DataFrame([feature_stats(fit, c, cvs, lift=False) for c in cand]).sort_values("chi2_df",
                                                                                                ascending=False)
            for alpha in (3.0, 10.0):
                g_fixed = (d2(sc.y, _fit_apply(fit, sc, STAGE2[line], alpha)) - base) * 1e4
                out = {"line": line, "split": nm, "alpha": alpha, "fixed_list": "+".join(STAGE2[line]),
                       "gain_fixed_bp": g_fixed}
                for k in (1, 2, 3):
                    top = st.feature.head(k).tolist()
                    out[f"top{k}"] = "+".join(top)
                    out[f"gain_top{k}_bp"] = (d2(sc.y, _fit_apply(fit, sc, top, alpha)) - base) * 1e4
                rows.append(out)
                print(f"  {line:9s} {nm} alpha {alpha:4.1f}: fixed {out['fixed_list']} {g_fixed:6.1f} bp | "
                      f"top1 {out['top1']} {out['gain_top1_bp']:6.1f} | top2 {out['top2']} {out['gain_top2_bp']:6.1f} | "
                      f"top3 {out['top3']} {out['gain_top3_bp']:6.1f}", flush=True)
    save(pd.DataFrame(rows), "stage2_forward")


def overall():
    """Overall emerging D2 (all lines together, as the leaderboard scores) of E_cell vs E_cell x stage 2 (OOF,
    3 fold shuffles) for several feature lists: the full STAGE2 list (an upper bound: chosen on both years) and the
    subset that survives selection on ONE year (forward check: motor gender, contents works_from_home)."""
    from metrics import d2_tweedie
    variants = [("full list, alpha 3", STAGE2, 3.0), ("full list, alpha 10", STAGE2, 10.0),
                ("one-year-selection survivors, alpha 3", {"motor": ["gender"], "contents": ["works_from_home"]}, 3.0),
                ("motor gender+fuel only, alpha 3", {"motor": ["gender", "fuel_type"]}, 3.0)]
    data = {line: load_line(line)[0] for line in LINES}
    y = np.concatenate([data[l].y.values for l in LINES])
    B = np.concatenate([data[l].E_cell.values for l in LINES])
    base = d2_tweedie(y, B)
    print(f"overall emerging D2 of E_cell: {base:.4f}")
    for nm, lists, alpha in variants:
        gains = []
        for s in (1, 2, 3):
            P = np.concatenate([_glm_stage2(data[l], lists[l], alpha, s) if l in lists else data[l].E_cell.values
                                for l in LINES])
            gains.append((d2_tweedie(y, P) - base) * 1e4)
        print(f"  {nm:40s}: overall gain {np.mean(gains):5.1f} bp (fold shuffles {', '.join(f'{x:.1f}' for x in gains)})",
              flush=True)


def weather_shift(name: str = "P2_P34_si05"):
    """How much do the 2025 weather values move the test predictions? Emerging-only fit (IAA_MARKET=part2) on all
    history, test predicted with its 2025 weather and with each address's 2024 weather instead (same model)."""
    os.environ["IAA_MARKET"] = "part2"
    from market_spec import use_spec
    use_spec("part2")
    import pipeline
    from configs import EXPERIMENTS
    from features import WEATHER
    from features import build_base
    cfg = EXPERIMENTS[name]
    spec = pipeline.get_spec()
    base = build_base()
    w = pd.read_parquet(RAW / "weather.parquet")
    w24 = w[w.year == 2024].set_index("address_id")
    for line in LINES:
        U = pipeline._load_unit(base, [line], cfg, spec)
        P = U["parts"][0]
        from sklearn.model_selection import GroupShuffleSplit   # early stopping on the quick 80/20 split
        a, b = next(GroupShuffleSplit(1, test_size=spec.quick_test_size, random_state=spec.seed).split(U["X"], groups=U["g"]))
        m, _ = pipeline._fit_predict(cfg, U["X"].iloc[a], U["y"][a], U["w"][a], U["e"][a], U["X"].iloc[b], U["e"][b],
                                     U["y"][b], U["w"][b], c=U["c"][a], cv_=U["c"][b], line=line)
        pt = m.predict(U["Xt"], U["te_e"])
        Xt2 = U["Xt"].copy()
        aid = P["te"].address_id.values
        for c in WEATHER:
            if c in Xt2:
                Xt2[c] = w24[c].reindex(aid).astype("float32").values
        p2 = m.predict(Xt2, U["te_e"])
        r = pt / p2
        nb = P["te"].new_business.astype(bool).values
        print(f"  [{line}] mean prediction 2025 weather / 2024 weather: {pt.sum() / p2.sum():.4f} "
              f"(NB {pt[nb].sum() / p2[nb].sum():.4f}, renewals {pt[~nb].sum() / p2[~nb].sum():.4f}); "
              f"row ratio p5/p50/p95 {np.quantile(r, 0.05):.3f} / {np.median(r):.3f} / {np.quantile(r, 0.95):.3f}; "
              f"corr(log ratio, log pred) {np.corrcoef(np.log(r), np.log(pt))[0, 1]:+.3f}", flush=True)


STAGE2_VARIANTS = {
    "core_a3": ({"motor": ["gender", "fuel_type"]}, 3.0),
    "full_a10": (STAGE2, 10.0),
    "full_a3": (STAGE2, 3.0),
}


def factors():
    """Per-test-row stage-2 factors for the final file (default --ref R2_FINAL): ridge Tweedie GLM on the emerging OOF
    (E_cell offset), applied to the 2025 rows, renormalised inside each (line x new_business) cell so that
    sum(test_pred x factor) = sum(test_pred) -- every probed subset level is unchanged. Writes
    <OUT>/stage2_factors.csv (row_id, line, new_business, one column per variant; lines without features get 1)."""
    from glum import GeneralizedLinearRegressor, TweedieDistribution
    out = []
    for line in LINES:
        tr, te = load_line(line)
        f = pd.DataFrame({"row_id": te.row_id.values, "line": line, "new_business": te.new_business.astype(bool).values})
        for vn, (lists, alpha) in STAGE2_VARIANTS.items():
            if line not in lists:
                f[vn] = 1.0
                continue
            cols = lists[line]
            M = pd.get_dummies(pd.concat([bands(tr[c]).rename(c) for c in cols], axis=1), dtype=float)
            Mt = pd.get_dummies(pd.concat([bands(te[c]).rename(c) for c in cols], axis=1), dtype=float).reindex(
                columns=M.columns, fill_value=0.0)
            y = tr.y.values
            yfit = np.minimum(y, np.quantile(y[tr.claims.values > 0], 0.995))
            yfit = yfit * y.sum() / yfit.sum()
            m = GeneralizedLinearRegressor(family=TweedieDistribution(1.5), link="log", alpha=alpha, l1_ratio=0.0)
            m.fit(M.values, yfit, offset=np.log(tr.E_cell.values))
            raw = np.exp(Mt.values @ m.coef_)
            p = te.test_pred.values
            k = pd.Series(p * raw).groupby(f.new_business.values).transform("sum").values / \
                pd.Series(p).groupby(f.new_business.values).transform("sum").values
            f[vn] = raw / k
            print(f"  [{line}] {vn}: relativities {pd.Series(np.exp(m.coef_), index=M.columns).round(3).to_dict()}; "
                  f"factor p1/p50/p99 {np.quantile(f[vn], 0.01):.3f} / {np.median(f[vn]):.3f} / {np.quantile(f[vn], 0.99):.3f}; "
                  f"check sum(p f)/sum(p) NB {np.sum((p * f[vn])[f.new_business]) / np.sum(p[f.new_business]):.6f}",
                  flush=True)
        out.append(f)
    F = pd.concat(out, ignore_index=True)
    save(F, "stage2_factors")
    print(f"wrote {OUT / 'stage2_factors.csv'}: {len(F):,} rows")


# ------------------------------------------------------------------------------------------------- 7b. quick screens
def screen(names, seeds):
    """Emerging-only quick screens (pipeline.quick, 80/20 policy-grouped) for each config over several split seeds;
    deltas vs the FIRST config, per line and overall. Run with IAA_MARKET=part2."""
    os.environ.setdefault("IAA_MARKET", "part2")
    import pipeline
    from configs import EXPERIMENTS
    res = {}
    for n in names:
        for s in seeds:
            print(f"== {n} seed {s}", flush=True)
            res[(n, s)] = pipeline.quick(n, EXPERIMENTS[n], seed=s)
    keys = ["motor", "buildings", "contents", "overall"]
    rows = []
    for n in names:
        for s in seeds:
            rows.append({"config": n, "seed": s, **{k: res[(n, s)][k] for k in keys}})
    R = pd.DataFrame(rows)
    b = R[R.config == names[0]].set_index("seed")[keys]
    D = []
    for n in names:
        x = R[R.config == n].set_index("seed")[keys]
        d = (x - b) * 1e4
        D.append({"config": n, **{f"{k}": x[k].mean() for k in keys},
                  **{f"d_{k}_bp": d[k].mean() for k in keys}, **{f"sd_{k}_bp": d[k].std() for k in keys}})
    D = pd.DataFrame(D)
    tag = "_".join(names)[:80]
    save(R, f"screen_raw_{tag}")
    save(D, f"screen_{tag}")
    print("\nQuick-screen means over seeds and paired deltas vs the first config (bp of D2):")
    print(D.to_string(index=False, float_format=lambda x: f"{x:.4f}" if abs(x) < 1 else f"{x:.1f}"))
    return D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("section")
    ap.add_argument("configs", nargs="*")
    ap.add_argument("--seeds", nargs="*", type=int, default=[11])
    ap.add_argument("--out", default=None)
    ap.add_argument("--ref", default=None, help="pooled result name to use as E instead of R2_P34_hn (e.g. R2_FINAL)")
    a = ap.parse_args()
    global OUT, REF
    if a.ref:
        REF = ROOT / f"experiments/results/part2pool/{a.ref}.pkl"
    if a.out:
        OUT = Path(a.out)
    fns = {"inventory": inventory, "scan": scan, "nbint": nb_interaction, "ids": ids, "shift": shift, "cross": cross, "target": target, "stage2": stage2, "forward": forward, "overall": overall, "wxshift": weather_shift, "factors": factors}
    for k in ([*fns] if a.section == "all" else [a.section]):
        if k == "screen":
            screen(a.configs, a.seeds)
        else:
            print(f"\n================ {k}", flush=True)
            fns[k]()


if __name__ == "__main__":
    main()
