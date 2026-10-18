"""Target preparation: develop to ultimate, cap large losses, trend to 2025, maturity weights.

Development uses a Bornhuetter-Ferguson style IBNR: each row gets the claims it is *expected* to still report,
  IBNR_row = a-priori frequency x sum over its days of (1 - completeness(age)) x late-claim severity(age) / 365,
so rows without reported claims also pick up IBNR, and nothing depends on the row's own (noisy) experience.
"""
import numpy as np
import pandas as pd

from market_spec import get_spec, warn_once


def __getattr__(name):
    """Market constants follow the active MarketSpec at access time (PEP 562).
    SUSPECT_CLAIMS (Part 1): an open motor bodily-injury claim with an 11.9M case estimate, 10x the next largest
    claim, treated as a data error (capped like any large loss, its excess NOT spread into the large-loss loading)."""
    s = get_spec()
    table = {"SUSPECT_CLAIMS": lambda: list(s.suspect_claims), "MATURE_YEARS": lambda: s.curve_years,
             "MAX_LAG": lambda: s.max_lag, "DAY0": lambda: s.day0}
    if name in table:
        return table[name]()
    raise AttributeError(f"module 'target' has no attribute {name!r}")


def caps(claims: pd.DataFrame, q: float) -> dict:
    """Per-line claim-level cap at quantile q of net incurred."""
    return {l: float(g.net_incurred.quantile(q)) for l, g in claims.groupby("line")} if q else {}


def _lags(claims):
    return (claims.report_date - claims.loss_date).dt.days


def completeness_curve(claims: pd.DataFrame, line: str) -> np.ndarray:
    """P(claim reported within `age` days of loss), age = 0..MAX_LAG, from mature loss years."""
    spec = get_spec()
    c = claims[(claims.line == line) & claims.loss_date.dt.year.isin(spec.curve_years)]
    lag = np.sort(_lags(c).values)
    return np.searchsorted(lag, np.arange(spec.max_lag + 1), side="right") / len(lag)


def late_severity_relativity(claims: pd.DataFrame, line: str, cap: float | None) -> np.ndarray:
    """Mean severity of claims reported later than `age` days, relative to all claims (mature years)."""
    MAX_LAG = get_spec().max_lag
    c = claims[(claims.line == line) & claims.loss_date.dt.year.isin(get_spec().curve_years)]
    sev = c.net_incurred.clip(upper=cap) if cap else c.net_incurred
    order = np.argsort(_lags(c).values)
    lag, sev = _lags(c).values[order], sev.values[order]
    tail_sum = np.cumsum(sev[::-1])[::-1]          # sum of sev for claims with lag >= lag[i]
    out = np.empty(MAX_LAG + 1)
    for a in range(MAX_LAG + 1):
        i = np.searchsorted(lag, a, side="right")   # claims with lag > a
        n = len(lag) - i
        out[a] = tail_sum[i] / n if n >= 30 else np.nan
    out = pd.Series(out).ffill().values
    return out / sev.mean()


def _daily(curve: np.ndarray) -> np.ndarray:
    """Map an age-indexed curve onto calendar days 2021-01-01 .. valuation date."""
    spec = get_spec()
    DAY0, VALUATION_DATE = spec.day0, spec.valuation_ts
    days = pd.date_range(DAY0, VALUATION_DATE, freq="D")
    age = np.clip((VALUATION_DATE - days).days, 0, spec.max_lag)
    return curve[age]


def _row_integral(pol: pd.DataFrame, daily: np.ndarray) -> np.ndarray:
    """Sum of a daily series over each row's period (days)."""
    DAY0 = get_spec().day0
    cs = np.concatenate([[0.0], np.cumsum(daily)])
    s = np.clip((pol.period_start - DAY0).dt.days.values, 0, len(daily) - 1)
    e = np.clip((pol.period_end - DAY0).dt.days.values, 0, len(daily) - 1)
    return cs[e + 1] - cs[s]


def trend_index(train: pd.DataFrame, test: pd.DataFrame, method: str, pp_by_year: pd.Series | None = None,
                target_year: int | None = None) -> dict:
    """Factor taking each training year's cost to the `target_year` level (test year for submissions; earlier for backtests).

    premium: ratio of mean full-year written premium (target year from test if 2025, else from training rows).
    claims:  log-linear fit of developed pure premium on years before the target (max 2021-23 to avoid immature 2024).
    """
    spec = get_spec()
    years = list(spec.train_years)
    target_year = spec.test_year if target_year is None else target_year
    if method == "none":
        return {y: 1.0 for y in years}
    if method == "premium":
        full = train[train.exposure > 0.99]
        idx = full.groupby(full.period_start.dt.year).written_premium.mean()
        if target_year == spec.test_year:
            idx[spec.test_year] = test.loc[test.exposure > 0.99, "written_premium"].mean()
        return {y: float(idx[target_year] / idx[y]) for y in years}
    if method == "claims":
        fit_years = [y for y in spec.mature_years if y < target_year]
        if len(fit_years) < 2:                    # a slope needs two points (review fix)
            return {y: 1.0 for y in years}
        pp = pp_by_year.loc[fit_years]
        slope = np.polyfit(pp.index.values, np.log(pp.values), 1)[0]
        return {y: float(np.exp(slope * (target_year - y))) for y in years}
    raise ValueError(method)


def _ibnr(df, claims, lc, line, cost, cap):
    """BF IBNR per row on the scale of `cost` (capped or not)."""
    comp = completeness_curve(claims, line)
    rel = late_severity_relativity(claims, line, cap)
    spec = get_spec()
    m = df.year <= spec.last_mature_year
    freq = (lc.loss_date.dt.year <= spec.last_mature_year).sum() / (df.loc[m, "exposure"] * df.loc[m, "completeness"]).sum()
    sev = cost[lc.loss_date.dt.year == spec.ref_sev_year].mean()          # recent, mostly mature severity
    return freq * sev * _row_integral(df, _daily((1 - comp) * rel)) / 365


def peril_matrix(pol: pd.DataFrame, claims: pd.DataFrame, line: str, tgt: pd.DataFrame, info: dict,
                 min_claims: int | None = None) -> tuple[np.ndarray, list]:
    """Per-row training cost split by peril, consistent with y_train (same cap, trend; IBNR spread by peril share).

    Perils with fewer than `min_claims` (default spec.peril_min_claims = 1000) claims are pooled into 'other'."""
    spec = get_spec()
    min_claims = spec.peril_min_claims if min_claims is None else min_claims
    lc = claims[(claims.line == line) & claims.row_id.notna()].copy()
    cost = lc.net_incurred.where(~lc.claim_id.isin(spec.suspect_claims), lc.paid_to_date)
    if info.get("cap"):
        cost = cost.clip(upper=info["cap"])
    counts = lc.peril.value_counts()
    lc["grp"] = lc.peril.where(lc.peril.map(counts) >= min_claims, "other")
    tab = cost.groupby([lc.row_id, lc.grp]).sum().unstack(fill_value=0.0)
    tab = tab.reindex(pol.row_id.values).fillna(0.0)
    share = tab.sum() / tab.values.sum()
    trend = pol.period_start.dt.year.map(info["trend"]).values
    M = tab.values * trend[:, None] + (tgt.ibnr.values * trend)[:, None] * share.values[None, :]
    return M, list(tab.columns)


def _segments(pol: pd.DataFrame, test: pd.DataFrame, line: str) -> tuple[pd.Series, pd.Series]:
    """Segment label per row of pol and test (NaN where a component is missing)."""
    sd = get_spec().segments.get(line)
    if sd is None:                                   # no segment definition: every row falls back to the pooled loading
        nan = pd.Series(np.nan, index=pol.index, dtype=object), pd.Series(np.nan, index=test.index, dtype=object)
        return nan
    need = [sd[k] for k in ("cat", "num", "col") if k in sd]
    absent = [c for c in need if c not in pol.columns or c not in test.columns]
    if absent:
        warn_once(f"_segments[{line}]: column(s) {absent} absent; pooled large-loss loading used")
        return pd.Series(np.nan, index=pol.index, dtype=object), pd.Series(np.nan, index=test.index, dtype=object)
    if sd["kind"] == "cross":
        cat, num = sd["cat"], sd["num"]
        def lab(d):
            band = pd.cut(d[num], [-np.inf, *sd["edges"], np.inf], right=False, labels=list(sd["labels"])).astype(object)
            return (d[cat].astype(object).astype(str) + "|" + band.astype(str)).where(band.notna() & d[cat].notna())
        return lab(pol), lab(test)
    col = sd["col"]
    edges = np.nanquantile(pol[col], list(sd["q"]))
    def lab(d):
        q = pd.Series(np.searchsorted(edges, d[col].values, side="right"), index=d.index)
        return (sd.get("prefix", "Q") + (q + 1).astype(str)).where(d[col].notna())
    return lab(pol), lab(test)


def segment_loading(pol, test, lc, cost, capped, line, K=None):
    """Credibility-shrunk large-loss loading per segment: 1 + (n r_s + K r) / (n + K), r = excess / capped (mature claims).
    Returns (load_train per pol row, load_test per test row, table)."""
    spec = get_spec()
    K = spec.segment_credibility_k if K is None else K
    sp, st = _segments(pol, test, line)
    mature = (lc.loss_date.dt.year <= spec.last_mature_year) & lc.row_id.notna()
    seg = lc.row_id.map(pd.Series(sp.values, index=pol.row_id.values))
    d = pd.DataFrame({"seg": seg, "capped": capped, "excess": (cost - capped).clip(lower=0)})[mature & seg.notna()]
    r = d.excess.sum() / d.capped.sum()
    t = d.groupby("seg").agg(n_claims=("capped", "size"), capped=("capped", "sum"), excess=("excess", "sum"))
    t["raw_ratio"] = t.excess / t.capped
    t["loading"] = 1 + (t.n_claims * t.raw_ratio + K * r) / (t.n_claims + K)
    t = t.reset_index().rename(columns={"seg": "segment"})
    m = t.set_index("segment").loading
    return (sp.map(m).fillna(1 + r).values.astype(float), st.map(m).fillna(1 + r).values.astype(float), t)


def prepare_target(pol: pd.DataFrame, test: pd.DataFrame, claims: pd.DataFrame, line: str, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Add training label, evaluation label and weight to a line's training rows.

    cfg keys: develop (bool), cap_q (float or None), trend ('premium'|'claims'|'none' or {line: method}),
              trend_to (target year, default spec.test_year), open_factor (float), weight_power (float).
    y_train follows cfg. y_eval is FIXED across experiments so CV scores are comparable:
    uncapped cost (suspect claim at paid-to-date), developed, premium-trended.
    """
    spec = get_spec()
    lc = claims[(claims.line == line) & claims.row_id.notna()].copy()
    suspect = lc.claim_id.isin(spec.suspect_claims)
    base = lc.net_incurred.where(~suspect, lc.paid_to_date)
    cost = base.where(lc.status != "open", base * cfg.get("open_factor", 1.0))
    cap = caps(claims[claims.line == line], cfg.get("cap_q")).get(line)
    capped = cost.clip(upper=cap) if cap else cost
    mature = lc.loss_date.dt.year <= spec.last_mature_year
    loading = float(cost[mature].sum() / capped[mature].sum())

    df = pol[["row_id", "period_start", "period_end", "exposure", "written_premium"]].copy()
    df["year"] = df.period_start.dt.year
    days = ((df.period_end - df.period_start).dt.days + 1).values
    df["completeness"] = _row_integral(df, _daily(completeness_curve(claims, line))) / days
    df["rep_train"] = df.row_id.map(capped.groupby(lc.row_id).sum()).fillna(0.0)
    df["rep_eval"] = df.row_id.map(base.groupby(lc.row_id).sum()).fillna(0.0)
    df["ibnr"] = _ibnr(df, claims, lc, line, capped, cap) if cfg.get("develop", True) else 0.0
    ibnr_eval = _ibnr(df, claims, lc, line, base, None)

    pp_year = (df.rep_train + df.ibnr).groupby(df.year).sum() / df.exposure.groupby(df.year).sum()
    method = cfg.get("trend", "premium")
    method = method[line] if isinstance(method, dict) else method
    tf = trend_index(pol, test, method, pp_year, cfg.get("trend_to", spec.test_year))
    tf_eval = trend_index(pol, test, "premium")
    df["y_train"] = (df.rep_train + df.ibnr) * df.year.map(tf)
    df["y_eval"] = (df.rep_eval + ibnr_eval) * df.year.map(tf_eval)
    df["y_dev"] = df.rep_eval + ibnr_eval            # developed actual, untrended (used by the level backtest)
    wp = cfg.get("weight_power", 0.0)
    df["w"] = df.completeness.clip(lower=0.05) ** wp if wp else 1.0
    k_rec = cfg.get("recency", 0.0)
    if k_rec:                                         # later policy years weigh more; mean weight unchanged
        w0 = df.w.astype(float)
        w1 = w0 * (1.0 + k_rec * (df.year - df.year.min()))
        df["w"] = w1 * (w0.mean() / w1.mean())
    info = {"line": line, "cap": cap, "loading": loading, "trend": tf, "trend_method": method,
            "ibnr_share": float(df.ibnr.sum() / max(df.rep_train.sum() + df.ibnr.sum(), 1e-9))}
    if cfg.get("loading") == "segment" and cap:
        info["loading_rows"], info["loading_test"], info["loading_table"] = segment_loading(pol, test, lc, cost, capped, line)
    if cfg.get("loading") == "model" and cap:             # per-row untrended totals for apply_model_loading
        rid = lc.row_id
        info["loading_inputs"] = {"y_total": df.row_id.map(cost.groupby(rid).sum()).fillna(0.0).values,
                                  "y_capped": df.row_id.map(capped.groupby(rid).sum()).fillna(0.0).values}
    return df[["row_id", "y_train", "y_eval", "y_dev", "w", "completeness", "ibnr"]], info


def apply_model_loading(tgt, info, X, Xt, tr, te, cfg):
    """cfg["loading"] == "model": fill info["loading_rows"/"loading_test"] from largeloss.model_loading.
    cfg is the target cfg (optional key "loading_params"). Call right after _line_data (X, Xt are not visible to
    prepare_target). No-op unless loading == "model" and the line is capped."""
    if cfg.get("loading") != "model" or "loading_inputs" not in info:
        return info
    from largeloss import model_loading
    from market_spec import get_spec
    li = info["loading_inputs"]
    info["loading_rows"], info["loading_test"], info["loading_diag"] = model_loading(
        X, Xt, li["y_total"], li["y_capped"], tr["exposure"].values, tgt["w"].values,
        tr[get_spec().group_col].values, info["loading"], cfg.get("loading_params"))
    return info
