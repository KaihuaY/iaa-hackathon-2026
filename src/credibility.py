"""Post-hoc credibility correction layer: Buhlmann experience rating + ridge segment layer."""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import GroupKFold

from metrics import d2_tweedie, tweedie_deviance

COMPLETENESS_2024 = {"motor": 0.920, "buildings": 0.939, "contents": 0.960}
K_GRID = [0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 1e9]
ALPHA_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 0.3, 1.0, 3.0]
MIN_LEVEL_ROWS = 500


def folds_for(train_df):
    """Exactly the folds the models used."""
    return list(GroupKFold(5).split(np.zeros(len(train_df)), groups=train_df["policy_id"]))


# ---------------------------------------------------------------- Part 1
def buhlmann_terms(train_df, test_df, oof, y_eval, completeness_2024, cap_q=0.995):
    """Prior-year (strictly earlier, same policy) E, A_n, A_c, P for train and test rows.
    A_c is built from y_eval (developed + trended, same scale as the OOF predictions P), capped at the
    cap_q quantile of positive y_eval; no completeness scaling is applied to it. A_n is the claim count
    divided by completeness_2024 for year-2024 rows.
    Returns (train_terms, test_terms, sev) with sev = sum(A_c)/sum(A_n) so A_n*sev is on P's scale."""
    oof = np.asarray(oof, float)
    y_eval = np.asarray(y_eval, float)
    pos = y_eval[y_eval > 0]
    cap = np.quantile(pos, cap_q) if len(pos) else np.inf
    comp = np.where(train_df["year"].to_numpy() == 2024, 1.0 / completeness_2024, 1.0)
    capped = np.minimum(y_eval, cap)
    capped = capped * (oof.sum() / capped.sum())  # population ratio of A_c to P is 1 (cap removes mass from A only)
    a_n = np.nan_to_num(train_df["claim_count"].to_numpy(float)) * comp
    sev = float(capped.sum() / a_n.sum()) if a_n.sum() > 0 else 0.0

    g = pd.DataFrame({
        "policy_id": train_df["policy_id"].to_numpy(),
        "year": train_df["year"].to_numpy(),
        "E": train_df["exposure"].to_numpy(float),
        "A_n": a_n,
        "A_c": capped,
        "P": oof,
    }).groupby(["policy_id", "year"], sort=True).sum()
    cols = ["E", "A_n", "A_c", "P"]
    cum = g.groupby(level=0).cumsum()
    prior = cum.groupby(level=0).shift(1).fillna(0.0)  # sums over earlier years only
    prior_tot = cum.groupby(level=0).tail(1).droplevel(1)  # all train years (for test rows)

    keys = pd.MultiIndex.from_arrays([train_df["policy_id"].to_numpy(), train_df["year"].to_numpy()])
    tr = prior.reindex(keys)
    train_terms = {c: tr[c].to_numpy() for c in cols}

    if test_df["year"].min() <= train_df["year"].max():
        raise ValueError("test years must be later than train years")
    te = prior_tot.reindex(test_df["policy_id"].to_numpy()).fillna(0.0)
    test_terms = {c: te[c].to_numpy() for c in cols}
    return train_terms, test_terms, sev


def buhlmann_adjust(mu, E, A, P, k):
    mu, E, A, P = (np.asarray(x, float) for x in (mu, E, A, P))
    ok = (E > 0) & (P > 0)
    Z = E / (E + k)
    with np.errstate(divide="ignore", invalid="ignore"):
        mult = 1 + Z * (A / np.where(ok, P, 1.0) - 1)
    mult = np.clip(np.where(ok, mult, 1.0), 0.25, 4.0)
    return mu * mult


def _dev_sums(y, mu, folds):
    d = tweedie_deviance(y, mu)
    return np.array([d[te].sum() for _, te in folds])


def _nested_choice(y, cands, folds):
    """cands: {param: adjusted_mu over all rows}. Choose param on 4 folds' rows (min deviance), apply to 5th.
    Returns (nested_mu, best_param_in_sample, chosen_param_per_fold)."""
    params = list(cands)
    fd = np.array([_dev_sums(y, cands[p], folds) for p in params])  # (P, 5)
    tot = fd.sum(1)
    best = params[int(np.argmin(tot))]
    out = np.empty(len(y))
    chosen = []
    for f, (_, te) in enumerate(folds):
        p = params[int(np.argmin(tot - fd[:, f]))]
        chosen.append(p)
        out[te] = cands[p][te]
    return out, best, chosen


def buhlmann_select(y, mu, terms, sev, folds, variant):
    """variant 'cost' (A=A_c) or 'count' (A=A_n*sev)."""
    A = terms["A_c"] if variant == "cost" else terms["A_n"] * sev
    cands = {k: buhlmann_adjust(mu, terms["E"], A, terms["P"], k) for k in K_GRID}
    nested, best, chosen = _nested_choice(y, cands, folds)
    return {"variant": variant, "k": best, "d2_insample": d2_tweedie(y, cands[best]),
            "d2_nested": d2_tweedie(y, nested), "nested_oof": nested, "fold_k": chosen}


# ---------------------------------------------------------------- Part 2
def segment_design(train_df, test_df, include_levels=False):
    """Sparse 0/1 design matrices (train, test) with identical columns; returns (Dtr, Dte, names)."""
    cols_tr, cols_te, names = [], [], []

    def add(name, a, b):
        cols_tr.append(np.asarray(a, bool))
        cols_te.append(np.asarray(b, bool))
        names.append(name)

    def has(c):
        return c in train_df.columns and c in test_df.columns

    for feat in (("make_model", "previous_insurer") if include_levels else ()):
        if not has(feat):
            continue
        a, b = train_df[feat].astype(object), test_df[feat].astype(object)
        vc = a.value_counts()
        keep = list(vc.index[vc >= MIN_LEVEL_ROWS])
        for lev in sorted(keep, key=str):
            add(f"{feat}={lev}", a == lev, b == lev)
        add(f"{feat}=_other", ~a.isin(keep), ~b.isin(keep))
    for c in ("pol_claims", "cust_claims"):
        if has(c):
            a, b = train_df[c].fillna(0), test_df[c].fillna(0)
            add(f"{c}=1", a == 1, b == 1)
            add(f"{c}>=2", a >= 2, b >= 2)
    if has("flood_zone"):
        add("flood_zone=5", train_df["flood_zone"] == 5, test_df["flood_zone"] == 5)
    if has("listed_building"):
        add("listed_building", train_df["listed_building"].fillna(False).astype(bool),
            test_df["listed_building"].fillna(False).astype(bool))
    if has("energy_rating"):
        add("energy_rating=missing", train_df["energy_rating"].isna(), test_df["energy_rating"].isna())
    if has("cover_type"):
        vc = train_df["cover_type"].value_counts()
        for lev in vc.index[1:]:
            add(f"cover_type={lev}", train_df["cover_type"] == lev, test_df["cover_type"] == lev)
    if has("sum_insured"):
        lo, hi = train_df["sum_insured"].quantile([0.1, 0.9])
        add("sum_insured_bottom_decile", train_df["sum_insured"] <= lo, test_df["sum_insured"] <= lo)
        add("sum_insured_top_decile", train_df["sum_insured"] >= hi, test_df["sum_insured"] >= hi)
    if has("bedrooms"):
        add("bedrooms=1", train_df["bedrooms"] == 1, test_df["bedrooms"] == 1)
        add("bedrooms>=5", train_df["bedrooms"] >= 5, test_df["bedrooms"] >= 5)
    if has("prior_claims_5y"):
        add("prior_claims_5y>=2", train_df["prior_claims_5y"] >= 2, test_df["prior_claims_5y"] >= 2)

    Dtr = sp.csr_matrix(np.column_stack(cols_tr).astype(np.float64))
    Dte = sp.csr_matrix(np.column_stack(cols_te).astype(np.float64))
    return Dtr, Dte, names


def fit_segment_layer(y, mu, D, alpha):
    from glum import GeneralizedLinearRegressor, TweedieDistribution
    m = GeneralizedLinearRegressor(family=TweedieDistribution(1.5), link="log", alpha=alpha,
                                   l1_ratio=0.0, fit_intercept=True)
    m.fit(D, np.asarray(y, float), offset=np.log(np.asarray(mu, float)))
    return m


def apply_segment_layer(model, mu, D):
    return model.predict(D, offset=np.log(np.asarray(mu, float)))


def segment_nested(y, mu, D, folds, alphas=ALPHA_GRID):
    """Nested OOF for each alpha: {alpha: (oof, d2)}. The alpha is then picked on this same nested
    D2, which is mildly optimistic (selection on the scores that get reported)."""
    res = {}
    for a in alphas:
        o = np.empty(len(y))
        for tr, te in folds:
            m = fit_segment_layer(y[tr], mu[tr], D[tr], a)
            o[te] = apply_segment_layer(m, mu[te], D[te])
        res[a] = (o, d2_tweedie(y, o))
    return res


# ---------------------------------------------------------------- Part 3
def _segment_defs(line, tr, sum_q):
    segs = {}
    if line == "motor":
        for mm in ("MM193", "MM251", "MM186", "MM124"):
            segs[f"make_model={mm}"] = (tr["make_model"] == mm).to_numpy()
        segs["cover_type=third_party_only"] = (tr["cover_type"] == "third_party_only").to_numpy()
    if line == "buildings":
        segs["sum_insured bottom decile"] = (tr["sum_insured"] <= sum_q[0]).to_numpy()
        segs["sum_insured top decile"] = (tr["sum_insured"] >= sum_q[1]).to_numpy()
        segs["bedrooms==1"] = (tr["bedrooms"] == 1).to_numpy()
        segs["listed_building"] = tr["listed_building"].fillna(False).astype(bool).to_numpy()
    if line == "contents":
        segs["flood_zone==5"] = (tr["flood_zone"] == 5).to_numpy()
    pc = tr["pol_claims"].fillna(0)
    segs["pol_claims=0"] = (pc == 0).to_numpy()
    segs["pol_claims=1"] = (pc == 1).to_numpy()
    segs["pol_claims>=2"] = (pc >= 2).to_numpy()
    return segs


def correct(name, use_buhlmann=True, use_segments=True, include_levels=False):
    import pipeline
    from data import LINES
    from features import CACHE

    base = pipeline.load(name)
    out_lines, checks, summary = {}, [], {}
    mults, ks, alphas = {}, {}, {}
    all_y, all_base, all_out = [], [], []
    for line in LINES:
        bl = base["lines"][line]
        tr = pd.read_parquet(CACHE / f"train_{line}.parquet")
        te = pd.read_parquet(CACHE / f"test_{line}.parquet")
        assert (tr["row_id"].to_numpy() == bl["row_id"]).all(), f"{line}: train row_id mismatch"
        assert (te["row_id"].to_numpy() == bl["test_row_id"]).all(), f"{line}: test row_id mismatch"
        y = np.asarray(bl["y_eval"], float)
        mu = np.asarray(bl["oof"], float)
        mu_t = np.asarray(bl["test_pred"], float)
        folds = folds_for(tr)
        d2_base = d2_tweedie(y, mu)
        cur, cur_t = mu, mu_t
        d2_b = d2_base
        ks[line] = {}
        print(f"\n=== {line} ===  base D2 {d2_base:.5f}")

        if use_buhlmann:
            ttr, tte, sev = buhlmann_terms(tr, te, mu, y, COMPLETENESS_2024[line])
            _ok = (ttr["E"] > 0) & (ttr["P"] > 0)
            print(f"  sanity: exposure-weighted mean A_c/P (train, E>0) = "
                  f"{np.sum(ttr['E'][_ok] * ttr['A_c'][_ok] / ttr['P'][_ok]) / np.sum(ttr['E'][_ok]):.4f}; sev={sev:.1f}")
            sels = {v: buhlmann_select(y, mu, ttr, sev, folds, v) for v in ("cost", "count")}
            for v, s in sels.items():
                print(f"  Buhlmann {v}: k={s['k']} in-sample D2 {s['d2_insample']:.5f} nested D2 {s['d2_nested']:.5f}")
            bv = max(sels, key=lambda v: sels[v]["d2_nested"])
            s = sels[bv]
            if s["d2_nested"] > d2_base:
                A_t = tte["A_c"] if bv == "cost" else tte["A_n"] * sev
                cur = s["nested_oof"]
                cur_t = buhlmann_adjust(mu_t, tte["E"], A_t, tte["P"], s["k"])
                d2_b = s["d2_nested"]
                ks[line] = {"variant": bv, "k": s["k"], "fold_k": s["fold_k"],
                            "d2_cost_nested": sels["cost"]["d2_nested"], "d2_count_nested": sels["count"]["d2_nested"]}
            else:
                ks[line] = {"variant": "none", "k": None}
            if ks[line]["variant"] != "none":
                _m = (cur / mu)[ttr["E"] > 0]
                q = np.percentile(_m, [5, 50, 95])
                print(f"  Buhlmann multiplier (E>0 train rows) p5/p50/p95: {q[0]:.3f} / {q[1]:.3f} / {q[2]:.3f}")
            else:
                print("  Buhlmann multiplier: none")
            print(f"  -> kept: {ks[line]['variant']}  D2 {d2_b:.5f}")

        d2_s = d2_b
        mults[line] = {}
        alphas[line] = None
        if use_segments:
            Dtr, Dte, names = segment_design(tr, te, include_levels)
            nest = segment_nested(y, cur, Dtr, folds)
            for a, (_, sc) in nest.items():
                print(f"  segments alpha={a:g}: nested D2 {sc:.5f}")
            ba = max(nest, key=lambda a: nest[a][1])
            final = fit_segment_layer(y, cur, Dtr, ba)
            cur, cur_t = nest[ba][0], apply_segment_layer(final, cur_t, Dte)
            d2_s, alphas[line] = nest[ba][1], ba
            mults[line] = {n: float(np.exp(c)) for n, c in zip(names, final.coef_)}
            mults[line]["_intercept"] = float(np.exp(final.intercept_))
            top = sorted(((n, m) for n, m in mults[line].items() if n != "_intercept"),
                         key=lambda t: -abs(np.log(t[1])))[:10]
            print(f"  chosen alpha={ba:g}; 10 largest multipliers (by |log|):")
            for n, m in top:
                print(f"     {n:40s} {m:.3f}")
        print(f"  SUMMARY {line}: base {d2_base:.5f} | Buhlmann(nested) {d2_b:.5f} | segments(nested) {d2_s:.5f}")
        summary[line] = (d2_base, d2_b, d2_s)

        sum_q = tr["sum_insured"].quantile([0.1, 0.9]).to_numpy() if "sum_insured" in tr else None
        for sname, mask in _segment_defs(line, tr, sum_q).items():
            checks.append((line, sname, int(mask.sum()), y[mask].sum() / mu[mask].sum(), y[mask].sum() / cur[mask].sum()))

        out_lines[line] = {"row_id": bl["row_id"], "y_eval": bl["y_eval"], "oof": cur, "test_row_id": bl["test_row_id"],
                           "test_pred": cur_t, "cv": d2_tweedie(y, cur)}
        all_y.append(y)
        all_base.append(mu)
        all_out.append(cur)

    y_all = np.concatenate(all_y)
    d2_before, d2_after = d2_tweedie(y_all, np.concatenate(all_base)), d2_tweedie(y_all, np.concatenate(all_out))
    print(f"\nOVERALL (nested, concatenated): before {d2_before:.5f} -> after {d2_after:.5f}")
    print("\nSegment check: actual/pred = sum(y_eval)/sum(pred)")
    print(f"  {'line':10s} {'segment':32s} {'rows':>8s} {'before':>8s} {'after':>8s}")
    for line, s, n, b, a in checks:
        print(f"  {line:10s} {s:32s} {n:8d} {b:8.3f} {a:8.3f}")

    res_out = {k: v for k, v in base.items() if k not in ("lines", "cv", "cv_nested", "name", "cfg", "holdout")}
    res_out.update({"name": name + "_cred", "lines": out_lines, "cv": d2_after, "cv_nested": d2_after,
                    "holdout": np.nan, "multipliers": mults, "k": ks, "alpha": alphas,
                    "cfg": {"model": "correction", "base": name, "use_buhlmann": use_buhlmann,
                            "use_segments": use_segments, "include_levels": include_levels, "desc": "Buhlmann experience rating + ridge segment layer"}})
    pipeline._save(res_out)
    print(f"saved {res_out['name']}")
    return res_out
