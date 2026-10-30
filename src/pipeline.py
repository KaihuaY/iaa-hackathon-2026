"""Run one experiment: CV (grouped by policy), 2023 time-holdout diagnostic, full fit, test predictions."""
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from features import build_base, model_matrix
from market_spec import get_spec
from metrics import d2_tweedie
from models import make
from target import apply_model_loading, peril_matrix, prepare_target



def __getattr__(name):
    if name == "RESULTS":                       # follows the active MarketSpec (PEP 562)
        return get_spec().results_path
    raise AttributeError(f"module 'pipeline' has no attribute {name!r}")


def _line_data(base, line, cfg):
    tr, te = base[f"train_{line}"], base[f"test_{line}"]
    tcfg = {**cfg.get("target", {}), **cfg.get("target_by_line", {}).get(line, {})}   # per-line target overrides (e.g. cap_q)
    tgt, info = prepare_target(tr, te, base["claims"], line, tcfg)
    fcfg = dict(cfg.get("features", {}))
    fcfg["drop"] = list(fcfg.get("drop", [])) + list(fcfg.get("drop_line", {}).get(line, []))
    X, Xt, cats = model_matrix(tr, te, fcfg)
    apply_model_loading(tgt, info, X, Xt, tr, te, tcfg)   # no-op unless target.loading == "model"
    return tr, te, tgt, info, X, Xt


def _aux(cfg, base, line, tr, tgt, info):
    """Extra per-row targets: claim counts (freq x sev) or the per-peril cost matrix (per-peril models)."""
    if cfg["model"] == "perperil":
        M, perils = peril_matrix(tr, base["claims"], line, tgt, info)
        print(f"  {line}: perils modelled {perils}", flush=True)
        return M
    return tr.claim_count.values


def _fit_predict(cfg, X, y, w, e, Xv, ev, yv=None, wv=None, n_iter=None, c=None, cv_=None, line=None):
    sw = cfg.get("target", {}).get("shift_weights")
    if sw:                                       # covariate-shift weights vs the rows this fit predicts (Xv)
        from shift import density_ratio_weights
        w = np.asarray(w) * density_ratio_weights(X, Xv, clip=tuple(sw.get("clip", (0.2, 5.0))))[0]
    params = cfg.get("params")
    if line is not None and cfg.get("params_by_line", {}).get(line):
        params = {**(params or {}), **cfg["params_by_line"][line]}
    m = make(cfg["model"], params)
    if cfg["model"] in ("freqsev", "perperil"):
        m.fit(X, y, w, e, Xv if n_iter is None else None, yv, wv, ev, n_iter=n_iter, counts=c, counts_val=cv_)
    elif cfg["model"] in ("lgbm", "glmboost", "catboost", "xgb", "nnboost"):
        m.fit(X, y, w, e, Xv if n_iter is None else None, yv, wv, ev, n_iter=n_iter)
    else:
        m.fit(X, y, w, e)
    return m, m.predict(Xv, ev)


def _concat_frames(Xs, Xts, lines):
    """Stack per-line design matrices (train and test share categorical levels) and add the categorical `line_id`."""
    from pandas.api.types import union_categoricals
    cols = [c for c in Xs[0].columns if all(c in f.columns for f in Xs[1:] + Xts)]
    allf = [f[cols] for f in Xs + Xts]
    out = {}
    for c in cols:
        ser = [f[c] for f in allf]
        if all(isinstance(x.dtype, pd.CategoricalDtype) for x in ser):
            u = union_categoricals([x.array for x in ser], sort_categories=True)
            out[c] = pd.Series(u, dtype=u.dtype)
        else:
            out[c] = pd.concat(ser, ignore_index=True)
    big = pd.DataFrame(out)
    nt = sum(len(f) for f in Xs)
    ids = np.concatenate([np.full(len(f), k) for k, f in enumerate(Xs)] + [np.full(len(f), k) for k, f in enumerate(Xts)])
    big["line_id"] = pd.Categorical.from_codes(ids, categories=list(lines))
    return big.iloc[:nt].reset_index(drop=True), big.iloc[nt:].reset_index(drop=True)


def _units(cfg, spec):
    """Groups of lines fitted together: one unit per line, except lines named in cfg['pool'] (one shared unit)."""
    pool = cfg.get("pool")
    if not pool:
        return [[l] for l in spec.lines]
    pooled = [l for l in pool if l in spec.lines]
    for l in pool:
        if l not in spec.lines:
            print(f"  pool: line {l!r} not in spec.lines - ignored", flush=True)
    if len(pooled) < 2:
        return [[l] for l in spec.lines]
    out, done = [], False
    for l in spec.lines:
        if l in pooled:
            if not done:
                out.append(pooled); done = True
        else:
            out.append([l])
    return out


def _offset_exposure(tr, te, tcfg, line=None):
    """Exposure for the log offset. target cfg `si_offset_power` k > 0 multiplies it by (sum_insured / train median)^k so
    the model prices per unit of sum insured (elasticity k) and the trees learn deviations; k = 0 (default) = plain exposure.
    Probe evidence 19 Oct: the 2025 buildings shift is concentrated in high-SI homes and the OOF under-predicts the top SI decile.
    `season_offset` alpha > 0 (needs `line`) also multiplies by seasonal_weight(period)^alpha (monthly peril-cost index)."""
    k = float(tcfg.get("si_offset_power", 0.0) or 0.0)
    alpha = float(tcfg.get("season_offset", 0.0) or 0.0)
    e_tr, e_te = tr.exposure.values.astype(float), te.exposure.values.astype(float)
    if k != 0.0 and "sum_insured" in tr:
        ref = float(np.nanmedian(tr.sum_insured.astype(float)))
        e_tr = e_tr * (tr.sum_insured.astype(float).fillna(ref).values / ref) ** k
        e_te = e_te * (te.sum_insured.astype(float).fillna(ref).values / ref) ** k
    if alpha > 0.0 and line is not None and "period_start" in tr and "period_end" in tr:
        from season import get_index, seasonal_weight
        idx = get_index(line)
        e_tr = e_tr * seasonal_weight(tr.period_start, tr.period_end, idx) ** alpha
        e_te = e_te * seasonal_weight(te.period_start, te.period_end, idx) ** alpha
    return e_tr, e_te


def _market_weights(cfg, tr):
    """Pooled market (rows with a `market` column): (fit, validation) weight multipliers per row, or None.
    cfg `market_weight_developed` m: developed rows weigh m, emerging rows 1; `market_weight_developed_val` (default m)
    is the developed weight in the early-stopping validation set. Absent key or no `market` column -> None (no change)."""
    m = cfg.get("market_weight_developed")
    if m is None or "market" not in tr:
        return None
    dev = (tr["market"].astype(str) == "developed").values
    mv = cfg.get("market_weight_developed_val", m)
    fit = np.where(dev, float(m), 1.0)
    k = cfg.get("weight_emerging_nb")             # Review 11: emerging new-business rows weigh k in the fit (not in validation)
    if k and "new_business" in tr:
        fit = np.where(~dev & tr["new_business"].astype(bool).values, fit * float(k), fit)
    return fit, np.where(dev, float(mv), 1.0)


def _load_unit(base, lines, cfg, spec):
    """Per-line data via _line_data (targets/weights/loadings untouched); for a pooled unit also the stacked arrays."""
    parts = []
    for line in lines:
        tr, te, tgt, info, X, Xt = _line_data(base, line, cfg)
        tcfg = {**cfg.get("target", {}), **cfg.get("target_by_line", {}).get(line, {})}
        e_tr, e_te = _offset_exposure(tr, te, tcfg, line=line)
        parts.append({"line": line, "tr": tr, "te": te, "tgt": tgt, "info": info, "X": X, "Xt": Xt,
                      "y": tgt.y_train.values, "w": tgt.w.values, "e": e_tr, "te_e": e_te,
                      "c": _aux(cfg, base, line, tr, tgt, info),
                      "load": info["loading"] if cfg.get("target", {}).get("cap_q") else 1.0})
        mw = _market_weights(cfg, tr)
        if mw is not None:                       # pooled market only; Part 1 never reaches this
            parts[-1]["wv"] = parts[-1]["w"] * mw[1]
            parts[-1]["w"] = parts[-1]["w"] * mw[0]
    if len(parts) == 1:
        p = parts[0]
        U = {k: p[k] for k in ("X", "Xt", "y", "w", "e", "c")}
        if "wv" in p:
            U["wv"] = p["wv"]
        U["g"], U["yr"], U["te_e"], U["fit_line"] = p["tr"][spec.group_col], p["tr"].period_start.dt.year.values, p["te_e"], p["line"]
    else:
        U = {k: np.concatenate([p[k] for p in parts]) for k in ("y", "w", "e", "c")}
        if all("wv" in p for p in parts):
            U["wv"] = np.concatenate([p["wv"] for p in parts])
        U["X"], U["Xt"] = _concat_frames([p["X"] for p in parts], [p["Xt"] for p in parts], lines)
        U["g"] = np.concatenate([p["tr"][spec.group_col].values for p in parts])
        U["yr"] = np.concatenate([p["tr"].period_start.dt.year.values for p in parts])
        U["te_e"] = np.concatenate([p["te_e"] for p in parts])
        U["fit_line"] = None
    U["parts"] = parts
    U["n"] = np.cumsum([0] + [len(p["tr"]) for p in parts])
    U["nt"] = np.cumsum([0] + [len(p["te"]) for p in parts])
    return U


def _fit_unit(cfg, spec, U, folds, holdout):
    """CV folds (grouped), holdout-year fit and full fit on one unit (a line, or the pooled lines). Raw predictions, no loading."""
    X, y, w, e, c, ln = U["X"], U["y"], U["w"], U["e"], U["c"], U["fit_line"]
    wv = U.get("wv", w)                          # validation weights (differ from w only in a pooled market)
    oof, iters = np.zeros(len(X)), []
    for a, b in GroupKFold(folds).split(X, groups=U["g"]):
        m, p = _fit_predict(cfg, X.iloc[a], y[a], w[a], e[a], X.iloc[b], e[b], y[b], wv[b], c=c[a], cv_=c[b], line=ln)
        oof[b] = p
        iters.append(getattr(m, "best_iter", None))
    ho_b = ho_p = None
    if holdout:
        a, ho_b = np.where(U["yr"] < spec.holdout_year)[0], np.where(U["yr"] == spec.holdout_year)[0]
        n_ho = int(np.mean([i for i in iters if i])) if iters[0] else None
        _, ho_p = _fit_predict(cfg, X.iloc[a], y[a], w[a], e[a], X.iloc[ho_b], e[ho_b], n_iter=n_ho, c=c[a], line=ln)
    n_full = int(np.mean(iters) * 1.1) if iters[0] else None
    m, pt = _fit_predict(cfg, X, y, w, e, U["Xt"], U["te_e"], n_iter=n_full, c=c, line=ln)
    return {"oof": oof, "iters": iters, "ho_b": ho_b, "ho_p": ho_p, "pt": pt, "m": m}


def quick(name: str, cfg: dict, seed: int | None = None) -> dict:
    """Fast screen: one 80/20 split grouped by policy (same split every run), one fit per line. No test predictions.
    Use to triage ideas; anything promising then gets the full 5-fold CV."""
    from sklearn.model_selection import GroupShuffleSplit
    spec = get_spec()
    seed = spec.seed if seed is None else seed
    base = build_base()
    ys, ps, out = [], [], {}
    em_pairs = []
    for lines in _units(cfg, spec):
        U = _load_unit(base, lines, cfg, spec)
        X, y, w, e, c = U["X"], U["y"], U["w"], U["e"], U["c"]
        a, b = next(GroupShuffleSplit(1, test_size=spec.quick_test_size, random_state=seed).split(X, groups=U["g"]))
        wv = U.get("wv", w)
        _, p = _fit_predict(cfg, X.iloc[a], y[a], w[a], e[a], X.iloc[b], e[b], y[b], wv[b], c=c[a], cv_=c[b], line=U["fit_line"])
        for k, P in enumerate(U["parts"]):
            line, info = P["line"], P["info"]
            sel = (b >= U["n"][k]) & (b < U["n"][k + 1])
            bl, pl = b[sel] - U["n"][k], p[sel]
            lb = info["loading_rows"][bl] if "loading_rows" in info else P["load"]
            out[line] = d2_tweedie(P["tgt"].y_eval.values[bl], pl * lb)
            ys.append(P["tgt"].y_eval.values[bl]); ps.append(pl * lb)
            print(f"  quick {line:9s} D2 {out[line]:.4f}", flush=True)
            if "market" in P["tr"]:              # pooled market: also score the emerging rows of the split
                em = P["tr"]["market"].astype(str).values[bl] == "emerging"
                ye, pe = P["tgt"].y_eval.values[bl][em], np.broadcast_to(pl * lb, pl.shape)[em]
                out[f"{line}_emerging"] = d2_tweedie(ye, pe)
                em_pairs.append((ye, pe))
                print(f"  quick {line:9s} emerging D2 {out[f'{line}_emerging']:.4f} | calib {pe.sum() / ye.sum():.3f} "
                      f"| rows {int(em.sum())}", flush=True)
    out["overall"] = d2_tweedie(np.concatenate(ys), np.concatenate(ps))
    print(f"  quick OVERALL D2 {out['overall']:.4f}", flush=True)
    if em_pairs:
        out["overall_emerging"] = d2_tweedie(np.concatenate([a for a, _ in em_pairs]), np.concatenate([b for _, b in em_pairs]))
        print(f"  quick OVERALL emerging D2 {out['overall_emerging']:.4f}", flush=True)
    return out


holdout_floor = 0.0
RATIO_LO, RATIO_HI = 0.5, 2.0


def run(name: str, cfg: dict, folds: int | None = None, holdout: bool = True, verbose: bool = True) -> dict:
    """Returns dict with per-line OOF/test predictions and scores; caches to experiments/results/<name>.pkl."""
    if cfg["model"] == "blend":
        return run_blend(name, cfg)
    t0 = time.time()
    spec = get_spec()
    folds = spec.n_folds if folds is None else folds
    base = build_base()
    res = {"name": name, "cfg": cfg, "lines": {}}
    for lines in _units(cfg, spec):
        U = _load_unit(base, lines, cfg, spec)
        F = _fit_unit(cfg, spec, U, folds, holdout)
        m = F["m"]
        imp = m.importance().head(25).to_dict() if hasattr(m, "importance") else {}
        for k, P in enumerate(U["parts"]):
            line, tr, te, tgt, info, load, e = P["line"], P["tr"], P["te"], P["tgt"], P["info"], P["load"], P["e"]
            iters = F["iters"]
            lr = info["loading_rows"] if "loading_rows" in info else load
            oof = F["oof"][U["n"][k]:U["n"][k + 1]] * lr
            cv = d2_tweedie(tgt.y_eval, oof)
            calib = float(oof.sum() / tgt.y_eval.sum())          # predicted / actual total cost (guardrail: ~1)
            ho = np.nan
            ho_extra = {}
            if holdout:
                sel = (F["ho_b"] >= U["n"][k]) & (F["ho_b"] < U["n"][k + 1])
                b = F["ho_b"][sel] - U["n"][k]
                ho = d2_tweedie(tgt.y_eval.values[b], F["ho_p"][sel] * (info["loading_rows"][b] if "loading_rows" in info else load))
                # per-row holdout predictions (for market-subset scoring; the scores above are unchanged)
                ho_extra = {"holdout_idx": b, "holdout_pred": F["ho_p"][sel] * (info["loading_rows"][b] if "loading_rows" in info else load)}
            pt = F["pt"][U["nt"][k]:U["nt"][k + 1]]
            pt_load = info.get("loading_test", load)                 # per-row array if segment loading, else scalar
            res["lines"][line] = {"row_id": tr.row_id.values, "y_eval": tgt.y_eval.values, "oof": oof,
                                  "test_row_id": te.row_id.values, "test_pred": pt * pt_load, "cv": cv, "holdout": ho, "calibration": calib,
                                  "info": info, "iters": iters, "importance": imp, **ho_extra}
            ratio = float((np.mean(pt * pt_load / te.exposure.values)) / np.mean(oof / e))
            res["lines"][line]["test_sanity"] = ratio
            if not (RATIO_LO <= ratio <= RATIO_HI):
                print(f"WARNING: {line} test predictions implausible (ratio {ratio:.2f}) - result marked unsafe", flush=True)
                res["lines"][line]["unsafe"] = True
                res["unsafe"] = True
            if ho < holdout_floor:
                print(f"WARNING: negative holdout on {line}", flush=True)
            if verbose:
                print(f"  {line:9s} CV D2 {cv:.4f} | {spec.holdout_year} holdout {ho:.4f} | calib {calib:.3f} | iters {iters} | loading {load:.3f} "
                      f"| test mean {np.mean(pt * pt_load):,.1f} | test/oof {ratio:.2f}", flush=True)
    res["lines"] = {l: res["lines"][l] for l in spec.lines}
    _summarise(res)
    res["minutes"] = (time.time() - t0) / 60
    _save(res)
    if res.get("unsafe"):
        print("UNSAFE RESULT", flush=True)
    return res


def _simplex(k: int, step: float = 0.05):
    """All weight vectors of length k on a `step` grid that sum to 1."""
    n = int(round(1 / step))
    if k == 1:
        return [np.array([1.0])]
    out = []
    for i in range(n + 1):
        for rest in _simplex(k - 1, step):
            v = np.r_[i / n, np.asarray(rest) * (1 - i / n)]
            out.append(v)
    return [v for v in {tuple(np.round(v, 6)) for v in out} if abs(sum(v) - 1) < 1e-6 and min(v) >= -1e-9]


def _best_weights(y, P, grid):
    scores = [d2_tweedie(y, P @ np.asarray(g)) for g in grid]
    i = int(np.argmax(scores))
    return np.asarray(grid[i]), scores[i]


def _stack_fit(y, P):
    """Tweedie (p=1.5) GLM of y on log model predictions: mu = exp(a + sum_k b_k log p_k)."""
    from glum import GeneralizedLinearRegressor, TweedieDistribution
    Z = np.log(np.clip(P, 1e-6, None))
    m = GeneralizedLinearRegressor(family=TweedieDistribution(1.5), link="log", alpha=1e-6, l1_ratio=0.0).fit(Z, y)
    return {"intercept": float(m.intercept_), "coef": np.asarray(m.coef_, float)}


def _stack_apply(w, P):
    return np.exp(w["intercept"] + np.log(np.clip(P, 1e-6, None)) @ w["coef"])


def _bin_edges(v, bins):
    """Interior cut points: quantiles if `bins` is an int, else the given edges."""
    if isinstance(bins, (list, tuple)):
        return np.asarray(sorted(bins), float)
    return np.unique(np.quantile(v[~np.isnan(v)], np.linspace(0, 1, int(bins) + 1)[1:-1]))


def _segment_labels(seg, line, P, T, ids, base):
    """Segment label (str) per OOF row and per test row, plus the definition. Train cut points are reused for test.

    seg['by'] == 'oof_decile': value = equal-weight mean of the parts' predictions per exposure.
    Otherwise `by` is a column of base['train_<line>'] / base['test_<line>'], matched on row_id.
    Segments with fewer than seg.get('min_rows', 50000) OOF rows are merged into 'rest'."""
    by, bins = seg["by"], seg.get("bins", 10)
    tr = base[f"train_{line}"].set_index("row_id").reindex(ids["oof"])
    te = base[f"test_{line}"].set_index("row_id").reindex(ids["test"])
    assert not tr.exposure.isna().any() and not te.exposure.isna().any(), "segment rows missing from base frames"
    if by == "oof_decile":
        v_tr = P.mean(axis=1) / np.clip(tr.exposure.values, 1e-9, None)
        v_te = T.mean(axis=1) / np.clip(te.exposure.values, 1e-9, None)
        numeric = True
    else:
        c_tr, c_te = tr[by], te[by]
        numeric = pd.api.types.is_numeric_dtype(c_tr) and not pd.api.types.is_bool_dtype(c_tr)
        v_tr, v_te = (c_tr.to_numpy(float), c_te.to_numpy(float)) if numeric else (c_tr, c_te)
    if numeric:
        edges = _bin_edges(v_tr, bins)

        def lab(v):
            out = np.array([f"q{i}" for i in np.searchsorted(edges, v, side="right")], dtype=object)
            out[np.isnan(v)] = "nan"
            return out
        a, b = lab(v_tr), lab(v_te)
        definition = {"by": by, "kind": "numeric", "edges": edges.tolist()}
    else:
        a = np.asarray(v_tr.astype(object).where(v_tr.notna(), "nan").astype(str), dtype=object)
        b = np.asarray(v_te.astype(object).where(v_te.notna(), "nan").astype(str), dtype=object)
        definition = {"by": by, "kind": "categorical"}
    names, counts = np.unique(a, return_counts=True)
    min_rows = seg.get("min_rows", 50000)
    small = set(names[counts < min_rows])
    known = set(names) - small
    a = np.array([x if x in known else "rest" for x in a], dtype=object)
    b = np.array([x if x in known else "rest" for x in b], dtype=object)
    definition["min_rows"] = min_rows
    definition["merged_into_rest"] = sorted(small)
    return a, b, definition


def run_blend(name: str, cfg: dict, folds: int | None = None) -> dict:
    """Per-line weights over cfg['parts'] (2 or more single models) on a simplex grid, chosen on OOF D².

    Also reports nested CV: weights are chosen on 4 folds and applied to the 5th (same GroupKFold by policy as the
    models), so the blend's own weight choice cannot flatter its score."""
    spec = get_spec()
    folds = spec.n_folds if folds is None else folds
    CACHE = spec.cache_path
    parts = [load(p) for p in cfg["parts"]]
    for part in parts:    # a blend part's OOF used full-sample weights, which would flatter the nested score
        # equal-weight seed averages choose nothing from the data, so they are allowed as parts
        assert part["cfg"]["model"] != "blend" or part["cfg"].get("average"), "blend parts must be single models"
    # Review 11 (pooled market, additive; both keys absent -> unchanged):
    #  weight_market "emerging": weights (and nested CV weights) fitted on that market's rows only;
    #  external_parts {exp: market}: results of another market (e.g. emerging-only part2), aligned by row_id; rows it
    #  does not predict (other market) take part 0's prediction. A blend may be an external part (its own weights were
    #  fitted on its OOF, a small optimism the nested score here does not remove).
    wm, ext = cfg.get("weight_market"), dict(cfg.get("external_parts", {}))
    assert not ext or wm, "external_parts need weight_market"
    assert not (wm and cfg.get("segments")), "weight_market with segments is not supported"
    names = list(cfg["parts"]) + list(ext)
    if ext:
        import pickle as _pk
        from market_spec import _resolve
        for en, emk in ext.items():
            with open(_resolve(emk).results_path / f"{en}.pkl", "rb") as f:
                parts.append(_pk.load(f))
    if cfg.get("average"):        # seed bagging: fixed equal weights, nothing chosen on the data
        grid = [np.full(len(parts), 1 / len(parts))]
    else:
        grid = _simplex(len(parts), cfg.get("step", 0.05 if len(parts) <= 3 else 0.1))
    for pname, part in zip(names, parts):    # a blend must never include an exploding part
        for line in spec.lines:
            pl = part["lines"][line]
            r = float(np.mean(pl["test_pred"]) / np.mean(pl["oof"]))
            if pl.get("unsafe") or not (RATIO_LO <= r <= RATIO_HI):
                raise ValueError(f"part {pname} unsafe on {line}: ratio {r:.2f}")
    res = {"name": name, "cfg": cfg, "lines": {}}
    seg_cfg = cfg.get("segments")
    base = build_base() if seg_cfg else None
    nested_y, nested_p = [], []
    nested_fm = []
    for line in spec.lines:
        L = [pt["lines"][line] for pt in parts]
        y = L[0]["y_eval"]
        ids = pd.read_parquet(CACHE / f"train_{line}.parquet", columns=["row_id", spec.group_col] + (["market"] if wm else []))
        nin = len(cfg["parts"])
        for l in L[:nin]:
            assert (l["row_id"] == ids.row_id.values).all(), "OOF rows misaligned"
            assert (l["test_row_id"] == L[0]["test_row_id"]).all(), "test rows misaligned"
        P = np.column_stack([l["oof"] for l in L[:nin]] + [
            pd.Series(np.asarray(l["oof"], float), index=l["row_id"]).reindex(L[0]["row_id"]).to_numpy() for l in L[nin:]])
        T = np.column_stack([l["test_pred"] for l in L[:nin]] + [
            pd.Series(np.asarray(l["test_pred"], float), index=l["test_row_id"]).reindex(L[0]["test_row_id"]).to_numpy()
            for l in L[nin:]])
        fm = (ids["market"].astype(str).values == wm) if wm else None
        if wm:
            assert fm.any(), f"no {wm} rows"
            assert np.isfinite(T).all(), "an external part misses test rows"
            assert np.isfinite(P[fm]).all(), "an external part misses weight-fit rows"
            P = np.where(np.isfinite(P), P, P[:, [0]])
        if cfg.get("stack"):       # meta-learner: Tweedie GLM on log predictions (geometric blend + intercept)
            fit_fn = lambda yy, PP: _stack_fit(yy, PP)
            apply_fn = _stack_apply
        else:
            fit_fn = lambda yy, PP: _best_weights(yy, PP, grid)[0]
            apply_fn = lambda w_, PP: PP @ w_
        fw = cfg.get("fixed_weights", {}).get(line)    # Review 11: {line: {part: weight}} -> not fitted on that line
        if fw is not None:
            wfix = np.array([float(fw.get(n, 0.0)) for n in names])
            assert abs(wfix.sum() - 1) < 1e-9 and not cfg.get("stack"), "fixed_weights must sum to 1 (no stacking)"
            fit_fn = lambda yy, PP, _w=wfix: _w
        w = fit_fn(y[fm], P[fm]) if wm else fit_fn(y, P)
        wcoef = lambda w_: w_["coef"] if cfg.get("stack") else w_
        if seg_cfg:
            sa, sb, sdef = _segment_labels(seg_cfg, line, P, T, {"oof": L[0]["row_id"], "test": L[0]["test_row_id"]}, base)
            labels = sorted(set(sa))
            print(f"  {line:9s} segments by {seg_cfg['by']}: " + ", ".join(f"{k}={int((sa == k).sum())}" for k in labels), flush=True)

            def seg_fit(idx):
                return {k: fit_fn(y[idx][sa[idx] == k], P[idx][sa[idx] == k]) for k in labels}

            def seg_apply(W_, PP, lab):
                out = np.zeros(len(PP))
                for k in labels:
                    m_ = lab == k
                    if m_.any():
                        out[m_] = apply_fn(W_[k], PP[m_])
                return out
            W = seg_fit(np.arange(len(y)))
            oof_pred, test_pred = seg_apply(W, P, sa), seg_apply(W, T, sb)
        else:
            oof_pred, test_pred = apply_fn(w, P), apply_fn(w, T)
        score = d2_tweedie(y, oof_pred)
        nest = np.zeros_like(y, dtype=float)
        for tr_idx, va_idx in GroupKFold(folds).split(y, groups=ids[spec.group_col].values):
            if seg_cfg:
                nest[va_idx] = seg_apply(seg_fit(tr_idx), P[va_idx], sa[va_idx])
            elif wm:
                tf = tr_idx[fm[tr_idx]]
                nest[va_idx] = apply_fn(fit_fn(y[tf], P[tf]), P[va_idx])
            else:
                nest[va_idx] = apply_fn(fit_fn(y[tr_idx], P[tr_idx]), P[va_idx])
        nested_y.append(y)
        nested_p.append(nest)
        wv = wcoef(w)
        res["lines"][line] = {**L[0], "oof": oof_pred, "test_pred": test_pred, "cv": score,
                              "cv_nested": d2_tweedie(y, nest), "holdout": np.nan,
                              "weight": float(wv[0]), "weights": dict(zip(names, map(float, wv)))}
        for k_ in ("holdout_idx", "holdout_pred"):     # part 0's per-row holdout must not pass as the blend's
            res["lines"][line].pop(k_, None)
        if wm:
            nested_fm.append((y[fm], nest[fm]))
            res["lines"][line].update({"oof_nested": nest, f"cv_{wm}": d2_tweedie(y[fm], oof_pred[fm]),
                                       f"cv_nested_{wm}": d2_tweedie(y[fm], nest[fm])})
            if all("holdout_pred" in l for l in L):   # holdout rows of the fit market predicted by every part
                hs = [pd.Series(np.asarray(l["holdout_pred"], float),
                                index=np.asarray(l["row_id"], dtype=object)[np.asarray(l["holdout_idx"])]) for l in L]
                rid0 = np.asarray(L[0]["row_id"], dtype=object)
                b0 = np.asarray(L[0]["holdout_idx"])
                b0 = b0[fm[b0]]
                b0 = b0[np.all([np.isin(rid0[b0], h.index) for h in hs], axis=0)]
                H = np.column_stack([h.reindex(rid0[b0]).to_numpy() for h in hs])
                res["lines"][line]["holdout_idx"] = b0
                res["lines"][line]["holdout_pred"] = apply_fn(w, H)
                res["lines"][line][f"holdout_{wm}"] = d2_tweedie(y[b0], res["lines"][line]["holdout_pred"]) if len(b0) else np.nan
            print(f"  {line:9s} {wm} rows: CV D2 {res['lines'][line][f'cv_{wm}']:.4f} | nested "
                  f"{res['lines'][line][f'cv_nested_{wm}']:.4f} | holdout {res['lines'][line].get(f'holdout_{wm}', np.nan):.4f}",
                  flush=True)
        elif not seg_cfg and all("holdout_pred" in l for l in L) and \
                all(np.array_equal(l["holdout_idx"], L[0]["holdout_idx"]) for l in L):
            # blend of the parts' holdout predictions with the full-OOF weights (diagnostic; weights saw 2024 OOF)
            res["lines"][line]["holdout_idx"] = L[0]["holdout_idx"]
            res["lines"][line]["holdout_pred"] = apply_fn(w, np.column_stack([l["holdout_pred"] for l in L]))
        if seg_cfg:
            res["lines"][line]["weights"] = {k: dict(zip(cfg["parts"], map(float, wcoef(W[k])))) for k in labels}
            res["lines"][line]["segments"] = {**sdef, "sizes": {k: int((sa == k).sum()) for k in labels}}
        wd = res["lines"][line]["weights"]
        wtxt = "; ".join(f"{k}: " + ",".join(f"{v:.2f}" for v in d.values()) for k, d in wd.items()) if seg_cfg else \
            ", ".join(f"{k}={v:.2f}" for k, v in wd.items())
        print(f"  {line:9s} weights {wtxt} | CV D2 {score:.4f} | nested {res['lines'][line]['cv_nested']:.4f}", flush=True)
    _summarise(res)
    res["cv_nested"] = d2_tweedie(np.concatenate(nested_y), np.concatenate(nested_p))
    print(f"  OVERALL   nested CV D2 {res['cv_nested']:.4f}")
    if wm:
        res[f"cv_nested_{wm}"] = d2_tweedie(np.concatenate([a for a, _ in nested_fm]), np.concatenate([b for _, b in nested_fm]))
        print(f"  OVERALL   {wm} rows nested CV D2 {res[f'cv_nested_{wm}']:.4f}")
    _save(res)
    return res


def _summarise(res):
    L = res["lines"].values()
    res["cv"] = d2_tweedie(np.concatenate([l["y_eval"] for l in L]), np.concatenate([l["oof"] for l in L]))
    ho = [l["holdout"] for l in L]
    res["holdout"] = float(np.mean(ho)) if not np.isnan(ho).any() else np.nan
    print(f"  OVERALL   CV D2 {res['cv']:.4f} | mean 2023 holdout {res['holdout']:.4f}")


def _save(res):
    RESULTS = get_spec().results_path
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / f"{res['name']}.pkl", "wb") as f:
        pickle.dump(res, f)


def load(name: str) -> dict:
    with open(get_spec().results_path / f"{name}.pkl", "rb") as f:
        return pickle.load(f)


def test_predictions(res: dict) -> pd.DataFrame:
    if res.get("unsafe"):
        raise ValueError("unsafe result; refusing to write a submission")
    return pd.concat([pd.DataFrame({"row_id": l["test_row_id"], "prediction": l["test_pred"]}) for l in res["lines"].values()],
                     ignore_index=True)
