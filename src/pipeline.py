"""Run one experiment: CV (grouped by policy), 2023 time-holdout diagnostic, full fit, test predictions."""
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from data import LINES
from features import CACHE, build_base, model_matrix
from metrics import d2_tweedie
from models import make
from target import peril_matrix, prepare_target

RESULTS = Path(__file__).resolve().parents[1] / "experiments" / "results"


def _line_data(base, line, cfg):
    tr, te = base[f"train_{line}"], base[f"test_{line}"]
    tgt, info = prepare_target(tr, te, base["claims"], line, cfg.get("target", {}))
    fcfg = dict(cfg.get("features", {}))
    fcfg["drop"] = list(fcfg.get("drop", [])) + list(fcfg.get("drop_line", {}).get(line, []))
    X, Xt, cats = model_matrix(tr, te, fcfg)
    return tr, te, tgt, info, X, Xt


def _aux(cfg, base, line, tr, tgt, info):
    """Extra per-row targets: claim counts (freq x sev) or the per-peril cost matrix (per-peril models)."""
    if cfg["model"] == "perperil":
        M, perils = peril_matrix(tr, base["claims"], line, tgt, info)
        print(f"  {line}: perils modelled {perils}", flush=True)
        return M
    return tr.claim_count.values


def _fit_predict(cfg, X, y, w, e, Xv, ev, yv=None, wv=None, n_iter=None, c=None, cv_=None):
    m = make(cfg["model"], cfg.get("params"))
    if cfg["model"] in ("freqsev", "perperil"):
        m.fit(X, y, w, e, Xv if n_iter is None else None, yv, wv, ev, n_iter=n_iter, counts=c, counts_val=cv_)
    elif cfg["model"] in ("lgbm", "glmboost", "catboost", "xgb"):
        m.fit(X, y, w, e, Xv if n_iter is None else None, yv, wv, ev, n_iter=n_iter)
    else:
        m.fit(X, y, w, e)
    return m, m.predict(Xv, ev)


def quick(name: str, cfg: dict, seed: int = 11) -> dict:
    """Fast screen: one 80/20 split grouped by policy (same split every run), one fit per line. No test predictions.
    Use to triage ideas; anything promising then gets the full 5-fold CV."""
    from sklearn.model_selection import GroupShuffleSplit
    base = build_base()
    ys, ps, out = [], [], {}
    for line in LINES:
        tr, te, tgt, info, X, Xt = _line_data(base, line, cfg)
        y, w, e = tgt.y_train.values, tgt.w.values, tr.exposure.values
        c = _aux(cfg, base, line, tr, tgt, info)
        load = info["loading"] if cfg.get("target", {}).get("cap_q") else 1.0
        a, b = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(X, groups=tr.policy_id))
        _, p = _fit_predict(cfg, X.iloc[a], y[a], w[a], e[a], X.iloc[b], e[b], y[b], w[b], c=c[a], cv_=c[b])
        out[line] = d2_tweedie(tgt.y_eval.values[b], p * load)
        ys.append(tgt.y_eval.values[b]); ps.append(p * load)
        print(f"  quick {line:9s} D2 {out[line]:.4f}", flush=True)
    out["overall"] = d2_tweedie(np.concatenate(ys), np.concatenate(ps))
    print(f"  quick OVERALL D2 {out['overall']:.4f}", flush=True)
    return out


def run(name: str, cfg: dict, folds: int = 5, holdout: bool = True, verbose: bool = True) -> dict:
    """Returns dict with per-line OOF/test predictions and scores; caches to experiments/results/<name>.pkl."""
    if cfg["model"] == "blend":
        return run_blend(name, cfg)
    t0 = time.time()
    base = build_base()
    res = {"name": name, "cfg": cfg, "lines": {}}
    for line in LINES:
        tr, te, tgt, info, X, Xt = _line_data(base, line, cfg)
        y, w, e = tgt.y_train.values, tgt.w.values, tr.exposure.values
        c = _aux(cfg, base, line, tr, tgt, info)
        load = info["loading"] if cfg.get("target", {}).get("cap_q") else 1.0
        oof, iters = np.zeros(len(tr)), []
        for k, (a, b) in enumerate(GroupKFold(folds).split(X, groups=tr.policy_id)):
            m, p = _fit_predict(cfg, X.iloc[a], y[a], w[a], e[a], X.iloc[b], e[b], y[b], w[b], c=c[a], cv_=c[b])
            oof[b] = p * load
            iters.append(getattr(m, "best_iter", None))
        cv = d2_tweedie(tgt.y_eval, oof)
        calib = float(oof.sum() / tgt.y_eval.sum())          # predicted / actual total cost (guardrail: ~1)
        ho = np.nan
        if holdout:
            yr = tr.period_start.dt.year.values
            a, b = np.where(yr <= 2022)[0], np.where(yr == 2023)[0]
            n_ho = int(np.mean([i for i in iters if i])) if iters[0] else None
            _, p = _fit_predict(cfg, X.iloc[a], y[a], w[a], e[a], X.iloc[b], e[b], n_iter=n_ho, c=c[a])
            ho = d2_tweedie(tgt.y_eval.values[b], p * load)
        n_full = int(np.mean(iters) * 1.1) if iters[0] else None
        m, pt = _fit_predict(cfg, X, y, w, e, Xt, te.exposure.values, n_iter=n_full, c=c)
        imp = m.importance().head(25).to_dict() if hasattr(m, "importance") else {}
        res["lines"][line] = {"row_id": tr.row_id.values, "y_eval": tgt.y_eval.values, "oof": oof,
                              "test_row_id": te.row_id.values, "test_pred": pt * load, "cv": cv, "holdout": ho, "calibration": calib,
                              "info": info, "iters": iters, "importance": imp}
        if verbose:
            print(f"  {line:9s} CV D2 {cv:.4f} | 2023 holdout {ho:.4f} | calib {calib:.3f} | iters {iters} | loading {load:.3f} "
                  f"| test mean {np.mean(pt * load):,.1f}", flush=True)
    _summarise(res)
    res["minutes"] = (time.time() - t0) / 60
    _save(res)
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


def run_blend(name: str, cfg: dict, folds: int = 5) -> dict:
    """Per-line weights over cfg['parts'] (2 or more single models) on a simplex grid, chosen on OOF D².

    Also reports nested CV: weights are chosen on 4 folds and applied to the 5th (same GroupKFold by policy as the
    models), so the blend's own weight choice cannot flatter its score."""
    parts = [load(p) for p in cfg["parts"]]
    for part in parts:    # a blend part's OOF used full-sample weights, which would flatter the nested score
        # equal-weight seed averages choose nothing from the data, so they are allowed as parts
        assert part["cfg"]["model"] != "blend" or part["cfg"].get("average"), "blend parts must be single models"
    if cfg.get("average"):        # seed bagging: fixed equal weights, nothing chosen on the data
        grid = [np.full(len(parts), 1 / len(parts))]
    else:
        grid = _simplex(len(parts), cfg.get("step", 0.05 if len(parts) <= 3 else 0.1))
    res = {"name": name, "cfg": cfg, "lines": {}}
    nested_y, nested_p = [], []
    for line in LINES:
        L = [pt["lines"][line] for pt in parts]
        y = L[0]["y_eval"]
        ids = pd.read_parquet(CACHE / f"train_{line}.parquet", columns=["row_id", "policy_id"])
        for l in L:
            assert (l["row_id"] == ids.row_id.values).all(), "OOF rows misaligned"
            assert (l["test_row_id"] == L[0]["test_row_id"]).all(), "test rows misaligned"
        P = np.column_stack([l["oof"] for l in L])
        T = np.column_stack([l["test_pred"] for l in L])
        if cfg.get("stack"):       # meta-learner: Tweedie GLM on log predictions (geometric blend + intercept)
            fit_fn = lambda yy, PP: _stack_fit(yy, PP)
            apply_fn = _stack_apply
        else:
            fit_fn = lambda yy, PP: _best_weights(yy, PP, grid)[0]
            apply_fn = lambda w_, PP: PP @ w_
        w = fit_fn(y, P)
        score = d2_tweedie(y, apply_fn(w, P))
        nest = np.zeros_like(y, dtype=float)
        for tr_idx, va_idx in GroupKFold(folds).split(y, groups=ids.policy_id.values):
            nest[va_idx] = apply_fn(fit_fn(y[tr_idx], P[tr_idx]), P[va_idx])
        nested_y.append(y)
        nested_p.append(nest)
        wv = w["coef"] if cfg.get("stack") else w
        res["lines"][line] = {**L[0], "oof": apply_fn(w, P), "test_pred": apply_fn(w, T), "cv": score,
                              "cv_nested": d2_tweedie(y, nest), "holdout": np.nan,
                              "weight": float(wv[0]), "weights": dict(zip(cfg["parts"], map(float, wv)))}
        wtxt = ", ".join(f"{k}={v:.2f}" for k, v in res["lines"][line]["weights"].items())
        print(f"  {line:9s} weights {wtxt} | CV D2 {score:.4f} | nested {res['lines'][line]['cv_nested']:.4f}", flush=True)
    _summarise(res)
    res["cv_nested"] = d2_tweedie(np.concatenate(nested_y), np.concatenate(nested_p))
    print(f"  OVERALL   nested CV D2 {res['cv_nested']:.4f}")
    _save(res)
    return res


def _summarise(res):
    L = res["lines"].values()
    res["cv"] = d2_tweedie(np.concatenate([l["y_eval"] for l in L]), np.concatenate([l["oof"] for l in L]))
    ho = [l["holdout"] for l in L]
    res["holdout"] = float(np.mean(ho)) if not np.isnan(ho).any() else np.nan
    print(f"  OVERALL   CV D2 {res['cv']:.4f} | mean 2023 holdout {res['holdout']:.4f}")


def _save(res):
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / f"{res['name']}.pkl", "wb") as f:
        pickle.dump(res, f)


def load(name: str) -> dict:
    with open(RESULTS / f"{name}.pkl", "rb") as f:
        return pickle.load(f)


def test_predictions(res: dict) -> pd.DataFrame:
    return pd.concat([pd.DataFrame({"row_id": l["test_row_id"], "prediction": l["test_pred"]}) for l in res["lines"].values()],
                     ignore_index=True)
