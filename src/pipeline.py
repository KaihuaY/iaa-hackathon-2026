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
from target import prepare_target

RESULTS = Path(__file__).resolve().parents[1] / "experiments" / "results"


def _line_data(base, line, cfg):
    tr, te = base[f"train_{line}"], base[f"test_{line}"]
    tgt, info = prepare_target(tr, te, base["claims"], line, cfg.get("target", {}))
    fcfg = dict(cfg.get("features", {}))
    fcfg["drop"] = list(fcfg.get("drop", [])) + list(fcfg.get("drop_line", {}).get(line, []))
    X, Xt, cats = model_matrix(tr, te, fcfg)
    return tr, te, tgt, info, X, Xt


def _fit_predict(cfg, X, y, w, e, Xv, ev, yv=None, wv=None, n_iter=None):
    m = make(cfg["model"], cfg.get("params"))
    if cfg["model"] in ("lgbm", "glmboost"):
        m.fit(X, y, w, e, Xv if n_iter is None else None, yv, wv, ev, n_iter=n_iter)
    else:
        m.fit(X, y, w, e)
    return m, m.predict(Xv, ev)


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
        load = info["loading"] if cfg.get("target", {}).get("cap_q") else 1.0
        oof, iters = np.zeros(len(tr)), []
        for k, (a, b) in enumerate(GroupKFold(folds).split(X, groups=tr.policy_id)):
            m, p = _fit_predict(cfg, X.iloc[a], y[a], w[a], e[a], X.iloc[b], e[b], y[b], w[b])
            oof[b] = p * load
            iters.append(getattr(m, "best_iter", None))
        cv = d2_tweedie(tgt.y_eval, oof)
        calib = float(oof.sum() / tgt.y_eval.sum())          # predicted / actual total cost (guardrail: ~1)
        ho = np.nan
        if holdout:
            yr = tr.period_start.dt.year.values
            a, b = np.where(yr <= 2022)[0], np.where(yr == 2023)[0]
            n_ho = int(np.mean([i for i in iters if i])) if iters[0] else None
            _, p = _fit_predict(cfg, X.iloc[a], y[a], w[a], e[a], X.iloc[b], e[b], n_iter=n_ho)
            ho = d2_tweedie(tgt.y_eval.values[b], p * load)
        n_full = int(np.mean(iters) * 1.1) if iters[0] else None
        m, pt = _fit_predict(cfg, X, y, w, e, Xt, te.exposure.values, n_iter=n_full)
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


def _best_weight(y, pa, pb, grid=np.linspace(0, 1, 21)):
    scores = [d2_tweedie(y, g * pa + (1 - g) * pb) for g in grid]
    return float(grid[int(np.argmax(scores))]), max(scores)


def run_blend(name: str, cfg: dict, folds: int = 5) -> dict:
    """Per-line weight w on cfg['parts'][0] vs (1-w) on parts[1], chosen on OOF D².

    Also reports nested CV: the weight is chosen on 4 folds and applied to the 5th (same GroupKFold by policy as the
    models), so the blend's own weight choice cannot flatter its score."""
    a, b = (load(p) for p in cfg["parts"])
    res = {"name": name, "cfg": cfg, "lines": {}}
    nested_y, nested_p = [], []
    for line in LINES:
        la, lb = a["lines"][line], b["lines"][line]
        y, pa, pb = la["y_eval"], la["oof"], lb["oof"]
        g, score = _best_weight(y, pa, pb)
        groups = pd.read_parquet(CACHE / f"train_{line}.parquet", columns=["policy_id"]).policy_id.values
        assert len(groups) == len(y)
        nest = np.zeros_like(y, dtype=float)
        for tr_idx, va_idx in GroupKFold(folds).split(y, groups=groups):
            gk, _ = _best_weight(y[tr_idx], pa[tr_idx], pb[tr_idx])
            nest[va_idx] = gk * pa[va_idx] + (1 - gk) * pb[va_idx]
        nested_y.append(y)
        nested_p.append(nest)
        res["lines"][line] = {**la, "oof": g * pa + (1 - g) * pb,
                              "test_pred": g * la["test_pred"] + (1 - g) * lb["test_pred"],
                              "cv": score, "cv_nested": d2_tweedie(y, nest), "holdout": np.nan, "weight": g}
        print(f"  {line:9s} weight on {cfg['parts'][0]} = {g:.2f} | CV D2 {score:.4f} | nested {res['lines'][line]['cv_nested']:.4f}")
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
