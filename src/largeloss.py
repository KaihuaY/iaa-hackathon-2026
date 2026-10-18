"""Model-based large-loss loading: loading(x) = 1 + E[excess | x] / E[capped | x], shrunk toward the flat factor."""
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

BASE = dict(objective="tweedie", tweedie_variance_power=1.5, learning_rate=0.03, num_leaves=7,
            min_data_in_leaf=5000, lambda_l2=100.0, feature_fraction=0.7, bagging_freq=0,
            verbose=-1, num_threads=4, deterministic=True, force_row_wise=True)


def _lgb_params(params, seed, n_feat):
    p = {**BASE, "seed": seed}
    for k in ("num_leaves", "min_data_in_leaf", "learning_rate", "lambda_l2", "feature_fraction", "tweedie_variance_power"):
        if params.get(k) is not None:
            p[k] = params[k]
    mono = params.get("monotone")
    if mono is not None:
        mono = list(mono)
        if len(mono) == n_feat:
            p["monotone_constraints"] = mono
    return p


def _num(X):
    if hasattr(X, "columns"):                              # category/object columns -> integer codes (NaN for missing)
        X = X.copy()
        for c in X.columns:
            if not np.issubdtype(X[c].dtype, np.number) and X[c].dtype != bool:
                X[c] = X[c].astype("category").cat.codes.astype(float).replace(-1, np.nan)
        return X.to_numpy(dtype=float)
    return np.asarray(X, dtype=float)


def _fit(X, y, expo, w, p, rounds):
    d = lgb.Dataset(X, y, weight=w, init_score=np.log(np.maximum(expo, 1e-9)))
    return lgb.train(p, d, rounds)


def _pred(m, X, expo):
    return m.predict(X) * np.maximum(expo, 1e-9)           # raw score excludes the init_score offset


def _oof_and_test(X, Xt, y, expo, expo_t, w, groups, p, rounds, folds):
    oof = np.zeros(len(y))
    for a, b in GroupKFold(folds).split(X, groups=groups):
        m = _fit(X[a], y[a], expo[a], w[a], p, rounds)
        oof[b] = _pred(m, X[b], expo[b])
    full = _fit(X, y, expo, w, p, rounds)
    return oof, _pred(full, Xt, expo_t)


def model_loading(X, Xt, y_total, y_capped, expo, w, groups, flat, params=None, seed=7, expo_test=None):
    """Returns (loading_rows, loading_test, diag). X, Xt: numeric arrays/DataFrames (same columns).
    params: shrink (0.5), rounds (300), folds (5), clip (2.0), monotone (list), capped_pred (OOF capped prediction for
    train rows, optionally a tuple (train_oof, test_pred)), plus LightGBM overrides.
    Test rows use unit exposure unless expo_test is given (exposure cancels in the ratio)."""
    params = dict(params or {})
    X, Xt = _num(X), _num(Xt)
    y_total = np.asarray(y_total, dtype=float)
    y_capped = np.asarray(y_capped, dtype=float)
    expo = np.asarray(expo, dtype=float)
    w = np.ones(len(X)) if w is None else np.asarray(w, dtype=float)
    groups = np.asarray(groups)
    et = np.ones(len(Xt)) if expo_test is None else np.asarray(expo_test, dtype=float)
    rounds, folds = int(params.get("rounds", 300)), int(params.get("folds", 5))
    k, cl = float(params.get("shrink", 0.5)), float(params.get("clip", 2.0))
    p = _lgb_params(params, seed, X.shape[1])
    excess = np.clip(y_total - y_capped, 0.0, None)

    ex_oof, ex_te = _oof_and_test(X, Xt, excess, expo, et, w, groups, p, rounds, folds)
    cp = params.get("capped_pred")
    if cp is not None:
        cp_oof, cp_te = (np.asarray(cp[0], float), np.asarray(cp[1], float)) if isinstance(cp, (tuple, list)) else (np.asarray(cp, float), None)
    else:
        cp_oof, cp_te = None, None
    if cp_oof is None or cp_te is None:                    # fit the capped model (also when only train OOF was given)
        o, t = _oof_and_test(X, Xt, y_capped, expo, et, w, groups, p, rounds, folds)
        cp_oof = o if cp_oof is None else cp_oof
        cp_te = t if cp_te is None else cp_te
    # test-side capped prediction is on unit exposure; ratio is exposure free
    eps = 1e-12
    raw_tr = 1.0 + ex_oof / np.maximum(cp_oof, eps)
    raw_te = 1.0 + ex_te / np.maximum(cp_te, eps)
    wm = w * expo
    mean_raw = float(np.sum(wm * raw_tr) / np.sum(wm))

    def shrink(r):
        return flat * (r / mean_raw) ** k

    lo, hi = flat / cl, flat * cl
    ltr, lte = shrink(raw_tr), shrink(raw_te)
    clipped = float(np.mean((ltr < lo) | (ltr > hi)))
    ltr, lte = np.clip(ltr, lo, hi), np.clip(lte, lo, hi)
    s = flat / (np.sum(wm * ltr) / np.sum(wm))               # exposure-weighted train mean == flat exactly
    ltr, lte = ltr * s, lte * s
    ltr_c = ltr
    corr = float(np.corrcoef(ex_oof, excess)[0, 1]) if excess.std() > 0 and ex_oof.std() > 0 else float("nan")
    qs = np.quantile(ltr_c, [0.01, 0.25, 0.5, 0.75, 0.99])
    diag = {"oof_corr_excess": corr, "loading_q": dict(zip(("q01", "q25", "q50", "q75", "q99"), map(float, qs))),
            "clipped_share": clipped, "mean_loading": float(np.sum(wm * ltr) / np.sum(wm)), "flat": float(flat)}
    print(f"  model_loading: flat {flat:.4f} mean {diag['mean_loading']:.4f} | q01/25/50/75/99 "
          + "/".join(f"{q:.3f}" for q in qs) + f" | OOF corr(excess) {corr:.3f} | clipped {clipped:.1%} | k={k}", flush=True)
    return ltr, lte, diag
