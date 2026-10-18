"""Covariate-shift importance weights: clipped density ratio p_target(x)/p_source(x) from a source-vs-target classifier."""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

SHIFT_PARAMS = dict(objective="binary", num_leaves=15, min_data_in_leaf=2000, learning_rate=0.05,
                    feature_fraction=0.7, lambda_l2=10.0, verbose=-1, num_threads=4)
N_ROUNDS = 200


def density_ratio_weights(X_src, X_tgt, cols=None, clip=(0.2, 5.0), n_folds=5, seed=7, params=None, verbose=True):
    """One weight per X_src row (mean exactly 1). Returns (weights, diagnostics dict: auc, ess_ratio, top_features)."""
    if cols is None:
        cols = [c for c in X_src.columns if c in X_tgt.columns]
    cols = [c for c in cols if X_src[c].nunique(dropna=False) > 1 or X_tgt[c].nunique(dropna=False) > 1]
    cols = [c for c in cols if pd.concat([X_src[c], X_tgt[c]]).nunique(dropna=False) > 1]
    n_s, n_t = len(X_src), len(X_tgt)
    Z = pd.concat([X_src[cols], X_tgt[cols]], axis=0, ignore_index=True)
    for c in cols:                                  # categories may differ between frames after concat
        if isinstance(X_src[c].dtype, pd.CategoricalDtype) or isinstance(X_tgt[c].dtype, pd.CategoricalDtype):
            Z[c] = Z[c].astype("category")
    lab = np.r_[np.zeros(n_s), np.ones(n_t)]
    p = {**SHIFT_PARAMS, "seed": seed, **(params or {})}
    oof, gain = np.zeros(len(Z)), np.zeros(len(cols))
    for a, b in StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(Z, lab):
        m = lgb.train(p, lgb.Dataset(Z.iloc[a], lab[a]), num_boost_round=N_ROUNDS)
        oof[b] = m.predict(Z.iloc[b])
        gain += m.feature_importance("gain")
    pr = np.clip(oof[:n_s], 1e-6, 1 - 1e-6)
    w = np.clip(pr / (1 - pr) * n_s / n_t, clip[0], clip[1])
    w = w / w.mean()
    auc = float(roc_auc_score(lab, oof))
    ess = float(w.sum() ** 2 / (w ** 2).sum() / n_s)
    top = dict(pd.Series(gain, index=cols).sort_values(ascending=False).head(10))
    diag = {"auc": auc, "ess_ratio": ess, "top_features": {k: float(v) for k, v in top.items()}}
    if verbose:
        print(f"shift weights: AUC {auc:.3f} ESS {ess:.3f}", flush=True)
    return w, diag
