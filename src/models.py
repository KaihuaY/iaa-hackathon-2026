"""Model wrappers with a common fit / predict interface (predictions are expected cost per row)."""
import lightgbm as lgb
import numpy as np
import pandas as pd

# Heavy regularisation: with ~86-96 % zero rows, small leaves chase chance claim-free pockets and the model
# drifts low (see experiment log, 17 Oct diagnostic). Large leaves + slow learning + rebalancing fix it.
LGBM_DEFAULT = dict(objective="tweedie", tweedie_variance_power=1.5, learning_rate=0.03, num_leaves=15,
                    min_data_in_leaf=20000, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
                    lambda_l2=50.0, max_cat_to_onehot=8, cat_smooth=50, verbose=-1, seed=7)


class ConstRate:
    """Exposure x average pure premium (the 'average cost' baseline)."""

    def fit(self, X, y, w, expo, **_):
        self.rate = np.sum(w * y) / np.sum(w * expo)
        return self

    def predict(self, X, expo):
        return self.rate * np.asarray(expo)


class LGBMTweedie:
    """Tweedie GBM. `margin` (log of a base prediction, e.g. a GLM) replaces the flat start when given.
    Predictions are rebalanced so the in-sample predicted total equals the actual total."""

    def __init__(self, params=None, rounds=3000, early_stop=200):
        self.params = {**LGBM_DEFAULT, **(params or {})}
        self.rounds, self.early_stop = rounds, early_stop
        self.best_iter = None

    def _start(self, expo, margin):
        # An init_score disables boost_from_average, so supply the start level ourselves.
        return margin if margin is not None else self.base + np.log(expo)

    def fit(self, X, y, w, expo, X_val=None, y_val=None, w_val=None, expo_val=None, n_iter=None,
            margin=None, margin_val=None):
        self.base = float(np.log(np.sum(w * y) / np.sum(w * expo)))
        dtr = lgb.Dataset(X, y, weight=w, init_score=self._start(expo, margin), categorical_feature="auto")
        if X_val is not None and n_iter is None:
            dva = lgb.Dataset(X_val, y_val, weight=w_val, init_score=self._start(expo_val, margin_val), reference=dtr)
            self.model = lgb.train(self.params, dtr, self.rounds, valid_sets=[dva],
                                   callbacks=[lgb.early_stopping(self.early_stop, verbose=False)])
            self.best_iter = self.model.best_iteration
        else:
            self.model = lgb.train(self.params, dtr, n_iter or self.rounds)
            self.best_iter = n_iter or self.rounds
        self.scale = 1.0
        self.scale = float(np.sum(w * y) / np.sum(w * self.predict(X, expo, margin)))
        return self

    def predict(self, X, expo, margin=None):
        return self.scale * np.exp(self.model.predict(X, num_iteration=self.best_iter, raw_score=True)
                                   + self._start(expo, margin))

    def importance(self):
        return pd.Series(self.model.feature_importance("gain"), index=self.model.feature_name()).sort_values(ascending=False)


class GLMTweedie:
    """Tweedie GLM (glum) on banded numerics and grouped categoricals, log(exposure) offset."""

    def __init__(self, alpha=1e-4, n_bins=10, top=30):
        self.alpha, self.n_bins, self.top = alpha, n_bins, top

    def _design(self, X, fit):
        out = pd.DataFrame(index=X.index)
        if fit:
            self.spec = {}
        for c in X.columns:
            s = X[c]
            if isinstance(s.dtype, pd.CategoricalDtype):
                if fit:
                    keep = s.value_counts().index[: self.top].astype(str).tolist()
                    self.spec[c] = ("cat", keep)
                keep = self.spec[c][1]
                v = s.astype(str).where(s.astype(str).isin(keep), "_other").where(s.notna(), "_missing")
                out[c] = pd.Categorical(v, categories=keep + ["_other", "_missing"])
            else:
                if fit:
                    edges = np.unique(np.nanquantile(s.astype(float), np.linspace(0, 1, self.n_bins + 1)))
                    self.spec[c] = ("num", edges)
                edges = self.spec[c][1]
                if len(edges) < 3:
                    continue
                b = np.clip(np.searchsorted(edges[1:-1], s.astype(float), side="right"), 0, len(edges) - 2)
                lab = np.where(s.isna(), "_missing", b.astype(str))
                out[c] = pd.Categorical(lab, categories=[str(i) for i in range(len(edges) - 1)] + ["_missing"])
        return out

    def fit(self, X, y, w, expo, **_):
        from glum import GeneralizedLinearRegressor, TweedieDistribution
        Z = self._design(X, fit=True)
        self.cols = Z.columns
        self.model = GeneralizedLinearRegressor(family=TweedieDistribution(1.5), link="log", alpha=self.alpha,
                                                l1_ratio=0.0, drop_first=True, max_iter=200)
        self.model.fit(Z, y, sample_weight=w, offset=np.log(expo))
        return self

    def predict(self, X, expo):
        Z = self._design(X, fit=False)[self.cols]
        return self.model.predict(Z, offset=np.log(expo))


class GLMBoost:
    """GLM first, then a regularised GBM boosting from the GLM's log-prediction (learns what the GLM misses)."""

    def __init__(self, params=None):
        params = dict(params or {})
        self.glm = GLMTweedie(**params.pop("glm", {}))
        self.gbm = LGBMTweedie(params)

    def fit(self, X, y, w, expo, X_val=None, y_val=None, w_val=None, expo_val=None, n_iter=None):
        self.glm.fit(X, y, w, expo)
        m = np.log(self.glm.predict(X, expo))
        mv = np.log(self.glm.predict(X_val, expo_val)) if X_val is not None and n_iter is None else None
        self.gbm.fit(X, y, w, expo, X_val, y_val, w_val, expo_val, n_iter=n_iter, margin=m, margin_val=mv)
        self.best_iter = self.gbm.best_iter
        return self

    def predict(self, X, expo):
        return self.gbm.predict(X, expo, margin=np.log(self.glm.predict(X, expo)))

    def importance(self):
        return self.gbm.importance()


def make(kind: str, params: dict | None = None):
    return {"const": lambda: ConstRate(), "lgbm": lambda: LGBMTweedie(params),
            "glm": lambda: GLMTweedie(**(params or {})), "glmboost": lambda: GLMBoost(params)}[kind]()
