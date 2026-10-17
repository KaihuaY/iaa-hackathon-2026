"""Advanced models (branch advanced-ml): CatBoost and XGBoost Tweedie with the common fit/predict interface.

Like LGBMTweedie: log(exposure) + average log-rate as the starting margin, early stopping on the validation fold,
and rebalancing so the in-sample predicted total equals the actual total.
"""
import numpy as np
import pandas as pd


def _cat_cols(X: pd.DataFrame) -> list:
    return [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]


class CatBoostTweedie:
    """CatBoost with ordered target statistics for categoricals (built to resist high-cardinality overfitting)."""

    DEFAULT = dict(loss_function="Tweedie:variance_power=1.5", learning_rate=0.06, depth=6, l2_leaf_reg=10,
                   min_data_in_leaf=500, one_hot_max_size=8, random_seed=7, verbose=0, thread_count=-1,
                   allow_writing_files=False)

    def __init__(self, params=None, rounds=3000, early_stop=150):
        self.params = {**self.DEFAULT, **(params or {})}
        self.rounds, self.early_stop = rounds, early_stop

    @staticmethod
    def _prep(X):
        X = X.copy()
        for c in _cat_cols(X):
            X[c] = X[c].astype(str).where(X[c].notna(), "_nan")
        return X

    def _pool(self, X, y, w, margin):
        from catboost import Pool
        return Pool(self._prep(X), y, weight=w, baseline=margin, cat_features=_cat_cols(X))

    def fit(self, X, y, w, expo, X_val=None, y_val=None, w_val=None, expo_val=None, n_iter=None):
        from catboost import CatBoostRegressor
        self.base = float(np.log(np.sum(w * y) / np.sum(w * expo)))
        tr = self._pool(X, y, w, self.base + np.log(expo))
        if X_val is not None and n_iter is None:
            va = self._pool(X_val, y_val, w_val, self.base + np.log(expo_val))
            self.model = CatBoostRegressor(iterations=self.rounds, od_type="Iter", od_wait=self.early_stop, **self.params)
            self.model.fit(tr, eval_set=va, use_best_model=True)
            self.best_iter = int(self.model.get_best_iteration()) + 1
        else:
            self.model = CatBoostRegressor(iterations=n_iter or 1000, **self.params)
            self.model.fit(tr)
            self.best_iter = n_iter or 1000
        self.scale = 1.0
        self.scale = float(np.sum(w * y) / np.sum(w * self.predict(X, expo)))
        return self

    def predict(self, X, expo):
        from catboost import Pool
        raw = self.model.predict(Pool(self._prep(X), cat_features=_cat_cols(X)), prediction_type="RawFormulaVal")
        return self.scale * np.exp(raw + self.base + np.log(expo))

    def importance(self):
        return pd.Series(self.model.get_feature_importance(), index=self.model.feature_names_).sort_values(ascending=False)


class XGBTweedie:
    """XGBoost hist trees with native categoricals, Tweedie objective."""

    DEFAULT = dict(objective="reg:tweedie", tweedie_variance_power=1.5, eta=0.05, max_depth=5, min_child_weight=200,
                   subsample=0.8, colsample_bytree=0.5, reg_lambda=50, tree_method="hist", max_cat_to_onehot=8,
                   max_bin=256, seed=7, nthread=-1)

    def __init__(self, params=None, rounds=3000, early_stop=150):
        self.params = {**self.DEFAULT, **(params or {})}
        self.rounds, self.early_stop = rounds, early_stop

    def _dm(self, X, y, w, margin):
        import xgboost as xgb
        return xgb.DMatrix(X, label=y, weight=w, base_margin=margin, enable_categorical=True)

    def fit(self, X, y, w, expo, X_val=None, y_val=None, w_val=None, expo_val=None, n_iter=None):
        import xgboost as xgb
        self.base = float(np.log(np.sum(w * y) / np.sum(w * expo)))
        dtr = self._dm(X, y, w, self.base + np.log(expo))
        if X_val is not None and n_iter is None:
            dva = self._dm(X_val, y_val, w_val, self.base + np.log(expo_val))
            self.model = xgb.train(self.params, dtr, self.rounds, evals=[(dva, "val")],
                                   early_stopping_rounds=self.early_stop, verbose_eval=False)
            self.best_iter = int(self.model.best_iteration) + 1
        else:
            self.model = xgb.train(self.params, dtr, n_iter or 1000)
            self.best_iter = n_iter or 1000
        self.scale = 1.0
        self.scale = float(np.sum(w * y) / np.sum(w * self.predict(X, expo)))
        return self

    def predict(self, X, expo):
        import xgboost as xgb
        d = xgb.DMatrix(X, base_margin=self.base + np.log(expo), enable_categorical=True)
        mu = self.model.predict(d, iteration_range=(0, self.best_iter))   # includes base_margin, on response scale
        return self.scale * mu

    def importance(self):
        return pd.Series(self.model.get_score(importance_type="gain")).sort_values(ascending=False)
