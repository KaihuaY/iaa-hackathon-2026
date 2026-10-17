"""Forward level backtest: can a trend method project costs one year ahead?

CV cannot judge the trend (its label already uses the premium trend), but a forward test can:
train on years < T with labels trended to year T, predict year-T rows, compare with year T's developed actual cost.
"""
import numpy as np
import pandas as pd

from data import LINES
from features import build_base, model_matrix
from metrics import d2_tweedie
from models import make
from target import prepare_target


def level_backtest(target_cfg: dict, methods=("premium", "claims", "none"), years=(2023, 2024),
                   model: str = "lgbm", n_iter: int = 300, features: dict | None = None) -> pd.DataFrame:
    base = build_base()
    rows = []
    for line in LINES:
        tr, te = base[f"train_{line}"], base[f"test_{line}"]
        X, _, _ = model_matrix(tr, te, features or {"history": True})
        yr = tr.period_start.dt.year.values
        for T in years:
            a, b = np.where(yr < T)[0], np.where(yr == T)[0]
            assert yr[a].max() < T, "backtest must not train on the target year"
            for meth in methods:
                tgt, info = prepare_target(tr, te, base["claims"], line, {**target_cfg, "trend": meth, "trend_to": T})
                m = make(model)
                if model == "lgbm":
                    m.fit(X.iloc[a], tgt.y_train.values[a], tgt.w.values[a], tr.exposure.values[a], n_iter=n_iter)
                else:
                    m.fit(X.iloc[a], tgt.y_train.values[a], tgt.w.values[a], tr.exposure.values[a])
                p = m.predict(X.iloc[b], tr.exposure.values[b]) * (info["loading"] if target_cfg.get("cap_q") else 1.0)
                actual = tgt.y_dev.values[b]
                rows.append({"line": line, "target_year": T, "trend": meth, "level_ratio": p.sum() / actual.sum(),
                             "d2": d2_tweedie(actual, p),
                             "factor_prev_year": info["trend"][T - 1]})
                print(f"  {line:9s} T={T} {meth:8s} level {rows[-1]['level_ratio']:.3f}  D2 {rows[-1]['d2']:.4f}", flush=True)
    return pd.DataFrame(rows)
