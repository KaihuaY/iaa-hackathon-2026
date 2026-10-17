"""Predict the public leaderboard score from local CV, using experiments that have both.

On 8 submissions (17 Oct) CV D² vs public LB: corr 0.995, rank agreement perfect, residual sd ~0.002.
"""
import numpy as np
import pandas as pd

from explog import LOG
from market_spec import get_spec

EXCLUDE = ("bug", "DIAGNOSTIC", "EXCLUDED")


MIN_HISTORY = 3
NO_HISTORY = "no LB history for this market yet"


def fit(log: pd.DataFrame | None = None) -> dict:
    """Fit on the active market's rows only (rows without a market column count as part1). With fewer than
    MIN_HISTORY scored rows, returns a model with n < MIN_HISTORY and no coefficients."""
    d = pd.read_csv(LOG) if log is None else log
    mk = d["market"].fillna("part1") if "market" in d else pd.Series("part1", index=d.index)
    d = d[mk == get_spec().name]
    d = d.assign(lb=pd.to_numeric(d.lb_public, errors="coerce"),
                 cvx=pd.to_numeric(d.get("cv_nested", d.cv), errors="coerce").fillna(d.cv))
    d = d[d.lb.notna() & ~d.decision.fillna("").str.contains("|".join(EXCLUDE))]
    if len(d) < MIN_HISTORY:
        return {"a": np.nan, "b": np.nan, "sd": np.nan, "n": len(d), "corr": np.nan, "best_lb": np.nan,
                "best_cv": np.nan, "best_exp": ""}
    b, a = np.polyfit(d.cvx, d.lb, 1)
    resid = d.lb - (a + b * d.cvx)
    best = d.loc[d.lb.idxmax()]
    return {"a": float(a), "b": float(b), "sd": float(resid.std(ddof=2)), "n": len(d),
            "corr": float(np.corrcoef(d.cvx, d.lb)[0, 1]), "best_lb": float(best.lb), "best_cv": float(best.cvx),
            "best_exp": best.exp}


def predict(cv: float, model: dict | None = None) -> tuple[float, float]:
    """LB anchored on the best submission: best LB + slope x (CV - best CV). Comparing like with like
    removes the fitted line's offset at the top end."""
    m = model or fit()
    if m["n"] < MIN_HISTORY:
        return float("nan"), float("nan")
    return m["best_lb"] + m["b"] * (cv - m["best_cv"]), m["sd"]


AUTO_REL = 0.05   # user rule (17 Oct): submit without asking only if predicted LB >= 5 % above the current best


def auto_ok(cv: float, model: dict | None = None, margin: float = 0.002) -> bool:
    """Gain must be positive, >= 5 % of |best|, and larger than the noise (review fix 17 Oct)."""
    m = model or fit()
    if m["n"] < MIN_HISTORY:
        return False
    gain = predict(cv, m)[0] - m["best_lb"]
    return gain > 0 and gain >= AUTO_REL * abs(m["best_lb"]) and gain > max(margin, m["sd"])


def verdict(cv: float, model: dict | None = None, margin: float = 0.002) -> str:
    m = model or fit()
    if m["n"] < MIN_HISTORY:
        return NO_HISTORY
    p, sd = predict(cv, m)
    gain = p - m["best_lb"]
    rel = gain / m["best_lb"]
    if auto_ok(cv, m):
        tag = "AUTO-SUBMIT (>= 5 % above best)"
    elif gain > margin:
        tag = "material - explain to user before submitting"
    else:
        tag = "below noise - do not submit"
    return (f"predicted LB {p:.4f} +/- {sd:.4f} (vs best {m['best_exp']} {m['best_lb']:.4f}: {gain:+.4f}, {rel:+.1%}): {tag}")
