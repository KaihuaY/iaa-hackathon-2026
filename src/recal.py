"""Monotone recalibration of a prediction scale (pred -> g(pred)), level-preserving on the fitting rows.

Each fitter returns a callable g(pred) -> calibrated pred. Out-of-range predictions are clipped to the fitted range
(no extrapolation). The ratio g/pred is bounded to [RATIO_LO, RATIO_HI] so a zero-heavy lowest block cannot produce a
zero prediction (deviance explodes), and the result is rescaled so sum(g(pred)) == sum(pred) on the fitting rows."""
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold

from metrics import d2_tweedie

RATIO_LO, RATIO_HI = 0.5, 2.0


class _Mapper:
    def __init__(self, xs, gs, scale):
        self.xs, self.gs, self.scale = np.asarray(xs, float), np.asarray(gs, float), float(scale)

    def __call__(self, pred):
        p = np.asarray(pred, float)
        return np.interp(p, self.xs, self.gs) * self.scale     # np.interp clips outside [xs[0], xs[-1]]


def _finish(xs, gs, pred, w):
    """Bound the ratio, keep g monotone, and rescale to preserve the weighted level of `pred`."""
    gs = np.clip(gs, RATIO_LO * xs, RATIO_HI * xs)
    gs = np.maximum.accumulate(gs)                              # re-impose monotone g after clipping
    m = _Mapper(xs, gs, 1.0)
    w = np.ones(len(pred)) if w is None else np.asarray(w, float)
    tot = (w * m(pred)).sum()
    m.scale = float((w * pred).sum() / tot) if tot > 0 else 1.0
    return m


def fit_isotonic(pred, y, w=None):
    pred, y = np.asarray(pred, float), np.asarray(y, float)
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip", y_min=0.0)
    iso.fit(pred, y, sample_weight=w)
    xs = np.asarray(iso.X_thresholds_, float)
    gs = np.asarray(iso.predict(xs), float)
    return _finish(xs, gs, pred, w)


def fit_bins(pred, y, n_bins=20, w=None):
    """Quantile bins of pred; per-bin mean actual vs mean pred, isotonic-smoothed (monotone g), interpolated."""
    pred, y = np.asarray(pred, float), np.asarray(y, float)
    w_ = np.ones(len(pred)) if w is None else np.asarray(w, float)
    edges = np.unique(np.quantile(pred, np.linspace(0, 1, n_bins + 1)))
    b = np.clip(np.searchsorted(edges, pred, side="right") - 1, 0, len(edges) - 2)
    k = len(edges) - 1
    sw = np.bincount(b, w_, k)
    keep = sw > 0
    sp = np.bincount(b, w_ * pred, k)
    px = (sp / np.where(keep, sw, 1))[keep]
    ratio = (np.bincount(b, w_ * y, k) / np.maximum(sp, 1e-12))[keep]
    gm = IsotonicRegression(increasing=True, y_min=0.0).fit(px, ratio * px, sample_weight=sw[keep]).predict(px)
    r = gm / px                                                  # smoothed ratio at the bin means
    xs = np.concatenate([[pred.min()], px, [pred.max()]])        # end anchors: keep the end-bin ratio out to the data range
    xs, uniq = np.unique(xs, return_index=True)
    gs = (np.concatenate([[r[0]], r, [r[-1]]]) * np.concatenate([[pred.min()], px, [pred.max()]]))[uniq]
    return _finish(xs, np.maximum.accumulate(gs), pred, w)


FITTERS = {"isotonic": fit_isotonic, "bins": fit_bins}


def nested_predictions(pred_oof, y, groups, fitter, folds=5):
    """Held-out recalibrated predictions: GroupKFold by `groups`, fit on the train part, apply to the held-out part."""
    pred_oof, y = np.asarray(pred_oof, float), np.asarray(y, float)
    out = np.zeros_like(pred_oof)
    for tr, va in GroupKFold(folds).split(y, groups=np.asarray(groups)):
        out[va] = fitter(pred_oof[tr], y[tr])(pred_oof[va])
    return out


def nested_eval(pred_oof, y, groups, fitter, folds=5):
    """Returns (d2_base, d2_recal) over all held-out rows."""
    out = nested_predictions(pred_oof, y, groups, fitter, folds)
    return d2_tweedie(np.asarray(y, float), pred_oof), d2_tweedie(np.asarray(y, float), out)
