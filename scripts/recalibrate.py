"""Nested-CV check of a monotone recalibration of a cached result's OOF predictions. Writes nothing."""
import argparse
import functools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:     # a caller may pre-seed another src (e.g. a main checkout holding the caches)
    sys.path[:0] = [str(ROOT / "src")]

import pipeline  # noqa: E402
import recal  # noqa: E402
from market_spec import get_spec  # noqa: E402
from metrics import d2_tweedie  # noqa: E402

PCTS = (1, 10, 25, 50, 75, 90, 99)


def by_covariate(a, spec, res):
    import features
    import submit
    base = features.build_base()
    fitter = functools.partial(recal.fit_covariate, n_bins=a.bins, monotone=not a.no_monotone)
    lines = a.lines if a.lines else list(spec.lines)
    print(f"== {a.exp} | by {a.by} | bins {a.bins} | monotone {not a.no_monotone} | lines {lines} ==")
    ys, ps, ns = [], [], []
    gfull = {}
    for line in spec.lines:
        L = res["lines"][line]
        y, p = np.asarray(L["y_eval"], float), np.asarray(L["oof"], float)
        if line in lines:
            tr = base[f"train_{line}"].set_index("row_id").loc[L["row_id"]]
            x, g = tr[a.by].to_numpy(float), tr[spec.group_col].values
            nest = recal.nested_predictions_cov(x, p, y, g, fitter)
            gfull[line] = fitter(x, p, y)
            b, r = d2_tweedie(y, p), d2_tweedie(y, nest)
            q = np.quantile(x, np.linspace(0.05, 0.95, 10))
            print(f"  {line:9s} base {b:.6f} | nested recal {r:.6f} | delta {r - b:+.6f}")
            print("            g@decile-mid " + " ".join(f"{v:.3f}" for v in gfull[line](q)))
        else:
            nest = p
        ys.append(y), ps.append(p), ns.append(nest)
    yy = np.concatenate(ys)
    b, r = d2_tweedie(yy, np.concatenate(ps)), d2_tweedie(yy, np.concatenate(ns))
    print(f"  OVERALL   base {b:.6f} | nested recal {r:.6f} | delta {r - b:+.6f}")
    if not a.apply:
        return
    tp = pipeline.test_predictions(res).copy()
    for line, gf in gfull.items():
        te = base[f"test_{line}"].set_index("row_id")[a.by]
        m = tp.row_id.isin(te.index)
        f = gf(te.loc[tp.row_id[m]].to_numpy(float))
        tp.loc[m, "prediction"] = tp.loc[m, "prediction"].to_numpy(float) * f
        sx = te.loc[tp.row_id[m]].to_numpy(float)
        q1, q3 = np.quantile(sx, [0.25, 0.75])
        print(f"  test {line:9s} mean factor: bottom SI quartile {f[sx <= q1].mean():.4f} | top SI quartile {f[sx >= q3].mean():.4f} | all {f.mean():.4f}")
    path = submit.write(tp, f"{a.exp}_sical")
    print(f"  wrote {path} (not uploaded)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--method", choices=list(recal.FITTERS), default=None, help="default: both")
    ap.add_argument("--by", default=None, help="covariate calibration by this base column (e.g. sum_insured)")
    ap.add_argument("--lines", nargs="*", default=None, help="with --by: lines to recalibrate (default all)")
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--no-monotone", action="store_true")
    ap.add_argument("--apply", action="store_true", help="with --by: write a calibrated submission file (no upload)")
    a = ap.parse_args()
    spec = get_spec()
    res = pipeline.load(a.exp)
    if a.by:
        return by_covariate(a, spec, res)
    for method in ([a.method] if a.method else list(recal.FITTERS)):
        fitter = recal.FITTERS[method]
        print(f"== {a.exp} | {method} ==")
        ys, ps, ns = [], [], []
        for line in spec.lines:
            L = res["lines"][line]
            ids = pd.read_parquet(spec.cache_path / f"train_{line}.parquet", columns=["row_id", spec.group_col])
            assert (L["row_id"] == ids.row_id.values).all()
            y, p, g = np.asarray(L["y_eval"], float), np.asarray(L["oof"], float), ids[spec.group_col].values
            nest = recal.nested_predictions(p, y, g, fitter)
            base, rec = d2_tweedie(y, p), d2_tweedie(y, nest)
            q = np.percentile(p, PCTS)
            curve = fitter(p, y)(q) / q
            print(f"  {line:9s} stored cv {L['cv']:.6f} | base {base:.6f} | nested recal {rec:.6f} | delta {rec - base:+.6f}")
            print("            ratio@pct " + " ".join(f"{pc}:{r:.3f}" for pc, r in zip(PCTS, curve)))
            ys.append(y), ps.append(p), ns.append(nest)
        yy = np.concatenate(ys)
        b, r = d2_tweedie(yy, np.concatenate(ps)), d2_tweedie(yy, np.concatenate(ns))
        print(f"  OVERALL   base {b:.6f} | nested recal {r:.6f} | delta {r - b:+.6f}")


if __name__ == "__main__":
    main()
