"""Nested-CV check of a monotone recalibration of a cached result's OOF predictions. Writes nothing."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]

import pipeline  # noqa: E402
import recal  # noqa: E402
from market_spec import get_spec  # noqa: E402
from metrics import d2_tweedie  # noqa: E402

PCTS = (1, 10, 25, 50, 75, 90, 99)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--method", choices=list(recal.FITTERS), default=None, help="default: both")
    a = ap.parse_args()
    spec = get_spec()
    res = pipeline.load(a.exp)
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
