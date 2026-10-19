"""Build a candidate file from a blend with probe-derived factors: per-line flat factors plus, for one line, factors by
quantile band of a numeric column (evidence: 19 Oct public-LB segment probes, docs/diagnostics_17oct.md §10b).

  python -P -E scripts/segment_calibrate.py X21_blend4_a1bag --level-factors motor=1.029 \
      --band-line buildings --band-col sum_insured --band-factors 1.0 1.05 1.08 1.20 [--tag sifix]

Bands are quantile bands of the column over that line's TEST rows (4 factors = quartiles, 10 = deciles).
The file is written under submissions/ and NOT uploaded.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("--level-factors", nargs="*", default=[], help="line=factor")
    ap.add_argument("--band-line", default=None)
    ap.add_argument("--band-col", default="sum_insured")
    ap.add_argument("--band-factors", nargs="*", type=float, default=[])
    ap.add_argument("--tag", default="segcal")
    a = ap.parse_args()
    import pipeline, submit
    from features import build_base
    res = pipeline.load(a.base)
    sub = pipeline.test_predictions(res).set_index("row_id")
    level = {k: float(v) for k, v in (s.split("=") for s in a.level_factors)}
    base = build_base()
    report = {}
    for line, L in res["lines"].items():
        rid = pd.Index(L["test_row_id"])
        f = np.full(len(rid), level.get(line, 1.0))
        if a.band_line == line and a.band_factors:
            te = base[f"test_{line}"].set_index("row_id").loc[rid]
            x = te[a.band_col].astype(float)
            q = pd.qcut(x.rank(method="first"), len(a.band_factors), labels=False)
            f = f * np.asarray(a.band_factors, float)[q.values]
            sizes = pd.Series(q.values).value_counts().sort_index().tolist()
            print(f"{line}: {a.band_col} bands {sizes} -> factors {a.band_factors}")
        sub.loc[rid, "prediction"] = sub.loc[rid, "prediction"].values * f
        report[line] = round(float(f.mean()), 4)
    print("mean factor by line:", report)
    out = submit.write(sub.reset_index(), f"{a.base}_{a.tag}")
    print("wrote", out, "(not uploaded)")


if __name__ == "__main__":
    main()
