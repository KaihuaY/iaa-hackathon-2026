"""Segment-level public-LB probe: scale only the test rows of one segment of one line and submit.

  python -P -E scripts/segment_probe.py <base_exp> --line motor --segment row_type=new_business --scale 1.06 [--submit]
  python -P -E scripts/segment_probe.py <base_exp> --line buildings --segment sum_insured_q=top --scale 1.08

The public LB scores a fixed subset of rows, so lb(scaled) - lb(base) is noise-free and tells whether the 2025 level
shift measured per line (level_probe.py) is concentrated in the segment. Segment syntax: `col=value` (categorical),
`col_q=top|bottom` (top/bottom quartile of a numeric column, cut on the test rows of that line).
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROBE_CSV = ROOT / "experiments" / "results" / "segment_probe.csv"


def segment_mask(te: pd.DataFrame, seg: str) -> np.ndarray:
    col, val = seg.split("=", 1)
    if col.endswith("_q"):
        x = te[col[:-2]].astype(float)
        q = x.quantile(0.75) if val == "top" else x.quantile(0.25)
        return (x >= q).values if val == "top" else (x <= q).values
    return (te[col].astype(str) == val).values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("--line", required=True)
    ap.add_argument("--segment", required=True)
    ap.add_argument("--scale", type=float, required=True)
    ap.add_argument("--submit", action="store_true")
    a = ap.parse_args()
    import pipeline, submit
    from features import build_base
    res = pipeline.load(a.base)
    te = build_base()[f"test_{a.line}"].set_index("row_id")
    sub = pipeline.test_predictions(res)
    L = res["lines"][a.line]
    rid = pd.Index(L["test_row_id"])
    mask = segment_mask(te.loc[rid].reset_index(), a.segment)
    if mask.sum() == 0 or mask.all():
        raise SystemExit(f"segment {a.segment} selects {mask.sum()} of {len(mask)} rows - nothing to probe")
    scaled_ids = set(rid[mask])
    hit = sub.row_id.isin(scaled_ids)
    sub.loc[hit, "prediction"] *= a.scale
    print(f"{a.line} segment {a.segment}: {mask.sum():,} of {len(mask):,} test rows ({mask.mean():.1%}) scaled x{a.scale}; "
          f"line mean prediction x{sub.loc[sub.row_id.isin(rid), 'prediction'].mean() / L['test_pred'].mean():.4f}")
    tag = f"{a.base}_seg_{a.line}_{a.segment.replace('=', '-')}_{a.scale}"
    path = submit.write(sub, tag)
    print("wrote", path)
    if a.submit:
        if submit.remaining_today() <= 0:
            raise SystemExit("no submissions left today")
        lb = submit.submit(path, f"segment probe {a.base} {a.line} {a.segment} x{a.scale}")
        row = pd.DataFrame([{"base": a.base, "line": a.line, "segment": a.segment, "scale": a.scale, "share": round(float(mask.mean()), 4),
                             "lb": lb, "time": f"{datetime.now():%Y-%m-%d %H:%M}"}])
        row.to_csv(PROBE_CSV, mode="a", header=not PROBE_CSV.exists(), index=False)
        print(f"{a.line} {a.segment} x{a.scale}: LB {lb}")
    else:
        print("Not uploading (no --submit).")


if __name__ == "__main__":
    main()
