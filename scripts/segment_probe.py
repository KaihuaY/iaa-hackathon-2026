"""Segment-level public-LB probe: scale only the test rows of one segment of one line and submit.

  python -P -E scripts/segment_probe.py <base_exp> --line motor --segment row_type=new_business --scale 1.06 [--submit]
  python -P -E scripts/segment_probe.py <base_exp> --line buildings --segment sum_insured_q=top --scale 1.08

The public LB scores a fixed subset of rows, so lb(scaled) - lb(base) is noise-free and tells whether the 2025 level
shift measured per line (level_probe.py) is concentrated in the segment. Segment syntax: `col=value` (categorical),
`col_q=top|bottom` (quartile), `col_d=top|bottom` (decile), `col_p=NN` (top NN percent), `col in a|b|c`,
`<datecol>_month in 1|2|3`, compound with " & ". Quantiles are cut on the test rows of that line. IAA_ROOT env overrides ROOT.
"""
import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("IAA_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROOT / "src"))
PROBE_CSV = ROOT / "experiments" / "results" / "segment_probe.csv"


def _atom_mask(te: pd.DataFrame, atom: str) -> np.ndarray:
    atom = atom.strip()
    m = re.fullmatch(r"(\w+)\s+in\s+(.+)", atom)
    if m:
        col, vals = m.group(1), [v.strip() for v in m.group(2).split("|")]
        if col.endswith("_month") and col not in te.columns and pd.api.types.is_datetime64_any_dtype(te[col[:-6]]):
            return te[col[:-6]].dt.month.isin([int(v) for v in vals]).values
        return te[col].astype(str).isin(vals).values
    col, val = atom.split("=", 1)
    col, val = col.strip(), val.strip()
    if col.endswith("_even") and col[:-5] in te.columns:          # split-half by parity of an id column (lumpiness test)
        last = te[col[:-5]].astype(str).str.extract(r"(\d)\D*$")[0].fillna("0").astype(int)   # parity of the last digit
        return ((last % 2) == int(val)).values
    if col.endswith("_q") and col[:-2] in te.columns and val in ("top", "bottom"):
        x = te[col[:-2]].astype(float)
        return (x >= x.quantile(0.75)).values if val == "top" else (x <= x.quantile(0.25)).values
    if col.endswith("_d") and col[:-2] in te.columns and val in ("top", "bottom"):
        x = te[col[:-2]].astype(float)
        return (x >= x.quantile(0.9)).values if val == "top" else (x <= x.quantile(0.1)).values
    if col.endswith("_p") and col[:-2] in te.columns:
        x = te[col[:-2]].astype(float)
        return (x >= x.quantile(1 - float(val) / 100)).values
    return (te[col].astype(str) == val).values


def segment_mask(te: pd.DataFrame, seg: str) -> np.ndarray:
    """AND of atoms separated by ' & '; atoms: col=value, col_q|col_d=top|bottom, col_p=NN, col in a|b|c."""
    mask = np.ones(len(te), dtype=bool)
    for atom in seg.split(" & "):
        mask &= _atom_mask(te, atom)
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("--line", required=True)
    ap.add_argument("--segment")
    ap.add_argument("--list-levels")
    ap.add_argument("--scale", type=float)
    ap.add_argument("--submit", action="store_true")
    a = ap.parse_args()
    import pipeline, submit
    from features import build_base
    res = pipeline.load(a.base)
    te = build_base()[f"test_{a.line}"].set_index("row_id")
    if a.list_levels:
        print(te[a.list_levels].value_counts(dropna=False).to_string())
        return
    if not a.segment or a.scale is None:
        raise SystemExit("--segment and --scale are required (unless --list-levels)")
    sub = pipeline.test_predictions(res)
    L = res["lines"][a.line]
    rid = pd.Index(L["test_row_id"])
    mask = segment_mask(te.loc[rid].rename_axis("row_id").reset_index(), a.segment)   # keep row_id as a column
    if mask.sum() == 0 or mask.all():
        raise SystemExit(f"segment {a.segment} selects {mask.sum()} of {len(mask)} rows - nothing to probe")
    scaled_ids = set(rid[mask])
    hit = sub.row_id.isin(scaled_ids)
    sub.loc[hit, "prediction"] *= a.scale
    print(f"{a.line} segment {a.segment}: {mask.sum():,} of {len(mask):,} test rows ({mask.mean():.1%}) scaled x{a.scale}; "
          f"line mean prediction x{sub.loc[sub.row_id.isin(rid), 'prediction'].mean() / L['test_pred'].mean():.4f}")
    seg_tag = re.sub(r"[^A-Za-z0-9_.+-]", "", a.segment.replace("=", "-").replace(" & ", "+").replace("|", "_"))
    tag = f"{a.base}_seg_{a.line}_{seg_tag}_{a.scale}"
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
