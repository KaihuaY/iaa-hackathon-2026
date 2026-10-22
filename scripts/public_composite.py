"""Public-only composite: multiply a submitted file's predictions by factors on test-row segments. Writes a file;
never uploads. Built for the visible public rank only — NOT a final-score candidate (see docs/final_pick_22oct.md).

  python -P -E scripts/public_composite.py --base submissions/20261018_2152_X30_blend4_si07_sifix.csv --tag pubA \
      --rule "buildings|region in R03|R08|R06|R11|R12|0.92" --rule "buildings|*|1.0" ...

Rule = "line|segment|factor"; segment uses scripts/segment_probe.py syntax, "*" = whole line. Rules apply in order
and multiply.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--rule", action="append", default=[])
    a = ap.parse_args()
    import submit
    from features import build_base
    from segment_probe import segment_mask
    sub = pd.read_csv(ROOT / a.base if not Path(a.base).is_absolute() else a.base).set_index("row_id")
    ref = sub.prediction.copy()
    base = build_base()
    for r in a.rule:
        line, rest = r.split("|", 1)
        seg, fac = rest.rsplit("|", 1)
        fac = float(fac)
        te = base[f"test_{line}"].set_index("row_id")
        te = te.loc[te.index.intersection(sub.index)]
        mask = np.ones(len(te), bool) if seg.strip() == "*" else segment_mask(te.rename_axis("row_id").reset_index(), seg)
        ids = te.index[mask]
        sub.loc[ids, "prediction"] *= fac
        print(f"{line:9s} {seg:45s} x{fac:<6} rows {mask.sum():>7,} ({mask.mean():.1%})")
    for line in ("motor", "buildings", "contents"):
        ids = base[f"test_{line}"].row_id
        print(f"  {line}: mean factor vs base {sub.loc[ids, 'prediction'].mean() / ref.loc[ids].mean():.4f}")
    print(f"corr vs base {np.corrcoef(ref, sub.prediction)[0, 1]:.4f}, rows {len(sub)}, NaN {int(sub.prediction.isna().sum())}")
    print("wrote", submit.write(sub.reset_index(), f"PUBLIC_ONLY_{a.tag}"), "(not uploaded)")


if __name__ == "__main__":
    main()
