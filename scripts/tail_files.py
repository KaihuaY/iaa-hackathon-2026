"""Build the three tail-test files requested by the user (19 Oct): nothing is uploaded here.

  python -P -E scripts/tail_files.py --base X21_blend4_a1bag --tail T3_tailload --motor T5_motor_p17 \
      --level-factors motor=1.029 buildings=1.093

 (a) <tail> + level factors                      : the tail model on its own
 (b) <base> x (tail loading_test / flat) + levels: the base blend re-weighted toward rows with higher predicted
                                                   large-loss propensity (the model-based loading's per-row ratio)
 (c) <base> with the motor line replaced by <motor>'s predictions + levels (power-1.7 motor)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _levels(res, level):
    sub = pipeline.test_predictions(res).set_index("row_id")
    for line, L in res["lines"].items():
        rid = pd.Index(L["test_row_id"])
        sub.loc[rid, "prediction"] = sub.loc[rid, "prediction"].values * level.get(line, 1.0)
    return sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="X21_blend4_a1bag")
    ap.add_argument("--tail", default="T3_tailload")
    ap.add_argument("--motor", default="T5_motor_p17")
    ap.add_argument("--level-factors", nargs="*", default=[])
    a = ap.parse_args()
    global pipeline
    import pipeline, submit
    level = {k: float(v) for k, v in (s.split("=") for s in a.level_factors)}
    base, tail, motor = pipeline.load(a.base), pipeline.load(a.tail), pipeline.load(a.motor)
    # (a) tail model + levels
    pa = _levels(tail, level)
    print("(a)", submit.write(pa.reset_index(), f"{a.tail}_tail_levels"))
    # (b) base x tail uplift ratio (per-row model loading / flat loading), renormalised per line to keep the level
    pb = _levels(base, level)
    for line, L in tail["lines"].items():
        info = L["info"]
        lt = np.asarray(info.get("loading_test", info["loading"]), float)
        if lt.ndim == 0:
            print(f"   {line}: scalar loading, no uplift"); continue
        ratio = lt / float(info["loading"]); ratio = ratio / ratio.mean()
        rid = pd.Index(L["test_row_id"])
        pb.loc[rid, "prediction"] = pb.loc[rid, "prediction"].values * ratio
        print(f"   {line}: uplift ratio q01/50/99 {np.quantile(ratio, [.01,.5,.99]).round(3).tolist()}")
    print("(b)", submit.write(pb.reset_index(), f"{a.base}_tail_uplift"))
    # (c) base with motor from the power-1.7 model
    pc = _levels(base, level)
    Lm = motor["lines"]["motor"]; rid = pd.Index(Lm["test_row_id"])
    pc.loc[rid, "prediction"] = np.asarray(Lm["test_pred"]) * level.get("motor", 1.0)
    print("(c)", submit.write(pc.reset_index(), f"{a.base}_motor_p17"))


if __name__ == "__main__":
    main()
