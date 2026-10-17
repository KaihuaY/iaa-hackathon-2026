"""Leaderboard level probe.

The public LB scores a FIXED subset of test rows, so submitting the same predictions with one line scaled by s
changes the score with zero sampling noise. Three points per line (s = 0.95, 1.00, 1.05) determine a parabola
lb(s) = a + b s + c s^2 whose maximum s* = -b/(2c) is the LB-optimal level for that line. The OOF D2 curve is
printed alongside as the local reference (sensitivity and curvature).

Usage: python -P -E scripts/level_probe.py <base_exp> [--lines motor buildings] [--scales 0.95 1.05]
                                           [--dry-run] [--submit] [--apply]
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROBE_CSV = ROOT / "experiments" / "results" / "level_probe.csv"
LOCAL_SCALES = [0.90, 0.95, 1.00, 1.05, 1.10]
CLIP = (0.85, 1.15)


def fit_optimum(scales, scores):
    """Quadratic fit -> (s*, interior_max, (a, b, c)). s* clipped to CLIP; interior_max False if c >= 0."""
    c, b, a = np.polyfit(np.asarray(scales, float), np.asarray(scores, float), 2)
    if c >= 0:  # no interior maximum: best end of the clip range
        f = lambda s: a + b * s + c * s * s
        return max(CLIP, key=f), False, (a, b, c)
    return float(np.clip(-b / (2 * c), *CLIP)), True, (a, b, c)


def scaled_submission(res, factors):
    """test_predictions with each line multiplied by factors[line]."""
    from pipeline import test_predictions
    df = test_predictions(res).copy()
    for line, s in factors.items():
        df.loc[df["row_id"].isin(res["lines"][line]["test_row_id"]), "prediction"] *= s
    return df


def local_curves(res, lines):
    from metrics import d2_tweedie
    L = res["lines"]
    for line in lines:
        y = L[line]["y_eval"]
        rows = []
        for s in LOCAL_SCALES:
            ys = np.concatenate([L[l]["y_eval"] for l in L])
            mu = np.concatenate([L[l]["oof"] * (s if l == line else 1.0) for l in L])
            rows.append((s, d2_tweedie(y, L[line]["oof"] * s), d2_tweedie(ys, mu)))
        print(f"\nLocal OOF curve, {line}:\n  scale   line D2   overall D2")
        for s, d, o in rows:
            print(f"  {s:5.2f}  {d:8.5f}  {o:9.5f}")
        print(f"  local curvature c = {fit_optimum(LOCAL_SCALES, [r[1] for r in rows])[2][2]:.5f}")


def log_probe(base, line, scale, lb):
    PROBE_CSV.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([dict(base=base, line=line, scale=scale, lb_public=lb, timestamp=f"{datetime.now():%Y-%m-%d %H:%M}")])
    row.to_csv(PROBE_CSV, mode="a", header=not PROBE_CSV.exists(), index=False)


def analyse(base, res):
    import explog
    log = pd.read_csv(explog.LOG)
    base_lb = log.loc[log["exp"] == base, "lb_public"].dropna()
    probes = pd.read_csv(PROBE_CSV) if PROBE_CSV.exists() else pd.DataFrame(columns=["base", "line", "scale", "lb_public"])
    probes = probes[(probes["base"] == base) & probes["lb_public"].notna()]
    opt = {}
    print(f"\nLB analysis for {base} (base LB at 1.00: {base_lb.iloc[-1] if len(base_lb) else 'missing'})")
    for line in res["lines"]:
        p = probes[probes["line"] == line].drop_duplicates("scale", keep="last")
        pts = dict(zip(p["scale"].astype(float), p["lb_public"].astype(float)))
        if len(base_lb):
            pts[1.0] = float(base_lb.iloc[-1])
        if len(pts) < 3:
            print(f"  {line}: {len(pts)} point(s), need 3")
            continue
        s, interior, (a, b, c) = fit_optimum(list(pts), list(pts.values()))
        gain = (a + b * s + c * s * s) - pts[1.0]
        if not interior:
            print(f"  WARNING {line}: c={c:.5f} >= 0, no interior maximum; using clip boundary")
        y = res["lines"][line]["y_eval"]
        oof = res["lines"][line]["oof"]
        from metrics import d2_tweedie
        lc = fit_optimum(LOCAL_SCALES, [d2_tweedie(y, oof * t) for t in LOCAL_SCALES])[2][2]
        print(f"  {line}: pts={sorted(pts.items())}\n    s*={s:.4f}  predicted LB gain={gain:+.5f}  LB c={c:.5f}  local OOF c={lc:.5f}")
        opt[line] = s
    return opt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("--lines", nargs="+", default=["motor", "buildings"])
    ap.add_argument("--scales", nargs="+", type=float, default=[0.95, 1.05])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    import pipeline, submit
    res = pipeline.load(a.base)
    files = []
    for line in a.lines:
        for s in a.scales:
            path = submit.write(scaled_submission(res, {line: s}), f"{a.base}_probe_{line}_{s}")
            files.append((line, s, path))
            print("wrote", path)
    local_curves(res, a.lines)
    if a.submit and not a.dry_run:
        for line, s, path in files:
            if submit.remaining_today() <= 0:
                print("No Kaggle submissions remaining today; stopping.")
                break
            lb = submit.submit(path, f"level probe {a.base} {line} x{s}")
            log_probe(a.base, line, s, lb)
            print(f"{line} x{s}: LB {lb}")
    else:
        print("\nNot uploading (no --submit, or --dry-run).")
    opt = analyse(a.base, res)
    if a.apply:
        factors = {l: opt.get(l, 1.0) for l in res["lines"]}
        print("Applying factors:", factors)
        print("wrote", submit.write(scaled_submission(res, factors), f"{a.base}_levelcal"))


if __name__ == "__main__":
    main()
