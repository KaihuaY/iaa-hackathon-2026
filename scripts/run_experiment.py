"""Run an experiment (CV + holdout + test predictions), log it, optionally submit to Kaggle.

  python scripts/run_experiment.py E2_lgbm            # CV only
  python scripts/run_experiment.py E2_lgbm --submit   # CV, then tests, then submit and record the LB score
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

import explog  # noqa: E402
import pipeline  # noqa: E402
import submit as sub  # noqa: E402
from configs import EXPERIMENTS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--no-holdout", action="store_true")
    ap.add_argument("--reuse", action="store_true", help="skip training, reuse cached results")
    a = ap.parse_args()
    cfg = EXPERIMENTS[a.exp]
    print(f"== {a.exp}: {cfg['desc']}", flush=True)
    res = pipeline.load(a.exp) if a.reuse else pipeline.run(a.exp, cfg, holdout=not a.no_holdout)
    path = sub.write(pipeline.test_predictions(res), a.exp)
    explog.record(res, cfg["desc"], cfg["proposer"], file=path.name,
                  verification="pytest gate + 5-fold CV + 2023 holdout")
    if a.submit:
        sub.run_tests()
        left = sub.remaining_today()
        if left <= 2:
            raise SystemExit(f"only {left} submissions left today; keeping them in reserve")
        score = sub.submit(path, f"{a.exp}: {cfg['desc']} | CV {res['cv']:.4f}")
        print(f"  PUBLIC LB: {score}")
        explog.update(a.exp, lb_public=score)


if __name__ == "__main__":
    main()
