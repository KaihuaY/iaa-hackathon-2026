"""Run an experiment (CV + holdout + test predictions), log it, optionally submit to Kaggle.

  python scripts/run_experiment.py X2_blend                       # local only: CV + predicted LB
  python scripts/run_experiment.py X2_blend --reuse --submit --approved
      # submits only with --approved (the user agreed after the improvement was explained)
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

import explog  # noqa: E402
import lbmodel  # noqa: E402
import pipeline  # noqa: E402
import submit as sub  # noqa: E402
from configs import EXPERIMENTS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--approved", action="store_true", help="user approved this submission")
    ap.add_argument("--no-holdout", action="store_true")
    ap.add_argument("--reuse", action="store_true", help="skip training, reuse cached results")
    a = ap.parse_args()
    cfg = EXPERIMENTS[a.exp]
    print(f"== {a.exp}: {cfg['desc']}", flush=True)
    res = pipeline.load(a.exp) if a.reuse else pipeline.run(a.exp, cfg, holdout=not a.no_holdout)
    path = sub.write(pipeline.test_predictions(res), a.exp)
    cvx = res.get("cv_nested", res["cv"])
    pred, sd = lbmodel.predict(cvx)
    explog.record(res, cfg["desc"], cfg["proposer"], file=path.name,
                  verification="pytest gate + 5-fold CV + 2023 holdout", lb_pred=round(pred, 5))
    print("  " + lbmodel.verdict(cvx), flush=True)
    if a.submit:
        if not a.approved:
            raise SystemExit("not submitting: submissions need the user's approval (--approved)")
        sub.run_tests()
        left = sub.remaining_today()
        if left <= 0:
            raise SystemExit("no submissions left today")
        score = sub.submit(path, f"{a.exp}: {cfg['desc']} | CV {res['cv']:.4f}")
        print(f"  PUBLIC LB: {score} (predicted {pred:.4f} ± {sd:.4f})")
        explog.update(a.exp, lb_public=score)


if __name__ == "__main__":
    main()
