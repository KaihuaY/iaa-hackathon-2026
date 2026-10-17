"""Run an experiment (CV + holdout + test predictions), log it, optionally submit to Kaggle.

  python scripts/run_experiment.py L16_relevant --quick        # fast 80/20 screen only (no log, no test preds)
  python scripts/run_experiment.py X3_blend                    # local only: 5-fold CV + predicted LB
  python scripts/run_experiment.py X3_blend --reuse --submit --approved   # user approved after explanation
  python scripts/run_experiment.py X4_blend --reuse --submit --auto       # only if predicted LB >= 5% above best
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
    ap.add_argument("--quick", action="store_true", help="fast 80/20 grouped split screen only")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--approved", action="store_true", help="user approved this submission")
    ap.add_argument("--auto", action="store_true", help="submit only if predicted LB >= 5%% above best")
    ap.add_argument("--no-holdout", action="store_true")
    ap.add_argument("--reuse", action="store_true", help="skip training, reuse cached results")
    a = ap.parse_args()
    cfg = EXPERIMENTS[a.exp]
    print(f"== {a.exp}: {cfg['desc']}", flush=True)
    if a.quick:
        pipeline.quick(a.exp, cfg)
        return
    res = pipeline.load(a.exp) if a.reuse else pipeline.run(a.exp, cfg, holdout=not a.no_holdout)
    path = sub.write(pipeline.test_predictions(res), a.exp)
    cvx = res.get("cv_nested", res["cv"])
    lbm = lbmodel.fit()
    pred, sd = lbmodel.predict(cvx, lbm)
    explog.record(res, cfg["desc"], cfg["proposer"], file=path.name,
                  verification="pytest gate + 5-fold CV + 2023 holdout", lb_pred=round(pred, 5) if pred == pred else "")
    print("  " + lbmodel.verdict(cvx, lbm), flush=True)
    if a.submit:
        if not (a.approved or (a.auto and lbmodel.auto_ok(cvx, lbm))):
            raise SystemExit("not submitting: needs the user's approval (--approved), "
                             "or --auto with predicted LB >= 5% above the current best")
        sub.run_tests()
        if sub.remaining_today() <= 0:
            raise SystemExit("no submissions left today")
        score = sub.submit(path, f"{a.exp}: {cfg['desc']} | CV {res['cv']:.4f}")
        print(f"  PUBLIC LB: {score} (predicted {pred:.4f} +/- {sd:.4f})")
        explog.update(a.exp, lb_public=score)


if __name__ == "__main__":
    main()
