"""Bayesian hyper-parameter search (Optuna TPE) on the quick 80/20 screen.

  python scripts/tune_optuna.py --base L17_tuned --trials 12

Each trial = one quick screen (~6 min, all lines). Trials are stored in experiments/optuna.db (resumable); the best
parameters are printed and saved to experiments/results/optuna_<base>.json for a follow-up full-CV config.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

import optuna  # noqa: E402

import pipeline  # noqa: E402
from configs import EXPERIMENTS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="L17_tuned")
    ap.add_argument("--trials", type=int, default=12)
    a = ap.parse_args()
    base = EXPERIMENTS[a.base]

    def objective(trial):
        params = {**base.get("params", {}),
                  "num_leaves": trial.suggest_int("num_leaves", 7, 63, log=True),
                  "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 2000, 30000, log=True),
                  "feature_fraction": trial.suggest_float("feature_fraction", 0.3, 0.9),
                  "lambda_l2": trial.suggest_float("lambda_l2", 1.0, 300.0, log=True),
                  "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.06, log=True),
                  "cat_smooth": trial.suggest_float("cat_smooth", 10.0, 200.0, log=True)}
        print(f"trial {trial.number}: {params}", flush=True)
        out = pipeline.quick(f"optuna_{trial.number}", {**base, "params": params})
        for k, v in out.items():
            trial.set_user_attr(k, v)
        return out["overall"]

    study = optuna.create_study(direction="maximize", study_name=f"quick_{a.base}",
                                storage=f"sqlite:///{(ROOT / 'experiments' / 'optuna.db').as_posix()}", load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=7))
    if not study.trials:   # seed the search with the current best so TPE starts from a known-good point
        study.enqueue_trial({"num_leaves": 15, "min_data_in_leaf": 8000, "feature_fraction": 0.5, "lambda_l2": 50.0,
                             "learning_rate": 0.03, "cat_smooth": 50.0})
    study.optimize(objective, n_trials=a.trials)
    best = {"value": study.best_value, "params": study.best_params, "n_trials": len(study.trials)}
    (ROOT / "experiments" / "results" / f"optuna_{a.base}.json").write_text(json.dumps(best, indent=2))
    print("BEST", json.dumps(best), flush=True)


if __name__ == "__main__":
    main()
