#!/usr/bin/env bash
# Re-run after the LightGBM start-level fix: CV only first, then submit the informative ones.
cd "$(dirname "$0")/.."
for e in E2_lgbm E3_dev E4_cap E5_hist E6_trend E7_glm_full; do python -P -E scripts/run_experiment.py $e; done
python -P -E scripts/run_experiment.py E8_blend
echo "DAY1B DONE"
