#!/usr/bin/env bash
# Day-1 experiment chain; each step independent so one failure doesn't stop the rest.
cd "$(dirname "$0")/.."
for e in E1_glm E2_lgbm E3_dev E4_cap E5_hist E6_trend; do python -P -E scripts/run_experiment.py $e --submit; done
python -P -E scripts/run_experiment.py E7_glm_full
python -P -E scripts/run_experiment.py E8_blend --submit
echo "DAY1 DONE"
