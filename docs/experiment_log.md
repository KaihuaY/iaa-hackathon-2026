# Experiment log

Every model run, how it was checked, and what was decided. `cv` = 5-fold Tweedie D² grouped by policy (all years, developed labels); `holdout_2023` = train 2021–22 → score 2023; `lb_public` = Kaggle public leaderboard. Proposer = who suggested the change (user / AI). Keep rule: Kaggle and CV must agree.

| exp | time | description | proposer | cv | cv_motor | cv_buildings | cv_contents | holdout_2023 | lb_public | decision | verification |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E0_const | 2026-10-16 23:52 | Average cost per line x exposure (pipeline / metric check vs host's 0.072) | AI | 0.08923 | 0.04159 | 0.03539 | 0.02797 | 0.03572 | 0.07088 | reference | pytest gate + 5-fold CV + 2023 holdout |
| E1_glm | 2026-10-16 23:59 | Tweedie GLM per line, banded factors, raw labels | user (decision board #1/#9) | 0.13964 | 0.09444 | 0.08741 | 0.08468 | 0.08009 | 0.10753 | baseline | pytest gate + 5-fold CV + 2023 holdout |
| E2_lgbm_bug | 2026-10-17 00:01 | Tweedie LightGBM per line, raw labels | user (decision board #1) | 0.12434 | 0.0682 | 0.07929 | 0.07505 | 0.06064 | 0.09852 | rejected: bug found | LB below GLM + test-mean 10-20% low exposed a LightGBM start-level bug (init_score disables boost_from_average). Fixed in models.py; calibration test + per-line calib check added |
