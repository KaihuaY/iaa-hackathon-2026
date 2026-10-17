# Experiment log

Every model run, how it was checked, and what was decided. `cv` = 5-fold Tweedie D² grouped by policy (all years, developed labels); `holdout_2023` = train 2021–22 → score 2023; `lb_public` = Kaggle public leaderboard. Proposer = who suggested the change (user / AI). Keep rule: Kaggle and CV must agree.

| exp | time | description | proposer | cv | cv_motor | cv_buildings | cv_contents | holdout_2023 | lb_public | decision | verification |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E0_const | 2026-10-16 23:52 | Average cost per line x exposure (pipeline / metric check vs host's 0.072) | AI | 0.08923 | 0.04159 | 0.03539 | 0.02797 | 0.03572 | 0.07088 |  | pytest gate + 5-fold CV + 2023 holdout |
