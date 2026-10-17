# IAA AI-Assisted Actuarial Hackathon 2026: Part 1 write-up (draft)

## 1. Summary

- **Task.** Predict the expected ultimate claim cost (risk premium, net of excess and recoveries, repudiated claims excluded) for 626,639 policy-period rows in 2025 across motor, buildings and contents. Metric: Tweedie D² (power 1.5).
- **Approach.** Clean the planted data errors, build a developed (BF IBNR), capped, 2025-trended target, then fit per-line Tweedie models (a GLM-boosted GBM, a monotone GBM, a frequency x severity GBM and CatBoost) combined by a per-line blend with a small credibility layer.
- **Validation.** Policy-grouped 5-fold CV, a 2023 holdout, and a CV-to-leaderboard predictor (corr ≈ 0.995, noise ±0.002) that decides what is worth submitting.
- **Public LB.** Best so far 0.12749 (`X10_blend4`, CV 0.15925, nested CV 0.15904; first on the leaderboard when submitted). The GLM baseline scored 0.10753.
- **Use of AI.** AI proposed most experiments and wrote most code. Every change passed a pytest gate, CV and human approval, and two independent AI code reviews caught real bugs (`docs/ai_review_log.md`).
- **Part 2.** The same pipeline is re-run on the emerging-market data, with data quality, development, trend, level and feature stability re-estimated rather than carried over. Part 2 results: [TBD].

## 2. Data understanding and quality

The EDA dashboard (`notebooks/01_eda.py`, section 2) records each problem. `src/data.py` fixes it and logs the row count (training rows; source `reports/01_eda_dashboard.html`, section 2).

| Issue | Fix (`src/data.py`) | Rows |
|---|---|---|
| Claims duplicated under a new `claim_id` | Drop duplicates on all other columns | 604 |
| `customer_id` lost leading zeros | Left-pad to 8 characters | 98,110 motor (test 25,554) |
| Exposure recorded in days (e.g. 365) | Recompute from period dates | 934 motor, 656 buildings, 729 contents |
| `sum_insured` x1000 unit error (buildings > 5m, contents > 1m) | Divide by 1000 | 1,395 buildings, 1,402 contents (test 370, 360) |
| `construction_type` labels (case, `T`/`W`) | Normalise to four levels | 641,380 buildings (test 166,647) |
| `years_licensed` impossible for `driver_age` | Cap at age minus 17; flag kept | 2,962 motor (test 755) |

**Leakage.** Two columns are excluded by construction (`LEAKY_COLS`) and by test. `renewal_decision` exists only in training, and `declined_by_insurer` rows claim at about ten times the normal rate (the insurer declines because of claims). `customer_service_contacts` includes claim-related calls: rows with zero contacts have zero claims, and test values resemble the no-claim rows. `written_premium` is also excluded, by user decision, as an independent technical price is wanted. The diagnostic `L13_prem` scored CV 0.15293 against 0.15209 for `L6_lgbm`; it is kept only as evidence.

**Claim to row matching.** Claims link by `policy_id` only and a policy has several rows (years, mid-term changes). Each claim is attached to the row whose period contains `loss_date` (`attach_claims`). A test asserts every matched claim falls inside its row period.

**2024 immaturity.** Valued at 31 December 2024, December losses are only about 67 to 80 % reported, and 2024 is about 95 % complete by count (EDA section 4). Motor 2024 cost is down about 23 % against frequency down about 10 %, so most of the gap is severity from late large claims and open case estimates. About a quarter of 2024 claims are open. Raw 2024 incurred would under-price recent years.

**The suspect claim.** One open motor bodily-injury claim (`CL30973963`) carries an 11.9m case estimate, ten times the next largest. It is treated as a data error: valued at paid-to-date in the evaluation label, capped in the training label, and its excess is not spread into the large-loss loading.

## 3. Target construction (`src/target.py`)

1. **Development to ultimate.** A Bornhuetter-Ferguson IBNR. Each row receives the claims it is expected still to report, from a report-lag completeness curve and a late-claim severity relativity fitted on the mature years 2021 to 2022. Rows with no reported claims also receive IBNR, so nothing depends on a row's own noisy experience.
2. **Large-loss cap.** A per-line claim cap with the excess returned as a loading. The first choice was the 99.5th percentile (user card #5). Sensitivity runs favoured the 99th (CV +0.0004, holdout +0.0004), while the 99.9th was worse by 0.0009. The loading is segment-specific (motor: cover x age band; home: sum-insured quartile) and credibility-shrunk with K = 200. Result in `L19_segload`: motor CV +0.0006, buildings holdout +0.003.
3. **Trend to 2025.** The user chose the premium change (+3 to 6 %) as the inflation index. CV cannot judge trend because the label already contains it, so a forward level backtest was built (`src/backtest.py`, `scripts/backtest_level.py`): train on years before T with labels trended to T, predict T, and compare the total with developed actuals. Level error was 1.8 % for the premium index against 3.0 % for a claims-based trend, so the claims trend was rejected (`L8_trendB`).
4. **Maturity weights and exposure.** Rows are weighted by completeness, so 2024 counts for less. Exposure enters as a log offset with `new_business` as a feature, because only new-business rows claim more per exposure year (EDA 8a).

`y_eval` is fixed across experiments (uncapped, developed, premium-trended) so CV scores are comparable.

## 4. Modelling

**Path.** GLM baseline (one Tweedie GLM per line, banded factors, CV 0.1396) to GBM. The first LightGBM scored below the GLM on the leaderboard. Three diagnostics shaped everything after it:

1. **Start-level bug.** Supplying `init_score` disabled LightGBM's `boost_from_average`, so boosting started at 1 cost unit a year and the test mean was 10 to 20 % low. Fixed in `models.py`. A calibration test now requires total predicted within 5 % of actual.
2. **Drift under heavy regularisation.** The Tweedie GBM chased chance claim-free pockets: in-sample calibration fell from 0.90 to 0.70 in a one-fold diagnostic. Fix: large leaves, slow learning, rebalancing. Later, with monotone constraints, regularisation could be relaxed (`L20_mono_lowreg`, minimum leaf 2,000).
3. **History as a calendar proxy.** Cumulative prior-claims features grew with the calendar year (holdout fell and the test mean jumped in `G5_hist`). Fix: a two-year look-back window, and a test that the window means the same in every year including 2025.

**Feature stability screen.** For each categorical, level claim frequencies in 2021 to 2022 are correlated with 2023. `previous_insurer`, `branch_id`, `email_domain` and `campaign_code` show roughly zero stability. `make_model` (300 levels) scores 0.23. Removing them lifted CV from 0.15253 to 0.15631 (+0.0038) and the 2023 holdout from 0.0995 to 0.1041. Removing line-irrelevant fields (`L16_relevant`) was neutral.

**Model families.** Tweedie LightGBM per line; GLM plus boosting from the GLM (`B15`); Poisson x Gamma frequency-severity GBMs (`F17`); CatBoost (`A1`). Monotone constraints (excess and no-claims discount down; prior claims and sum insured up) cost nothing in CV (0.15756 against 0.15743) and are actuarially defensible. Training power 1.35 added +0.0003 (`P20_135`).

**Blend.** Per-line weights are chosen on CV and scored by nested CV, where weights are chosen out of fold. A blend of blends is refused. The best family differs by line: the GLM-based model carries the larger weight for motor (0.65 in `X1`), while the standalone GBM is stronger for buildings and contents, where interactions such as soil x dry summer matter.

**Credibility layer.** A Bühlmann experience adjustment plus a ridge segment layer (`src/credibility.py`). Its measured worth is small: +0.0001 nested in the blend (`X8` against `X9`), although an in-sample scan suggested up to +0.006. Kept as a marginal member and reported as such.

**What did not work.**

| Idea | Result | Why |
|---|---|---|
| Per-peril GBMs summed | 0.1615 vs 0.1616 (buildings -0.0013) | A single GBM already captures peril signal |
| PCA on weather (user question) | Neutral | Raw weather kept for interpretability |
| Stacking meta-learner | Nested 0.1584, same as grid blend | Simpler grid kept |
| XGBoost | 0.1607 vs 0.1628 | Weaker on every line |
| Richer claim history by peril | 0.1626 vs 0.1628 | Neutral |
| Address-level history | Rejected before modelling | Addresses are about 1:1 with policies |
| Optuna (12 trials) | 0.1639 vs 0.1634 | Within quick-screen noise |
| Cap at 99.9th percentile | CV -0.0009 | Rejected |

## 5. Validation and trust

- **Grouped 5-fold CV** by `policy_id` across all years (user card #4), so one policy never sits in both train and validation.
- **2023 holdout** (train 2021 to 2022), used as a second opinion.
- **CV to LB predictor** (`src/lbmodel.py`): corr ≈ 0.995 across submissions, noise ±0.002. A candidate whose predicted gain is below the noise is not submitted (`X7_blend4` predicted +0.0005; `X11_blend4` +0.0001). Auto-submit requires gain > 0, at least 5 % of |best|, and more than max(0.002, residual sd). A submission without explicit approval is refused, and a test proves it.
- **Approval gates.** Every submission needs user approval; the user's decisions (`docs/decisions.json`) set the structure.
- **Automated tests** (`tests/`): Tweedie D² equals sklearn's; each cleaning rule and claim de-duplication; customer join complete; leaky columns, ids, targets and `written_premium` absent from features, with identical train/test columns; history uses prior years only with a stable window including 2025; every claim inside its row period; development, loading and trend sane; models calibrated (total within 5 %) and freq x sev handles nil claims; submission format (626,639 rows, no negatives); submit gate and LB predictor; monotone constraints; credibility layers recover a known effect; level-probe fit. Main files: `tests/test_pipeline.py`, `tests/test_models.py`, `tests/test_credibility.py`, `tests/test_level_probe.py`.
- **Independent AI reviews** (`docs/ai_review_log.md`). Review 1 (a separate read-only agent on the modelling code) found three bugs (an auto-submit rule with no noise check, a Gamma severity crash on zero-cost claims, a claims trend fitted through one point), two risks now asserted in code (blend-part alignment, blend-of-blend flattering nested CV) and a minor leakage (section 8). Review 2 (the credibility layer, written by a smaller model to a spec) found a scale mismatch that biased the Bühlmann ratio 11 to 24 % low (mean A/P 0.80 before, 0.98 to 0.99 after) and a segment design that fitted stability-screen noise. Unit tests on synthetic data could not have caught either, so real-data sanity prints were added.

## 6. Use of AI

| Stage | What AI did | How it was verified |
|---|---|---|
| EDA | Built the dashboard, listed the planted errors, flagged both leakage columns and the 2024 immaturity | Register counts come from code; the user approved the fixes (card #11) |
| Feature engineering | Proposed experience rating, history windows, the stability screen | History tests; holdout and test-mean checks |
| Modelling | Proposed and ran the logged experiments; the user set direction through 12 decision cards | Pytest gate, 5-fold CV and 2023 holdout on every run; LB predictor |
| Diagnostics | Traced the start-level bug, drift and calendar proxy from LB and calibration anomalies | A regression test added for each |
| Code review | Separate agents reviewed code before first use | Each finding has a fix and, where possible, a test |

`docs/experiment_log.md` records proposer, verification and decision for every run; failed AI proposals (stacking, XGBoost, tuning, per-peril, richer history, claims trend) were rejected.

**Cases where AI output was wrong and caught:** the LightGBM start level (LB below GLM, test mean 10 to 20 % low); history features acting as a calendar proxy (holdout drop); the auto-submit rule that could pass a no-better model; Gamma severity crashing on nil claims; the credibility layer mixing raw and developed scales; a quick screen that overstated the CatBoost buildings gain, corrected by full CV; and `E3_dev`, voided because it ran with the start-level bug.

## 7. Results

| Step | Experiment | CV | Public LB |
|---|---|---|---|
| Constant rate x exposure | `E0_const` | 0.0892 | 0.0709 |
| Tweedie GLM, raw labels | `E1_glm` | 0.1396 | 0.1075 |
| + development, experience, trend, weights | `G6_trend` | 0.1477 | 0.1172 |
| + GBM boosting from GLM | `B6_boost` | 0.1515 | 0.1197 |
| Standalone regularised GBM | `L6_lgbm` | 0.1521 | 0.1235 |
| Per-line blend | `X1_blend` | 0.1552 | 0.1249 |
| De-noised, tuned, bagged 3-way blend | `X6_blend3bag` | 0.1585 | 0.1263 |
| 4-way blend + CatBoost + credibility | `X10_blend4` | 0.1593 (nested 0.1590) | **0.1275** |

Per-line CV for `X10`: motor 0.1104, buildings 0.1150, contents 0.1031. CV exceeds LB by a steady margin of about 0.03, which is why the predictor works. `E2_lgbm_bug` (LB 0.0985) is the buggy run and is excluded.

**Calibration evidence** (out-of-fold, `X6` blend, `docs/diagnostics_17oct.md` section 1). Calibration slope: motor 0.962, buildings 1.043, contents 0.999. Actual/predicted by year stays within 0.98 to 1.02 for motor and contents and 1.00 to 1.01 for buildings; by risk decile it is 0.95 to 1.14 (motor), 0.84 to 1.07 (buildings) and 0.91 to 1.11 (contents). Slope recalibration adds at most 0.0002 D², so the model is well calibrated.

**Plateau analysis** (`docs/diagnostics_17oct.md`).
- **Level sensitivity.** Scaling a line's predictions by 0.95 or 1.05 costs only 0.0003 to 0.0007 D² (motor 0.1099 at x1.00, 0.1092 at x0.95, 0.1096 at x1.05). CV cannot judge the 2025 level, hence the leaderboard level probe (results: [TBD: 18 Oct]). Implied 2025 pure premium against 2024 developed and trended actual: motor 0.948, buildings 0.974, contents 1.029 (partly weather: fewer hail and freezing days in 2025).
- **Ceiling.** A compound-Poisson simulation treating `X6` as the truth gives the D² a perfect model would reach: motor 0.0941 (`X6` scores 0.0910), buildings 0.0958 (0.1016), contents 0.0888 (0.0988). `X6` already scores about what noise allows, so noise is not the binding limit; more gain needs new signal.
- **Persistence.** 2023 to 2024 actual/predicted by 2021 to 2022 claim count: buildings 0.983 (0 claims, 93 % of rows), 1.200 (1), 1.519 (2+); contents 0.980, 1.277, 1.616; motor 0.999, 1.007, 1.079. Experience is somewhat under-used despite the history features.
- **Segment scan.** Largest out-of-fold deviations: motor `make_model` MM193 1.84x, MM186 1.68x, MM251 1.49x, MM124 1.41x (about 3,300 rows each); third-party-only motor 1.09x (88k rows); buildings bottom and top sum-insured decile 0.85x and 1.15x; contents flood zone 5 1.27x. The in-sample recoverable bound was 0.006, but nested validation delivered about +0.0001 to +0.0003 per step: `make_model` effects do not generalise across policy folds (a few large claims) and the prior-claim segments are too small a share of deviance.

Reading: further feature or algorithm work is inside the ±0.002 LB noise; the remaining effort belongs in level and adaptation.

## 8. Limitations and Part 2 plan

- **Mild leakage in target parameters.** Cap, loading, IBNR curves and trend are estimated on all training years, including CV validation rows. They are aggregate parameters, so the effect is judged small (review finding 6), but CV is slightly optimistic. The 2023 holdout and the LB are the independent checks.
- **Level uncertainty.** The 2025 level rests on one premium index with a backtest error of 1.8 %. 2025 weather is windier (wind index about 10 % above average), an effect on buildings that the trend does not capture.
- **Immature 2024.** Development curves assume 2021 to 2022 reporting behaviour persists.
- **Part 2 readiness.** The emerging market will differ in mix, level and possibly data quality. Checks to re-run: the data-quality register and cleaning tests; population stability of rating factors against the old market; the stability screen on new categorical fields; the report-lag completeness curve; the forward level backtest and level probe; calibration by line; and the CV-to-LB predictor refitted on new submissions. A read-only audit found about 90 hard-coded Part 1 assumptions, grouped as calendar, lines and perils, data-quality fixes and schema (`docs/part1_assumptions_inventory.md`). They now live in a `MarketSpec` dataclass (`src/market_spec.py`), selected by the environment variable `IAA_MARKET`, with a column alias map; the Part 1 defaults are proven byte-identical by `tests/test_golden_part1.py`. `scripts/new_market_report.py` produces a readiness report before any modelling: schema diff, data-quality register, drift (PSI), claims development and an auto-generated adaptation checklist. A dress rehearsal on a synthetic market (two lines, years shifted, currency x50, one renamed and one dropped column; `docs/rehearsal_report_b.md`) broke on three assumptions on the first run; these were fixed, and the second run completed the pipeline end to end with a quick-screen D² of 0.121 for motor and 0.113 for buildings. Part 2 results: [TBD].
