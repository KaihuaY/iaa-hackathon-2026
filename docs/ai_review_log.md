# AI review log

Independent reviews of AI-written code and analysis: what was checked, what was found, and what changed.
Complements `docs/experiment_log.md` (experiments) and the pytest suite (automated guardrails).

## 2026-10-17 · Review 1: new modelling code (frequency × severity, quick screen, LB predictor, trend backtest)

**How:** a separate AI agent did a read-only review of `src/models.py`, `src/pipeline.py`, `src/lbmodel.py`,
`src/target.py` and `scripts/run_experiment.py`. It looked for correctness bugs only. A human-directed fix and a test
followed for each finding.

| # | Finding | Severity | Fix | Test |
|---|---|---|---|---|
| 1 | 5 % auto-submit rule had no noise check; a tiny or ≤ 0 best score could let a no-better model auto-submit | Bug | Gain must be > 0, ≥ 5 % of \|best\|, and > max(0.002, residual sd) | `test_lb_predictor_and_submit_gate` |
| 2 | Blend parts aligned only by length | Risk | Assert row_id equality for OOF and test rows | (assert in `run_blend`) |
| 3 | Nested blend CV flattered if a part is itself a blend | Risk | Refuse blend-of-blend | (assert in `run_blend`) |
| 4 | Gamma severity crashes on zero-cost claims (count > 0, cost 0) | Bug (latent) | Severity fitted on cost > 0 only | `test_freqsev_calibrated_and_handles_zero_cost_claims` |
| 5 | Claims trend fitted through one point when backtesting to ≤ 2022 | Bug (latent) | Needs ≥ 2 years, else no trend | n/a |
| 6 | Target-construction parameters (cap, loading, IBNR, trend) estimated on all training years, including CV validation rows | Minor leakage | Documented; effect small (aggregate parameters, not row targets). Not changed. | n/a |

The reviewer confirmed as correct: Poisson objective override, severity start level, rebalancing, frequency early
stopping labels, loading only with a cap, the GroupKFold folds, LB-predictor anchoring, per-line trend handling, and
that nothing can be submitted without explicit approval or the 5 % rule.

## 2026-10-17 · Earlier self-caught issues (found from diagnostics, before this log existed)

| Issue | How it was caught | Fix |
|---|---|---|
| LightGBM started boosting from rate 1 CU/yr (init_score disables boost_from_average) | LB below GLM, test mean 10–20 % low | Start from the average rate; calibration test |
| Tweedie GBM drifted low by chasing chance claim-free pockets | One-fold diagnostic: in-sample calibration 0.90 → 0.70 | Large leaves, slow learning, rebalancing |
| Cumulative claim history acted as a calendar-year proxy | Holdout drop and test-mean jump in G5 | 2-year look-back window; stability test |
| Noise features (`previous_insurer` etc.) ranked as important | Year-to-year stability screen | Features removed: CV +0.004 |

## 2026-10-17 · Review 2: credibility correction layer (code written by a smaller model to a spec; reviewed before first run)

| # | Finding | Severity | Fix | Evidence |
|---|---|---|---|---|
| 1 | Bühlmann actual `A` was raw reported claim cost while expected `P` was on the developed, trended scale, so `A/P` was biased 11–24 % low and the test rows (longest history) would have been pushed down hardest | Bug | `A` built from the evaluation target (same scale), then rescaled so the population ratio is 1 | Sanity print of mean `A/P`: 0.80 before, 0.98–0.99 after |
| 2 | Segment design included ~80 `make_model` / `previous_insurer` level columns the stability screen had classed as noise; they forced the ridge penalty to its maximum and shrank the real indicators too | Design | Structural indicators only by default; levels opt-in (`--levels`) | Nested D² falls when levels are included |
| 3 | Unit tests passed on synthetic data with no trend, so neither issue could be caught by tests alone | Process | Real-data sanity prints added to the run | — |

Outcome: correction layer worth +0.0001 nested (in-sample scan had suggested up to +0.006; the rare-segment deviations are
mostly policy-level noise). Kept as a marginal blend member.

## 2026-10-18 · Review 3: smooth log-GLM extrapolation caught by the holdout and test-mean checks

| # | Finding | Severity | How caught | Fix |
|---|---|---|---|---|
| 1 | The new smooth GLM (log1p + squared terms + interactions, written by a smaller model to spec) scored a normal 5-fold CV (buildings 0.1117) but a 2023 time-holdout D² of −0.66 and a 2025 test-prediction mean of 24,016 vs ~400: quadratic log terms and products extrapolate explosively outside the fit range | Bug (would have corrupted any blend containing it) | Not by CV (out-of-fold rows are interpolation). By the 2023 holdout printed alongside every run and by the per-line test-mean print | Winsorise inputs to the fit range, clip standardised terms, bound the linear predictor; plus a new pipeline guard: any model whose test pure premium per exposure is outside 0.5–2× its OOF value is marked unsafe, cannot be blended, cannot be written to a submission |

Lesson for the write-up: nested CV is necessary but not sufficient; a forward holdout and a test-distribution sanity check
are the guardrails that catch extrapolation, and they must be automatic because a blend would otherwise hide a broken part
behind a small weight on one line.

## 2026-10-18 · Review 4: the smooth GLM still failed the 2023 holdout after the extrapolation fix — root cause

After Review 3's fix the smooth GLM (G14) and smooth-boost (B19) scored sane 5-fold CVs (0.1562 / 0.1570), sane test
means (test/OOF 0.93–1.04) and sane test tails (max/median comparable with CatBoost), yet the buildings 2023 holdout was
still −0.59. A targeted diagnostic (script written by a smaller model to a spec; findings verified against the feature
ranges in the real training and test sets) found:

| # | Finding | Evidence | Decision |
|---|---|---|---|
| 1 | Not an outlier problem: the 20 worst rows carry 8 % of the deviance; removing rows with pred > 50× median makes the holdout *worse* (−0.61) | deviance decomposition on the reproduced holdout (−0.587) | Clipping/winsorising cannot fix it |
| 2 | The cumulative-exposure history counters (`cust_expo`, `pol_expo`, `cust_{motor,buildings,contents}_expo`) are calendar-time proxies: mean 0.00 (2021), 1.67 (2022), 2.99 (2023), 2.95 (2024). Trained on ≤2022 only, every 2023 row sits in a region of the quadratic terms the fit never saw; the five largest GLM coefficients are their linear/squared terms (±1.1 to ±3.8, offsetting each other) | 55–69 % of 2023 rows outside the ≤2022 range; dropping the five counters moves the holdout from −0.587 to **+0.108** (≈ the 5-fold CV of 0.111) | Smooth GLM family drops the counters (`EXPO_COUNTERS`); trees keep them (they map out-of-range values to the last leaf; tree holdouts are +0.06–0.07) |
| 3 | The 2025 test set is **in range**: 0 % of test rows above the training max on any history feature; test mean `cust_expo` 2.95 = the 2023–24 level, which is half the training rows. So the test predictions of G14/B19 were not actually extrapolating — the failure is specific to the ≤2022-only fit | feature-range check on the full training vs test matrices | A part must still pass the time-shift test to be in the final blend: B19 is replaced by B20 (no counters), G14 had zero blend weight and is dropped |
| 4 | Bands-mode GLM on the same holdout: +0.085 — quantile banding is inherently shift-robust, which is why the original GLM base never showed the problem | — | Documented as a design trade-off: smooth terms gain CV (0.1562 vs 0.1522) but need shift-safe inputs |

Lesson: random K-fold CV cannot see a feature that drifts with calendar time; only the forward holdout can. Any feature
built from cumulative history must be either windowed and saturating (as the 2-year window is from 2023 on) or excluded
from models with unbounded functional forms.

## 2026-10-18 · Review 5: verification of the round-5 code written by smaller models (no defects found, one risk noted)

| # | Item | Verification | Outcome |
|---|---|---|---|
| 1 | Pipeline refactor for pooled lines (`_units/_load_unit/_fit_unit`; fold, holdout and full-fit logic unified; loading applied after the raw fit) | Golden-hash test on the Part 1 pipeline + 91-test suite after merge | Byte-identical outputs for every config without `pool`; merged |
| 2 | Neural residual learner (`NNBoost`): Tweedie deviance formula, exposure offset through the GLM margin, ±3 residual clamp, early stopping, rebalancing | Line-by-line read against `LGBMTweedie`/`GLMBoost`; synthetic tests (interaction recovery, calibration, unseen category); real quick screen (0.1623, early stop at epoch 1–2) | Correct; weak as a model — retune N2 queued, rejection if < the LightGBM residual |
| 3 | Covariate-shift weights hook (`_fit_predict`): weights computed against the rows each fit predicts; mean 1; no-op without the key | Synthetic tests (same-distribution ≈ 1, shifted target, no-op); golden hash | Correct; **judge only on the 2023 holdout** (CV folds are training-distributed, so CV cannot score it) |
| 4 | Risk: `_concat_frames` drops columns not shared by the pooled lines | Noted at review; checked on the first pooled quick screen | — |
| 5 | `largeloss._num` (smaller model) used `np.issubdtype` on pandas Categorical columns → `TypeError` on the first real run (P27) | Caught by the run, not by the synthetic tests (which used object columns) | Fixed with `pd.api.types.is_numeric_dtype`; regression test with a real `Categorical` added |
