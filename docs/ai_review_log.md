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
