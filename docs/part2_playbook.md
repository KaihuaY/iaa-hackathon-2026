# Part 2 playbook (data released 23 Oct, about 1 week)

Run everything from the repo root in PowerShell. Log every decision as: observation, evidence file, action, residual risk.

## Hour 0: intake
```
kaggle competitions download -c <PART2_SLUG> -p data/part2 ; Expand-Archive data/part2/*.zip data/part2 -Force
python -P -E scripts/new_market_report.py data/part2 --ref data/raw/public --out docs/part2_readiness.md
```
- Never write into `data/raw/public`. Keep `data/` out of git.
- Read `docs/part2_readiness.md` in order: schema diff, DQ register, level vs Part 1, development, drift, checklist.
- Create `docs/part2_decisions.md` with first impressions: lines, years, currency scale, renames, immature years, test size.

## Hour 1: write the `part2` spec
Append to `src/market_spec.py` after `REHEARSAL`: `PART2 = register(dataclasses.replace(PART1, name="part2", competition="<slug>", data_dir="data/part2", ...))`. Set `$env:IAA_MARKET="part2"`. Caches and results go to `experiments/cache/part2` and `experiments/results/part2`.

| Field | Decide from the report |
|---|---|
| `competition`, `data_dir`, `expected_rows` | Slug from the Part 2 Kaggle URL; `data/part2`; `expected_rows=None` uses the sample file. |
| `lines` | Tables `policies_*`/`test_*` present. Drop absent lines from `segments`, `irrelevant`, `completeness_last_year`. |
| `train_years`, `test_year` | Section 3 portfolio table. |
| `valuation_date` | Latest claim date or report date in section 4; ask if unclear. |
| `mature_years`, `ref_sev_year`, `curve_years` | Section 4 `open_share` and lag p90/p99. Mature means open_share under 5%. Curve years are the two oldest. |
| `immature_years`, `completeness_last_year` | Years with open_share above 15% or a count drop. Take completeness from the lag curve, not from Part 1's .92/.939/.96. |
| `holdout_year`, `backtest_years` | Last mature year; the last two years. |
| `max_lag` | Section 4 `lag_max`. |
| `column_aliases` | Schema diff rename hints, as `{new: canonical Part 1 name}`. Check each by dtype and range. |
| `customer_id_width` | None unless ids are zero-padded strings (the report shows orphan joins). |
| `exposure_days_check`, tolerance | On only if the DQ register shows exposure in days. |
| `si_unit_error` | Set per line from the sum_insured distribution in local currency. Wrong thresholds divide real data. Use `{}` if no x1000 errors appear. |
| `min_licence_age` | Local legal age, if licence data exists. |
| `construction_map` | Add every label in the "inconsistent labels" row. |
| `leaky_cols` | Columns missing from test or tied to post-period events. Check names against the test schema. |
| `excluded_claim_statuses` | Status values in the claims table. |
| `suspect_claims` | Claim ids over 10x the next largest. Empty tuple if none. |
| `peril_min_claims` | Scale to claim count; pooled "other" must not swallow every peril. |
| `segments`, `segment_credibility_k` | Need the named columns to exist. Re-pick bands from the age and sum_insured distributions. |
| `noise_features`, `irrelevant`, `monotone` | **Re-derive with the stability screen; do not copy.** Start with `()`, `{}` and the actuarial-prior subset of `monotone` whose columns exist and have PSI under 0.1. |
| `tuned` | Scale `min_data_in_leaf` to row count (Part 1: 8000 on about 600k rows). |

## Hour 2: build and screen
```
$env:IAA_MARKET="part2"
python -P -E -c "import features; features.build_base(refresh=True)"
python -P -E scripts/run_experiment.py L17_tuned --quick
```
- Check calibration (predicted vs actual by line and year), zero share vs Part 1, and test-mean vs train-mean prediction.
- Hard-coded Part 1 items still open (rehearsal "not fixed" list, `docs/rehearsal_report.md`):
  - `src/credibility.py` lines 124, 198 use `make_model`/`previous_insurer`. Run corrections without `--levels`.
  - `src/credibility.py` 123-147 and 183-211 segment columns are no-ops if absent. Confirm they apply.
  - `src/market.py` column lists and `_NONNEG` use `driver_age`; the `[2021..2024]` years flag (line 502) is cosmetic.
  - `tests/test_pipeline.py` asserts `626_639` rows and Part 1 data. `submit.run_tests` runs the whole suite and will block `--submit`. Make those tests spec-aware, or skip them under `IAA_MARKET`, and record the change.
  - `experiments/log.csv`, `level_probe.csv`, `backtest_level.csv` and `backtest_level.py` are shared across markets. Copy Part 1 files aside first, or filter by market.

**Calendar-proxy rule.** The readiness report section 7 (and `calendar_proxy_features.csv`) lists numeric features whose yearly mean drifts by more than 1 sd or that are almost all zero in the first year (cumulative-exposure history counters are the usual case). Drop flagged features from any model with unbounded terms (smooth GLM, NN); trees tolerate them. Keep the forward (latest-year) holdout as the gate: random K-fold cannot see time drift.

## Day 1: first submission
1. Minimal chain: `python -P -E scripts/run_experiment.py B15_denoise_mm` (GLM plus GBM), then `L20_mono_lowreg`.
2. Write a small `part2` blend config in `experiments/configs.py`, `parts=[...]`.
3. Tests: `python -P -E -m pytest -q tests`.
4. Submit only with user approval: `python -P -E scripts/run_experiment.py <name> --submit --approved`. The limit is 5 per day.
5. Then the full chain, as time allows: `python -P -E scripts/reproduce.py --final <part2 blend> --plan`, then drop `--plan`. The X10 chain is B15_denoise_mm, L20_mono_lowreg(+_cred), F17_freqsev/F17_s101/F17_s202 (F17_bag), A1_catboost, X10_blend4. Re-derive its noise lists first.

## Days 2-6: adaptation loop
- Stability screen: re-derive `noise_features`, `irrelevant`, `monotone`. Log the evidence.
- Trend and level: `python -P -E scripts/backtest_level.py`, compare trend methods, choose, record the 2025-equivalent level risk.
- Level probe: `python -P -E scripts/level_probe.py <base> --dry-run`, then `--submit` with approval, then `--apply`.
- Blend with `run_experiment.py`, select with `python -P -E scripts/final_select.py --top 3 --dry-run`. Select the final on the Kaggle site by hand.
- After each experiment, add a row in `docs/experiment_log.md` (hypothesis, CV, holdout, decision). Add every AI-written change and the check that caught or cleared it to `docs/ai_review_log.md`.

## What the judges need
| Question | Narrative | Evidence |
|---|---|---|
| What changed | Schema, units, years, development, drift, level vs Part 1 | `docs/part2_readiness.md` |
| What we did | Each spec field and rule, with its reason | `src/market_spec.py` (PART2), `docs/part2_decisions.md`, `docs/experiment_log.md` |
| How we checked | AI outputs reviewed, tests, Part 1 regression | `docs/ai_review_log.md`, `tests/test_golden_part1.py` |
| Remaining uncertainty | Level (immature year, trend), thin segments, untested hard-coded items, LB-only evidence | The decisions log and `level_probe.csv`. State it as ranges. |
| Process | The rehearsal proved the pipeline on a synthetic market | `docs/rehearsal_report.md`, `docs/rehearsal_report_b.md` |

## Lessons from the Part 1 end-game (19 Oct debate — apply from day 1 in Part 2)
- **Leaderboard-derived factors are mostly noise unless the segment is large.** On a public subset of ~20–25 % of the
  test rows, the Tweedie-optimal multiplier of a whole line has sampling SD ≈ 0.05 and of a sum-insured quartile ≈ 0.10–
  0.13. Treat every probe reading with credibility Z = τ²/(τ² + σ²) (τ from year-to-year A/E variation, σ from a
  public-sized bootstrap of the OOF predictions — `docs/diagnostics_17oct.md` §10d) and never probe cells under ~10 %
  of a line. Read probes with the exact Tweedie response: ΔLB = −α(s^−0.5 − 1) − b(s^0.5 − 1), b ∝ Σμ^0.5.
- **Nested-CV gains of +0.0005 are invisible on a small public board** (P(≤ 0) ≈ 0.17): keep choosing models by
  nested CV and the forward holdout; use submissions only for the test-year level/structure.
- **Real sum insured.** Nominal SI as a feature plus a premium-index trend double-counts SI growth (+3 %/yr here);
  deflate SI to the target year (SI / (1+g)^(target − year)) or trend by a claims-cost index when it backtests better.
- **SI elasticity in the offset** (`target_by_line: {"buildings": {"si_offset_power": 0.7}}`) was the one mechanism
  that moved buildings CV (+0.0003–0.0005 per family); screen k ∈ {0.3, 0.5, 0.7} per home line on day 1.
