# Part 2 decisions: the emerging market spec and the first submission (27 Oct)

Market: `data/part2/emerging` (only the emerging rows are scored). Spec: `PART2` in `src/market_spec.py`; select with
`$env:IAA_MARKET="part2"`. Caches go to `experiments/cache/part2`, results to `experiments/results/part2`, files to
`submissions/part2/`. Evidence: `docs/part2_readiness.md` (sections 1-5) and the checks below.

## 1. First impressions

- Three lines as in Part 1. History 2023-2024 only (two years; Part 1 had four). Test year 2025.
- History rows: motor 21,680 + 25,603, buildings 12,580 + 14,867, contents 15,834 + 18,732. Test rows: 90,521 / 52,295 / 66,247
  (209,063 = `sample_submission`). The test book is 3.5x the 2024 book.
- **Test is 76 % new business**, against 12-15 % in history. In history, new-business rows start mid-year (7.5 % start in
  January, mean exposure 0.47). In test, 89-93 % of rows start in January 2025 (renewals on 1 January; new business spread across the month, none on the 1st; mean exposure 0.88-0.92). Only 24 % of test
  policy_ids appear in history.
- Schema: motor has `driver_age_band` instead of `driver_age`, and has no `annual_mileage` or telematics columns. Home lines have
  `policyholder_age_band`. Customers have no `date_of_birth` or `credit_band`, and have a new `gender` column. The bands are
  categorical, not renamed ages, so there are no `column_aliases`.
- Level vs Part 1: motor frequency x1.29 and severity x0.58; contents pure premium x0.80; buildings x0.98.

## 2. PART2 field table

| Field | Value | Evidence | Reason |
|---|---|---|---|
| competition | `iaa-ai-assisted-hackathon-2026-part-2` | Kaggle URL | Slug |
| data_dir | `data/part2/emerging` | Download layout | `developed/` is byte-identical to Part 1 |
| lines | motor, buildings, contents (inherited) | `policies_*`/`test_*` present | All three lines present |
| train_years / test_year | (2023, 2024) / 2025 | Readiness §3 | Only years in history |
| valuation_date | 2024-12-31 | Maximum report_date is 2024-12-30 | Day after the last report |
| curve_years | (2023,) | Readiness §4: 2023 open share 4 % (bld 4.1 %, cnt 0 %, motor 3.6 %) | The only mature year. Its lags are right-censored at 366-730 days, so the bias is small (2023 lag p99 is 189-344 days) |
| mature_years / ref_sev_year | (2023,) / 2023 | Same | Open share under 5 % |
| immature_years | (2024,) | 2024 open share 20-25 % (contents 12 %) | Above 15 % |
| completeness_last_year | motor 0.923, buildings 0.944, contents 0.961 | §4 below | Computed from the curve. Part 1's values are not reused |
| holdout_year / backtest_years | 2024 / (2024,) | Two-year history | Forward test: fit on 2023, score 2024. This is the only forward split |
| max_lag | 730 (inherited) | lag_max is 653 days | Covers every observed lag |
| column_aliases | {} | Bands are strings | Not a rename. They are kept as new categoricals |
| customer_id_width | 8 (inherited) | All 135,856 ids have 8 characters | No padding errors found (0 rows), so the rule is kept as a guard |
| exposure_days_check | True (inherited) | 47 / 33 / 34 rows record exposure in days | Repaired from dates. 188 rows with exposure 1.0139 (370/365) agree with their dates and are left alone |
| si_unit_error | buildings 5e6, contents 1e6 (inherited) | 53 / 57 history rows and 104 / 135 test rows are above the thresholds. Divided by 1000, they land at the median | The thresholds suit this currency scale |
| min_licence_age | 17 (inherited) | No numeric age | The repair is skipped (logged). Check: 0 rows have years_licensed > band upper bound - 17 |
| construction_map | Part 1 map + brk->brick, stn->stone, wod->timber, mix->non_standard | Codes BRK/WOD/MIX/STN. `clean_policies` strips and lower-cases before mapping | Gives the Part 1 canonical labels. 0 NaN after mapping (test asserts this) |
| normalise_labels (new field) | ("occupation",) | 79 % of rows have case or whitespace variants ('Retired', 'retired', 'Retired ') | 47,924 customer rows merged to the most frequent spelling. Part 1 keeps `()` and is unchanged |
| leaky_cols | renewal_decision, customer_service_contacts (inherited) | renewal_decision is absent from test. Contacts are post-inception | Same rule as Part 1 |
| excluded_claim_statuses | repudiated (inherited) | Status values: closed 8,082, open 1,056, repudiated 120 | Same as Part 1 |
| suspect_claims | () | Largest claims: 361k, 340k, 304k (buildings fire) | No claim is 10x the next one |
| peril_min_claims | 60 | Claim counts: buildings subsidence 52 / fire 70, contents flood 86 | Pools only buildings subsidence into "other". "Other" does not swallow the perils |
| segment_credibility_k | 200 (inherited) | Credibility is per claim count, so it is scale-free | Kept. Large claims are rare (37 motor claims above the 99 % cap in 2023) |
| segments | motor: quartiles of `vehicle_value_group` (prefix VVG); home: SI quartiles (inherited) | There is no numeric driver age, and `_segments` "cross" needs a numeric column. 2023 excess/capped by VVG quartile: 0.133 / 0.116 / 0.073 / 0.032 | Uses an existing kind (no new code). Monotone and plausible: bodily injury does not scale with vehicle value |
| noise_features | () | Not yet screened | Re-derive with the stability screen (playbook) |
| irrelevant | Part 1 lists (inherited) | Every listed column exists in Part 2 (addresses table identical, document_preference present) | Domain lists. A stability re-screen is still open |
| monotone | Part 1 directions minus annual_mileage | Column absent | Other directions still apply |
| tuned | min_data_in_leaf 300, feature_fraction 0.5 | Quick screen L17: leaf 150 / 300 / 600 gives 0.1018 / 0.1021 / 0.0981 | Part 1 used 8,000 on 0.7-1.0M rows per line. Part 2 has 27-47k rows (about 1/20) |
| expected_rows | None | sample_submission has 209,063 rows | Validation checks against the sample |

## 3. Data-quality findings and actions (from `data.build_all` log)

| Issue | Rows | Action |
|---|---|---|
| Exposure recorded in days | motor 47, buildings 33, contents 34 (test 0) | Recomputed from dates (existing rule) |
| sum_insured x1000 | buildings 53 (test 104), contents 57 (test 135) | Divided by 1000 (existing rule, same thresholds) |
| construction_type codes BRK/WOD/MIX/STN | All buildings rows | Mapped to Part 1 labels (spec map) |
| occupation case/whitespace variants | 47,924 customer rows | New spec-gated `data.normalise_labels` |
| Exposure 1.0139 (370 days) | 188 | None: consistent with dates |
| Claims with loss dates 2021-22 (before any policy period) | 34 (plus 2 from 2023) that attach to no row | None: they match no row, so they do not enter the target |
| Duplicate claims under a new id | 0 | Rule kept |
| Missing values about 12 % in many columns (occupation, vehicle_age, engine_size_cc, smoke_alarms, ...) | Readiness §2 | LightGBM handles NaN. The GLM uses a `_missing` level or indicator |
| smoke_alarms None | 3,266 history rows | Mapped to NaN (existing rule) |

## 4. Completeness of 2024 claims (method)

`completeness_last_year` is used only by the Bühlmann credibility layer, where 2024 claim counts are divided by it.
Method, mirroring Part 1:
1. Build the reporting-lag curve from loss-year 2023 claims (`target.completeness_curve`, curve_years=(2023,), using the
   cleaned, attached `build_base()` claims).
2. For each 2024 row, average the curve over the days of its period at their age on 2024-12-31 (`target._row_integral` /
   `_daily`, as `prepare_target` computes `completeness`).
3. Take the **unweighted mean over 2024 rows**. Check: the same computation under PART1 gives 0.9195 / 0.9388 / 0.9596.
   That reproduces Part 1's 0.920 / 0.939 / 0.960 exactly. The exposure-weighted mean gives 0.932 / 0.950 / 0.970, so the
   Part 1 values were unweighted.

Part 2 results: motor 0.9230, buildings 0.9442, contents 0.9610 (exposure-weighted 0.935 / 0.955 / 0.971). Independent
check: 2024 / 2023 reported claim frequency is 0.921 / 0.973 / 0.974. Motor agrees. Home lines report fast. Curve values at
30 / 90 / 365 days: motor 0.84 / 0.95 / 0.99, buildings 0.88 / 0.97 / 0.996, contents 0.94 / 0.99 / 0.996.
`tests/test_market_spec.py::test_part2_completeness_matches_lag_curve` recomputes the values from the cache.

## 5. Code changes (Part 1 behaviour unchanged; golden test passes)

| File | Change | Why |
|---|---|---|
| `src/market_spec.py` | New field `normalise_labels=()`. `PART2` registered after REHEARSAL. Guarded `market_spec_pool` import hook at the end | Spec for the new market. The pooled spec is registered by another module |
| `src/data.py` | New `normalise_labels()`, applied to customers in `build_all` for the spec's columns | occupation variants. An empty tuple returns the same frame (Part 1 identical) |
| `src/market.py` | `driver_age_band`, `policyholder_age_band`, `gender` added to `_STR` (expected categorical) | Readiness type checks. `_NONNEG` already skips absent columns |
| `src/submit.py` | `write()`: other markets write to `submissions/<market>/<UTC>Z_<name>.csv` (Part 1 path unchanged). `run_tests()` uses a per-market pass marker. `_code_state` also watches `configs_part2.py` | Keeps files apart. **Bug fixed:** a part2 test pass, with the Part 1 data tests skipped, could otherwise satisfy the Part 1 submit gate |
| `src/features.py`, `src/target.py` | No change | They already skip absent columns (`customer_age`, licence repair, engineered ratios) |
| `tests/test_pipeline.py` | `test_submission_validation` is spec-aware and no longer marked `part1_data` | Runs under every market: 626,639 rows for Part 1, the sample length otherwise |
| `tests/test_market_spec.py` | 4 tests: PART2 fields, `normalise_labels`, real-data cleaning, completeness recompute | Guards the decisions above |
| `experiments/configs_part2.py` | New Part 2 configs (`P2_*`). They read `REGISTRY["part2"]` explicitly | Leaf sizes scaled by about 1/20. Part 1-only columns removed from interactions |
| `experiments/configs.py` | 6-line hook at the end merges `configs_part2.EXPERIMENTS` and `configs_pool.EXPERIMENTS` (if present) | Other sessions' uncommitted edits left intact |

## 6. Screen results

Quick screen = one 80/20 policy-grouped split (D², uncapped developed labels). Full = 5-fold GroupKFold by policy.
Holdout = fit on 2023, score 2024. On a single split, buildings (about 1,000 claims) is very noisy: the constant scores -0.049.

| Config | Quick motor / bld / cnt / overall | Full CV motor / bld / cnt / overall | 2024 holdout motor / bld / cnt | Note |
|---|---|---|---|---|
| P2_E0 constant | .050 / -.049 / .037 / .068 | .037 / .021 / .040 / .099 | .089 / -.038 / .051 | Reference |
| P2_B15 (GLM a=1e-4 + boost) | -.005 / -.484 / -.240 / -.158 | | | GLM overfit, see below |
| P2_G banded GLM a=0.001 / 0.01 / 0.1 | overall -.169 / -.159 / -.107 | | | |
| GLM alpha sweep (0.1-100, scratch) | bands best at a=10: .062 / -.081 / .038. Smooth best at a=10-30: .053-.058 / -.010-.002 / .102-.085 | | | Chosen alpha: 10 |
| P2_B15_a10 | .062 / -.052 / .095 / .085 | | | |
| P2_L17 leaf 150 / 300 / 600 | .1018 / .1021 / .0981 overall | | | Leaf 300 goes into the spec |
| P2_L20 (leaf 100, monotone, segment loading) | .072 / -.007 / .107 / .1058 | | | |
| P2_L20_nohist | .068 / -.025 / .086 / .0936 | | | Keep experience rating |
| P2_L20_nan1 (2023 history unknown) | .067 / -.021 / .113 / .1005 | | | Rejected |
| P2_P34_si00 / si05 | .1001 / .1059 overall (bld -.020 / -.010, cnt .113 / .125) | | | SI^0.5 helps both home lines |
| P2_B27 smooth (a=1e-4) si00 / si05 | -.047 / -.049 overall | | | GLM overfit |
| P2_B27_a10_si00 / si05 | .1054 / .1067 overall | | | |
| P2_F22 / F22_si05 | .1064 / .1092 overall | | | |
| P2_L20_si05 | .072 / .003 / .119 / .1116 | .0781 / .0662 / .0918 / .1406 | .1201 / .0087 / .0771 | Blend part |
| P2_F22_si05 | | .0751 / .0706 / .0926 / .1411 | .0924 / -.0081 / .0408 | Blend part. Buildings holdout negative |
| P2_B27_a10_si05 | | .0828 / .0772 / .1052 / .1489 | .1145 / .0379 / .0938 | Best single model. The boost often stops at 1-9 rounds, so the smooth GLM does the work |
| **P2_X3_blend** | | .0854 / .0878 / .1065 / **.1537** (nested .0840 / .0842 / .1049 / **.1517**) | n/a | First submission candidate |

Blend weights (L20 / F22 / B27): motor .20 / .28 / .52, buildings 0 / .45 / .55, contents 0 / .25 / .75.

Root cause of the GLM failure: with ~400 one-hot/band parameters and 1-7k claims per line, ridge alpha 1e-4 (Part 1)
gives fitted rates from 2.7 to 12,000 on buildings (median 175, mean 400). The Tweedie deviance per row is O(sqrt(cost)),
so a useful ridge is O(1-30). A smaller model's defaults would not have shown this. The quick screen exposed it at once.

### Diagnostics of the candidates (full run)

| Model | Line | Calib (OOF/actual) | Train PP (trended) | Test PP | Test/train | Unsafe | NB share test / train | NB relativity OOF / test | NB actual (y/expo) |
|---|---|---|---|---|---|---|---|---|---|
| P2_X3_blend | motor | 0.981 | 484.6 | 544.7 | 1.124 | no | .760 / .138 | 1.283 / 1.263 | 1.603 |
| | buildings | 1.033 | 459.7 | 509.7 | 1.109 | no | .759 / .138 | 1.050 / 1.130 | 0.849 |
| | contents | 0.980 | 97.2 | 108.7 | 1.118 | no | .761 / .136 | 1.108 / 1.149 | 1.307 |
| P2_B27_a10_si05 | motor / bld / cnt | 0.983 / 1.036 / 0.979 | | 535.9 / 501.6 / 108.5 | 1.106 / 1.091 / 1.116 | no | | 1.322 / 1.045 / 1.084 (OOF) | |

Test/train of about 1.11 comes mostly from the new-business mix. For motor: (.76 x 1.26 + .24) / (.14 x 1.26 + .86) = 1.13.
Premium-index trend factors (`trend="premium"`): motor 2023 1.076 / 2024 1.081, buildings 1.125 / 1.058, contents 1.110 / 1.049.

OOF actual/expected by year x new business (blend): motor 2023 renewal 0.94, NB 1.63, 2024 renewal 1.05, NB 0.99.
Buildings: 2023 1.20 / 1.08, 2024 0.76 / 0.60 (the 2023 fire year). Contents: 1.08 / 1.01 / 0.94 / 1.31.

## 7. First submission (not uploaded; needs user approval)

`submissions/part2/20261027_1559Z_P2_X3_blend.csv` (written by `submit.write`, validated against sample_submission:
209,063 rows, identical row_id order, all finite and >= 0).
sha256 `20d27bf8c64cf5268b9639866f2d9162cfee431979463aca6df5a81129058561`.

| Line | Mean prediction per row | Predicted PP (per exposure) | 2024 reported PP (raw, undeveloped, uncapped) | Ratio |
|---|---|---|---|---|
| motor | 475.39 | 544.7 | 382.6 | 1.42 |
| buildings | 461.73 | 509.7 | 306.7 | 1.66 |
| contents | 96.81 | 108.7 | 86.8 | 1.25 |

The ratios to raw 2024 are large. They are the product of development (1/0.92-0.96 plus IBNER on 20 % open claims),
the 2023-24 average instead of 2024 alone (buildings 2023 was 514 vs 2024 307, a fire year), the premium trend
(+5-12 %), and the new-business mix (+11-13 %). Against the developed and trended history mean the ratios are 1.11-1.12.

## 8. Open risks

1. **New business, 76 % of test.** History new business is mid-year starters with half-year exposure. Test new business is a
   January 2025 bulk book. The model's new-business relativity (motor 1.26, bld 1.13, cnt 1.15 in test) is learned from about 2.6-3.9k
   rows per year and swings by year (motor NB A/E 1.63 in 2023, 0.99 in 2024). This is the largest level lever: ±20 % on
   motor. Candidate for a whole-line level probe once approved. Do not tune it on the small public board without credibility
   weighting (playbook lessons).
2. **Two-year history.** One mature year. There is no trend fit (`claims` trend needs two mature years), so the premium index
   is the only trend. That index mixes in the new-business price mix: test written premium is up 11 % for motor.
3. **Buildings fire volatility.** 2023 fire cost 3.2M vs 1.1M in 2024. The large-loss loading (1.161) comes from 2023 only.
   The buildings holdout is near zero for every model (best 0.038) and the 2024 A/E is 0.76.
4. **Segments changed.** Motor large-loss segments now use vehicle value group, not cover x age. It is untested on the leaderboard.
5. **Experience rating.** History features are 0 for all 2023 rows (first year). Test rows have up to 2 years of history.
   The 2024 history is immature (playbook: biased low). No-history and first-year-unknown variants screened worse.
6. **Not re-derived yet:** noise_features, irrelevant, monotone (stability screen), calendar-proxy rule (readiness §7), the
   `expenses` and `reference_policies` tables, and `gender` / band leakage review (gender is a plain policyholder attribute).
7. The `experiments/log.csv` column `holdout_2023` holds the 2024 holdout for part2 rows (explog column name is fixed).
