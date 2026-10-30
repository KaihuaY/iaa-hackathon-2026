# Part 2 feature hunt: structure in the emerging data that the model may miss (agent H1, 28 Oct)

Reference model: `R2_P34_hn` (pooled P34, all round-2 fixes). Its 5-fold OOF on the **emerging** history rows (row_id
without the "D" prefix) is the "expected" E. The stage-2 results are repeated on the final blend `R2_FINAL`.
Code: `scripts/part2_feature_hunt.py`, sections `inventory scan nbint ids shift cross target stage2 forward overall
wxshift factors screen`. `--ref R2_FINAL` switches the base. Logs: `experiments/logs/hunt_*.log`. CSVs:
`experiments/logs/part2_feature_hunt/` (and `…/R2_FINAL/`).

**How to read the numbers.** The level is settled by the public probes, so everything here is about **shape**.

- **The base.** E is rescaled inside each (year × new-business) cell so that the cell's a/e is 1 (`E_cell`).
  A level's a/e = Σ y_eval / Σ E_cell.
- **Noise of one level.** z = log(a/e) / √((1 + cv_sev²) / claims), with cv_sev = 2.18 (motor), 3.14 (buildings),
  1.69 (contents). At 150 claims a level's a/e has SD ≈ 0.20 / 0.27 / 0.16. So the "a/e outside 0.8-1.25 on
  ≥ 100 claims" flag is mostly noise: 80 of 839 such levels fall outside, but only 7 have |z| > 2 (max 2.76).
- **Per-feature statistics.**
  - chi²/df = Σz² / (levels − 1). Random features give 0.4-2.2. Of the 225 used feature × line pairs, 14 exceed 2.2;
    chance alone gives about 11.
  - `year_corr` = claim-weighted correlation of level log a/e between 2023 and 2024. Random features fall within ±0.4.
  - `lift` = out-of-fold (5 policy-grouped folds) D² gain, in basis points (bp, ×1e-4), of credibility-weighted level
    factors on top of E_cell.
- **Nulls for the lift.** Random 10/40-level features: motor -12 ± 11 bp, buildings +5 ± 17, contents -4 ± 10. Random
  2-level features: motor -3.6 ± 4.1, contents -1.2 ± 7.7, buildings -2.7 ± 23.
- **Starting point.** Emerging OOF D² of E_cell: motor 0.0989, buildings 0.1305, contents 0.1230; overall 0.1767
  (R2_P34_hn) and 0.1782 (R2_FINAL).

## Ranked findings

| # | Finding | Evidence (emerging OOF, E_cell base) | Expected value | Recommended action |
|---|---|---|---|---|
| 1 | **The pooled model applies developed-market relativities where emerging ones differ, the same way in both emerging years.** Robust core: **motor gender** (frequency effect). Supporting: motor fuel_type, contents works_from_home. Fragile (severity-only): contents gender, buildings trees_within_5m | **Motor gender:** M 1.068 / F 0.925 (3,943 / 3,105 claims). 2023 1.061 / 0.934, 2024 1.074 / 0.917. NB 1.081 / 0.911, renewals 1.066 / 0.927. **Frequency** a/e 1.047 / 0.946, z ≈ 4.2. Developed rows have no gender, so the pooled P34 (min leaf 2,000) learns it from 4.6 % of its rows. **Motor fuel:** petrol 0.924 (3,738 claims), diesel 1.081 (2,091), hybrid 1.142 (1,084); year_corr 0.83. The same model's developed-row fuel a/e is 0.99-1.03. Fuel is partly severity: 4 of the 5 largest emerging motor claims are diesel. **Contents works_from_home:** True 1.217 (266 claims; 1.255 / 1.178), frequency a/e 1.067; developed 0.969. **Contents gender:** F 1.106 / M 0.886 (518 / 480 claims, both years), but frequency a/e 1.008 / 0.992, so severity only. **Buildings trees:** True 1.521 (123 claims; 1.664 / 1.300), frequency a/e 0.999, capped-99 % a/e 1.286 (severity; subsidence 10 of 123 vs 42 of 933 claims); developed 1.008. **Stage-2 GLM on the R2_FINAL OOF** (honest 5-fold, 3 shuffles, level-neutral per cell), alpha 3: motor +11.5 bp (random null -7.2), contents +24.9 (-3.1), buildings +34.2 (-2.9; buildings nulls swing ±25). | **Overall emerging D² +21.1 bp** (full list, alpha 3; an upper bound because the list was chosen on both years). +13.5 bp at alpha 10. **+4.5 bp for motor gender + fuel alone.** Year splits (fit on one year, score the other): motor +8.0 / +12.1, buildings +54.9 / +44.0, contents +40.5 / +32.7. Selection made on one year only keeps just motor gender (+5.3) and contents works_from_home (+17.8 with basement) | **Top recommendation: an emerging stage-2 relativity layer on the final file** (§8). No new pooled training needed. Factors are ready in `experiments/logs/part2_feature_hunt/R2_FINAL/stage2_factors.csv`. Ship the motor core (`core_a3`), or `full_a10` if a paired public reading agrees |
| 2 | **2025 weather is shifted** (colder, drier, less hail). Dropping the weather columns is **not** a confirmed gain | PSI test vs 2024: dry_summer 0.95, mean_temperature 0.86, freezing 0.81, hail 0.73, wildfire 0.48, rainfall 0.46. Means 2023 / 2024 / 2025: freezing 21.0 / 17.5 / 28.9, hail 13.9 / 17.1 / 11.4, temperature 15.3 / 15.8 / 14.2. All inside historical ranges. 100 % of test addresses have 2025 weather, and it is in the test matrix. Per address, hail is mostly year noise (year-to-year correlation 0.20-0.28); freezing (0.70-0.90) and rainfall (0.85-0.95) are mostly geography. Weather is about 9 % of buildings split gain. In the emerging-only P34, the 2025 weather moves test predictions by ×0.997 / 1.022 / 0.985 (motor / bld / cnt) against the same addresses' 2024 weather; row-level p5-p95 0.92-1.07 / 0.94-1.14 / 0.88-1.09 | Quick screen, all 7 weather columns dropped, over 8 split seeds: overall **+16.2 ± 14.5 bp** (SE), buildings +55 ± 42, contents -28 ± 31, motor +9 ± 10. The first 3 seeds gave +42; the next 5 gave +1 | No action now. If a pooled slot is free, `R2_P34_hn` + drop the 7 WEATHER columns is the one weather variant worth a 5-fold CV (low priority) |
| 3 | `customer_service_contacts` is a claim-driven leak in history and is **absent** from test as a signal, so it is correctly excluded | History a/e: motor ≥ 3 contacts 4.55 (4,713 claims), ≤ 1 contact 0.24. History rows with claims average 3.4 contacts, no-claim rows 0.42. Test: Poisson(0.40), flat by NB (0.396 / 0.401), the same as history no-claim rows. 3+ contacts on 0.8 % of test rows vs 9.6 % of history rows | 0 (it would break the model on test). Its in-sample "lift" is +3,500 bp | Keep it in `leaky_cols` |
| 4 | Test NB looks like history NB on every rating attribute. It differs only in timing | PSI(test NB vs 2024 history NB) ≤ 0.01 on driver/policyholder age band, years_licensed, ncb_years, discount_applied, marketing_segment, campaign_code, channel, payment_method, days_quote_to_inception (means 13.2 / 13.2 days). On the columns that separate history NB from renewals (age band, years_licensed, ncb_years, discount_applied, marketing_segment), PSI(test NB vs history renewals) is 0.11-0.32. Channel, payment and quote days do not separate NB from renewals in either year. Differences: start_month (PSI 3.6-4.0; 88-91 % January, median start day 9) and exposure (0.88-0.92 vs 0.47-0.48) | 0 for shape. The NB **level** is the known issue, already probed | None |
| 5 | NB × feature interactions (lines_held, voluntary_excess, licence_points) are a 2023-only artefact | Motor NB lines_held = 2: a/e 1.607 on 147 claims (2023 2.217, 2024 0.919). voluntary_excess 500: 1.836 (2.673 / 0.982). licence_points 3: 1.854 (2.456 / 1.109). These are the known 2023 NB anomaly (raw NB a/e 1.66 in 2023, 0.94 in 2024) | 0 | Do not add NB interactions |
| 6 | No id carries information | NB = exactly "policy_id not in history": 68,823 / 39,716 / 50,441 NB test rows, none seen. Every test renewal is a 2024 policy with renewal_count + 1 (21,698 / 12,579 / 15,806 rows). Test NB customers are all new (cust_expo = 0). row_id runs in blocks by line and year, ordered by start date within a year (Spearman 0.67-0.71 in 2024). Policy, customer and address ids are random: a/e by id decile chi²/df 0.32-1.42, \|year_corr\| ≤ 0.40. claim_id vs loss_date Spearman +0.013 | 0 | None |
| 7 | Cross-table: no third-party or repairer structure; other-line claim history is thin and already in the model | third_party_id: 2,654 unique "TP" ids, 0 match a customer, 0 repeats. repairer_id: motor only, 986 ids, median 5 claims each, region purity 0.50. Same-year association: P(contents claim \| buildings claim) / P(\| none) = 1.32 (708 buildings-claim customer-years); motor \| contents 1.23. 2024 buildings rows with a 2023 **contents** claim: 296 rows, 21 claims, a/e 4.47. Contents rows with a 2023 buildings claim: 381 rows, 12 claims, a/e 2.79. Addresses with > 1 customer: 0.8 % of history address-years, 1.6 % of test; shared-address a/e motor 1.086 (159 claims) | Small (2-4 % of home renewals) | `cust_<line>_claims` already carry it (contents `cust_buildings_rate` chi²/df 4.24, but lift only +1.6 bp). Nothing new |
| 8 | Unused raw columns: none has signal on the emerging rows | chi²/df (motor / bld / cnt): previous_insurer 1.12 / 1.02 / 0.76 (11 motor levels outside 0.8-1.25, consistent with SE 0.18 at 175 claims a level); campaign_code 1.25 / 0.41 / 1.44; branch_id 0.30 / 1.25 / 0.66; email_domain 1.58 / 1.15 / 0.52; customer_since_year 0.09 / 0.59 / 0.34; make_model 0.72 (300 levels); written_premium 0.51 / 0.20 / 0.69; year_built 0.63 / 0.62 / 0.67. compulsory_excess has one value in emerging. `customer_tenure` = `renewal_count` exactly (duplicate column). campaign_code is an acquisition marker: each of 16 codes covers about 7 % of NB rows and 0.7-2 % of renewals | 0 | Keep the pooled noise list (emerging-only screen, §7: neutral) |
| 9 | Target construction for mid-year NB is consistent | Reported frequency 2024 / 2023 by start quarter vs 2024 completeness: motor renewals Q1 0.926 vs 0.946. NB Q1-Q4 0.78 / 0.74 / 0.96 / 0.77 vs 0.93 / 0.91 / 0.87 / 0.65 (the 2023 NB year is high, finding 5). The IBNR share rises with late starts (motor 2024 NB Q4 34 %), as BF should. OOF a/e of NB by start month shows no trend (motor Jan 0.94, Feb 1.22, … Oct 0.58, Nov 3.39; \|z\| ≤ 2.04 on 16-109 claims a month) | 0 | None |
| 10 | Seasonality. Home claims are winter-heavy, so mid-year NB history rows carry more winter per unit exposure than test's January starters | 2023 frequency index by loss month: buildings Jan 1.32, Dec 1.63, May / Nov 0.76. Contents Dec 1.29, Jul-Sep 0.83-0.85. Motor flat (0.91-1.08). A July starter's buildings index is about 1.03, an October starter's 1.13 | A few % on the buildings NB level, absorbed by the probed NB factor. Seasonal-offset screen: home lines -0.7 bp (§7) | None (as in Part 1) |

## 1. Inventory: what the model sees

`model_matrix` with the R2_P34_hn features (`history True`, `hist_nan_2021`, drop `make_model` + noise list
previous_insurer / branch_id / email_domain / campaign_code, `rich_history` off, `premium` off) feeds 76 columns
(motor), 68 (buildings) and 69 (contents). The `irrelevant` lists are **not** applied by the R2 configs. Motor sees
property fields (basement, bedrooms, flood_zone, roof_type, property_age, …). None of them is out of line (motor
property_age 34-46 years a/e 1.25 on 720 claims, z 2.49, one of 7 levels with |z| > 2).

Raw columns never fed to the model (every line): row_id, policy_id, customer_id, address_id, period_start, period_end,
exposure (an offset, not a feature), compulsory_excess (constant in emerging), written_premium,
customer_service_contacts and renewal_decision (leaky), surname, customer_since_year (as customer_tenure),
email_domain, previous_insurer, campaign_code, branch_id, year_built (as property_age); motor also make_model. Claims
table: repairer_id and third_party_id are unused (no structure, finding 7). The cache also holds 24 `hx_*` per-peril
history columns, which are off (screen §7).

## 2. Residual scan of the used features

Features whose chi²/df and stability both clear the null (all others: chi²/df < 2.2, or |year_corr| < 0.4, or a lift
inside 2 null SD):

| line | feature | chi²/df | stability | lift bp (null) | frequency / severity |
|---|---|---|---|---|---|
| motor | gender | 6.21 | same in both years and in NB / renewals | +8.3 (2-level null -3.6 ± 4.1) | frequency (z ≈ 4.2) |
| motor | fuel_type | 3.55 | year_corr 0.83 | +14.6 (4-level null -5.2 ± 7.2) | mixed (hybrid frequency 1.05) |
| contents | works_from_home | 3.73 | 1.255 / 1.178 | +22.7 (null -1.2 ± 7.7) | mixed (frequency 1.067) |
| contents | gender | 3.21 | 1.099 / 1.114 | +12.3 | severity only |
| buildings | trees_within_5m | 2.43 | 1.664 / 1.300 | +23.1 (null -2.7 ± 22.8) | severity only |
| contents | elevation_m | 1.41 | year_corr 0.71 | +21.8 | lowest decile 1.345 (145 claims); weak |

## 3-6. Ids, shifts, cross-table, target

See findings 4-10. Two details:

- **Carried renewals.** 24 % of test rows are renewals of 2024 policies with the same policy_id. They differ from the
  76 % new rows exactly as history renewals differ from history NB: driver age 17-24 is 16.1 % of carried renewals and
  27.6 % of new rows.
- **History window.** Test renewals have twice the history window (pol_expo 1.68 vs 0.94), because 2025 rows see 2023
  and 2024. The pooled model knows that range from the developed rows (LOOKBACK 2).
- **Lines per customer-year** are identical in history and test (1 / 2 / 3 lines: 59.6 / 29.4 / 11.0 % vs
  59.7 / 29.4 / 10.9 %).

## 7. Quick screens (emerging-only, IAA_MARKET=part2, `pipeline.quick` 80/20 policy-grouped, paired over split seeds)

Baseline `P2_P34_si05` (seed 11 reproduces the documented 0.1059). The configs are appended to
`experiments/configs_part2.py` as `H1_*`. Δ = mean paired difference vs the baseline, in bp of D² (SD of the paired
difference over seeds). Seeds 11-13; weather also on 14-18.

| config | change | Δ motor | Δ buildings | Δ contents | Δ overall | verdict |
|---|---|---|---|---|---|---|
| H1_P34_nogender | drop gender | -5.5 (54) | +9.7 (193) | +3.2 (97) | +3.7 (40) | neutral. A single 80/20 split cannot resolve a 5-10 bp effect |
| H1_P34_dn | drop previous_insurer, branch_id, email_domain, campaign_code | +21.5 (33) | -24.7 (35) | -24.8 (66) | -4.4 (23) | neutral: the noise list costs nothing |
| H1_P34_rich | + hx_* rich history | -2.3 (39) | -65.5 (106) | +19.9 (19) | -19.9 (33) | reject |
| H1_P34_nowx | drop the 7 weather columns | +9.4 (28) | +55.0 (118) | -27.5 (86) | **+16.2 (41)**, 8 seeds | unresolved (1.1 SE); finding 2 |
| H1_P34_nomonth | drop start_month and row_type | +7.9 (31) | -38.7 (118) | +4.8 (63) | -8.6 (30) | neutral |
| H1_P34_season | seasonal exposure offset, home lines | 0 | +2.8 (29) | -7.6 (12) | -0.7 (11) | reject (as in Part 1) |

**Resolution limit.** The quick screen's paired SD is 11-41 bp overall, so it cannot see effects of the size found in
finding 1 (5-20 bp). The stage-2 tests are better suited: they use all 109k emerging rows out-of-fold, with a
random-feature null of about ±4-8 bp per line.

## 8. Proposed implementation: emerging stage-2 relativity layer (no src change made)

Definition, per line L with feature list F_L, fitted on the emerging history rows of the final model's result
(`R2_FINAL`):

1. **Base.** b_i = OOF prediction rescaled per (year × new_business) cell to the cell's Σ y_eval (`E_cell`).
2. **Target.** y_i = y_eval capped at the 99.5 % quantile of the line's claim rows, then multiplied by
   Σ y_eval / Σ capped (the level is kept).
3. **Model.** glum `GeneralizedLinearRegressor(TweedieDistribution(1.5), link="log", alpha=α, l1_ratio=0,
   fit_intercept=True)` on the one-hot of F_L, with offset log b_i.
4. **Apply.** On 2025 rows, f_i = exp(x_i · β). Renormalise inside each (line × new_business) cell so that
   Σ p_i f_i = Σ p_i, where p = the file's predictions. Then f_i ← f_i / k_cell, and the final prediction is p_i × f_i.
   Every probed subset level (motor NB / renewals, buildings, contents) is unchanged.

| variant | F_L | α | relativities (before renormalisation) | test factor p1-p99 | OOF overall gain |
|---|---|---|---|---|---|
| `core_a3` (recommended) | motor: gender, fuel_type | 3 | F 0.971 / M 1.029; petrol 0.968, diesel 1.016, hybrid 1.040, electric 0.978 | 0.944-1.075 | +4.5 bp |
| `full_a10` | + contents: gender, works_from_home; buildings: trees_within_5m | 10 | motor F 0.982 / M 1.019, petrol 0.980, hybrid 1.019; contents F 1.031 / M 0.970, wfh 1.039 / 0.963; buildings trees 1.082 / 0.924 | motor 0.965-1.042, cnt 0.950-1.090, bld 0.980-1.148 | +13.5 bp |
| `full_a3` (upper bound) | same | 3 | contents F 1.063 / M 0.940, wfh 1.084 / 0.922; buildings trees 1.165 / 0.859 | cnt 0.897-1.194, bld 0.959-1.303 | +21.1 bp |

**Ready to use.** Factors for all 209,063 test rows, in sample-submission-independent row_id form:
`experiments/logs/part2_feature_hunt/R2_FINAL/stage2_factors.csv` (columns row_id, line, new_business, core_a3,
full_a10, full_a3; rebuild with `scripts/part2_feature_hunt.py factors --ref R2_FINAL --out …`).

**Caveats.**
- The features were chosen after looking at both history years, so +21 bp is optimistic. The core (motor gender) has
  a frequency z of 4.2 and survives a one-year-only selection.
- Buildings trees and contents gender are severity-only effects on 123 and about 1,000 claims.
- Gender as a rating factor is a policy question (it is already a model feature here; in the EU it would not be
  allowed). Say so in the write-up if it is used.

**Pooled-run alternative (needs a src change; proposed diff for `features.model_matrix`, before the categorical
loop):**

```python
    if cfg.get("market_copies") and "market" in train.columns:
        # emerging-only copies (NaN on developed rows) so trees can learn market-specific relativities
        for F, raw in ((X, train), (Xt, test)):
            em = raw["market"].astype(str).values == "emerging"
            for c in cfg["market_copies"]:
                if c in raw.columns:
                    F[f"em_{c}"] = raw[c].where(em).values
```

Config: `R2_P34_hn` + `features.market_copies = ["fuel_type", "works_from_home", "trees_within_5m"]`. I expect less
than the post-hoc layer: gender is already such a copy (NaN on every developed row) and the P34 (min leaf 2,000) still
leaves it at a/e 1.068 / 0.925.
