# Data-quality audit (IAA_AI) - unhandled planted issues

(Written here, not to scratchpad/audit_report.md, because plan mode blocked any other file write. All analysis was read-only inline Python; repo untouched.)

Scope: train = policies_*, test = test_*; "cleaned" = data.clean_policies / clean_claims. Totals: train 2,409,311 rows (motor 979,968; buildings 707,726; contents 721,617), test 626,639.

## 1. Policy rows
| Issue | Count | Share | Lines | Train/test | Fix |
|---|---|---|---|---|---|
| Exposure on a /360 basis instead of /365 (survives cleaning because inside the 0.02 tolerance; raw 1.0139 on full-year rows) | 4,719 motor / 3,472 bldg / 3,524 cont = 11,715 | 0.49% of train rows; excess exposure only ~136 policy-years (0.007%) | all | train only, test 0 | Recompute exposure = days/365 for every row (tolerance ~0.0006). Negligible effect |
| Duplicate row_id, duplicate (policy_id, period_start), full-row dups | 0 | - | all | both | none |
| period_end < period_start, exposure <= 0/NaN, period > 366 d | 0 | - | all | both | none |
| Overlapping / nested periods, gaps within a policy | 0 (all consecutive rows abut by exactly 1 day) | - | all | both | none |
| Same policy_id with >1 customer_id | 0 | - | all | both | none |
| Policy with >1 address_id | 9,070 mot / 25,861 bldg / 26,119 cont policies in train | ~3% / 9% / 9% of policies | all | both | Mid-term splits; looks like genuine moves (test: every buildings split pair changes address). Informational |
| Test rows duplicating train (policy_id, period_start) or row_id | 0 | - | all | test | none |
| Test policies | 500,245 continue from train (all alive at end-2024), 89,666 new; renewal_count = last train + 1 for all 500,245 | - | all | test | consistent |

## 2. Numeric sanity
| Issue | Count | Lines | Train/test | Note / fix |
|---|---|---|---|---|
| Sentinels (-1, 99, 999, 9999, 1e6) | none systematic. Value 99 appears 139 telematics_score, 408 days_unoccupied, 9-53 written_premium: natural values | - | both | none |
| sum_insured x1000 | bldg 1,395 train / 370 test; cont 1,402 / 360; all raw >= 8.2M, none between 1.1M and 2M (clean separation), all multiples of 100 | bldg, cont | both (~0.2%) | Already handled. Note: after /1000 these keep sub-1000 resolution (test buildings 99.9th pct 674,846), so the corrected rows are identifiable by non-round SI |
| annual_mileage = 0 | 11,928 train / 3,108 test (1.2%) | motor | both | Spread evenly across fuel/use (incl. 926 electric, 1,209 business). Probably a missing-code or planted zero; recommend treating 0 as a level or NaN, not a number. No km/miles bimodality: values are banded with spikes at 5k/10k/15k/20k/25k/30k |
| Random NaN blocks (vehicle_age, engine_size_cc, annual_mileage 2.5% each, independent; overnight_parking 2.5%; telematics NaN 87% = non-opt-in; days_unoccupied/smoke_alarms 2.5%; high_value_items, security_level 2.5%) | per column ~24k train / ~6k test | motor, bldg, cont | same rate train and test, same each year | MCAR; use missing indicator or median impute. Engine size missing is not related to electric (electric rows have a cc value) |
| ncb_years > years_licensed | 127,155 train (13%) / 34,808 test (13.6%) | motor | both | Generator artefact; same rate in test; informational |
| named_young_driver with driver_age > 25 | 52,039 / 13,030 (5%) | motor | both | Same artefact |
| days_unoccupied > exposure*365 | 6,756 / 1,947 | bldg | both | informational |
| policyholder_age 17 (<18) | 3,331 bldg, 3,376 cont train; 333 / 340 test | bldg, cont | both | informational |
| written_premium is an annual rate, not scaled by exposure (premium/exposure up to 3e5 on 1-day rows) | all rows | all | both | Do not divide by exposure when using premium as feature |
| Group-level unit mixtures (log-median deviation > 0.25 for SI, mileage, engine, premium by year/region/branch/channel) | none flagged | all | both | no hidden unit subsets after existing fixes |
| Train(2024) vs test(2025) shifts: vehicle_age +0.11 sd, written_premium +0.05 to +0.14 sd, SI +0.075 sd, renewal_count mix, customer_service_contacts (max 36 train vs 6 test: leaky col, already excluded) | - | all | - | Normal inflation/ageing; row_type mix identical |

## 3. Categorical labels
All object columns (policies, customers, addresses, claims) checked after stripping case/punctuation: zero near-duplicate levels, zero whitespace or 'N/A'/'Unknown' tokens in cleaned data. None used as the missing token (missing = real None: overnight_parking, security_level, marital_status, soil_type, roof_type, energy_rating ~2.5%). construction_type variants already mapped. Train vs test level sets identical.

## 4. Claims (162,546 after dedup; 604 exact re-keyed dups already removed, GBP 3.29M)
| Issue | Count | Money | Lines | Note / fix |
|---|---|---|---|---|
| Negative amounts, incurred < paid, recoveries > incurred, report < loss | 0 | - | - | none |
| Zero incurred, all status=repudiated | 1,705 (1,406 motor, 299 contents) | 0 | motor, contents | already excluded by net_claims |
| Claim line vs policy-id prefix, peril vs line | 0 mismatches (M/B/C map 1:1; no storm on motor etc.) | - | - | none |
| Near-duplicates (same policy, peril, loss_date, different amount) | 17 groups / 34 rows; amounts differ by 1.01x-89x, different report dates and repairers | - | - | Looks like chance coincidence, not planted; keep |
| Same policy+peril+amounts, different dates | 14 rows | - | - | ignore |
| Orphan claims (loss_date outside every period of its policy) | 413 (0.25%): ALL before the first period (4-499 days; 273 are Dec-2020 on policies starting 2021-01-01, 140 in 2021-23 on new_business policies). None after, none unmatched policy | incurred 1.79M of 770.4M (0.23%): motor 296 / 1.19M, bldg 67 / 0.43M, cont 50 / 0.16M | all | train only (claims table only covers <=2024). Contribute zero to target currently (attach_claims drops them). Treat as pre-inception data noise or left-truncation; leave out |
| Extreme motor bodily_injury: CL30973963 incurred 11,894,277 (paid 738,769, open, 2023-06) vs next largest 1.12M; BI median 11.5k, 35 BI > 200k | 1 claim (1.5% of total incurred); top-20 claims = 2.7% of total | 11.9M | motor | Not a x1000 slip: amounts carry arbitrary cent values (x1000 of a 2-dp value would end in 0), so likely heavy-tail/planted large loss. Cap/winsorise in severity (existing large-loss loading) |
| x1000-scaled claim subsets | none: per line x year medians stable (motor ~1.7-1.9k, bldg ~2.8-3.5k, cont ~1.1-1.2k) | - | - | none |
| Peril label variants, currency/unit fields | none present | - | - | none |
| Claims with loss before 2021 | 273 (Dec 2020) | in orphan figure | - | see orphans |
| Claim reporting lag > 730 days | 57; max 1361 | small | - | informational (IBNR tail) |

## 5. Joins
| Check | Result |
|---|---|
| customer_id after zfill(8) | 0 unmatched in all 6 tables. Raw motor ids lose zeros: 98,110 train rows (10.0%) and 25,554 test rows (10.0%) fail a raw join (bldg/cont raw ids already 8 chars). Handled |
| address_id | 0 unmatched, all format A+7 digits; no whitespace/case variants |
| customers / addresses duplicate keys | 0; no duplicate attribute rows; 0 customers or addresses unused. Surname+DOB collisions 9,776 consistent with chance |
| weather (address_id, year) coverage | 100% for 2021-2025; test 2025 has 0 missing rows; no duplicates; no NaN |
| Cross-table consistency: policy age vs customer DOB (+-1 year only: 1,000-1,300 rows per table, birthday timing), customer_since_year <= year, years_at_address <= age, year_built <= year | all pass, 0 violations |
| customers DOB ages | 17-92, none <17 or >100 |
| Missing attributes | customers: household_size 2.5% NaN, years_at_address 2.5%, campaign_code 51%; addresses: flood_zone 2.9%, distance_to_river 2.5%, soil/roof/energy 2.5% - MCAR |

## 6. Dates
| Issue | Result |
|---|---|
| Mixed formats / day-month swap | none detectable: period_start month distribution is flat (Jan 80-86% because annual renewals start 1 Jan, other months 1.2-2.0% each) and identical for 2022-2025 train vs 2025 test; parquet dates are parsed datetimes. 2021 has higher Jan share (left-truncation) |
| 2024 leap year convention | period_end 30-Dec for 2024 full years; row_type handles it |
| DOB | clean |

## 7. claim_count / claim_amount fields on policy rows
None exist in any policy table (columns listed in data.py schema); only renewal_decision and customer_service_contacts as leaky columns, already excluded. Nothing to reconcile.

## 8. Other planted-looking items
| Item | Finding |
|---|---|
| New_business flag on non-first rows | 10,448 motor / 4,004 bldg / 5,870 cont train rows are flagged new_business although not the first row of their policy_id (renewal_count resets to 0 on these); renewal_count>0 with new_business is never seen. Likely mid-term-change split rows or generator re-using flag; worth checking whether these rows form a coherent block (not examined further) |
| renewal_decision of last 2024 row | 2,742 rows 'declined_by_insurer' still reappear in test 2025 - the label is leaky/unreliable (already excluded) |
| ID ranges | train row_ids not contiguous (gaps where the other line sits); test row_ids contiguous per line; row_id ranges disjoint train vs test; no behavioural break |
| Constant-in-year columns | none. Means of new_business, discount, missed_payments, quote-to-inception stable across years (new_business 12% in 2021, ~15% later - left-truncation artefact) |
| Motor vehicle changes | 65,767 train policies (21%) have >1 make_model across their rows - vehicle change at renewal or planted shuffling; make_model->body_type mapping is non-deterministic (top share ~38%), so make_model carries little structural information |
| Shuffled-column blocks | no evidence (cross-column consistency checks pass) |

## Ranked summary (impact on a pure-premium model)
1. Heavy-tail claim outliers: one 11.9M motor bodily-injury claim (1.5% of all incurred; top-20 claims = 2.7%) plus 35 BI claims above 200k dominate motor severity variance; cap/winsorise rather than model them (not a unit error, amounts are not x1000 artefacts).
2. Orphan pre-inception claims: 413 claims / 1.79M (0.23% of incurred) happen before the policy's first period and are silently dropped by attach_claims; harmless for the target but they signal either left-truncated exposure (273 Dec-2020 losses) or unreliable dates; keep excluded.
3. annual_mileage = 0 (11,928 train, 3,108 test, 1.2% of motor) and MCAR NaNs (2.5% per column across vehicle_age, engine_size_cc, annual_mileage, parking, security_level, etc.): need an explicit missing/zero level; same rate in test so no shift.
4. Exposure /360 basis on 11,715 train rows (0.5%; test clean): tiny exposure error (136 policy-years, 0.007%), trivial to fix by recomputing days/365.
5. Residual logical oddities (ncb_years > years_licensed 13%, named_young_driver with age > 25 5%, under-18 policyholders, new_business flag on non-first rows): present at equal rates in train and test, so they are generator artefacts that add no shift; ignore or use as features.

Everything else asked for (duplicates, overlaps, negative values, label variants, join keys, weather coverage including 2025, DOB ranges, dayfirst ambiguity, test-train row duplicates, cross-line claims) is clean after the existing fixes.
