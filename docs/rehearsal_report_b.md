# New market readiness report

Dataset: `data\rehearsal`  
Reference (Part 1): `data/raw/public`

## 1. Schema differences

| table | status | column | detail |
|---|---|---|---|
| customers | missing_column | previous_insurer | expected, not present |
| policies_motor | missing_column | driver_age | possible rename to 'age_driver' |
| policies_motor | new_column | age_driver | int64; possible rename of 'driver_age' |
| test_motor | missing_column | driver_age | possible rename to 'age_driver' |
| test_motor | new_column | age_driver | int64; possible rename of 'driver_age' |

## 2. Data-quality register

Rows with `kind=cleaning` are Part 1 fixes (`src/data.py`) applied to the new data; `generic` are new checks. Zero-row checks are omitted.

| table | line | kind | issue | column | rows | share |
|---|---|---|---|---|---|---|
| customers |  | generic | missing rate > 1% | occupation | 13462 | 0.0247 |
| customers |  | generic | missing rate > 1% | marital_status | 13762 | 0.0252 |
| customers |  | generic | missing rate > 1% | household_size | 13444 | 0.0247 |
| customers |  | generic | missing rate > 1% | years_at_address | 13701 | 0.0251 |
| customers |  | generic | missing rate > 1% | campaign_code | 279085 | 0.5119 |
| addresses |  | generic | missing rate > 1% | flood_zone | 16823 | 0.0292 |
| addresses |  | generic | missing rate > 1% | distance_to_river_km | 14373 | 0.0250 |
| addresses |  | generic | missing rate > 1% | soil_type | 14392 | 0.0250 |
| addresses |  | generic | missing rate > 1% | roof_type | 14401 | 0.0250 |
| addresses |  | generic | missing rate > 1% | energy_rating | 14443 | 0.0251 |
| claims | claims | generic | missing rate > 1% | repairer_id | 56258 | 0.4046 |
| claims | claims | generic | missing rate > 1% | third_party_id | 97639 | 0.7022 |
| policies_motor | motor | generic | missing rate > 1% | vehicle_age | 24427 | 0.0249 |
| policies_motor | motor | generic | missing rate > 1% | engine_size_cc | 24694 | 0.0252 |
| policies_motor | motor | generic | missing rate > 1% | annual_mileage | 24711 | 0.0252 |
| policies_motor | motor | generic | missing rate > 1% | overnight_parking | 24371 | 0.0249 |
| policies_motor | motor | generic | missing rate > 1% | telematics_score | 858223 | 0.8758 |
| policies_motor | motor | generic | exposure > 1.01 | exposure | 3973 | 0.0041 |
| test_motor | motor | generic | missing rate > 1% | vehicle_age | 6316 | 0.0248 |
| test_motor | motor | generic | missing rate > 1% | engine_size_cc | 6440 | 0.0253 |
| test_motor | motor | generic | missing rate > 1% | annual_mileage | 6271 | 0.0246 |
| test_motor | motor | generic | missing rate > 1% | overnight_parking | 6398 | 0.0251 |
| test_motor | motor | generic | missing rate > 1% | telematics_score | 221126 | 0.8672 |
| policies_buildings | buildings | generic | missing rate > 1% | days_unoccupied | 17587 | 0.0249 |
| policies_buildings | buildings | generic | missing rate > 1% | smoke_alarms | 17867 | 0.0252 |
| policies_buildings | buildings | generic | sum_insured > 2.5e+08 | sum_insured | 1395 | 0.0020 |
| policies_buildings | buildings | generic | exposure > 1.01 | exposure | 3039 | 0.0043 |
| policies_buildings | buildings | generic | inconsistent casing/whitespace ('Brick', 'non_standard', 'stone', 'W', 'T', 'brick') | construction_type | 707726 | 1.0000 |
| test_buildings | buildings | generic | missing rate > 1% | days_unoccupied | 4586 | 0.0249 |
| test_buildings | buildings | generic | missing rate > 1% | smoke_alarms | 4602 | 0.0250 |
| test_buildings | buildings | generic | sum_insured > 2.5e+08 | sum_insured | 370 | 0.0020 |
| test_buildings | buildings | generic | inconsistent casing/whitespace ('Brick', 'W', 'non_standard', 'stone', 'brick', 'BRICK') | construction_type | 184155 | 1.0000 |
| policies_motor | motor | cleaning | exposure recorded in days, not years |  | 934 | 0.0010 |
| policies_motor | motor | cleaning | years_licensed impossible for driver_age |  | 2962 | 0.0030 |
| test_motor | motor (test) | cleaning | years_licensed impossible for driver_age |  | 755 | 0.0030 |
| policies_buildings | buildings | cleaning | exposure recorded in days, not years |  | 656 | 0.0009 |
| policies_buildings | buildings | cleaning | sum_insured x1000 unit error |  | 1395 | 0.0020 |
| policies_buildings | buildings | cleaning | construction_type inconsistent labels |  | 641380 | 0.9063 |
| test_buildings | buildings (test) | cleaning | sum_insured x1000 unit error |  | 370 | 0.0020 |
| test_buildings | buildings (test) | cleaning | construction_type inconsistent labels |  | 166647 | 0.9049 |
| claims | claims | cleaning | duplicate claims under new claim_id |  | 520 | 0.0037 |

## 3. Portfolio and target overview

| set | line | year | rows | exposure | claims | cost | premium | frequency | severity | pure_premium | loss_ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|
| train | motor | 2019 | 216008 | 188,004.0 | 27,354.0 | 5,294,872,054.0 | 5,809,509,819.0 | 0.1455 | 193,568.5 | 28,163.6 | 0.9114 |
| train | motor | 2020 | 254685 | 199,713.1 | 29,288.0 | 5,744,536,245.5 | 6,998,068,332.0 | 0.1467 | 196,139.6 | 28,763.9 | 0.8209 |
| train | motor | 2021 | 254577 | 199,695.2 | 27,990.0 | 5,865,385,185.0 | 7,237,392,607.0 | 0.1402 | 209,552.9 | 29,371.7 | 0.8104 |
| train | motor | 2022 | 254698 | 199,799.7 | 25,864.0 | 4,533,348,844.5 | 7,447,067,992.5 | 0.1294 | 175,276.4 | 22,689.5 | 0.6087 |
| test | motor | 2023 | 254986 | 199,983.6 |  |  |  |  |  |  |  |
| train | buildings | 2019 | 155729 | 140,792.9 | 6,195.0 | 2,702,641,709.5 | 2,311,663,522.0 | 0.0440 | 436,261.8 | 19,195.9 | 1.1691 |
| train | buildings | 2020 | 183864 | 149,883.0 | 6,683.0 | 3,020,859,115.0 | 2,888,718,503.5 | 0.0446 | 452,021.4 | 20,154.8 | 1.0457 |
| train | buildings | 2021 | 184132 | 149,997.7 | 6,817.0 | 3,241,623,561.5 | 3,035,308,820.5 | 0.0454 | 475,520.5 | 21,611.2 | 1.0680 |
| train | buildings | 2022 | 184001 | 149,927.5 | 6,571.0 | 3,367,697,477.5 | 3,197,324,141.5 | 0.0438 | 512,509.1 | 22,462.2 | 1.0533 |
| test | buildings | 2023 | 184155 | 150,243.9 |  |  |  |  |  |  |  |


### Level versus Part 1 (train, all years)

| line | frequency_new | frequency_ref | frequency_ratio | severity_new | severity_ref | severity_ratio | pure_premium_new | pure_premium_ref | pure_premium_ratio |
|---|---|---|---|---|---|---|---|---|---|
| buildings | 0.0445 | 0.0445 | 1.0000 | 469,535.6 | 9,390.7 | 50.0 | 20,881.8 | 417.6 | 50.0 |
| motor | 0.1404 | 0.1404 | 1.0000 | 194,017.4 | 3,880.3 | 50.0 | 27,233.0 | 544.7 | 50.0 |

## 4. Claims development

| line | loss_year | n_claims | open_share | lag_p50 | lag_p90 | lag_p99 | lag_max |
|---|---|---|---|---|---|---|---|
| buildings | all | 26442 | 0.0716 | 9.0000 | 34.0 | 239.0 | 758 |
| buildings | 2018 | 46 | 0.0000 | 249.0 | 413.0 | 513.7 | 543 |
| buildings | 2019 | 6204 | 0.0027 | 9.0000 | 33.0 | 243.9 | 630 |
| buildings | 2020 | 6797 | 0.0081 | 9.0000 | 35.0 | 322.1 | 758 |
| buildings | 2021 | 6824 | 0.0295 | 9.0000 | 35.0 | 267.8 | 600 |
| buildings | 2022 | 6571 | 0.2465 | 8.0000 | 30.0 | 112.3 | 288 |
| motor | all | 112606 | 0.0661 | 9.0000 | 42.0 | 377.9 | 1361 |
| motor | 2018 | 198 | 0.0101 | 265.5 | 483.6 | 624.2 | 675 |
| motor | 2019 | 27745 | 0.0116 | 9.0000 | 48.0 | 460.6 | 1361 |
| motor | 2020 | 30102 | 0.0202 | 9.0000 | 44.0 | 425.0 | 1001 |
| motor | 2021 | 28390 | 0.0384 | 9.0000 | 44.0 | 353.0 | 657 |
| motor | 2022 | 26171 | 0.2073 | 8.0000 | 34.0 | 138.3 | 358 |

## 5. Drift vs Part 1

Max PSI 8.3095; shift (>0.25): 4, watch (0.1-0.25): 0, columns compared: 88.

| line | column | psi | flag | unseen_share | new_levels |
|---|---|---|---|---|---|
| buildings | sum_insured | 8.3095 | shift |  |  |
| motor | written_premium | 8.2831 | shift |  |  |
| buildings | written_premium | 8.2831 | shift |  |  |
| weather | year | 6.0777 | shift | 0.4000 | 2019, 2020 |
| buildings | exposure | 0.0000 | ok |  |  |
| motor | exposure | 0.0000 | ok |  |  |
| motor | years_licensed | 0.0000 | ok |  |  |
| motor | driver_age | 0.0000 | ok |  |  |
| motor | body_type | 0.0000 | ok | 0.0000 |  |
| motor | transmission | 0.0000 | ok | 0.0000 |  |
| motor | seats | 0.0000 | ok | 0.0000 |  |
| motor | vehicle_colour | 0.0000 | ok | 0.0000 |  |
| motor | fuel_type | 0.0000 | ok | 0.0000 |  |
| motor | driver_assist | 0.0000 | ok | 0.0000 |  |
| motor | cover_type | 0.0000 | ok | 0.0000 |  |
| motor | annual_mileage | 0.0000 | ok |  |  |
| motor | overnight_parking | 0.0000 | ok | 0.0000 |  |
| motor | vehicle_use | 0.0000 | ok | 0.0000 |  |
| motor | named_young_driver | 0.0000 | ok | 0.0000 |  |
| motor | ncb_years | 0.0000 | ok |  |  |
| motor | vehicle_value_group | 0.0000 | ok |  |  |
| motor | vehicle_age | 0.0000 | ok |  |  |
| motor | make_model | 0.0000 | ok | 0.0000 |  |
| motor | engine_size_cc | 0.0000 | ok |  |  |
| motor | channel | 0.0000 | ok | 0.0000 |  |
| motor | compulsory_excess | 0.0000 | ok | 0.0000 |  |
| motor | voluntary_excess | 0.0000 | ok | 0.0000 |  |
| motor | telematics_score | 0.0000 | ok |  |  |
| motor | telematics_opt_in | 0.0000 | ok | 0.0000 |  |
| motor | renewal_count | 0.0000 | ok |  |  |
| motor | new_business | 0.0000 | ok | 0.0000 |  |
| motor | days_quote_to_inception | 0.0000 | ok |  |  |
| motor | licence_inconsistent | 0.0000 | ok | 0.0000 |  |
| motor | discount_applied | 0.0000 | ok | 0.0000 |  |
| buildings | policyholder_age | 0.0000 | ok |  |  |
| buildings | construction_type | 0.0000 | ok | 0.0000 |  |
| buildings | occupancy | 0.0000 | ok | 0.0000 |  |
| buildings | days_unoccupied | 0.0000 | ok |  |  |
| buildings | smoke_alarms | 0.0000 | ok | 0.0000 |  |
| motor | payment_method | 0.0000 | ok | 0.0000 |  |

## 6. Adaptation checklist

1. [ ] column `previous_insurer` missing in `customers`: feature list must drop it  
   evidence: `schema_diff: customers / previous_insurer / missing_column / expected, not present`
2. [ ] column `driver_age` missing in `policies_motor`: feature list must drop it  
   evidence: `schema_diff: policies_motor / driver_age / missing_column / possible rename to 'age_driver'`
3. [ ] new column `age_driver` in `policies_motor`: screen as a candidate feature and check for leakage  
   evidence: `schema_diff: policies_motor / age_driver / new_column / int64; possible rename of 'driver_age'`
4. [ ] column `driver_age` missing in `test_motor`: feature list must drop it  
   evidence: `schema_diff: test_motor / driver_age / missing_column / possible rename to 'age_driver'`
5. [ ] new column `age_driver` in `test_motor`: screen as a candidate feature and check for leakage  
   evidence: `schema_diff: test_motor / age_driver / new_column / int64; possible rename of 'driver_age'`
6. [ ] `occupation` in `customers` is 2.5% missing: define imputation / missing-level handling  
   evidence: `dq_register: customers / missing rate > 1% / occupation / rows=13462 share=2.47%`
7. [ ] `marital_status` in `customers` is 2.5% missing: define imputation / missing-level handling  
   evidence: `dq_register: customers / missing rate > 1% / marital_status / rows=13762 share=2.52%`
8. [ ] `household_size` in `customers` is 2.5% missing: define imputation / missing-level handling  
   evidence: `dq_register: customers / missing rate > 1% / household_size / rows=13444 share=2.47%`
9. [ ] `years_at_address` in `customers` is 2.5% missing: define imputation / missing-level handling  
   evidence: `dq_register: customers / missing rate > 1% / years_at_address / rows=13701 share=2.51%`
10. [ ] `campaign_code` in `customers` is 51.2% missing: define imputation / missing-level handling  
   evidence: `dq_register: customers / missing rate > 1% / campaign_code / rows=279085 share=51.19%`
11. [ ] `flood_zone` in `addresses` is 2.9% missing: define imputation / missing-level handling  
   evidence: `dq_register: addresses / missing rate > 1% / flood_zone / rows=16823 share=2.92%`
12. [ ] `distance_to_river_km` in `addresses` is 2.5% missing: define imputation / missing-level handling  
   evidence: `dq_register: addresses / missing rate > 1% / distance_to_river_km / rows=14373 share=2.50%`
13. [ ] `soil_type` in `addresses` is 2.5% missing: define imputation / missing-level handling  
   evidence: `dq_register: addresses / missing rate > 1% / soil_type / rows=14392 share=2.50%`
14. [ ] `roof_type` in `addresses` is 2.5% missing: define imputation / missing-level handling  
   evidence: `dq_register: addresses / missing rate > 1% / roof_type / rows=14401 share=2.50%`
15. [ ] `energy_rating` in `addresses` is 2.5% missing: define imputation / missing-level handling  
   evidence: `dq_register: addresses / missing rate > 1% / energy_rating / rows=14443 share=2.51%`
16. [ ] `repairer_id` in `claims` is 40.5% missing: define imputation / missing-level handling  
   evidence: `dq_register: claims / missing rate > 1% / repairer_id / rows=56258 share=40.46%`
17. [ ] `third_party_id` in `claims` is 70.2% missing: define imputation / missing-level handling  
   evidence: `dq_register: claims / missing rate > 1% / third_party_id / rows=97639 share=70.22%`
18. [ ] `vehicle_age` in `policies_motor` is 2.5% missing: define imputation / missing-level handling  
   evidence: `dq_register: policies_motor / missing rate > 1% / vehicle_age / rows=24427 share=2.49%`
19. [ ] `engine_size_cc` in `policies_motor` is 2.5% missing: define imputation / missing-level handling  
   evidence: `dq_register: policies_motor / missing rate > 1% / engine_size_cc / rows=24694 share=2.52%`
20. [ ] `annual_mileage` in `policies_motor` is 2.5% missing: define imputation / missing-level handling  
   evidence: `dq_register: policies_motor / missing rate > 1% / annual_mileage / rows=24711 share=2.52%`
21. [ ] `overnight_parking` in `policies_motor` is 2.5% missing: define imputation / missing-level handling  
   evidence: `dq_register: policies_motor / missing rate > 1% / overnight_parking / rows=24371 share=2.49%`
22. [ ] `telematics_score` in `policies_motor` is 87.6% missing: define imputation / missing-level handling  
   evidence: `dq_register: policies_motor / missing rate > 1% / telematics_score / rows=858223 share=87.58%`
23. [ ] exposure > 1.01 in `policies_motor`: add a validation/cleaning rule  
   evidence: `dq_register: policies_motor / exposure > 1.01 / exposure / rows=3973 share=0.41%`
24. [ ] `vehicle_age` in `test_motor` is 2.5% missing: define imputation / missing-level handling  
   evidence: `dq_register: test_motor / missing rate > 1% / vehicle_age / rows=6316 share=2.48%`
25. [ ] `engine_size_cc` in `test_motor` is 2.5% missing: define imputation / missing-level handling  
   evidence: `dq_register: test_motor / missing rate > 1% / engine_size_cc / rows=6440 share=2.53%`
26. [ ] `annual_mileage` in `test_motor` is 2.5% missing: define imputation / missing-level handling  
   evidence: `dq_register: test_motor / missing rate > 1% / annual_mileage / rows=6271 share=2.46%`
27. [ ] `overnight_parking` in `test_motor` is 2.5% missing: define imputation / missing-level handling  
   evidence: `dq_register: test_motor / missing rate > 1% / overnight_parking / rows=6398 share=2.51%`
28. [ ] `telematics_score` in `test_motor` is 86.7% missing: define imputation / missing-level handling  
   evidence: `dq_register: test_motor / missing rate > 1% / telematics_score / rows=221126 share=86.72%`
29. [ ] `days_unoccupied` in `policies_buildings` is 2.5% missing: define imputation / missing-level handling  
   evidence: `dq_register: policies_buildings / missing rate > 1% / days_unoccupied / rows=17587 share=2.49%`
30. [ ] `smoke_alarms` in `policies_buildings` is 2.5% missing: define imputation / missing-level handling  
   evidence: `dq_register: policies_buildings / missing rate > 1% / smoke_alarms / rows=17867 share=2.52%`
31. [ ] sum_insured > 2.5e+08 in `policies_buildings`: add a validation/cleaning rule  
   evidence: `dq_register: policies_buildings / sum_insured > 2.5e+08 / sum_insured / rows=1395 share=0.20%`
32. [ ] exposure > 1.01 in `policies_buildings`: add a validation/cleaning rule  
   evidence: `dq_register: policies_buildings / exposure > 1.01 / exposure / rows=3039 share=0.43%`
33. [ ] `construction_type` in `policies_buildings` has inconsistent labels: add a normalisation map  
   evidence: `dq_register: policies_buildings / inconsistent casing/whitespace ('Brick', 'non_standard', 'stone', 'W', 'T', 'brick') / construction_type / rows=707726 share=100.00%`
34. [ ] `days_unoccupied` in `test_buildings` is 2.5% missing: define imputation / missing-level handling  
   evidence: `dq_register: test_buildings / missing rate > 1% / days_unoccupied / rows=4586 share=2.49%`
35. [ ] `smoke_alarms` in `test_buildings` is 2.5% missing: define imputation / missing-level handling  
   evidence: `dq_register: test_buildings / missing rate > 1% / smoke_alarms / rows=4602 share=2.50%`
36. [ ] sum_insured > 2.5e+08 in `test_buildings`: add a validation/cleaning rule  
   evidence: `dq_register: test_buildings / sum_insured > 2.5e+08 / sum_insured / rows=370 share=0.20%`
37. [ ] `construction_type` in `test_buildings` has inconsistent labels: add a normalisation map  
   evidence: `dq_register: test_buildings / inconsistent casing/whitespace ('Brick', 'W', 'non_standard', 'stone', 'brick', 'BRICK') / construction_type / rows=184155 share=100.00%`
38. [ ] Part 1 fix still needed (exposure recorded in days, not years) in `policies_motor`: 934 rows, keep/re-validate the rule  
   evidence: `dq_register: policies_motor / exposure recorded in days, not years / - / rows=934 share=0.10%`
39. [ ] Part 1 fix still needed (years_licensed impossible for driver_age) in `policies_motor`: 2,962 rows, keep/re-validate the rule  
   evidence: `dq_register: policies_motor / years_licensed impossible for driver_age / - / rows=2962 share=0.30%`
40. [ ] Part 1 fix still needed (years_licensed impossible for driver_age) in `test_motor`: 755 rows, keep/re-validate the rule  
   evidence: `dq_register: test_motor / years_licensed impossible for driver_age / - / rows=755 share=0.30%`
41. [ ] Part 1 fix still needed (exposure recorded in days, not years) in `policies_buildings`: 656 rows, keep/re-validate the rule  
   evidence: `dq_register: policies_buildings / exposure recorded in days, not years / - / rows=656 share=0.09%`
42. [ ] Part 1 fix still needed (sum_insured x1000 unit error) in `policies_buildings`: 1,395 rows, keep/re-validate the rule  
   evidence: `dq_register: policies_buildings / sum_insured x1000 unit error / - / rows=1395 share=0.20%`
43. [ ] Part 1 fix still needed (construction_type inconsistent labels) in `policies_buildings`: 641,380 rows, keep/re-validate the rule  
   evidence: `dq_register: policies_buildings / construction_type inconsistent labels / - / rows=641380 share=90.63%`
44. [ ] Part 1 fix still needed (sum_insured x1000 unit error) in `test_buildings`: 370 rows, keep/re-validate the rule  
   evidence: `dq_register: test_buildings / sum_insured x1000 unit error / - / rows=370 share=0.20%`
45. [ ] Part 1 fix still needed (construction_type inconsistent labels) in `test_buildings`: 166,647 rows, keep/re-validate the rule  
   evidence: `dq_register: test_buildings / construction_type inconsistent labels / - / rows=166647 share=90.49%`
46. [ ] Part 1 fix still needed (duplicate claims under new claim_id) in `claims`: 520 rows, keep/re-validate the rule  
   evidence: `dq_register: claims / duplicate claims under new claim_id / - / rows=520 share=0.37%`
47. [ ] `written_premium` (motor) has shifted (PSI 8.28): re-check relativity, consider re-binning or dropping  
   evidence: `drift_report: motor / written_premium / psi=8.28`
48. [ ] `sum_insured` (buildings) has shifted (PSI 8.31): re-check relativity, consider re-binning or dropping  
   evidence: `drift_report: buildings / sum_insured / psi=8.31`
49. [ ] `written_premium` (buildings) has shifted (PSI 8.28): re-check relativity, consider re-binning or dropping  
   evidence: `drift_report: buildings / written_premium / psi=8.28`
50. [ ] `year` (weather) has shifted (PSI 6.08): re-check relativity, consider re-binning or dropping  
   evidence: `drift_report: weather / year / psi=6.08`
51. [ ] new level(s) 2019, 2020 in `year` (weather), 40.0% of rows: category handling (map to other / grouped level)  
   evidence: `drift_report: weather / year / unseen_share=0.400`
52. [ ] buildings: latest loss year 2022 looks immature (open share/claim count): develop it before using as a target  
   evidence: `claims_development: buildings / loss_year=2022 / open_share=0.247 n=6571`
53. [ ] motor: latest loss year 2022 looks immature (open share/claim count): develop it before using as a target  
   evidence: `claims_development: motor / loss_year=2022 / open_share=0.207 n=26171`
54. [ ] buildings: severity is 50.00x Part 1: recalibrate the base level (do not carry Part 1 intercept)  
   evidence: `portfolio_summary: buildings severity new=4.695e+05 ref=9391`
55. [ ] buildings: pure_premium is 50.00x Part 1: recalibrate the base level (do not carry Part 1 intercept)  
   evidence: `portfolio_summary: buildings pure_premium new=2.088e+04 ref=417.6`
56. [ ] motor: severity is 50.00x Part 1: recalibrate the base level (do not carry Part 1 intercept)  
   evidence: `portfolio_summary: motor severity new=1.94e+05 ref=3880`
57. [ ] motor: pure_premium is 50.00x Part 1: recalibrate the base level (do not carry Part 1 intercept)  
   evidence: `portfolio_summary: motor pure_premium new=2.723e+04 ref=544.7`
58. [ ] training years are [2019, 2020, 2021, 2022] (Part 1: 2021-2024): revisit backtest folds and year trend  
   evidence: `portfolio_summary: train years`

## Dress-rehearsal run log, phase B (17 Oct 2026, after the fixes)

Fixes applied after the first run (see `docs/rehearsal_report.md`, "Dress-rehearsal run log"): (1) the env-selected spec is
resolved lazily so `IAA_MARKET` can pick a market registered later in `src/market_spec.py`; (2) `MarketSpec.column_aliases`
renames new column names to the canonical Part 1 names at load time (`age_driver` -> `driver_age`); (3) missing columns are
tolerated (logged once) in cleaning, enrichment, history features, segment loading and the credibility layer; (4) the
readiness report's sum-insured anomaly check uses the active spec's currency threshold.

| Step (IAA_MARKET=rehearsal) | Outcome |
|---|---|
| `features.build_base(refresh=True)` | OK; cache written under `experiments/cache/rehearsal/` only: train_motor (979,968 × 121), train_buildings (707,726 × 108), claims 137,122 |
| `pipeline.quick("rehearsal_L17", L17_tuned)` | OK; quick 80/20 D²: motor 0.1209, buildings 0.1126, overall 0.1183 |
| `new_market_report.py data/rehearsal --ref data/raw/public` | OK; currency-aware check now flags 0.20 % of buildings rows (the genuine ×1000 errors), not 60 % |
| `pytest tests` with `IAA_MARKET` unset | 57 passed, including `test_golden_part1` (Part 1 outputs byte-identical) |

Still hard-coded (to watch on the real Part 2 data): the readiness tool's reference schema and column lists are Part 1 by
design; `experience_features` requires `customer_id`; claims cleaning needs `line`, `peril`, `net_incurred`, `status`,
`incurred_to_date`; `src/market.py` fixes `LINES` at import.
