# New market readiness report

Dataset: `data\part2\emerging`  
Reference (Part 1): `data/raw/public`

## 1. Schema differences

| table | status | column | detail |
|---|---|---|---|
| customers | missing_column | date_of_birth | expected, not present |
| customers | missing_column | credit_band | expected, not present |
| customers | new_column | gender | string |
| policies_motor | missing_column | driver_age | possible rename to 'driver_age_band' |
| policies_motor | missing_column | annual_mileage | expected, not present |
| policies_motor | missing_column | telematics_opt_in | expected, not present |
| policies_motor | missing_column | telematics_score | expected, not present |
| policies_motor | new_column | driver_age_band | string; possible rename of 'driver_age' |
| test_motor | missing_column | driver_age | possible rename to 'driver_age_band' |
| test_motor | missing_column | annual_mileage | expected, not present |
| test_motor | missing_column | telematics_opt_in | expected, not present |
| test_motor | missing_column | telematics_score | expected, not present |
| test_motor | new_column | driver_age_band | string; possible rename of 'driver_age' |
| policies_buildings | missing_column | policyholder_age | possible rename to 'policyholder_age_band' |
| policies_buildings | new_column | policyholder_age_band | string; possible rename of 'policyholder_age' |
| test_buildings | missing_column | policyholder_age | possible rename to 'policyholder_age_band' |
| test_buildings | new_column | policyholder_age_band | string; possible rename of 'policyholder_age' |
| policies_contents | missing_column | policyholder_age | possible rename to 'policyholder_age_band' |
| policies_contents | new_column | policyholder_age_band | string; possible rename of 'policyholder_age' |
| test_contents | missing_column | policyholder_age | possible rename to 'policyholder_age_band' |
| test_contents | new_column | policyholder_age_band | string; possible rename of 'policyholder_age' |
| expenses | new_table |  | table not in Part 1 |
| reference_policies | new_table |  | table not in Part 1 |

## 2. Data-quality register

Rows with `kind=cleaning` are Part 1 fixes (`src/data.py`) applied to the new data; `generic` are new checks. Zero-row checks are omitted.

| table | line | kind | issue | column | rows | share |
|---|---|---|---|---|---|---|
| customers |  | generic | missing rate > 1% | occupation | 16517 | 0.1216 |
| customers |  | generic | missing rate > 1% | marital_status | 16348 | 0.1203 |
| customers |  | generic | missing rate > 1% | household_size | 16037 | 0.1180 |
| customers |  | generic | missing rate > 1% | years_at_address | 16226 | 0.1194 |
| customers |  | generic | missing rate > 1% | campaign_code | 27433 | 0.2019 |
| customers |  | generic | inconsistent casing/whitespace ('Retired', 'Student', 'retired', 'student', 'Retired ', 'Care worker') | occupation | 107364 | 0.7903 |
| addresses |  | generic | missing rate > 1% | flood_zone | 4037 | 0.0295 |
| addresses |  | generic | missing rate > 1% | distance_to_river_km | 16421 | 0.1199 |
| addresses |  | generic | missing rate > 1% | soil_type | 16480 | 0.1204 |
| addresses |  | generic | missing rate > 1% | roof_type | 16337 | 0.1193 |
| addresses |  | generic | missing rate > 1% | energy_rating | 16515 | 0.1206 |
| claims | claims | generic | missing rate > 1% | repairer_id | 4283 | 0.4626 |
| claims | claims | generic | missing rate > 1% | third_party_id | 6545 | 0.7070 |
| policies_motor | motor | generic | missing rate > 1% | vehicle_age | 5604 | 0.1185 |
| policies_motor | motor | generic | missing rate > 1% | engine_size_cc | 5745 | 0.1215 |
| policies_motor | motor | generic | missing rate > 1% | overnight_parking | 5697 | 0.1205 |
| policies_motor | motor | generic | exposure > 1.01 | exposure | 188 | 0.0040 |
| test_motor | motor | generic | missing rate > 1% | vehicle_age | 11070 | 0.1223 |
| test_motor | motor | generic | missing rate > 1% | engine_size_cc | 11004 | 0.1216 |
| test_motor | motor | generic | missing rate > 1% | overnight_parking | 10859 | 0.1200 |
| policies_buildings | buildings | generic | missing rate > 1% | days_unoccupied | 3291 | 0.1199 |
| policies_buildings | buildings | generic | missing rate > 1% | smoke_alarms | 3266 | 0.1190 |
| policies_buildings | buildings | generic | sum_insured > 5e+06 | sum_insured | 53 | 0.0019 |
| policies_buildings | buildings | generic | exposure > 1.01 | exposure | 117 | 0.0043 |
| test_buildings | buildings | generic | missing rate > 1% | days_unoccupied | 6240 | 0.1193 |
| test_buildings | buildings | generic | missing rate > 1% | smoke_alarms | 6341 | 0.1213 |
| test_buildings | buildings | generic | sum_insured > 5e+06 | sum_insured | 104 | 0.0020 |
| policies_contents | contents | generic | missing rate > 1% | security_level | 4238 | 0.1226 |
| policies_contents | contents | generic | missing rate > 1% | high_value_items | 4067 | 0.1177 |
| policies_contents | contents | generic | sum_insured > 1e+06 | sum_insured | 57 | 0.0016 |
| policies_contents | contents | generic | exposure > 1.01 | exposure | 146 | 0.0042 |
| test_contents | contents | generic | missing rate > 1% | security_level | 8050 | 0.1215 |
| test_contents | contents | generic | missing rate > 1% | high_value_items | 7954 | 0.1201 |
| test_contents | contents | generic | sum_insured > 1e+06 | sum_insured | 135 | 0.0020 |
| policies_motor | motor | cleaning | exposure recorded in days, not years |  | 47 | 0.0010 |
| policies_buildings | buildings | cleaning | exposure recorded in days, not years |  | 33 | 0.0012 |
| policies_buildings | buildings | cleaning | sum_insured x1000 unit error |  | 53 | 0.0019 |
| policies_buildings | buildings | cleaning | construction_type inconsistent labels |  | 27447 | 1.0000 |
| test_buildings | buildings (test) | cleaning | sum_insured x1000 unit error |  | 104 | 0.0020 |
| test_buildings | buildings (test) | cleaning | construction_type inconsistent labels |  | 52295 | 1.0000 |
| policies_contents | contents | cleaning | exposure recorded in days, not years |  | 34 | 0.0010 |
| policies_contents | contents | cleaning | sum_insured x1000 unit error |  | 57 | 0.0016 |
| test_contents | contents (test) | cleaning | sum_insured x1000 unit error |  | 135 | 0.0020 |

## 3. Portfolio and target overview

| set | line | year | rows | exposure | claims | cost | premium | frequency | severity | pure_premium | loss_ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|
| train | motor | 2023 | 21680 | 18,865.8 | 3,557.0 | 8,162,761.2 | 13,247,857.6 | 0.1885 | 2,294.8 | 432.7 | 0.6162 |
| train | motor | 2024 | 25603 | 20,105.1 | 3,491.0 | 7,693,126.8 | 15,669,551.5 | 0.1736 | 2,203.7 | 382.6 | 0.4910 |
| test | motor | 2025 | 90521 | 79,006.9 |  |  |  |  |  |  |  |
| train | buildings | 2023 | 12580 | 11,373.3 | 519.0 | 5,848,688.1 | 4,353,303.3 | 0.0456 | 11,269.1 | 514.2 | 1.3435 |
| train | buildings | 2024 | 14867 | 12,096.1 | 537.0 | 3,710,061.2 | 5,459,609.6 | 0.0444 | 6,908.9 | 306.7 | 0.6795 |
| test | buildings | 2025 | 52295 | 47,376.3 |  |  |  |  |  |  |  |
| train | contents | 2023 | 15834 | 14,108.6 | 491.0 | 1,285,654.5 | 1,974,906.3 | 0.0348 | 2,618.4 | 91.1 | 0.6510 |
| train | contents | 2024 | 18732 | 14,951.3 | 507.0 | 1,297,699.8 | 2,460,397.4 | 0.0339 | 2,559.6 | 86.8 | 0.5274 |
| test | contents | 2025 | 66247 | 59,016.4 |  |  |  |  |  |  |  |


### Level versus Part 1 (train, all years)

| line | frequency_new | frequency_ref | frequency_ratio | severity_new | severity_ref | severity_ratio | pure_premium_new | pure_premium_ref | pure_premium_ratio |
|---|---|---|---|---|---|---|---|---|---|
| buildings | 0.0450 | 0.0445 | 1.0117 | 9,051.8 | 9,390.7 | 0.9639 | 407.3 | 417.6 | 0.9752 |
| contents | 0.0343 | 0.0401 | 0.8573 | 2,588.5 | 2,777.3 | 0.9320 | 88.9 | 111.2 | 0.7991 |
| motor | 0.1809 | 0.1404 | 1.2885 | 2,249.7 | 3,880.3 | 0.5798 | 406.9 | 544.7 | 0.7470 |

## 4. Claims development

| line | loss_year | n_claims | open_share | lag_p50 | lag_p90 | lag_p99 | lag_max |
|---|---|---|---|---|---|---|---|
| buildings | all | 1059 | 0.1483 | 8.0000 | 31.0 | 181.9 | 493 |
| buildings | 2021 | 1 | 0.0000 | 483.0 | 483.0 | 483.0 | 483 |
| buildings | 2022 | 2 | 0.0000 | 464.0 | 487.2 | 492.4 | 493 |
| buildings | 2023 | 519 | 0.0405 | 8.0000 | 33.0 | 188.8 | 422 |
| buildings | 2024 | 537 | 0.2533 | 8.0000 | 30.0 | 110.0 | 224 |
| contents | all | 1016 | 0.0581 | 6.0000 | 22.0 | 71.7 | 542 |
| contents | 2021 | 1 | 0.0000 | 481.0 | 481.0 | 481.0 | 481 |
| contents | 2022 | 4 | 0.0000 | 507.0 | 538.4 | 541.6 | 542 |
| contents | 2023 | 499 | 0.0000 | 6.0000 | 23.2 | 73.1 | 520 |
| contents | 2024 | 512 | 0.1152 | 6.0000 | 19.0 | 42.0 | 64 |
| motor | all | 7183 | 0.1169 | 9.0000 | 41.0 | 330.5 | 653 |
| motor | 2021 | 4 | 0.0000 | 486.5 | 532.1 | 546.4 | 548 |
| motor | 2022 | 22 | 0.0000 | 488.0 | 546.8 | 564.4 | 565 |
| motor | 2023 | 3608 | 0.0363 | 9.0000 | 46.0 | 344.0 | 653 |
| motor | 2024 | 3549 | 0.1998 | 8.0000 | 36.0 | 136.5 | 330 |

## 5. Drift vs Part 1

Max PSI 17.4813; shift (>0.25): 16, watch (0.1-0.25): 14, columns compared: 101.

| line | column | psi | flag | unseen_share | new_levels |
|---|---|---|---|---|---|
| buildings | construction_type | 17.5 | shift |  |  |
| addresses | region | 14.3 | shift | 1.0000 | E03, E06, E02, E05, E01 |
| weather | wildfire_index | 6.6728 | shift |  |  |
| weather | year | 3.2432 | shift | 0.0000 |  |
| customers | occupation | 2.5450 | shift | 0.5019 | retired, student, Retired , Student , Self-employed retired |
| weather | mean_temperature_c | 2.3298 | shift |  |  |
| customers | customer_since_year | 2.0638 | shift |  |  |
| weather | rainfall_mm | 1.8019 | shift |  |  |
| weather | hail_days | 1.6749 | shift |  |  |
| weather | freezing_days | 1.4278 | shift |  |  |
| weather | dry_summer_index | 1.3351 | shift |  |  |
| motor | channel | 1.1302 | shift | 0.0000 |  |
| contents | channel | 1.1190 | shift | 0.0000 |  |
| buildings | channel | 1.1130 | shift | 0.0000 |  |
| motor | vehicle_age | 0.7414 | shift |  |  |
| customers | campaign_code | 0.4407 | shift | 0.0000 |  |
| buildings | discount_applied | 0.2241 | watch | 0.0000 |  |
| motor | discount_applied | 0.2199 | watch | 0.0000 |  |
| contents | discount_applied | 0.2195 | watch | 0.0000 |  |
| addresses | soil_type | 0.1752 | watch | 0.0000 |  |
| customers | marital_status | 0.1678 | watch | 0.0000 |  |
| contents | security_level | 0.1652 | watch | 0.0000 |  |
| motor | overnight_parking | 0.1608 | watch | 0.0000 |  |
| addresses | energy_rating | 0.1600 | watch | 0.0000 |  |
| customers | household_size | 0.1578 | watch | 0.0000 |  |
| customers | marketing_segment | 0.1574 | watch | 0.0000 |  |
| addresses | roof_type | 0.1570 | watch | 0.0000 |  |
| buildings | smoke_alarms | 0.1548 | watch | 0.0000 |  |
| contents | high_value_items | 0.1526 | watch | 0.0000 |  |
| weather | wind_exposure_index | 0.1495 | watch |  |  |
| motor | fuel_type | 0.0858 | ok | 0.0000 |  |
| contents | tenure | 0.0739 | ok | 0.0000 |  |
| contents | renewal_count | 0.0685 | ok |  |  |
| buildings | written_premium | 0.0675 | ok |  |  |
| motor | renewal_count | 0.0650 | ok |  |  |
| buildings | renewal_count | 0.0634 | ok |  |  |
| motor | ncb_years | 0.0493 | ok |  |  |
| motor | years_licensed | 0.0428 | ok |  |  |
| motor | written_premium | 0.0352 | ok |  |  |
| contents | written_premium | 0.0248 | ok |  |  |

## 6. Adaptation checklist

1. [ ] column `date_of_birth` missing in `customers`: feature list must drop it  
   evidence: `schema_diff: customers / date_of_birth / missing_column / expected, not present`
2. [ ] column `credit_band` missing in `customers`: feature list must drop it  
   evidence: `schema_diff: customers / credit_band / missing_column / expected, not present`
3. [ ] new column `gender` in `customers`: screen as a candidate feature and check for leakage  
   evidence: `schema_diff: customers / gender / new_column / string`
4. [ ] column `driver_age` missing in `policies_motor`: feature list must drop it  
   evidence: `schema_diff: policies_motor / driver_age / missing_column / possible rename to 'driver_age_band'`
5. [ ] column `annual_mileage` missing in `policies_motor`: feature list must drop it  
   evidence: `schema_diff: policies_motor / annual_mileage / missing_column / expected, not present`
6. [ ] column `telematics_opt_in` missing in `policies_motor`: feature list must drop it  
   evidence: `schema_diff: policies_motor / telematics_opt_in / missing_column / expected, not present`
7. [ ] column `telematics_score` missing in `policies_motor`: feature list must drop it  
   evidence: `schema_diff: policies_motor / telematics_score / missing_column / expected, not present`
8. [ ] new column `driver_age_band` in `policies_motor`: screen as a candidate feature and check for leakage  
   evidence: `schema_diff: policies_motor / driver_age_band / new_column / string; possible rename of 'driver_age'`
9. [ ] column `driver_age` missing in `test_motor`: feature list must drop it  
   evidence: `schema_diff: test_motor / driver_age / missing_column / possible rename to 'driver_age_band'`
10. [ ] column `annual_mileage` missing in `test_motor`: feature list must drop it  
   evidence: `schema_diff: test_motor / annual_mileage / missing_column / expected, not present`
11. [ ] column `telematics_opt_in` missing in `test_motor`: feature list must drop it  
   evidence: `schema_diff: test_motor / telematics_opt_in / missing_column / expected, not present`
12. [ ] column `telematics_score` missing in `test_motor`: feature list must drop it  
   evidence: `schema_diff: test_motor / telematics_score / missing_column / expected, not present`
13. [ ] new column `driver_age_band` in `test_motor`: screen as a candidate feature and check for leakage  
   evidence: `schema_diff: test_motor / driver_age_band / new_column / string; possible rename of 'driver_age'`
14. [ ] column `policyholder_age` missing in `policies_buildings`: feature list must drop it  
   evidence: `schema_diff: policies_buildings / policyholder_age / missing_column / possible rename to 'policyholder_age_band'`
15. [ ] new column `policyholder_age_band` in `policies_buildings`: screen as a candidate feature and check for leakage  
   evidence: `schema_diff: policies_buildings / policyholder_age_band / new_column / string; possible rename of 'policyholder_age'`
16. [ ] column `policyholder_age` missing in `test_buildings`: feature list must drop it  
   evidence: `schema_diff: test_buildings / policyholder_age / missing_column / possible rename to 'policyholder_age_band'`
17. [ ] new column `policyholder_age_band` in `test_buildings`: screen as a candidate feature and check for leakage  
   evidence: `schema_diff: test_buildings / policyholder_age_band / new_column / string; possible rename of 'policyholder_age'`
18. [ ] column `policyholder_age` missing in `policies_contents`: feature list must drop it  
   evidence: `schema_diff: policies_contents / policyholder_age / missing_column / possible rename to 'policyholder_age_band'`
19. [ ] new column `policyholder_age_band` in `policies_contents`: screen as a candidate feature and check for leakage  
   evidence: `schema_diff: policies_contents / policyholder_age_band / new_column / string; possible rename of 'policyholder_age'`
20. [ ] column `policyholder_age` missing in `test_contents`: feature list must drop it  
   evidence: `schema_diff: test_contents / policyholder_age / missing_column / possible rename to 'policyholder_age_band'`
21. [ ] new column `policyholder_age_band` in `test_contents`: screen as a candidate feature and check for leakage  
   evidence: `schema_diff: test_contents / policyholder_age_band / new_column / string; possible rename of 'policyholder_age'`
22. [ ] new table `expenses`: inspect and decide whether to join it  
   evidence: `schema_diff: expenses / - / new_table / table not in Part 1`
23. [ ] new table `reference_policies`: inspect and decide whether to join it  
   evidence: `schema_diff: reference_policies / - / new_table / table not in Part 1`
24. [ ] `occupation` in `customers` is 12.2% missing: define imputation / missing-level handling  
   evidence: `dq_register: customers / missing rate > 1% / occupation / rows=16517 share=12.16%`
25. [ ] `marital_status` in `customers` is 12.0% missing: define imputation / missing-level handling  
   evidence: `dq_register: customers / missing rate > 1% / marital_status / rows=16348 share=12.03%`
26. [ ] `household_size` in `customers` is 11.8% missing: define imputation / missing-level handling  
   evidence: `dq_register: customers / missing rate > 1% / household_size / rows=16037 share=11.80%`
27. [ ] `years_at_address` in `customers` is 11.9% missing: define imputation / missing-level handling  
   evidence: `dq_register: customers / missing rate > 1% / years_at_address / rows=16226 share=11.94%`
28. [ ] `campaign_code` in `customers` is 20.2% missing: define imputation / missing-level handling  
   evidence: `dq_register: customers / missing rate > 1% / campaign_code / rows=27433 share=20.19%`
29. [ ] `occupation` in `customers` has inconsistent labels: add a normalisation map  
   evidence: `dq_register: customers / inconsistent casing/whitespace ('Retired', 'Student', 'retired', 'student', 'Retired ', 'Care worker') / occupation / rows=107364 share=79.03%`
30. [ ] `flood_zone` in `addresses` is 2.9% missing: define imputation / missing-level handling  
   evidence: `dq_register: addresses / missing rate > 1% / flood_zone / rows=4037 share=2.95%`
31. [ ] `distance_to_river_km` in `addresses` is 12.0% missing: define imputation / missing-level handling  
   evidence: `dq_register: addresses / missing rate > 1% / distance_to_river_km / rows=16421 share=11.99%`
32. [ ] `soil_type` in `addresses` is 12.0% missing: define imputation / missing-level handling  
   evidence: `dq_register: addresses / missing rate > 1% / soil_type / rows=16480 share=12.04%`
33. [ ] `roof_type` in `addresses` is 11.9% missing: define imputation / missing-level handling  
   evidence: `dq_register: addresses / missing rate > 1% / roof_type / rows=16337 share=11.93%`
34. [ ] `energy_rating` in `addresses` is 12.1% missing: define imputation / missing-level handling  
   evidence: `dq_register: addresses / missing rate > 1% / energy_rating / rows=16515 share=12.06%`
35. [ ] `repairer_id` in `claims` is 46.3% missing: define imputation / missing-level handling  
   evidence: `dq_register: claims / missing rate > 1% / repairer_id / rows=4283 share=46.26%`
36. [ ] `third_party_id` in `claims` is 70.7% missing: define imputation / missing-level handling  
   evidence: `dq_register: claims / missing rate > 1% / third_party_id / rows=6545 share=70.70%`
37. [ ] `vehicle_age` in `policies_motor` is 11.9% missing: define imputation / missing-level handling  
   evidence: `dq_register: policies_motor / missing rate > 1% / vehicle_age / rows=5604 share=11.85%`
38. [ ] `engine_size_cc` in `policies_motor` is 12.2% missing: define imputation / missing-level handling  
   evidence: `dq_register: policies_motor / missing rate > 1% / engine_size_cc / rows=5745 share=12.15%`
39. [ ] `overnight_parking` in `policies_motor` is 12.0% missing: define imputation / missing-level handling  
   evidence: `dq_register: policies_motor / missing rate > 1% / overnight_parking / rows=5697 share=12.05%`
40. [ ] exposure > 1.01 in `policies_motor`: add a validation/cleaning rule  
   evidence: `dq_register: policies_motor / exposure > 1.01 / exposure / rows=188 share=0.40%`
41. [ ] `vehicle_age` in `test_motor` is 12.2% missing: define imputation / missing-level handling  
   evidence: `dq_register: test_motor / missing rate > 1% / vehicle_age / rows=11070 share=12.23%`
42. [ ] `engine_size_cc` in `test_motor` is 12.2% missing: define imputation / missing-level handling  
   evidence: `dq_register: test_motor / missing rate > 1% / engine_size_cc / rows=11004 share=12.16%`
43. [ ] `overnight_parking` in `test_motor` is 12.0% missing: define imputation / missing-level handling  
   evidence: `dq_register: test_motor / missing rate > 1% / overnight_parking / rows=10859 share=12.00%`
44. [ ] `days_unoccupied` in `policies_buildings` is 12.0% missing: define imputation / missing-level handling  
   evidence: `dq_register: policies_buildings / missing rate > 1% / days_unoccupied / rows=3291 share=11.99%`
45. [ ] `smoke_alarms` in `policies_buildings` is 11.9% missing: define imputation / missing-level handling  
   evidence: `dq_register: policies_buildings / missing rate > 1% / smoke_alarms / rows=3266 share=11.90%`
46. [ ] sum_insured > 5e+06 in `policies_buildings`: add a validation/cleaning rule  
   evidence: `dq_register: policies_buildings / sum_insured > 5e+06 / sum_insured / rows=53 share=0.19%`
47. [ ] exposure > 1.01 in `policies_buildings`: add a validation/cleaning rule  
   evidence: `dq_register: policies_buildings / exposure > 1.01 / exposure / rows=117 share=0.43%`
48. [ ] `days_unoccupied` in `test_buildings` is 11.9% missing: define imputation / missing-level handling  
   evidence: `dq_register: test_buildings / missing rate > 1% / days_unoccupied / rows=6240 share=11.93%`
49. [ ] `smoke_alarms` in `test_buildings` is 12.1% missing: define imputation / missing-level handling  
   evidence: `dq_register: test_buildings / missing rate > 1% / smoke_alarms / rows=6341 share=12.13%`
50. [ ] sum_insured > 5e+06 in `test_buildings`: add a validation/cleaning rule  
   evidence: `dq_register: test_buildings / sum_insured > 5e+06 / sum_insured / rows=104 share=0.20%`
51. [ ] `security_level` in `policies_contents` is 12.3% missing: define imputation / missing-level handling  
   evidence: `dq_register: policies_contents / missing rate > 1% / security_level / rows=4238 share=12.26%`
52. [ ] `high_value_items` in `policies_contents` is 11.8% missing: define imputation / missing-level handling  
   evidence: `dq_register: policies_contents / missing rate > 1% / high_value_items / rows=4067 share=11.77%`
53. [ ] sum_insured > 1e+06 in `policies_contents`: add a validation/cleaning rule  
   evidence: `dq_register: policies_contents / sum_insured > 1e+06 / sum_insured / rows=57 share=0.16%`
54. [ ] exposure > 1.01 in `policies_contents`: add a validation/cleaning rule  
   evidence: `dq_register: policies_contents / exposure > 1.01 / exposure / rows=146 share=0.42%`
55. [ ] `security_level` in `test_contents` is 12.2% missing: define imputation / missing-level handling  
   evidence: `dq_register: test_contents / missing rate > 1% / security_level / rows=8050 share=12.15%`
56. [ ] `high_value_items` in `test_contents` is 12.0% missing: define imputation / missing-level handling  
   evidence: `dq_register: test_contents / missing rate > 1% / high_value_items / rows=7954 share=12.01%`
57. [ ] sum_insured > 1e+06 in `test_contents`: add a validation/cleaning rule  
   evidence: `dq_register: test_contents / sum_insured > 1e+06 / sum_insured / rows=135 share=0.20%`
58. [ ] Part 1 fix still needed (exposure recorded in days, not years) in `policies_motor`: 47 rows, keep/re-validate the rule  
   evidence: `dq_register: policies_motor / exposure recorded in days, not years / - / rows=47 share=0.10%`
59. [ ] Part 1 fix still needed (exposure recorded in days, not years) in `policies_buildings`: 33 rows, keep/re-validate the rule  
   evidence: `dq_register: policies_buildings / exposure recorded in days, not years / - / rows=33 share=0.12%`
60. [ ] Part 1 fix still needed (sum_insured x1000 unit error) in `policies_buildings`: 53 rows, keep/re-validate the rule  
   evidence: `dq_register: policies_buildings / sum_insured x1000 unit error / - / rows=53 share=0.19%`
61. [ ] Part 1 fix still needed (construction_type inconsistent labels) in `policies_buildings`: 27,447 rows, keep/re-validate the rule  
   evidence: `dq_register: policies_buildings / construction_type inconsistent labels / - / rows=27447 share=100.00%`
62. [ ] Part 1 fix still needed (sum_insured x1000 unit error) in `test_buildings`: 104 rows, keep/re-validate the rule  
   evidence: `dq_register: test_buildings / sum_insured x1000 unit error / - / rows=104 share=0.20%`
63. [ ] Part 1 fix still needed (construction_type inconsistent labels) in `test_buildings`: 52,295 rows, keep/re-validate the rule  
   evidence: `dq_register: test_buildings / construction_type inconsistent labels / - / rows=52295 share=100.00%`
64. [ ] Part 1 fix still needed (exposure recorded in days, not years) in `policies_contents`: 34 rows, keep/re-validate the rule  
   evidence: `dq_register: policies_contents / exposure recorded in days, not years / - / rows=34 share=0.10%`
65. [ ] Part 1 fix still needed (sum_insured x1000 unit error) in `policies_contents`: 57 rows, keep/re-validate the rule  
   evidence: `dq_register: policies_contents / sum_insured x1000 unit error / - / rows=57 share=0.16%`
66. [ ] Part 1 fix still needed (sum_insured x1000 unit error) in `test_contents`: 135 rows, keep/re-validate the rule  
   evidence: `dq_register: test_contents / sum_insured x1000 unit error / - / rows=135 share=0.20%`
67. [ ] `vehicle_age` (motor) has shifted (PSI 0.74): re-check relativity, consider re-binning or dropping  
   evidence: `drift_report: motor / vehicle_age / psi=0.74`
68. [ ] `channel` (motor) has shifted (PSI 1.13): re-check relativity, consider re-binning or dropping  
   evidence: `drift_report: motor / channel / psi=1.13`
69. [ ] `construction_type` (buildings) has shifted (PSI 17.48): re-check relativity, consider re-binning or dropping  
   evidence: `drift_report: buildings / construction_type / psi=17.48`
70. [ ] `channel` (buildings) has shifted (PSI 1.11): re-check relativity, consider re-binning or dropping  
   evidence: `drift_report: buildings / channel / psi=1.11`
71. [ ] `channel` (contents) has shifted (PSI 1.12): re-check relativity, consider re-binning or dropping  
   evidence: `drift_report: contents / channel / psi=1.12`
72. [ ] `occupation` (customers) has shifted (PSI 2.54): re-check relativity, consider re-binning or dropping  
   evidence: `drift_report: customers / occupation / psi=2.54`
73. [ ] `customer_since_year` (customers) has shifted (PSI 2.06): re-check relativity, consider re-binning or dropping  
   evidence: `drift_report: customers / customer_since_year / psi=2.06`
74. [ ] `campaign_code` (customers) has shifted (PSI 0.44): re-check relativity, consider re-binning or dropping  
   evidence: `drift_report: customers / campaign_code / psi=0.44`
75. [ ] `region` (addresses) has shifted (PSI 14.30): re-check relativity, consider re-binning or dropping  
   evidence: `drift_report: addresses / region / psi=14.30`
76. [ ] `year` (weather) has shifted (PSI 3.24): re-check relativity, consider re-binning or dropping  
   evidence: `drift_report: weather / year / psi=3.24`
77. [ ] `hail_days` (weather) has shifted (PSI 1.67): re-check relativity, consider re-binning or dropping  
   evidence: `drift_report: weather / hail_days / psi=1.67`
78. [ ] `freezing_days` (weather) has shifted (PSI 1.43): re-check relativity, consider re-binning or dropping  
   evidence: `drift_report: weather / freezing_days / psi=1.43`
79. [ ] `dry_summer_index` (weather) has shifted (PSI 1.34): re-check relativity, consider re-binning or dropping  
   evidence: `drift_report: weather / dry_summer_index / psi=1.34`
80. [ ] `wildfire_index` (weather) has shifted (PSI 6.67): re-check relativity, consider re-binning or dropping  
   evidence: `drift_report: weather / wildfire_index / psi=6.67`
81. [ ] `rainfall_mm` (weather) has shifted (PSI 1.80): re-check relativity, consider re-binning or dropping  
   evidence: `drift_report: weather / rainfall_mm / psi=1.80`
82. [ ] `mean_temperature_c` (weather) has shifted (PSI 2.33): re-check relativity, consider re-binning or dropping  
   evidence: `drift_report: weather / mean_temperature_c / psi=2.33`
83. [ ] new level(s) retired, student, Retired , Student , Self-employed retired in `occupation` (customers), 50.2% of rows: category handling (map to other / grouped level)  
   evidence: `drift_report: customers / occupation / unseen_share=0.502`
84. [ ] new level(s) E03, E06, E02, E05, E01 in `region` (addresses), 100.0% of rows: category handling (map to other / grouped level)  
   evidence: `drift_report: addresses / region / unseen_share=1.000`
85. [ ] 14 column(s) on watch (PSI 0.1-0.25): motor.overnight_parking, motor.discount_applied, buildings.smoke_alarms, buildings.discount_applied, contents.security_level, contents.high_value_items, contents.discount_applied, customers.marital_status, customers.household_size, customers.marketing_segment, addresses.soil_type, addresses.roof_type, addresses.energy_rating, weather.wind_exposure_index  
   evidence: `drift_report: flag=watch`
86. [ ] buildings: latest loss year 2024 looks immature (open share/claim count): develop it before using as a target  
   evidence: `claims_development: buildings / loss_year=2024 / open_share=0.253 n=537`
87. [ ] contents: latest loss year 2024 looks immature (open share/claim count): develop it before using as a target  
   evidence: `claims_development: contents / loss_year=2024 / open_share=0.115 n=512`
88. [ ] motor: latest loss year 2024 looks immature (open share/claim count): develop it before using as a target  
   evidence: `claims_development: motor / loss_year=2024 / open_share=0.200 n=3549`
89. [ ] contents: pure_premium is 0.80x Part 1: recalibrate the base level (do not carry Part 1 intercept)  
   evidence: `portfolio_summary: contents pure_premium new=88.9 ref=111.2`
90. [ ] motor: frequency is 1.29x Part 1: recalibrate the base level (do not carry Part 1 intercept)  
   evidence: `portfolio_summary: motor frequency new=0.1809 ref=0.1404`
91. [ ] motor: severity is 0.58x Part 1: recalibrate the base level (do not carry Part 1 intercept)  
   evidence: `portfolio_summary: motor severity new=2250 ref=3880`
92. [ ] motor: pure_premium is 0.75x Part 1: recalibrate the base level (do not carry Part 1 intercept)  
   evidence: `portfolio_summary: motor pure_premium new=406.9 ref=544.7`
93. [ ] training years are [2023, 2024] (Part 1: 2021-2024): revisit backtest folds and year trend  
   evidence: `portfolio_summary: train years`

## 7. Calendar-proxy features

_Not computed (needs the engineered training frame: run with the market spec active)._

