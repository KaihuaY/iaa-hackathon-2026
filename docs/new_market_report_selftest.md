# New market readiness report

Dataset: `data\raw\public`  
Reference (Part 1): `data/raw/public`

## 1. Schema differences

_No schema differences vs Part 1._

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
| claims | claims | generic | missing rate > 1% | repairer_id | 80360 | 0.4926 |
| claims | claims | generic | missing rate > 1% | third_party_id | 121741 | 0.7462 |
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
| policies_buildings | buildings | generic | sum_insured > 1e7 | sum_insured | 1395 | 0.0020 |
| policies_buildings | buildings | generic | exposure > 1.01 | exposure | 3039 | 0.0043 |
| policies_buildings | buildings | generic | inconsistent casing/whitespace ('Brick', 'non_standard', 'stone', 'W', 'T', 'brick') | construction_type | 707726 | 1.0000 |
| test_buildings | buildings | generic | missing rate > 1% | days_unoccupied | 4586 | 0.0249 |
| test_buildings | buildings | generic | missing rate > 1% | smoke_alarms | 4602 | 0.0250 |
| test_buildings | buildings | generic | sum_insured > 1e7 | sum_insured | 370 | 0.0020 |
| test_buildings | buildings | generic | inconsistent casing/whitespace ('Brick', 'W', 'non_standard', 'stone', 'brick', 'BRICK') | construction_type | 184155 | 1.0000 |
| policies_contents | contents | generic | missing rate > 1% | security_level | 18087 | 0.0251 |
| policies_contents | contents | generic | missing rate > 1% | high_value_items | 18088 | 0.0251 |
| policies_contents | contents | generic | sum_insured > 1e7 | sum_insured | 1395 | 0.0019 |
| policies_contents | contents | generic | exposure > 1.01 | exposure | 3082 | 0.0043 |
| test_contents | contents | generic | missing rate > 1% | security_level | 4677 | 0.0249 |
| test_contents | contents | generic | missing rate > 1% | high_value_items | 4707 | 0.0251 |
| test_contents | contents | generic | sum_insured > 1e7 | sum_insured | 359 | 0.0019 |
| policies_motor | motor | cleaning | customer_id lost leading zeros |  | 98110 | 0.1001 |
| policies_motor | motor | cleaning | exposure recorded in days, not years |  | 934 | 0.0010 |
| policies_motor | motor | cleaning | years_licensed impossible for driver_age |  | 2962 | 0.0030 |
| test_motor | motor (test) | cleaning | customer_id lost leading zeros |  | 25554 | 0.1002 |
| test_motor | motor (test) | cleaning | years_licensed impossible for driver_age |  | 755 | 0.0030 |
| policies_buildings | buildings | cleaning | exposure recorded in days, not years |  | 656 | 0.0009 |
| policies_buildings | buildings | cleaning | sum_insured x1000 unit error |  | 1395 | 0.0020 |
| policies_buildings | buildings | cleaning | construction_type inconsistent labels |  | 641380 | 0.9063 |
| test_buildings | buildings (test) | cleaning | sum_insured x1000 unit error |  | 370 | 0.0020 |
| test_buildings | buildings (test) | cleaning | construction_type inconsistent labels |  | 166647 | 0.9049 |
| policies_contents | contents | cleaning | exposure recorded in days, not years |  | 729 | 0.0010 |
| policies_contents | contents | cleaning | sum_insured x1000 unit error |  | 1402 | 0.0019 |
| test_contents | contents (test) | cleaning | sum_insured x1000 unit error |  | 360 | 0.0019 |
| claims | claims | cleaning | duplicate claims under new claim_id |  | 604 | 0.0037 |

## 3. Portfolio and target overview

| set | line | year | rows | exposure | claims | cost | premium | frequency | severity | pure_premium | loss_ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|
| train | motor | 2021 | 216008 | 188,004.0 | 27,354.0 | 105,897,441.1 | 116,190,196.4 | 0.1455 | 3,871.4 | 563.3 | 0.9114 |
| train | motor | 2022 | 254685 | 199,712.6 | 29,288.0 | 114,890,724.9 | 139,961,366.6 | 0.1467 | 3,922.8 | 575.3 | 0.8209 |
| train | motor | 2023 | 254577 | 199,695.2 | 27,990.0 | 117,307,703.7 | 144,747,852.1 | 0.1402 | 4,191.1 | 587.4 | 0.8104 |
| train | motor | 2024 | 254698 | 199,800.2 | 25,864.0 | 90,666,976.9 | 148,941,359.8 | 0.1294 | 3,505.5 | 453.8 | 0.6087 |
| test | motor | 2025 | 254986 | 199,983.6 |  |  |  |  |  |  |  |
| train | buildings | 2021 | 155729 | 140,792.9 | 6,195.0 | 54,052,834.2 | 46,233,270.4 | 0.0440 | 8,725.2 | 383.9 | 1.1691 |
| train | buildings | 2022 | 183864 | 149,882.6 | 6,683.0 | 60,417,182.3 | 57,774,370.1 | 0.0446 | 9,040.4 | 403.1 | 1.0457 |
| train | buildings | 2023 | 184132 | 149,997.7 | 6,817.0 | 64,832,471.2 | 60,706,176.4 | 0.0454 | 9,510.4 | 432.2 | 1.0680 |
| train | buildings | 2024 | 184001 | 149,927.9 | 6,571.0 | 67,353,949.5 | 63,946,482.8 | 0.0438 | 10,250.2 | 449.2 | 1.0533 |
| test | buildings | 2025 | 184155 | 150,243.9 |  |  |  |  |  |  |  |
| train | contents | 2021 | 158848 | 140,941.1 | 5,582.0 | 14,861,300.7 | 19,864,972.6 | 0.0396 | 2,662.4 | 105.4 | 0.7481 |
| train | contents | 2022 | 187632 | 149,997.1 | 5,942.0 | 15,820,781.9 | 24,713,824.1 | 0.0396 | 2,662.5 | 105.5 | 0.6402 |
| train | contents | 2023 | 187646 | 149,985.1 | 6,158.0 | 17,539,466.9 | 26,059,827.9 | 0.0411 | 2,848.2 | 116.9 | 0.6730 |
| train | contents | 2024 | 187491 | 149,977.1 | 5,988.0 | 17,516,035.1 | 27,456,066.1 | 0.0399 | 2,925.2 | 116.8 | 0.6380 |
| test | contents | 2025 | 187498 | 150,123.8 |  |  |  |  |  |  |  |


### Level versus Part 1 (train, all years)

| line | frequency_new | frequency_ref | frequency_ratio | severity_new | severity_ref | severity_ratio | pure_premium_new | pure_premium_ref | pure_premium_ratio |
|---|---|---|---|---|---|---|---|---|---|
| buildings | 0.0445 | 0.0445 | 1.0000 | 9,390.7 | 9,390.7 | 1.0000 | 417.6 | 417.6 | 1.0000 |
| contents | 0.0401 | 0.0401 | 1.0000 | 2,777.3 | 2,777.3 | 1.0000 | 111.2 | 111.2 | 1.0000 |
| motor | 0.1404 | 0.1404 | 1.0000 | 3,880.3 | 3,880.3 | 1.0000 | 544.7 | 544.7 | 1.0000 |

## 4. Claims development

| line | loss_year | n_claims | open_share | lag_p50 | lag_p90 | lag_p99 | lag_max |
|---|---|---|---|---|---|---|---|
| buildings | all | 26442 | 0.0716 | 9.0000 | 34.0 | 239.0 | 759 |
| buildings | 2020 | 46 | 0.0000 | 249.0 | 413.0 | 512.7 | 542 |
| buildings | 2021 | 6204 | 0.0027 | 9.0000 | 33.0 | 243.0 | 629 |
| buildings | 2022 | 6797 | 0.0081 | 9.0000 | 34.4 | 322.1 | 759 |
| buildings | 2023 | 6824 | 0.0295 | 9.0000 | 35.0 | 268.8 | 601 |
| buildings | 2024 | 6571 | 0.2465 | 8.0000 | 30.0 | 112.3 | 289 |
| contents | all | 24102 | 0.0336 | 6.0000 | 23.0 | 63.0 | 565 |
| contents | 2020 | 30 | 0.0000 | 355.5 | 468.1 | 524.9 | 540 |
| contents | 2021 | 5650 | 0.0000 | 6.0000 | 24.0 | 67.5 | 565 |
| contents | 2022 | 6105 | 0.0000 | 7.0000 | 24.0 | 62.0 | 541 |
| contents | 2023 | 6242 | 0.0030 | 6.0000 | 23.0 | 61.6 | 557 |
| contents | 2024 | 6075 | 0.1302 | 6.0000 | 22.6 | 49.3 | 102 |
| motor | all | 112606 | 0.0661 | 9.0000 | 42.0 | 377.0 | 1361 |
| motor | 2020 | 198 | 0.0101 | 265.5 | 482.6 | 623.2 | 674 |
| motor | 2021 | 27745 | 0.0116 | 9.0000 | 48.0 | 459.6 | 1361 |
| motor | 2022 | 30102 | 0.0202 | 9.0000 | 44.0 | 425.0 | 1001 |
| motor | 2023 | 28390 | 0.0384 | 9.0000 | 44.0 | 353.1 | 658 |
| motor | 2024 | 26171 | 0.2073 | 8.0000 | 34.0 | 138.3 | 359 |

## 5. Drift vs Part 1

Max PSI 0.0000; shift (>0.25): 0, watch (0.1-0.25): 0, columns compared: 109.

| line | column | psi | flag | unseen_share | new_levels |
|---|---|---|---|---|---|
| motor | exposure | 0.0000 | ok |  |  |
| motor | driver_age | 0.0000 | ok |  |  |
| motor | years_licensed | 0.0000 | ok |  |  |
| motor | ncb_years | 0.0000 | ok |  |  |
| motor | vehicle_value_group | 0.0000 | ok |  |  |
| motor | vehicle_age | 0.0000 | ok |  |  |
| motor | make_model | 0.0000 | ok | 0.0000 |  |
| motor | engine_size_cc | 0.0000 | ok |  |  |
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
| motor | telematics_opt_in | 0.0000 | ok | 0.0000 |  |
| motor | telematics_score | 0.0000 | ok |  |  |
| motor | voluntary_excess | 0.0000 | ok | 0.0000 |  |
| motor | compulsory_excess | 0.0000 | ok | 0.0000 |  |
| motor | channel | 0.0000 | ok | 0.0000 |  |
| motor | payment_method | 0.0000 | ok | 0.0000 |  |
| motor | missed_payments_12m | 0.0000 | ok | 0.0000 |  |
| motor | new_business | 0.0000 | ok | 0.0000 |  |
| motor | renewal_count | 0.0000 | ok |  |  |
| motor | days_quote_to_inception | 0.0000 | ok |  |  |
| motor | discount_applied | 0.0000 | ok | 0.0000 |  |
| motor | written_premium | 0.0000 | ok |  |  |
| motor | licence_inconsistent | 0.0000 | ok | 0.0000 |  |
| buildings | exposure | 0.0000 | ok |  |  |
| buildings | policyholder_age | 0.0000 | ok |  |  |
| buildings | sum_insured | 0.0000 | ok |  |  |
| buildings | construction_type | 0.0000 | ok | 0.0000 |  |
| buildings | occupancy | 0.0000 | ok | 0.0000 |  |
| buildings | days_unoccupied | 0.0000 | ok |  |  |
| buildings | smoke_alarms | 0.0000 | ok | 0.0000 |  |
| buildings | accidental_damage_cover | 0.0000 | ok | 0.0000 |  |

## 6. Adaptation checklist

1. [ ] `occupation` in `customers` is 2.5% missing: define imputation / missing-level handling  
   evidence: `dq_register: customers / missing rate > 1% / occupation / rows=13462 share=2.47%`
2. [ ] `marital_status` in `customers` is 2.5% missing: define imputation / missing-level handling  
   evidence: `dq_register: customers / missing rate > 1% / marital_status / rows=13762 share=2.52%`
3. [ ] `household_size` in `customers` is 2.5% missing: define imputation / missing-level handling  
   evidence: `dq_register: customers / missing rate > 1% / household_size / rows=13444 share=2.47%`
4. [ ] `years_at_address` in `customers` is 2.5% missing: define imputation / missing-level handling  
   evidence: `dq_register: customers / missing rate > 1% / years_at_address / rows=13701 share=2.51%`
5. [ ] `campaign_code` in `customers` is 51.2% missing: define imputation / missing-level handling  
   evidence: `dq_register: customers / missing rate > 1% / campaign_code / rows=279085 share=51.19%`
6. [ ] `flood_zone` in `addresses` is 2.9% missing: define imputation / missing-level handling  
   evidence: `dq_register: addresses / missing rate > 1% / flood_zone / rows=16823 share=2.92%`
7. [ ] `distance_to_river_km` in `addresses` is 2.5% missing: define imputation / missing-level handling  
   evidence: `dq_register: addresses / missing rate > 1% / distance_to_river_km / rows=14373 share=2.50%`
8. [ ] `soil_type` in `addresses` is 2.5% missing: define imputation / missing-level handling  
   evidence: `dq_register: addresses / missing rate > 1% / soil_type / rows=14392 share=2.50%`
9. [ ] `roof_type` in `addresses` is 2.5% missing: define imputation / missing-level handling  
   evidence: `dq_register: addresses / missing rate > 1% / roof_type / rows=14401 share=2.50%`
10. [ ] `energy_rating` in `addresses` is 2.5% missing: define imputation / missing-level handling  
   evidence: `dq_register: addresses / missing rate > 1% / energy_rating / rows=14443 share=2.51%`
11. [ ] `repairer_id` in `claims` is 49.3% missing: define imputation / missing-level handling  
   evidence: `dq_register: claims / missing rate > 1% / repairer_id / rows=80360 share=49.26%`
12. [ ] `third_party_id` in `claims` is 74.6% missing: define imputation / missing-level handling  
   evidence: `dq_register: claims / missing rate > 1% / third_party_id / rows=121741 share=74.62%`
13. [ ] `vehicle_age` in `policies_motor` is 2.5% missing: define imputation / missing-level handling  
   evidence: `dq_register: policies_motor / missing rate > 1% / vehicle_age / rows=24427 share=2.49%`
14. [ ] `engine_size_cc` in `policies_motor` is 2.5% missing: define imputation / missing-level handling  
   evidence: `dq_register: policies_motor / missing rate > 1% / engine_size_cc / rows=24694 share=2.52%`
15. [ ] `annual_mileage` in `policies_motor` is 2.5% missing: define imputation / missing-level handling  
   evidence: `dq_register: policies_motor / missing rate > 1% / annual_mileage / rows=24711 share=2.52%`
16. [ ] `overnight_parking` in `policies_motor` is 2.5% missing: define imputation / missing-level handling  
   evidence: `dq_register: policies_motor / missing rate > 1% / overnight_parking / rows=24371 share=2.49%`
17. [ ] `telematics_score` in `policies_motor` is 87.6% missing: define imputation / missing-level handling  
   evidence: `dq_register: policies_motor / missing rate > 1% / telematics_score / rows=858223 share=87.58%`
18. [ ] exposure > 1.01 in `policies_motor`: add a validation/cleaning rule  
   evidence: `dq_register: policies_motor / exposure > 1.01 / exposure / rows=3973 share=0.41%`
19. [ ] `vehicle_age` in `test_motor` is 2.5% missing: define imputation / missing-level handling  
   evidence: `dq_register: test_motor / missing rate > 1% / vehicle_age / rows=6316 share=2.48%`
20. [ ] `engine_size_cc` in `test_motor` is 2.5% missing: define imputation / missing-level handling  
   evidence: `dq_register: test_motor / missing rate > 1% / engine_size_cc / rows=6440 share=2.53%`
21. [ ] `annual_mileage` in `test_motor` is 2.5% missing: define imputation / missing-level handling  
   evidence: `dq_register: test_motor / missing rate > 1% / annual_mileage / rows=6271 share=2.46%`
22. [ ] `overnight_parking` in `test_motor` is 2.5% missing: define imputation / missing-level handling  
   evidence: `dq_register: test_motor / missing rate > 1% / overnight_parking / rows=6398 share=2.51%`
23. [ ] `telematics_score` in `test_motor` is 86.7% missing: define imputation / missing-level handling  
   evidence: `dq_register: test_motor / missing rate > 1% / telematics_score / rows=221126 share=86.72%`
24. [ ] `days_unoccupied` in `policies_buildings` is 2.5% missing: define imputation / missing-level handling  
   evidence: `dq_register: policies_buildings / missing rate > 1% / days_unoccupied / rows=17587 share=2.49%`
25. [ ] `smoke_alarms` in `policies_buildings` is 2.5% missing: define imputation / missing-level handling  
   evidence: `dq_register: policies_buildings / missing rate > 1% / smoke_alarms / rows=17867 share=2.52%`
26. [ ] sum_insured > 1e7 in `policies_buildings`: add a validation/cleaning rule  
   evidence: `dq_register: policies_buildings / sum_insured > 1e7 / sum_insured / rows=1395 share=0.20%`
27. [ ] exposure > 1.01 in `policies_buildings`: add a validation/cleaning rule  
   evidence: `dq_register: policies_buildings / exposure > 1.01 / exposure / rows=3039 share=0.43%`
28. [ ] `construction_type` in `policies_buildings` has inconsistent labels: add a normalisation map  
   evidence: `dq_register: policies_buildings / inconsistent casing/whitespace ('Brick', 'non_standard', 'stone', 'W', 'T', 'brick') / construction_type / rows=707726 share=100.00%`
29. [ ] `days_unoccupied` in `test_buildings` is 2.5% missing: define imputation / missing-level handling  
   evidence: `dq_register: test_buildings / missing rate > 1% / days_unoccupied / rows=4586 share=2.49%`
30. [ ] `smoke_alarms` in `test_buildings` is 2.5% missing: define imputation / missing-level handling  
   evidence: `dq_register: test_buildings / missing rate > 1% / smoke_alarms / rows=4602 share=2.50%`
31. [ ] sum_insured > 1e7 in `test_buildings`: add a validation/cleaning rule  
   evidence: `dq_register: test_buildings / sum_insured > 1e7 / sum_insured / rows=370 share=0.20%`
32. [ ] `construction_type` in `test_buildings` has inconsistent labels: add a normalisation map  
   evidence: `dq_register: test_buildings / inconsistent casing/whitespace ('Brick', 'W', 'non_standard', 'stone', 'brick', 'BRICK') / construction_type / rows=184155 share=100.00%`
33. [ ] `security_level` in `policies_contents` is 2.5% missing: define imputation / missing-level handling  
   evidence: `dq_register: policies_contents / missing rate > 1% / security_level / rows=18087 share=2.51%`
34. [ ] `high_value_items` in `policies_contents` is 2.5% missing: define imputation / missing-level handling  
   evidence: `dq_register: policies_contents / missing rate > 1% / high_value_items / rows=18088 share=2.51%`
35. [ ] sum_insured > 1e7 in `policies_contents`: add a validation/cleaning rule  
   evidence: `dq_register: policies_contents / sum_insured > 1e7 / sum_insured / rows=1395 share=0.19%`
36. [ ] exposure > 1.01 in `policies_contents`: add a validation/cleaning rule  
   evidence: `dq_register: policies_contents / exposure > 1.01 / exposure / rows=3082 share=0.43%`
37. [ ] `security_level` in `test_contents` is 2.5% missing: define imputation / missing-level handling  
   evidence: `dq_register: test_contents / missing rate > 1% / security_level / rows=4677 share=2.49%`
38. [ ] `high_value_items` in `test_contents` is 2.5% missing: define imputation / missing-level handling  
   evidence: `dq_register: test_contents / missing rate > 1% / high_value_items / rows=4707 share=2.51%`
39. [ ] sum_insured > 1e7 in `test_contents`: add a validation/cleaning rule  
   evidence: `dq_register: test_contents / sum_insured > 1e7 / sum_insured / rows=359 share=0.19%`
40. [ ] Part 1 fix still needed (customer_id lost leading zeros) in `policies_motor`: 98,110 rows, keep/re-validate the rule  
   evidence: `dq_register: policies_motor / customer_id lost leading zeros / - / rows=98110 share=10.01%`
41. [ ] Part 1 fix still needed (exposure recorded in days, not years) in `policies_motor`: 934 rows, keep/re-validate the rule  
   evidence: `dq_register: policies_motor / exposure recorded in days, not years / - / rows=934 share=0.10%`
42. [ ] Part 1 fix still needed (years_licensed impossible for driver_age) in `policies_motor`: 2,962 rows, keep/re-validate the rule  
   evidence: `dq_register: policies_motor / years_licensed impossible for driver_age / - / rows=2962 share=0.30%`
43. [ ] Part 1 fix still needed (customer_id lost leading zeros) in `test_motor`: 25,554 rows, keep/re-validate the rule  
   evidence: `dq_register: test_motor / customer_id lost leading zeros / - / rows=25554 share=10.02%`
44. [ ] Part 1 fix still needed (years_licensed impossible for driver_age) in `test_motor`: 755 rows, keep/re-validate the rule  
   evidence: `dq_register: test_motor / years_licensed impossible for driver_age / - / rows=755 share=0.30%`
45. [ ] Part 1 fix still needed (exposure recorded in days, not years) in `policies_buildings`: 656 rows, keep/re-validate the rule  
   evidence: `dq_register: policies_buildings / exposure recorded in days, not years / - / rows=656 share=0.09%`
46. [ ] Part 1 fix still needed (sum_insured x1000 unit error) in `policies_buildings`: 1,395 rows, keep/re-validate the rule  
   evidence: `dq_register: policies_buildings / sum_insured x1000 unit error / - / rows=1395 share=0.20%`
47. [ ] Part 1 fix still needed (construction_type inconsistent labels) in `policies_buildings`: 641,380 rows, keep/re-validate the rule  
   evidence: `dq_register: policies_buildings / construction_type inconsistent labels / - / rows=641380 share=90.63%`
48. [ ] Part 1 fix still needed (sum_insured x1000 unit error) in `test_buildings`: 370 rows, keep/re-validate the rule  
   evidence: `dq_register: test_buildings / sum_insured x1000 unit error / - / rows=370 share=0.20%`
49. [ ] Part 1 fix still needed (construction_type inconsistent labels) in `test_buildings`: 166,647 rows, keep/re-validate the rule  
   evidence: `dq_register: test_buildings / construction_type inconsistent labels / - / rows=166647 share=90.49%`
50. [ ] Part 1 fix still needed (exposure recorded in days, not years) in `policies_contents`: 729 rows, keep/re-validate the rule  
   evidence: `dq_register: policies_contents / exposure recorded in days, not years / - / rows=729 share=0.10%`
51. [ ] Part 1 fix still needed (sum_insured x1000 unit error) in `policies_contents`: 1,402 rows, keep/re-validate the rule  
   evidence: `dq_register: policies_contents / sum_insured x1000 unit error / - / rows=1402 share=0.19%`
52. [ ] Part 1 fix still needed (sum_insured x1000 unit error) in `test_contents`: 360 rows, keep/re-validate the rule  
   evidence: `dq_register: test_contents / sum_insured x1000 unit error / - / rows=360 share=0.19%`
53. [ ] Part 1 fix still needed (duplicate claims under new claim_id) in `claims`: 604 rows, keep/re-validate the rule  
   evidence: `dq_register: claims / duplicate claims under new claim_id / - / rows=604 share=0.37%`
54. [ ] buildings: latest loss year 2024 looks immature (open share/claim count): develop it before using as a target  
   evidence: `claims_development: buildings / loss_year=2024 / open_share=0.247 n=6571`
55. [ ] contents: latest loss year 2024 looks immature (open share/claim count): develop it before using as a target  
   evidence: `claims_development: contents / loss_year=2024 / open_share=0.130 n=6075`
56. [ ] motor: latest loss year 2024 looks immature (open share/claim count): develop it before using as a target  
   evidence: `claims_development: motor / loss_year=2024 / open_share=0.207 n=26171`
