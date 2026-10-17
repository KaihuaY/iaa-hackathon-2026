# Level-frequency stability screen - market `rehearsal`

stability = exposure-weighted correlation of level claim frequencies, years 2019-2020 vs 2022; spread = exposure-weighted sd of level frequencies / overall frequency. Levels need >= 300 rows in both periods. noise: stability < 0.2 and spread < 0.1; watch: 0.2-0.5. Weather variables vary by year by design: do not drop them on this screen alone.

## motor

noise: soil_type, make_model, branch_id, bedrooms, distance_to_river_km, campaign_code, email_domain, rainfall_mm, energy_rating, seats, flood_zone

watch: fuel_type, wildfire_index, dry_summer_index, property_age, elevation_m

| feature | kind | levels | stability | spread |
|---|---|---|---|---|
| cover_type | cat | 3 | 1.00 | 0.235 |
| voluntary_excess | num | 4 | 1.00 | 0.151 |
| marital_status | cat | 4 | 1.00 | 0.150 |
| vehicle_use | cat | 3 | 1.00 | 0.063 |
| cust_expo | num | 4 | 1.00 | 0.098 |
| missed_payments_12m | num | 4 | 1.00 | 0.094 |
| channel | cat | 3 | 1.00 | 0.075 |
| discount_applied | num | 3 | 1.00 | 0.101 |
| licence_points | num | 4 | 1.00 | 0.051 |
| urban_density | cat | 4 | 0.99 | 0.086 |
| prior_claims_5y | num | 4 | 0.99 | 0.136 |
| customer_age | num | 8 | 0.99 | 0.365 |
| driver_age | num | 8 | 0.99 | 0.361 |
| annual_mileage | num | 6 | 0.99 | 0.169 |
| credit_band | cat | 5 | 0.99 | 0.086 |
| row_type | cat | 4 | 0.99 | 0.099 |
| marketing_segment | cat | 12 | 0.99 | 0.323 |
| years_licensed | num | 8 | 0.99 | 0.358 |
| occupation | cat | 20 | 0.99 | 0.196 |
| years_at_address | num | 8 | 0.98 | 0.126 |
| body_type | cat | 5 | 0.98 | 0.092 |
| pol_expo | num | 3 | 0.97 | 0.094 |
| cust_motor_expo | num | 3 | 0.97 | 0.094 |
| household_size | num | 7 | 0.97 | 0.068 |
| ncb_years | num | 8 | 0.97 | 0.391 |
| region | cat | 12 | 0.96 | 0.118 |
| telematics_score | num | 8 | 0.96 | 0.145 |
| customer_tenure | num | 7 | 0.95 | 0.107 |
| renewal_count | num | 7 | 0.95 | 0.107 |
| vehicle_value_group | num | 8 | 0.92 | 0.056 |
| days_quote_to_inception | num | 7 | 0.91 | 0.027 |
| overnight_parking | cat | 3 | 0.90 | 0.014 |
| engine_size_cc | num | 8 | 0.88 | 0.053 |
| property_type | cat | 4 | 0.86 | 0.014 |
| freezing_days | num | 8 | 0.80 | 0.045 |
| vehicle_colour | cat | 8 | 0.72 | 0.017 |
| mean_temperature_c | num | 8 | 0.69 | 0.051 |
| wind_exposure_index | num | 8 | 0.66 | 0.028 |
| vehicle_age | num | 8 | 0.66 | 0.047 |
| document_preference | cat | 3 | 0.55 | 0.008 |
| hail_days | num | 8 | 0.54 | 0.017 |
| fuel_type | cat | 4 | 0.48 | 0.007 |
| wildfire_index | num | 7 | 0.45 | 0.022 |
| dry_summer_index | num | 8 | 0.41 | 0.048 |
| property_age | num | 8 | 0.37 | 0.013 |
| elevation_m | num | 8 | 0.35 | 0.010 |
| soil_type | cat | 5 | 0.17 | 0.009 |
| make_model | cat | 300 | 0.08 | 0.067 |
| branch_id | num | 8 | 0.02 | 0.005 |
| bedrooms | num | 6 | 0.00 | 0.006 |
| distance_to_river_km | num | 8 | -0.01 | 0.009 |
| campaign_code | cat | 15 | -0.24 | 0.028 |
| email_domain | cat | 5 | -0.27 | 0.005 |
| rainfall_mm | num | 8 | -0.44 | 0.010 |
| energy_rating | cat | 7 | -0.49 | 0.006 |
| seats | num | 4 | -0.82 | 0.002 |
| flood_zone | num | 5 | -0.83 | 0.003 |
| transmission | cat | 2 |  | 0.002 |
| driver_assist | cat | 2 |  | 0.028 |
| named_young_driver | cat | 2 |  | 0.000 |
| telematics_opt_in | cat | 2 |  | 0.032 |
| payment_method | cat | 2 |  | 0.077 |
| new_business | cat | 2 |  | 0.104 |
| licence_inconsistent | cat | 2 |  | 0.001 |
| marketing_consent | cat | 2 |  | 0.006 |
| roof_type | cat | 2 |  | 0.000 |
| basement | cat | 2 |  | 0.003 |
| trees_within_5m | cat | 2 |  | 0.001 |
| listed_building | cat | 2 |  | 0.001 |
| solar_panels | cat | 2 |  | 0.001 |
| pol_claims | num | 1 |  |  |
| cust_claims | num | 1 |  |  |
| cust_motor_claims | num | 1 |  |  |
| cust_buildings_expo | num | 2 |  | 0.059 |
| cust_buildings_claims | num | 2 |  | 0.008 |
| lines_held | num | 2 |  | 0.035 |
| pol_rate | num | 1 |  |  |
| cust_rate | num | 1 |  |  |
| cust_motor_rate | num | 1 |  |  |
| cust_buildings_rate | num | 1 |  |  |
| start_month | num | 2 |  | 0.048 |

## buildings

noise: wildfire_index, hail_days, campaign_code, occupation, customer_age, policyholder_age, years_at_address, renewal_count, customer_tenure, household_size, days_quote_to_inception, urban_density, cust_buildings_expo, pol_expo, cust_expo, marital_status

watch: soil_type, freezing_days, email_domain, branch_id, mean_temperature_c

| feature | kind | levels | stability | spread |
|---|---|---|---|---|
| prior_claims_5y | num | 4 | 1.00 | 0.159 |
| flood_zone | num | 5 | 1.00 | 0.240 |
| construction_type | cat | 4 | 1.00 | 0.113 |
| voluntary_excess | num | 4 | 0.99 | 0.104 |
| channel | cat | 3 | 0.99 | 0.074 |
| energy_rating | cat | 7 | 0.99 | 0.247 |
| elevation_m | num | 8 | 0.99 | 0.172 |
| missed_payments_12m | num | 4 | 0.97 | 0.096 |
| row_type | cat | 4 | 0.96 | 0.026 |
| property_type | cat | 4 | 0.96 | 0.084 |
| distance_to_river_km | num | 8 | 0.96 | 0.141 |
| discount_applied | num | 3 | 0.95 | 0.065 |
| sum_insured | num | 8 | 0.94 | 0.092 |
| property_age | num | 8 | 0.94 | 0.272 |
| credit_band | cat | 5 | 0.94 | 0.081 |
| wind_exposure_index | num | 8 | 0.93 | 0.052 |
| region | cat | 12 | 0.87 | 0.112 |
| bedrooms | num | 6 | 0.86 | 0.090 |
| rainfall_mm | num | 8 | 0.74 | 0.048 |
| marketing_segment | cat | 12 | 0.73 | 0.061 |
| days_unoccupied | num | 8 | 0.71 | 0.050 |
| document_preference | cat | 3 | 0.67 | 0.005 |
| dry_summer_index | num | 8 | 0.62 | 0.028 |
| licence_points | num | 4 | 0.61 | 0.012 |
| soil_type | cat | 5 | 0.42 | 0.020 |
| freezing_days | num | 8 | 0.34 | 0.047 |
| email_domain | cat | 5 | 0.28 | 0.014 |
| branch_id | num | 8 | 0.26 | 0.024 |
| mean_temperature_c | num | 8 | 0.26 | 0.048 |
| wildfire_index | num | 7 | 0.13 | 0.022 |
| hail_days | num | 8 | 0.09 | 0.025 |
| campaign_code | cat | 15 | -0.03 | 0.040 |
| occupation | cat | 20 | -0.06 | 0.022 |
| customer_age | num | 8 | -0.23 | 0.011 |
| policyholder_age | num | 8 | -0.31 | 0.015 |
| years_at_address | num | 8 | -0.40 | 0.016 |
| renewal_count | num | 7 | -0.44 | 0.020 |
| customer_tenure | num | 7 | -0.44 | 0.020 |
| household_size | num | 7 | -0.51 | 0.014 |
| days_quote_to_inception | num | 7 | -0.51 | 0.026 |
| urban_density | cat | 4 | -0.54 | 0.006 |
| cust_buildings_expo | num | 3 | -0.67 | 0.002 |
| pol_expo | num | 3 | -0.67 | 0.002 |
| cust_expo | num | 4 | -0.86 | 0.026 |
| marital_status | cat | 4 | -0.98 | 0.004 |
| occupancy | cat | 2 |  | 0.067 |
| smoke_alarms | num | 2 |  | 0.005 |
| accidental_damage_cover | cat | 2 |  | 0.156 |
| mortgage | cat | 2 |  | 0.006 |
| payment_method | cat | 2 |  | 0.081 |
| new_business | cat | 2 |  | 0.026 |
| marketing_consent | cat | 2 |  | 0.003 |
| roof_type | cat | 2 |  | 0.029 |
| basement | cat | 2 |  | 0.005 |
| trees_within_5m | cat | 2 |  | 0.002 |
| listed_building | cat | 2 |  | 0.005 |
| solar_panels | cat | 2 |  | 0.009 |
| pol_claims | num | 2 |  | 0.137 |
| cust_claims | num | 1 |  |  |
| cust_motor_expo | num | 2 |  | 0.034 |
| cust_motor_claims | num | 1 |  |  |
| cust_buildings_claims | num | 2 |  | 0.137 |
| lines_held | num | 2 |  | 0.043 |
| pol_rate | num | 1 |  |  |
| cust_rate | num | 1 |  |  |
| cust_motor_rate | num | 1 |  |  |
| cust_buildings_rate | num | 1 |  |  |
| start_month | num | 2 |  | 0.009 |

