# Level-frequency stability screen - market `part1`

stability = exposure-weighted correlation of level claim frequencies, years 2021-2022 vs 2023; spread = exposure-weighted sd of level frequencies / overall frequency. Levels need >= 300 rows in both periods. noise: stability < 0.2 and spread < 0.1; watch: 0.2-0.5. Weather variables vary by year by design: do not drop them on this screen alone.

## motor

noise: wildfire_index, energy_rating, email_domain, bedrooms, rainfall_mm, previous_insurer, distance_to_river_km, seats, flood_zone

watch: campaign_code, soil_type, wind_exposure_index, fuel_type, property_age, make_model, branch_id

| feature | kind | levels | stability | spread |
|---|---|---|---|---|
| cover_type | cat | 3 | 1.00 | 0.235 |
| voluntary_excess | num | 4 | 1.00 | 0.151 |
| prior_claims_5y | num | 4 | 1.00 | 0.136 |
| urban_density | cat | 4 | 1.00 | 0.086 |
| body_type | cat | 5 | 1.00 | 0.092 |
| missed_payments_12m | num | 4 | 1.00 | 0.094 |
| marital_status | cat | 4 | 1.00 | 0.150 |
| annual_mileage | num | 6 | 1.00 | 0.169 |
| cust_expo | num | 4 | 1.00 | 0.098 |
| discount_applied | num | 3 | 1.00 | 0.101 |
| row_type | cat | 4 | 1.00 | 0.099 |
| customer_tenure | num | 7 | 0.99 | 0.107 |
| renewal_count | num | 7 | 0.99 | 0.107 |
| driver_age | num | 8 | 0.99 | 0.361 |
| customer_age | num | 8 | 0.99 | 0.365 |
| years_licensed | num | 8 | 0.99 | 0.358 |
| credit_band | cat | 5 | 0.99 | 0.086 |
| channel | cat | 3 | 0.99 | 0.075 |
| property_type | cat | 4 | 0.99 | 0.014 |
| years_at_address | num | 8 | 0.99 | 0.126 |
| lines_held | num | 3 | 0.99 | 0.041 |
| marketing_segment | cat | 12 | 0.99 | 0.323 |
| pol_expo | num | 3 | 0.99 | 0.094 |
| cust_motor_expo | num | 3 | 0.99 | 0.094 |
| occupation | cat | 20 | 0.98 | 0.196 |
| licence_points | num | 4 | 0.98 | 0.051 |
| document_preference | cat | 3 | 0.97 | 0.008 |
| ncb_years | num | 8 | 0.96 | 0.391 |
| vehicle_use | cat | 3 | 0.96 | 0.063 |
| household_size | num | 7 | 0.95 | 0.068 |
| region | cat | 12 | 0.94 | 0.118 |
| telematics_score | num | 8 | 0.92 | 0.145 |
| vehicle_value_group | num | 8 | 0.91 | 0.056 |
| vehicle_age | num | 8 | 0.85 | 0.047 |
| engine_size_cc | num | 8 | 0.84 | 0.053 |
| overnight_parking | cat | 3 | 0.81 | 0.014 |
| dry_summer_index | num | 8 | 0.81 | 0.048 |
| freezing_days | num | 8 | 0.81 | 0.045 |
| days_quote_to_inception | num | 7 | 0.81 | 0.027 |
| elevation_m | num | 8 | 0.75 | 0.010 |
| mean_temperature_c | num | 8 | 0.71 | 0.051 |
| hail_days | num | 8 | 0.68 | 0.017 |
| vehicle_colour | cat | 8 | 0.58 | 0.017 |
| campaign_code | cat | 15 | 0.49 | 0.028 |
| soil_type | cat | 5 | 0.48 | 0.009 |
| wind_exposure_index | num | 8 | 0.42 | 0.028 |
| fuel_type | cat | 4 | 0.32 | 0.007 |
| property_age | num | 8 | 0.27 | 0.013 |
| make_model | cat | 300 | 0.23 | 0.067 |
| branch_id | num | 8 | 0.21 | 0.005 |
| wildfire_index | num | 7 | 0.16 | 0.022 |
| energy_rating | cat | 7 | 0.10 | 0.006 |
| email_domain | cat | 5 | 0.09 | 0.005 |
| bedrooms | num | 6 | 0.06 | 0.006 |
| rainfall_mm | num | 8 | -0.04 | 0.010 |
| previous_insurer | cat | 40 | -0.12 | 0.019 |
| distance_to_river_km | num | 8 | -0.27 | 0.009 |
| seats | num | 4 | -0.47 | 0.002 |
| flood_zone | num | 5 | -0.65 | 0.003 |
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
| cust_claims | num | 2 |  | 0.217 |
| cust_motor_claims | num | 1 |  |  |
| cust_buildings_expo | num | 2 |  | 0.059 |
| cust_buildings_claims | num | 2 |  | 0.008 |
| cust_contents_expo | num | 1 |  | 0.054 |
| cust_contents_claims | num | 2 |  | 0.000 |
| pol_rate | num | 1 |  |  |
| cust_rate | num | 2 |  | 0.223 |
| cust_motor_rate | num | 1 |  |  |
| cust_buildings_rate | num | 1 |  |  |
| cust_contents_rate | num | 1 |  |  |
| start_month | num | 2 |  | 0.048 |

## buildings

noise: email_domain, occupation, household_size, cust_buildings_expo, pol_expo, previous_insurer, branch_id, mean_temperature_c, campaign_code, document_preference, dry_summer_index

watch: hail_days, policyholder_age, renewal_count, customer_tenure, customer_age, region, wind_exposure_index, urban_density, years_at_address, row_type, freezing_days

| feature | kind | levels | stability | spread |
|---|---|---|---|---|
| construction_type | cat | 4 | 1.00 | 0.113 |
| credit_band | cat | 5 | 1.00 | 0.081 |
| property_age | num | 8 | 1.00 | 0.272 |
| prior_claims_5y | num | 4 | 1.00 | 0.159 |
| lines_held | num | 3 | 0.99 | 0.039 |
| flood_zone | num | 5 | 0.99 | 0.240 |
| elevation_m | num | 8 | 0.99 | 0.172 |
| voluntary_excess | num | 4 | 0.99 | 0.104 |
| missed_payments_12m | num | 4 | 0.99 | 0.096 |
| discount_applied | num | 3 | 0.99 | 0.065 |
| licence_points | num | 4 | 0.98 | 0.012 |
| channel | cat | 3 | 0.97 | 0.074 |
| distance_to_river_km | num | 8 | 0.97 | 0.141 |
| energy_rating | cat | 7 | 0.97 | 0.247 |
| sum_insured | num | 8 | 0.95 | 0.092 |
| property_type | cat | 4 | 0.94 | 0.084 |
| rainfall_mm | num | 8 | 0.94 | 0.048 |
| soil_type | cat | 5 | 0.92 | 0.020 |
| bedrooms | num | 6 | 0.89 | 0.090 |
| days_unoccupied | num | 8 | 0.86 | 0.050 |
| cust_expo | num | 3 | 0.86 | 0.026 |
| days_quote_to_inception | num | 7 | 0.83 | 0.026 |
| marketing_segment | cat | 12 | 0.70 | 0.061 |
| wildfire_index | num | 7 | 0.62 | 0.022 |
| marital_status | cat | 4 | 0.51 | 0.004 |
| hail_days | num | 8 | 0.50 | 0.025 |
| policyholder_age | num | 8 | 0.48 | 0.015 |
| renewal_count | num | 7 | 0.48 | 0.020 |
| customer_tenure | num | 7 | 0.48 | 0.020 |
| customer_age | num | 8 | 0.48 | 0.011 |
| region | cat | 12 | 0.43 | 0.112 |
| wind_exposure_index | num | 8 | 0.43 | 0.052 |
| urban_density | cat | 4 | 0.39 | 0.006 |
| years_at_address | num | 8 | 0.28 | 0.016 |
| row_type | cat | 4 | 0.27 | 0.026 |
| freezing_days | num | 8 | 0.26 | 0.047 |
| email_domain | cat | 5 | 0.16 | 0.014 |
| occupation | cat | 20 | 0.10 | 0.022 |
| household_size | num | 7 | 0.05 | 0.014 |
| cust_buildings_expo | num | 3 | -0.04 | 0.002 |
| pol_expo | num | 3 | -0.04 | 0.002 |
| previous_insurer | cat | 40 | -0.08 | 0.037 |
| branch_id | num | 8 | -0.08 | 0.024 |
| mean_temperature_c | num | 8 | -0.16 | 0.048 |
| campaign_code | cat | 15 | -0.34 | 0.040 |
| document_preference | cat | 3 | -0.61 | 0.005 |
| dry_summer_index | num | 8 | -0.83 | 0.028 |
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
| cust_contents_expo | num | 2 |  | 0.005 |
| cust_contents_claims | num | 2 |  | 0.017 |
| pol_rate | num | 1 |  |  |
| cust_rate | num | 1 |  |  |
| cust_motor_rate | num | 1 |  |  |
| cust_buildings_rate | num | 1 |  |  |
| cust_contents_rate | num | 1 |  |  |
| start_month | num | 2 |  | 0.009 |

## contents

noise: branch_id, campaign_code, previous_insurer, licence_points, soil_type

watch: high_value_items, years_at_address, household_size, wind_exposure_index, wildfire_index, email_domain, rainfall_mm

| feature | kind | levels | stability | spread |
|---|---|---|---|---|
| channel | cat | 3 | 1.00 | 0.075 |
| voluntary_excess | num | 4 | 1.00 | 0.173 |
| flood_zone | num | 5 | 0.99 | 0.196 |
| policyholder_age | num | 8 | 0.99 | 0.171 |
| customer_age | num | 8 | 0.99 | 0.171 |
| document_preference | cat | 3 | 0.99 | 0.010 |
| prior_claims_5y | num | 4 | 0.99 | 0.136 |
| cust_contents_expo | num | 3 | 0.98 | 0.022 |
| pol_expo | num | 3 | 0.98 | 0.022 |
| credit_band | cat | 5 | 0.98 | 0.100 |
| marketing_segment | cat | 12 | 0.98 | 0.177 |
| elevation_m | num | 8 | 0.98 | 0.120 |
| discount_applied | num | 3 | 0.98 | 0.073 |
| security_level | cat | 3 | 0.97 | 0.041 |
| mean_temperature_c | num | 8 | 0.97 | 0.101 |
| row_type | cat | 4 | 0.97 | 0.049 |
| urban_density | cat | 4 | 0.96 | 0.075 |
| region | cat | 12 | 0.96 | 0.157 |
| distance_to_river_km | num | 8 | 0.96 | 0.110 |
| freezing_days | num | 8 | 0.95 | 0.095 |
| missed_payments_12m | num | 4 | 0.94 | 0.102 |
| dry_summer_index | num | 8 | 0.93 | 0.088 |
| marital_status | cat | 4 | 0.92 | 0.056 |
| lines_held | num | 3 | 0.91 | 0.030 |
| renewal_count | num | 7 | 0.91 | 0.047 |
| customer_tenure | num | 7 | 0.91 | 0.047 |
| occupation | cat | 20 | 0.86 | 0.097 |
| property_type | cat | 4 | 0.80 | 0.026 |
| energy_rating | cat | 7 | 0.77 | 0.029 |
| property_age | num | 8 | 0.75 | 0.031 |
| cust_expo | num | 3 | 0.73 | 0.030 |
| sum_insured | num | 8 | 0.64 | 0.051 |
| hail_days | num | 8 | 0.61 | 0.034 |
| bedrooms | num | 6 | 0.58 | 0.014 |
| days_quote_to_inception | num | 7 | 0.54 | 0.032 |
| high_value_items | num | 6 | 0.47 | 0.022 |
| years_at_address | num | 8 | 0.46 | 0.048 |
| household_size | num | 7 | 0.41 | 0.039 |
| wind_exposure_index | num | 8 | 0.39 | 0.048 |
| wildfire_index | num | 7 | 0.38 | 0.075 |
| email_domain | cat | 5 | 0.35 | 0.015 |
| rainfall_mm | num | 8 | 0.22 | 0.024 |
| branch_id | num | 8 | 0.15 | 0.018 |
| campaign_code | cat | 15 | 0.12 | 0.041 |
| previous_insurer | cat | 40 | -0.07 | 0.039 |
| licence_points | num | 4 | -0.15 | 0.008 |
| soil_type | cat | 5 | -0.83 | 0.004 |
| tenure | cat | 2 |  | 0.008 |
| accidental_damage_cover | cat | 2 |  | 0.329 |
| works_from_home | cat | 2 |  | 0.028 |
| lodgers | cat | 2 |  | 0.026 |
| bicycle_cover | cat | 2 |  | 0.009 |
| payment_method | cat | 2 |  | 0.075 |
| new_business | cat | 2 |  | 0.048 |
| marketing_consent | cat | 2 |  | 0.002 |
| roof_type | cat | 2 |  | 0.014 |
| basement | cat | 2 |  | 0.011 |
| trees_within_5m | cat | 2 |  | 0.005 |
| listed_building | cat | 2 |  | 0.001 |
| solar_panels | cat | 2 |  | 0.005 |
| pol_claims | num | 2 |  | 0.126 |
| cust_claims | num | 1 |  |  |
| cust_motor_expo | num | 2 |  | 0.043 |
| cust_motor_claims | num | 1 |  |  |
| cust_buildings_expo | num | 2 |  | 0.011 |
| cust_buildings_claims | num | 2 |  | 0.032 |
| cust_contents_claims | num | 2 |  | 0.126 |
| pol_rate | num | 1 |  |  |
| cust_rate | num | 1 |  |  |
| cust_motor_rate | num | 1 |  |  |
| cust_buildings_rate | num | 1 |  |  |
| cust_contents_rate | num | 1 |  |  |
| start_month | num | 2 |  | 0.017 |

