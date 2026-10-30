# Part 2: office premium build-up for the 20 reference policies

Generated 2026-10-29 22:05 by `scripts/office_premium.py` from `submissions/part2/20261030_0145Z_P2_F11v3_robust_nb105_ren103_si103_fl105.csv` (risk premiums) and `expenses.parquet` (column `emerging`). All amounts in CU.

## Convention

The risk premium RP is our predicted expected ultimate claim cost for the row's period (net of excess and recoveries), so it already reflects the row's exposure. The office premium is built on that same row basis:

```
CAT = cat_reinsurance x RP            (buildings and contents only; 0 for motor)
F   = fixed_expense_per_policy        (CU per policy, not pro-rated)
D   = 1 - variable_expense - commission[channel] - premium_tax - target_profit_margin
P   = (RP + CAT + F) / D              (primary)

Alternative (premium tax on top of the net premium):
P_net = (RP + CAT + F) / (1 - variable_expense - commission[channel] - target_profit_margin)
P_alt = P_net x (1 + premium_tax)
```

Every item whose unit is "share of premium" is read as a share of the office premium actually charged, so it sits in the denominator: after paying variable expenses, commission, premium tax and the profit margin out of P, exactly RP + CAT + F is left to pay claims, reinsurance and fixed costs.

### Assumptions used

| Item | emerging | Unit | How it enters |
|---|---:|---|---|
| `fixed_expense_per_policy` | 85.00 | CU | added per policy (CU), not pro-rated |
| `variable_expense` | 15.0% | share of premium | denominator (share of office premium) |
| `commission_direct` | 0.0% | share of premium | denominator, if channel = direct |
| `commission_comparison_site` | blank | share of premium | denominator, if channel = comparison_site (blank in table: taken as 0, assumption) |
| `commission_broker` | 20.0% | share of premium | denominator, if channel = broker |
| `premium_tax` | 8.0% | share of premium | denominator (share of office premium) |
| `cat_reinsurance` | 12.0% | share of property risk premium | x risk premium, buildings and contents only |
| `target_profit_margin` | 8.0% | share of premium | denominator (share of office premium) |

1. RP is the submission prediction for the row (row basis, i.e. already for its exposure). RP p.a. = RP / exposure is shown for readability only.
2. Catastrophe reinsurance is a cost loading on property risk premium (buildings, contents); motor carries none. It is added to the risk premium, not netted off it: we price the gross risk and buy the cover.
3. The fixed expense is a per-policy cost and is charged in full on each reference row, whatever its exposure (see the note on pro-rating below).
4. Variable expense, commission, premium tax and profit margin are shares of the office premium (gross of tax) and enter the denominator.
5. Commission follows the policy's `channel`. A blank commission rate (comparison site in the emerging market) is taken as 0; no reference policy is sold through that channel, so this does not affect any number here.
6. `written_premium` is an annual rate (in the training data its mean is flat across exposure bands, it does not scale with exposure), so the comparison column uses an annual-equivalent office premium: P p.a. = (RP p.a. x (1 + cat rate) + F) / D, i.e. one full year of cover and one fixed expense. Ratio = P p.a. / written premium.
7. No investment income, no allowance for claims-handling costs beyond what the risk premium already includes, and no rounding or minimum premium.

## Worked examples

**M00185311** (motor, direct, exposure 0.9644 years, cover `third_party_only`, current written premium 872.31 CU a year).

1. Risk premium for the row: RP = 726.75 CU (annualised: 726.75 / 0.9644 = 753.58 CU a year).
2. Catastrophe reinsurance: motor is not a property line, so CAT = 0.
3. Fixed expense per policy: F = 85.00 CU (not pro-rated).
4. Costs to recover before percentage loadings: RP + CAT + F = 726.75 + 0.00 + 85.00 = 811.75 CU.
5. Share-of-premium loadings: 15.0% variable + 0.0% commission (direct) + 8.0% premium tax + 8.0% profit = 31.0%; denominator D = 1 - 0.3100 = 0.6900.
6. Office premium P = 811.75 / 0.6900 = **1,176.45 CU** for the row.
7. Check: 726.75 risk + 0.00 cat + 85.00 fixed + variable 176.47 + commission 0.00 + premium tax 94.12 + profit 94.12 = 1,176.45 CU = P.
8. Tax-on-top alternative: net premium = 811.75 / 0.7700 = 1,054.23; x (1 + 0.08) = 1,138.56 CU.
9. Comparison with the current price: annual-equivalent office premium = (753.58 x 1.00 + 85.00) / 0.6900 = 1,215.34 CU a year, 1.39 x the written premium of 872.31.

**B00229821** (buildings, broker, exposure 0.9918 years, sum insured 378,000 CU, current written premium 285.60 CU a year).

1. Risk premium for the row: RP = 276.12 CU (annualised: 276.12 / 0.9918 = 278.40 CU a year).
2. Catastrophe reinsurance: CAT = 12.0% x 276.12 = 33.13 CU.
3. Fixed expense per policy: F = 85.00 CU (not pro-rated).
4. Costs to recover before percentage loadings: RP + CAT + F = 276.12 + 33.13 + 85.00 = 394.26 CU.
5. Share-of-premium loadings: 15.0% variable + 20.0% commission (broker) + 8.0% premium tax + 8.0% profit = 51.0%; denominator D = 1 - 0.5100 = 0.4900.
6. Office premium P = 394.26 / 0.4900 = **804.60 CU** for the row.
7. Check: 276.12 risk + 33.13 cat + 85.00 fixed + variable 120.69 + commission 160.92 + premium tax 64.37 + profit 64.37 = 804.60 CU = P.
8. Tax-on-top alternative: net premium = 394.26 / 0.5700 = 691.68; x (1 + 0.08) = 747.01 CU.
9. Comparison with the current price: annual-equivalent office premium = (278.40 x 1.12 + 85.00) / 0.4900 = 809.82 CU a year, 2.84 x the written premium of 285.60.

## Office premium by reference policy

RP, CAT, P (primary) and P (tax on top) are for the row's period; P p.a. is the annual equivalent used for the comparison with the current written premium.

| row_id | line | channel | exposure | sum insured / cover | RP | RP p.a. | CAT | F | D | P (primary) | P (tax on top) | P p.a. | written p.a. | P p.a. / written |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M00161473 | motor | broker | 0.9644 | third_party_fire_theft | 1,271.60 | 1,318.54 | 0.00 | 85.00 | 0.4900 | 2,768.57 | 2,570.40 | 2,864.37 | 1,131.78 | 2.53 |
| M00176777 | motor | broker | 0.9918 | comprehensive | 1,015.23 | 1,023.63 | 0.00 | 85.00 | 0.4900 | 2,245.38 | 2,084.65 | 2,262.51 | 443.61 | 5.10 |
| M00185311 | motor | direct | 0.9644 | third_party_only | 726.75 | 753.58 | 0.00 | 85.00 | 0.6900 | 1,176.45 | 1,138.56 | 1,215.34 | 872.31 | 1.39 |
| M00167496 | motor | broker | 0.9836 | comprehensive | 609.23 | 619.39 | 0.00 | 85.00 | 0.4900 | 1,416.79 | 1,315.38 | 1,437.53 | 463.95 | 3.10 |
| M00129413 | motor | broker | 0.3315 | comprehensive | 139.51 | 420.83 | 0.00 | 85.00 | 0.4900 | 458.18 | 425.38 | 1,032.31 | 354.15 | 2.91 |
| M00118569 | motor | direct | 0.9397 | third_party_fire_theft | 883.74 | 940.44 | 0.00 | 85.00 | 0.6900 | 1,403.96 | 1,358.75 | 1,486.15 | 1,095.48 | 1.36 |
| M00153851 | motor | broker | 0.9616 | comprehensive | 302.51 | 314.59 | 0.00 | 85.00 | 0.4900 | 790.83 | 734.22 | 815.48 | 414.72 | 1.97 |
| B00229821 | buildings | broker | 0.9918 | 378,000 | 276.12 | 278.40 | 33.13 | 85.00 | 0.4900 | 804.60 | 747.01 | 809.82 | 285.60 | 2.84 |
| B00218282 | buildings | broker | 0.9726 | 222,000 | 356.84 | 366.89 | 42.82 | 85.00 | 0.4900 | 989.10 | 918.30 | 1,012.08 | 230.80 | 4.39 |
| B00250679 | buildings | broker | 0.1096 | 236,000 | 26.66 | 243.23 | 3.20 | 85.00 | 0.4900 | 234.40 | 217.62 | 729.42 | 180.72 | 4.04 |
| B00223369 | buildings | broker | 0.9945 | 222,000 | 406.32 | 408.56 | 48.76 | 85.00 | 0.4900 | 1,102.19 | 1,023.30 | 1,107.33 | 204.56 | 5.41 |
| B00215809 | buildings | direct | 0.9753 | 362,000 | 542.35 | 556.09 | 65.08 | 85.00 | 0.6900 | 1,003.53 | 971.20 | 1,025.82 | 237.22 | 4.32 |
| B00215707 | buildings | broker | 0.9863 | 316,000 | 200.53 | 203.32 | 24.06 | 85.00 | 0.4900 | 631.83 | 586.60 | 638.20 | 272.44 | 2.34 |
| B00217584 | buildings | broker | 0.9671 | 198,000 | 1,285.15 | 1,328.87 | 154.22 | 85.00 | 0.4900 | 3,110.95 | 2,888.27 | 3,210.88 | 323.36 | 9.93 |
| C00259568 | contents | broker | 1.0000 | 22,100 | 49.59 | 49.59 | 5.95 | 85.00 | 0.4900 | 286.81 | 266.28 | 286.81 | 83.18 | 3.45 |
| C00283470 | contents | broker | 0.9726 | 35,700 | 69.22 | 71.17 | 8.31 | 85.00 | 0.4900 | 331.69 | 307.94 | 336.14 | 117.97 | 2.85 |
| C00270331 | contents | broker | 0.9726 | 33,400 | 34.45 | 35.42 | 4.13 | 85.00 | 0.4900 | 252.20 | 234.15 | 254.42 | 140.29 | 1.81 |
| C00314425 | contents | broker | 0.0685 | 27,200 | 4.40 | 64.23 | 0.53 | 85.00 | 0.4900 | 183.53 | 170.39 | 320.27 | 102.87 | 3.11 |
| C00302985 | contents | broker | 0.9918 | 22,900 | 62.99 | 63.51 | 7.56 | 85.00 | 0.4900 | 317.46 | 294.73 | 318.65 | 115.88 | 2.75 |
| C00286287 | contents | broker | 0.9671 | 71,100 | 92.17 | 95.31 | 11.06 | 85.00 | 0.4900 | 384.15 | 356.65 | 391.32 | 269.23 | 1.45 |

## Summary by line

| line | n | sum RP | sum CAT | sum F | sum P (primary) | sum P (tax on top) | mean P p.a. / written |
|---|---:|---:|---:|---:|---:|---:|---:|
| motor | 7 | 4,948.56 | 0.00 | 595.00 | 10,260.16 | 9,627.35 | 2.62 |
| buildings | 7 | 3,093.97 | 371.28 | 595.00 | 7,876.61 | 7,352.32 | 4.75 |
| contents | 6 | 312.82 | 37.54 | 510.00 | 1,755.84 | 1,630.16 | 2.57 |
| total | 20 | 8,355.35 | 408.81 | 1,700.00 | 19,892.61 | 18,609.82 | 3.35 |

**Fixed-expense floor.** With zero risk premium the office premium is still F / D: 173.47 CU (broker), 123.19 CU (direct). 5 of the 20 current written premiums are below this floor (5 contents), so for low-premium property policies the fixed expense, not the risk premium, drives the price and the ratio to the current premium.

## Why tax inside the denominator is the primary convention

The `expenses` table gives premium tax as a share of premium, like the other percentage items, so the primary build makes the tax 8.0% of the premium the customer pays, and commission and profit are also measured on that gross premium. The alternative treats the tax as a levy added on top of a net premium (P_alt = P_net x (1 + t)): the tax is then 8.0% of the net premium, i.e. 7.4% of the gross, and commission, variable expense and profit are earned on the net premium only. The alternative is therefore always lower; the ratio P_alt / P = (1 + t)(D) / (D + t) ranges from 0.9284 to 0.9678 here (about 6.6% lower). We recommend the primary convention because it applies every "share of premium" item to the same base, the premium charged, which is how the table states them; the tax-on-top figure is the right one only if the local tax is legally levied on the premium net of tax.

## Note: pro-rating the fixed expense

We charge the full fixed expense per policy on each row because it is stated per policy (issuing, administration and claims set-up costs are incurred once, however short the period). The alternative pro-rates it by exposure: P = (RP + CAT + F x exposure) / D. It matters only for short rows. Caveat: a row is a policy-year segment, so a policy that starts late in 2025 continues in a 2026 row; charging F on both would double count it, and in that case pro-rating (or charging F once per policy term) is the better choice. Rows that end early because the policy was cancelled or lapsed have incurred the full set-up cost, which supports charging F in full.

| row_id | line | exposure | P (F in full) | P (F pro-rated) |
|---|---|---:|---:|---:|
| M00129413 | motor | 0.3315 | 458.18 | 342.21 |
| B00250679 | buildings | 0.1096 | 234.40 | 79.94 |
| C00314425 | contents | 0.0685 | 183.53 | 21.94 |
