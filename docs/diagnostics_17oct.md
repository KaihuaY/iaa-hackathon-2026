# Diagnostics behind the "global optimum" plan (17 Oct 2026)

All figures are out-of-fold (5-fold grouped by policy) on the X6 blend unless stated. Code: the snippets live in the
session log; the reusable parts are `scripts/level_probe.py` (level sensitivity) and `src/credibility.py`.

## 1. Calibration of X6 (is the model mis-shaped?)
| Line | Calibration slope (1 = ideal) | Actual/pred by risk decile | By year | By row type |
|---|---|---|---|---|
| motor | 0.962 | 0.95–1.14 (decile 0 at 1.14) | 0.98–1.02 | 0.97–1.04 |
| buildings | 1.043 | 0.84–1.07 | 1.00–1.01 | 0.95–1.04 |
| contents | 0.999 | 0.91–1.11 | 0.98–1.02 | 0.98–1.01 |

Slope recalibration adds ≤ 0.0002 D². Conclusion: the model is well calibrated; gains must come from new signal.

## 2. Level sensitivity (what a level error costs)
D² when all of a line's predictions are scaled by k:

| Line | ×0.90 | ×0.95 | ×1.00 | ×1.05 | ×1.10 |
|---|---|---|---|---|---|
| motor | 0.1073 | 0.1092 | 0.1099 | 0.1096 | 0.1084 |
| buildings | 0.1122 | 0.1133 | 0.1135 | 0.1129 | 0.1118 |
| contents | 0.1015 | 0.1025 | 0.1027 | 0.1022 | 0.1011 |

A 5 % level error costs 0.0003–0.0007 per line. CV cannot judge the 2025 level (its label uses the premium trend),
hence the leaderboard level probe. Implied 2025 pure premium vs 2024 developed+trended actual: motor 0.948,
buildings 0.974, contents 1.029 (part weather response: 2025 has fewer hail and freezing days).

## 3. Ceiling check (is noise binding?)
Compound-Poisson simulation with the real severity distribution, treating X6's predictions as the truth, mature years:

| Line | X6 D² on reported claims | D² a perfect model would score if X6 were the truth |
|---|---|---|
| motor | 0.0910 | 0.0941 |
| buildings | 0.1016 | 0.0958 |
| contents | 0.0988 | 0.0888 |

X6 already scores about what pure noise around its own predictions allows, so the ceiling is not binding; further
gains need more discrimination (new signal), not less noise.

## 4. Persistent unexplained risk (experience is under-used)
2023–24 actual/predicted by the policy's 2021–22 claim count, although the model has 2-year history features:

| Line | 0 claims (share) | 1 claim | 2+ claims |
|---|---|---|---|
| motor | 0.999 (80 %) | 1.007 (16 %) | 1.079 (4 %) |
| buildings | 0.983 (93 %) | 1.200 (6.5 %) | 1.519 (0.5 %) |
| contents | 0.980 (94 %) | 1.277 (5.9 %) | 1.616 (0.4 %) |

## 5. Segment scan (where is the model systematically off?)
OOF actual/pred by every feature level (≥ 2,000 rows), ranked by deviance recoverable if the level were fixed.
Top items: motor `make_model` MM193 1.84×, MM251 1.49×, MM186 1.68×, MM124 1.41× (≈3,300 rows each); motor
`cover_type=third_party_only` 1.09× (88k rows); buildings sum-insured bottom/top decile 0.85×/1.15×, bedrooms 1 /
5+ 0.86×/1.12×, listed buildings 1.15×; contents flood zone 5 1.27×. Upper bound on recoverable D² (in-sample,
overlaps ignored): 0.006.

## 6. What the plan delivered (nested CV, each vs its predecessor)
| Step | Gain | Note |
|---|---|---|
| Segment-specific large-loss loading (L19) | +0.0003 | TPO loadings 1.13–1.37 vs comprehensive 1.08–1.13 |
| Lower regularisation + monotone constraints (L20) | +0.0002 | buildings +0.0006 |
| Tweedie training power 1.35 (P20_135) | +0.0003 | |
| Credibility layer (Bühlmann + ridge segments) | +0.0001 | k = 128 chosen: the segment deviations are mostly policy-level noise |
| 4-way blend with CatBoost (X10) | +0.0005 vs X6 | public LB 0.12749, #1 |

Lesson: the in-sample scan's 0.006 bound was real but did not survive nested validation; `make_model` effects do
not generalise across policy folds (a few large claims), and the prior-claim segments are too small a share of the
deviance to matter beyond ~0.0001–0.0003.
