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

## 7. Level probe result and why a flat factor is right (18 Oct 00:40 UTC)
Public-LB probe of X10 (same rows, scaled per line): motor ×0.95/1.00/1.05 → 0.12710/0.12749/0.12752 (optimum ×1.029);
buildings ×0.95/1.00/1.05/1.10 → 0.12690/0.12749/0.12781/0.12791 (optimum ×1.093). Calibrated file scored **0.12797**.
Could the buildings shortfall be a weather under-reaction (2025 is the windiest year; 29 % of 2025 rows are in the top wind
quintile vs 20 % in training)? OOF actual/pred by wind quintile is 1.02 / 1.06 / 1.01 / 0.99 / 0.96 (top quintile slightly
*over*-predicted), rainfall 0.96–1.05, dry-summer 0.99–1.02: the model's weather response is calibrated, so the 2025
shortfall is a year-level effect, not a weather one. A weather-dependent uplift would push the wrong way; the flat per-line
factor is the right correction. Motor's hail/freezing calibration is within ±2 %.

## 8. Round 5 (18 Oct): what the fine-tuning and novel-technique tracks delivered
Baselines: P20_135 quick 0.1639 / full CV 0.1582 / 2023 holdout 0.1056; smooth GLM (no counters) quick 0.1622;
smooth-boost B20 quick 0.1628 / CV 0.1574; best blend X16 nested 0.1597.

| Idea | Result | Verdict |
|---|---|---|
| Smooth-GLM ridge alpha 1e-5 / 1e-3 (G16/G17) | quick 0.1621 / 0.1622 | flat |
| + motor / home interaction sets (G18/G19) | 0.1621 / 0.1622 | nothing left in hand-picked interactions |
| Recency weighting k = 0.25 / 0.5 (P22) | 0.1635 / 0.1634 | worse than 0.1639 |
| Exposure as a feature (P23) | 0.1636 | worse |
| Neural residual on the smooth GLM, CANN pattern (N1; N2 regularised retune) | 0.1623 / 0.1623, early stop at epoch 1–2 | +0.0001 over its GLM base, below the LightGBM residual (0.1628): no diversity worth a blend slot |
| Pooled buildings + contents model (P25 LightGBM, B23 smooth-boost) | 0.1606 / 0.1609; contents 0.084–0.086 | contents collapses; rejected |
| Covariate-shift importance weights (P24, clip 0.2–5; tight 0.5–2) | CV 0.1581 both; holdout 0.1021 / 0.1056 | the train-vs-target classifier reaches AUC 1.0 against 2023 and 0.995 against 2025 through the calendar-proxy history features, so the "shift" it finds is calendar time (ESS 0.27–0.41) — a recency weighting in disguise, and recency does not help |

Seven consecutive ideas below +0.0002 nested: the stop rule is met twice over. The model plateau is X16 at 0.1597
(bootstrap: better than every earlier candidate on 96–100 % of row subsets). What remains for Part 1 is the final
pick (user decision on 22 Oct) and the deliverables; the one untested mechanism-backed idea is a model-based
large-loss loading (expected ≤ +0.0002).

## 9. Why are the leaders ahead? (18 Oct 23:45 UTC; leader 0.12996, us #4 0.12815, local X21 ≈ 0.1290 predicted)
Checks run before deciding on a probe batch:

| Hypothesis | Check | Result |
|---|---|---|
| Competitors use 2025 reported claims | claims file date range; claims joined to test policies | report dates end 2024-12-30; **0** claims with a 2025 loss date on test policies — nobody has future claims |
| The 2025 buildings shortfall (probe ×1.093) is a trend-method artefact | 2025 factors from the premium index vs the claims-cost log-linear trend | buildings 1.054 vs 1.064 (1 % apart); motor 1.026 vs 0.991 (the claims trend points the *other* way from the ×1.029 probe) — not the explanation |
| 2025 weather is out of the training range (trees cap, GLM clips) | test vs train range of every weather feature, by line | 0 % of 2025 rows above the training max; wind index mean 4.88 vs 4.35–4.60 by year but inside the range; response calibrated by quintile (§7) |
| The leaders keep probing the public board | our probe history vs the leaders' continuing submissions | our only end-game jump (+0.0005) came from 4 probes; we stopped on 18 Oct; contents was never probed and buildings' optimum sits at the probed edge |

Conclusion: the 2025 effect is a level shift observable only through the public board (possibly planted, possibly
specific to the public subset); the remaining lever is a small probe batch — base X21, buildings ×1.09/×1.15,
contents ×1.03/×1.06, then the calibrated candidate — approved by the user for 19 Oct. The private-LB caveat stands:
the public-vs-private level gap (sd ≈ 0.005) is larger than any of these differences, so the final ranking will be
decided largely by how the 2025 level transfers, which no one can observe before the close.

## 10. 19 Oct probe batch on the X21 base (public LB; same rows, noise-free differences)
| File | Public LB | Reading |
|---|---|---|
| X21 raw | 0.12744 | = X10 raw (0.12749) despite +0.0007 nested CV: the local gains since X10 do not transfer to the public rows |
| buildings ×1.09 / ×1.15 | 0.12783 / 0.12772 | parabola optimum **×1.098** (X10 probe: ×1.093) — the buildings level shift is stable across models; +0.0004 |
| contents ×1.03 / ×1.06 | 0.12748 / 0.12748 | flat (+0.00004): no contents level shift |
Implication: a calibrated X21 would score ≈0.1280, below the submitted FINAL_avg3_levelcal (0.12815); line-level
calibration is exhausted; the leaders' +0.002 is structure the public rows reward and our CV cannot see. Next:
segment probes (motor new business, motor full-year exposure shape, buildings SI top/bottom quartile).
