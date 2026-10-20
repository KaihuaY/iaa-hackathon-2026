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

### 10b. Segment probes on the X21 base (19 Oct; base 0.12744; 5 submissions)
| Segment scaled | Share of line | LB | Δ vs base |
|---|---|---|---|
| motor new business ×1.06 | 15 % | 0.12744 | 0 |
| motor full-year exposure rows ×1.04 | 59 % | 0.12751 | +0.00007 (same as the flat motor curve → no exposure-shape effect) |
| buildings SI top quartile ×1.08 | 25 % | 0.12785 | +0.00041 (= the entire flat ×1.09 gain, with a third of the level change) |
| buildings SI top quartile ×1.16 | 25 % | 0.12808 | +0.00064 — parabola optimum ≈ ×1.22, gain ≈ +0.0007 |
| buildings SI bottom quartile ×1.08 | 25 % | 0.12734 | −0.00010 |
Reading: the 2025 buildings shift is a **sum-insured structure**, not a level — high-SI homes are under-priced, low-SI
homes slightly over-priced. The blend's own OOF shows the same pattern in training (actual/pred by SI decile 0.91 …
1.11; contents 0.97 … 1.05), i.e. the models under-respond to sum insured (monotone constraints + heavy regularisation
flatten the top of the SI response) and the 2025 test has ~7 % higher SI (indexation), which turns a shape error into
a level shift. Fixes under test: sum insured in the exposure offset (P33/B26, elasticity k) and a nested SI-decile
calibration of the blend (`scripts/recalibrate.py --by sum_insured`).

### 10c. Decision and candidate (19 Oct 00:35 UTC)
Nested SI-decile calibration of X21 fitted on the training years gains +0.00003 (g 0.93 → 1.04 across SI), i.e. the
training data does not contain the 2025 high-SI shift the public rows show (top quartile wants ≈ ×1.22). The fix is
therefore probe-derived, not model-derived: `scripts/segment_calibrate.py X21_blend4_a1bag --level-factors
motor=1.029 --band-line buildings --band-col sum_insured --band-factors 1.0 1.05 1.08 1.20` →
`submissions/20261018_2021_X21_blend4_a1bag_sifix.csv` (buildings mean factor 1.0825; +0.5 % vs the submitted final;
correlation 0.994). Expected public ≈ 0.1284. The user approved one or two submissions on 20 Oct for this candidate
only (no more mapping probes); the top-quartile factor is hedged below the parabola optimum because the effect may be
weaker on the private rows. Scheduled for 20 Oct 00:03 UTC.

### 10d. How much of a segment factor measured on the public rows is noise? (19 Oct 12:00 UTC, debate check)
Closed-form Tweedie-optimal multiplier s = Σ y·μ^−0.5 / Σ μ^0.5 of X21's OOF predictions, on 300 public-sized
random samples (20 % of the line's test size) of the 2023 and 2024 rows:

| Segment | sampling SD of s | full-year s, 2023 → 2024 |
|---|---|---|
| buildings SI Q0 / Q3 | 0.100–0.105 / 0.116–0.130 | 0.860 → 1.045 / 0.994 → 1.032 |
| buildings top SI decile | 0.165–0.213 | 1.094 → 0.945 |
| buildings whole line | 0.054–0.059 | 0.971 → 0.998 |
| contents SI Q0 / Q3 / top decile / line | 0.07–0.10 / 0.10–0.14 / 0.16–0.29 / 0.05–0.06 | — |

Reading: the public board's "Q3 wants ×1.22" is ≈1.7 SD of pure sampling noise, and even whole-year segment
optima swing 15–20 % between adjacent years (large losses). Segment factors fitted on the public rows are mostly
not expected to transfer to the private rows; line-level factors are twice as precise. SI indexation is uniform
(+3 %/yr buildings, +2 %/yr contents, identical across quartiles and years), so the public tilt is not an
exposure-base artefact. Implication for the final: credibility-shrink probe-derived segment tilts
(Z = τ²/(τ²+σ²) with τ ≈ 0.05 from the OOF SI miscalibration and σ ≈ 0.12 → Z ≈ 0.15), keep the line levels, and
prefer the model-driven tilt (X30's SI offset) over the board-fitted one for the file ticked as final.

### 10f. Combining every saved experiment (19 Oct evening, user request)
Pool: 80 single models and equal-weight seed bags (weighted blends excluded — their saved OOF weights were fitted
in-sample, which inflated a first run to +0.0008). Nested CV, GroupKFold(5) by policy, X21 reproduced exactly.
| Combiner | overall nested D² | vs X21 (0.15975) | P(better) on 25 % subsets |
|---|---|---|---|
| greedy ensemble selection | 0.15984 | +0.00009 | 0.69 |
| penalised optimised weights | 0.15898 | −0.00075 | 0.01 |
| geometric optimised weights | 0.15895 | −0.00077 | 0.00 |
Reading: no combination of what we have beats the 4-part blend; free weight optimisation over 80 near-identical
parts overfits. Together with the public-score fit (docs/lb_response_fit.md) this closes "combine the submissions":
the best view is X21/X30 plus credibility-shrunk level/tilt factors. The written file GB_grand_blend.csv is not used.

### 10e. 20 Oct batch (00:03 UTC; base for probes = X21 raw 0.12744; same rows, noise-free differences)
| # | File | Public LB | Δ vs base / reading |
|---|---|---|---|
| 1 | X21 × buildings SI bands [1.0,1.05,1.08,1.20], motor ×1.029 | 0.12827 | +0.00012 over the 17 Oct final |
| 2 | X30 (SI^0.7 offset) × residual bands [1.01,1.055,1.085,1.16], motor ×1.029 | **0.12838** | new best (+0.00023) |
| 3 | contents SI top quartile ×1.10 | 0.12751 | +0.00007 → s* ≈ 1.16, Z ≈ 0.12 → shrunk ≈ ×1.02 |
| 4 | motor value groups 15–20 ×1.06 | 0.12749 | +0.00005 → s* ≈ 1.09 (incl. the line's 3 %), shrunk ≈ +1–2 % relative |
| 5 | buildings wind-jump regions (R03,R08,R06,R11,R12) ×1.08 | 0.12733 | −0.00011 → s* ≈ 1.00 for these regions vs ×1.095 for the line: the weather-level story is rejected |
| 6 | buildings SI Q4, even-id half ×1.20 | 0.12743 | −0.00001 → s* ≈ 1.09 in the even half vs ≈ 1.25 for all of Q4 (implied ≈ 1.4 in the odd half): **lumpy** — the Q4 gain sits in a few rows of one half |
Exact Tweedie reading: ΔLB = −α(s^−0.5−1) − b(s^0.5−1), s* = α/b, b ∝ Σμ^0.5 (buildings line b = 0.180).
Conclusion: the high-SI uplift on the public rows is carried by a handful of claims (split-half test), and the
high-wind regions do not want any uplift at all — both point to public-row noise, not a 2025 mechanism. Supports
ticking the credibility-shrunk file.

### 10e (cont.). Tail tests, slots 7–9 (20 Oct 00:40 UTC; user request: models for the large outliers)
Reference: X21 × line levels (motor ×1.029, buildings ×1.093) ≈ 0.1279 (X21 raw 0.12744 + the probed line gains).
| File | Public LB | vs ≈0.1279 | Local CV |
|---|---|---|---|
| T3 model-based large-loss loading + levels | 0.12709 | −0.0008 | 0.1579 (below baseline) |
| X21 × per-row large-loss propensity uplift + levels | 0.12816 | ≈ +0.0002 | uplift not CV-validated (excess-model OOF corr 0.012) |
| X21 with power-1.7 motor + levels | 0.12668 | −0.0012 | motor 0.1082 < 0.1091 |
Reading: the outlier models lose on the public rows exactly as CV predicted; the small public gain of the
propensity uplift (range 0.97–1.10 per row) is the size of public-row noise (σ ≈ 0.0003 for a file-level
difference) and has no CV support → not adopted. Large losses are not predictable from these features.
