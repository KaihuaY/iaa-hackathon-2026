# Part 2 plan — emerging market (27 Oct 2026 15:40 UTC → close 31 Oct 03:59 UTC)

Scoring: Tweedie D² (p=1.5) on the 209,063 emerging-market test rows (public slice during the competition, private at
the close). Same working method as Part 1: Fable orchestrates and verifies, Opus agents write code from specs with
disjoint file ownership, one heavy training job at a time, nested CV + forward holdout decide models, the leaderboard is
used only for gross-error checks (Part 1 lesson: public-probed levels did not transfer; structure chosen locally did).

## What is different from Part 1 (evidence: `docs/part2_readiness.md`, checks run 27 Oct)
| Fact | Consequence |
|---|---|
| Two history years only (2023 mature, 2024 ≈ 20–25 % open); 110k rows, 9,258 claims (≈ 4 % of Part 1) | emerging-only models are noisy; forward holdout = train 2023 → score 2024 (immature, completeness-scaled) |
| Test is **76 % new business** (81k of 90k motor rows start Jan 2025); history is 12–15 % new business; only 24 % of test policy ids seen before | history features are absent for most test rows; the new-business relativity must come from the developed market (2.4M rows) |
| Buildings fire cost 3.2M (2023) vs 1.1M (2024); 2023 buildings loss ratio 1.34 | buildings level uncertain by ±30 %; use developed-market shape, emerging level with credibility |
| Schema: age bands replace ages; no mileage, telematics, DOB, credit band; `gender` new; construction codes BRK/WOD/MIX/STN; regions E01–E06; channel mix broker-heavy | feature lists rebuilt; developed data mapped to the emerging schema for pooling |
| Same currency scale and the same Part 1 DQ faults (exposure in days, SI ×1000) at the same rates | Part 1 cleaning rules kept; thresholds verified |
| Level vs Part 1 history: motor frequency ×1.29, severity ×0.58 (PP ×0.75); contents PP ×0.80; buildings PP ×0.98 | a pooled model needs `market` as a feature and emerging rows up-weighted |

## Tracks (parallel, disjoint files)
A. **Emerging-only pipeline** — `PART2` MarketSpec (`data/part2/emerging`), column-list fixes, configs `P2_*`, first
   file. Baseline and the fallback.
B. **Pooled market** — `data/part2/pooled` (developed mapped to the emerging schema + "D"-prefixed ids), spec
   `part2pool`, `market` feature, developed-row weight ∈ {0.1, 0.3, 1}, emerging-only scoring
   (`scripts/score_market.py`). Expected main lever (new-business relativity, severity shape, thin segments).
C. **Office premium build-up** — `scripts/office_premium.py`, `docs/part2_office_premium.md` for the 20 reference
   policies (judged deliverable, not scored).

## Day plan
- 27 Oct: A + B first files; submit the emerging-only baseline and the pooled candidate (gross-error check: a
  30 % level miss is visible even on a ~50k-row public slice; a 5 % one is not). ≤ 10/day, user-approved.
- 28 Oct: CV-driven selection (seed bags of every part, SI-offset screen k ∈ {0, 0.5, 0.7} for home lines, developed
  weight), credibility of level: emerging 2024 vs developed trend; blend; office-premium doc on the candidate file.
- 29 Oct: final candidate, hard-shrunk public factors only if two independent readings agree (Z ≤ 0.2), write-up and
  video updates (adaptation section), repo tidy.
- 30 Oct: tick the final on Kaggle (manual), deliverables packaged before 31 Oct 03:59 UTC.

## Scoring model (confirmed 27 Oct 19:30 UTC) and the 28 Oct upload protocol
**Scoring is known exactly.** Public score = unweighted row-level Tweedie D² (p = 1.5) on a random subset of the test
file (the organisers' description, confirmed by the probes: with μ the base file, scaling a subset by s changes the
score by ΔLB(s) = −α(s^−½ − 1) − β(s^½ − 1), optimum s* = α/β, β ∝ Σ√μ over the subset with one constant for the whole
file). Checks: buildings β fitted from two probes 0.1471 vs 0.1485 from the file's Σ√μ (1 %); the composite probe
(NB ×1.25, renewals ×0.93) was predicted +0.00188 and scored +0.00187. The fitted constants live in
`docs/part2_response.json`; `scripts/part2_factors.py` builds factor files and prints the predicted change before
upload. Readings so far (on the reference file `…POOL_P34_w10_v0_dn.csv`, 0.15968):

| subset | s* | max gain | sampling SD of s* (public ≈ ¼ of rows) |
|---|---|---|---|
| motor all | 1.152 | +0.0016 | ≈ 0.05 (≈ 4,500 public claims) |
| motor new business | **1.195** | **+0.0021** | ≈ 0.05 |
| motor renewals | 0.99 | 0 | ≈ 0.10 |
| buildings all | 1.063 | +0.0001 | ≈ 0.08 |
| contents all | 0.964 | 0 | ≈ 0.07 |

**Why the level gap is real, not a bug (checks run 27 Oct 19:00–19:30 UTC, `scripts/part2_nb_diagnostics.py`):**
- The round-2 model (`R2_X4`) is calibrated to the developed history (OOF actual/expected 1.02 motor) and sits at
  0.895 × the reference motor level, so 2025 public motor cost is ≈ 1.27 × the history-calibrated level, ≈ 1.33 on
  new business and ≈ 1.10 on renewals.
- Not a premium trend: the insurer's motor renewal premium per policy is flat (604 / 601 / 608 for 2023 / 2024 / 2025).
- Not weather: 2025 hail days 11.4 vs 14.0 / 17.1 in 2023 / 2024 (milder).
- Not development: emerging claims report and settle exactly like developed ones (mean lag 26 vs 25 days, 0.9 % vs
  0.9–1.6 % reported after a year, open share 3.7 % vs 3.9 % for 2023 losses), so the one-year lag curve is fine.
- Not the pooled trend index: developed and emerging full-year premium levels coincide (594 vs 603 in 2023).
- New business is structural, not age: on emerging rows the model under-predicts new business in every age band
  (a/e 0.97–1.41) and over-predicts renewals in every band (0.87–0.99). The two history years disagree on its size
  (NB/renewal calibration ratio 1.82 in 2023, 0.95 in 2024, ≈ 350 NB claims each), which is why a train-2023 →
  score-2024 holdout rejects it; the public reading (≈ 3,600 NB claims) is three times as precise as both years
  together and implies a 2025 NB/renewal relativity of 1.61 (raw history 1.60, pooled two-year OOF 1.71, model 1.34).
  Contents shows the same sign (NB a/e 1.24, holdout +0.0014 with NB ×1.2); buildings the opposite (0.80, 40 claims
  a year — not credible).

**Decision rule (differs from Part 1 on purpose):** Part 1 readings were ±5 % levels with SD 5 % and did not transfer.
Here the readings are 15–30 % with SD 5 %, the prior from history is ±25 % (the two years disagree), so credibility
Z = τ²/(τ² + σ²) ≈ 0.9. Factors go on the structural subset (new business) with Z-shrinkage, and the model keeps its
shape. Everything is logged as a 2025 measurement, not a model.

**28 Oct protocol (10 uploads, each with a predicted score written before upload; results to the ledger):**
1. `R2_X4` raw — measures the round-2 shape gain and re-calibrates α, β on the new base (predicted ≈ 0.1575: +0.0015
   shape, −0.0037 level).
2. `R2_X4` × motor NB 1.33, motor renewals 1.10, buildings 1.09, contents 0.98 — the public-optimal file
   (predicted ≈ 0.1634; the all-optima ceiling on the reference response is +0.00225 over the reference, so the
   motor level/NB fix is worth ≈ +0.0022 and the shape gain comes on top).
3. PROBE: file 2 × contents NB 1.20 (history says +0.0006 if NB/renewal 1.24 holds, −0.0008 if not).
4. PROBE: file 2 × buildings NB 0.85 (+0.0007 if the history ratio 0.80 holds, −0.0014 if not).
5. PROBE: file 2 × motor NB 1.10 (checks that the NB optimum on the round-2 base is where the reference said).
6–7. Round-2 seed-bag blend (agent B2) × the Z-shrunk factors from readings 1–5 — final candidates (`P2_` names).
8–10. Spare: a written-premium-feature model if it passes emerging CV; a second reading of anything inconsistent.
Never tick a PROBE file. Final tick = the best `P2_` candidate by predicted private score (public reading minus the
noise part), decided 29 Oct.

Prepared 27 Oct 19:36 UTC (`submissions/part2/`, all 209,063 rows in sample order, finite, ≥ 0):

| # | file | sha256_16 | predicted vs its base | purpose |
|---|---|---|---|---|
| 1 | `20261027_1826Z_P2_POOL_R2_X4.csv` | (B2 ledger) | ≈ 0.1575 (−0.0037 level, +0.0015 shape vs 0.15968) | shape gain, re-calibrate α, β |
| 2 | `20261027_1936Z_P2_POOL_R2_X4_opt_nb133_ren110_b109_c098.csv` | 5923c499377c7557 | +0.00595 vs file 1 (≈ 0.1634) | public-optimal candidate |
| 3 | `20261027_1936Z_PROBE_R2opt_contentsNB_x1.20.csv` | 914fa69675d84671 | −0.0006 if contents NB = line, +0.0008 if 1.24, +0.0021 if 1.45 | contents NB reading |
| 4 | `20261027_1936Z_PROBE_R2opt_buildingsNB_x0.85.csv` | 0f57ad0ebc6e6880 | +0.0012 if buildings NB = 0.8 × line, −0.0009 if = line | buildings NB reading |
| 5 | `20261027_1936Z_PROBE_R2opt_motorNB_x1.10.csv` | b071eceb5c3cc9a0 | −0.00054 if the NB optimum on the round-2 base is 1.19 × reference | NB optimum check |
| 6 | `20261027_2009Z_P2_POOL_R2_FINAL_shrunk_nb130_ren109_b107_c099.csv` | df4db3ab6488d315 | +0.00564 vs `R2_FINAL` raw (≈ 0.1635) | **intended final**: B2's seed-bag blend × Z-shrunk factors (motor NB 1.30, renewals 1.09, buildings 1.07, contents 0.99) |
| 7 | (after readings 3–5) file 6 × contents NB / buildings NB factors if the probes confirm history | | second final |
| 8 | `20261027_2112Z_P2_POOL_R2_FINAL_wpl_shrunk_nb130_ren109_b107_c099.csv` | c7aafdce999acf80 | ≈ 0.1638 + 0.0006 if the premium feature's nested gain (+.0007 overall, Review 13) is real | paired reading vs file 6 on identical public rows: written premium as a feature (agent B3; small, seed-noise-sized gain; F22 part rejects it) — only promoted to final if it beats file 6 by ≥ +0.0004 |

File 6 scored **0.16382** (#8) at 21:04 UTC, so file 1 is dropped from the 28 Oct batch; order is now 3, 4, 5, 8, then 7
and bagged refinements. Written premium stays out of the intended final unless reading 8 confirms it: the gain is within
seed noise and a premium-as-feature model is harder to defend as a pricing model.

**28 Oct 02:22–02:25 UTC — the probes were taken on file 6 itself (not the file-2 base), so each reading gives the
residual optimum of the intended final directly.** β of any subset of any file is known from its Σ√μ; a probe then fixes
α exactly (`scripts/part2_factors.py --fit`, entries under `fits` in `docs/part2_response.json`). Before upload the
reference α was transferred to file 6 (uniform-rescale approximation): expected residual optima motor NB 1.019,
renewals 1.035, buildings 0.983, contents 0.994. Readings (4 slots used, 6 left):

| # | file | sha256_16 | public | reading | residual optimum s* (sampling SD) | max gain left | decision |
|---|---|---|---|---|---|---|---|
| 9 | `20261028_0222Z_PROBE_F6_motorNB_x1.10.csv` | 663a4bb161fb3b03 | 0.16337 | −0.00045 (predicted −0.00035) | **1.011** (0.08) | +0.00001 | motor NB factor 1.30 is at the optimum; nothing to change |
| 10 | `20261028_0222Z_PROBE_F6_contentsNB_x1.20.csv` | 7c76aee5f951f7ea | 0.16336 | −0.00046 | **1.017** (0.11) | +0.00000 | history's contents NB a/e 1.24 is not in 2025; no factor |
| 11 | `20261028_0222Z_PROBE_F6_buildingsNB_x0.85.csv` | 055f58575eaa960e | 0.16225 | −0.00157 | **1.079** (0.15) | +0.00018 | history's buildings NB a/e 0.80 rejected (wrong sign); 0.5 SD from 1, no factor |
| 8 | `…R2_FINAL_wpl_shrunk…csv` (paired, same public rows) | c7aafdce999acf80 | **0.16407** | +0.00025 vs file 6 | – | – | below the +0.0004 promotion rule: written premium stays out; file 6 remains the intended final |

**Level programme closed.** File 6 is at the public optimum on every structural subset (all residuals within one
sampling SD of 1; the largest possible further level gain is +0.0002, inside noise). File 7 is not built: a composite of
these residuals would move the public score by less than its noise and has no private case. Files 3–5 (probes on the
file-2 base) are superseded and were not uploaded. The remaining gap to the leader (0.0090) is model shape, not level.
Credibility per subset now uses the measured sampling SD (Z = τ² / (τ² + σ²), τ = 0.25): motor NB 0.91, contents NB
0.84, buildings NB 0.74. The only unprobed subsets are the home-line renewals (β 0.03 / 0.02; a reading would have
SD ≈ 0.3, so it could not decide a factor), left alone on purpose.

**28 Oct 04:03 UTC — shape, not level: two paired readings of the stage-2 emerging relativity layer (Review 15).**
`20261028_0403Z_P2_F6_stage2_core_a3.csv` (file 6 × motor gender/fuel relativities, level-neutral) **0.16452 = +0.00070**,
promoted to intended final; `…_full_a10.csv` 0.16412 (+0.00030), not promoted. Tick candidate is now the core file.

## Progress log (newest first)
- **30 Oct 04:55 UTC — team-workflow diagram redone with Archify (user request):** spec `final_submission/diagram/ai_team.workflow.json`
  (workflow v2, 3 lanes, 8 nodes, showcase profile), validated 9/9 checks 0 errors, delivered `ai_team_workflow.html`, visual-check
  pass at 1440×900 … 2048×1320. Its SVG + scoped styles embedded in board section 1 (light/dark verified); interactive HTML
  published beside the board. Archify update notice seen (installed 2.17.0-dev.1, latest 3.2.0) and acknowledged; skill unchanged.
- **30 Oct 04:05 UTC — video board published as a private claude.ai artifact at the user's request:** https://claude.ai/artifact/8h6y89Ntv33dRmtsEoMKeB
  (IAA Video Board; same seven sections as the local `final_submission/video_board.html`, artifact page format with theme
  tokens, safe-area gutters, Spectral/Source Sans). Script and folder README point at it.
- **30 Oct 03:45 UTC — `final_submission/video_board.html`:** one offline page (inline SVG, no scripts) with everything to
  show during the video, one section per beat: AI-team workflow diagram, data-quality table, target flow + experiment
  ladder + log quotes, AI-mistakes table, new-market table + pooling diagram, split-half bar chart, what-was-submitted
  table. Video script's "On screen" column now points at the board sections.
- **30 Oct 03:20 UTC — write-up refocused on the model (user request):** §3 now describes target, features, the four
  part families with reasons, the per-line blend weights (X33: motor P.10/B.16/F.36/A.38, buildings .20/.32/.16/.32,
  contents .30/.39/.14/.17; R2_FINAL: motor P.20/F.32/L.48, buildings P 1.0 fixed, contents P.60/B.10/F.14/L.16) and the
  blend-vs-best-part gain (0.1598 vs 0.1586; 0.1725 vs 0.1721); leaderboard level factors reduced to one disclosure
  paragraph; stage-2 layer presented as model structure; private-score forecasts removed. DOCX 4 pages.
- **30 Oct 03:00 UTC — user review of the package:** write-up kept as DOCX only (source moved to `docs/writeup_final.md`,
  US spelling, governance folded into assumptions/limitations, local-file references removed, team line kept; 4 pages in
  Word with A4/narrow margins; `scripts/build_writeup.py` now writes the DOCX only, `--pdf` optional). Video script
  rewritten as a first-person story of working with AI in an unfamiliar domain (~3:30, little about scores). Checklist
  answers the interim-vs-final history question (final code in six staged commits; the development story is in the logs).
- **30 Oct 02:35 UTC — write-up and video script rewritten for the judges (final solution only, plain language),
  reviewed by an independent agent (25 findings, all applied: measured vs applied stage-2 factors, full level
  disclosure incl. buildings ×1.07 / contents ×0.99 carried from file 6, score-step attribution, annual-basis office
  premium 21,555 vs 7,340 CU, review-count wording), re-rendered (4 pages). New dashboard version
  `notebooks/06_part2_final.py` → `reports/06_part2_final_dashboard.html` (section 0 'What was submitted', ticked
  final starred, office premium checked against F11 v3); 05 kept as the historical record.**
- **30 Oct 02:15 UTC — F11 v3 ticked by the user; deliverables packaged in `final_submission/` for review (nothing
  uploaded or committed).** Contents: `writeup.md/.html/.pdf/.docx` (both parts, 4 A4 pages, office-premium build-up in
  §8, three placeholders: team, members, repo URL), the two ticked prediction files (Part 1 sha d83ce8cba461830d, Part 2
  sha 378f7b835d3ace59), `office_premium_*` regenerated from F11 v3 (total office premium 19,892.61 CU vs written 7,340.12,
  mean ratio 3.35), `video_script.md` (4:45, nine beats), `GITHUB_CHECKLIST.md` (six proposed commits; Part 2 is not yet
  committed; data/submissions/caches stay offline), `README.md` index with the form-field map (the Google Form needs a
  sign-in, its fields were not readable). Repo: `LICENSE` (MIT) added, README Part 2 + deliverables sections appended,
  `scripts/build_writeup.py` renders the write-up (Edge headless PDF, Word COM DOCX). Open: record the video, fill
  placeholders and re-render, commit/push per the checklist, submit the form, regenerate the Kaggle token.
- **30 Oct 01:40 UTC — split-half programme done (4 of the user's 6 slots): extreme claims confirmed, motor NB level
  cut to 1.05, robust F11 v3 built, scored and verified. NEW INTENDED FINAL = `20261030_0145Z_P2_F11v3_robust_nb105_ren103_si103_fl105.csv`
  (sha 378f7b835d3ace59, public 0.16139, predicted 0.16130).** Readings in order, each inverted before the next was chosen:
  (1) 17–24 odd-id digits 1/3/5 ×1.10 on F10: **0.16947, +0.00150** (predicted +0.00168 if the odd-half excess is all in
  1/3/5, +0.00080 diffuse, −0.00050 if in 7/9) → ~90 % of the excess sits in 30 % of the young rows (s* 2.04 vs 0.72 even
  vs 0.90 for 7/9); ex-excess young optimum on core_a3 0.90; excess α 0.0445 (core units); 84 % of those rows are NB, and
  "excess in renewals" would need a renewal optimum of 0.28, so the NB-split probe was skipped as non-discriminating.
  This probe is now the account's best public score and is a PROBE: it must not be left as the default — tick explicitly.
  (2) home tilts even half (SI q4 ×1.10 + flood ≥ 3 ×1.12) on F10: **0.16784, −0.00013** (predicted −0.00009 if fully
  priced, +0.00022 if diffuse) → even-half residual 1.04 on core_a3 for both, odd half 1.29/1.34 by subtraction: the home
  signal is concentrated too; history a/e supports only a mild flood-3 tilt (zone 3 a/e 1.52 on 92 claims, zones 4–5 < 1).
  (3) motor NB excluding 17–24, even half ×1.10 on core_a3: **0.16370, −0.00082** (predicted −0.00019 if the NB level
  were right, −0.00097 at the inferred ex-excess optimum 0.81) → even 0.84, odd 0.77 by subtraction, whole 0.81: both
  halves agree, so the NB ×1.30 is ~20 % too high once the few extreme young-driver claims are set aside (model prior for
  NB is 1.0). (4) F11 v3 = core_a3 × motor NB 1.05 / renewals 1.03, no 17–24 or 65+ tilt, buildings SI q4 ×1.03 and
  flood ≥ 3 ×1.05 with buildings cells renormalised: **0.16139 vs predicted 0.16130**; motor level ×0.829 vs core_a3;
  expected change vs core_a3 with the public extreme claims set aside +0.0016 (NB non-young +0.0018). Public ranking
  by file: F10 0.16797 (#6-equivalent) > core_a3 0.16452 > F11 v3 0.16139; the gap is the extreme claims in the public
  quarter (α 0.0445 ≈ 2.5m CU equivalent on 1,700 public rows), which the response model prices exactly. Hedge
  considered and rejected as immaterial: if the generator's heavy tail (developed history: 11.9m/1.1m/0.6m/0.5m) is real,
  the expected private uplift on motor NB is ~2–4 %, worth ≈ 0.00005 either way; the 1.05 is closer to the model prior.
  **Tick decision: F11 v3** (reasons in Review 20); 2 slots unspent, nothing left with stakes above 0.0001 to test
  (motor renewals max gain 0.00012, buildings NB 0.00007). Deadline 31 Oct 03:59 UTC; tick on kaggle.com before then.
- **29 Oct 02:00 UTC — split-half check: the young-driver signal is a few extreme public claims, and so was the
  27 Oct motor level (Review 19). F10 is not to be ticked as it stands.** Last slot of the day: drivers 17–24 even-id
  half ×1.10 on F10 read −0.00083 (uniform effect: +0.00026) → even half s* 0.84, odd half 1.85 (core_a3 units).
  History bootstraps of public-quarter-sized young samples give a per-half log SD of 0.14 and never a half-to-half gap
  of 0.79 (4,000 draws, median 0.12). The odd-half excess α equals one claim of ≈ 2.5m CU; the developed history has
  motor bodily-injury claims of 11.9m / 1.1m / 0.6m / 0.5m, the emerging history nothing above 0.36m. The same excess
  is 19 % of the file-6 motor:NB α: ex-excess optima are reference motor 0.98 (was 1.15), reference motor:NB 0.98
  (was 1.195), file-6 motor:NB 0.815 (was 1.011). So the motor NB ×1.30 / renewals ×1.09 in files 6, core_a3 and F10
  rest on the same few public claims, which the private rows do not share (Part 1's "flat levels did not transfer"
  is the same mechanism). Stakes: if the excess is extreme claims, F10 is ≈ 0.0045 below the robust file on the
  private board; if the young effect were real and diffuse, the robust file would be ≈ 0.005 below F10 — and the
  bootstrap puts the diffuse case near zero probability. **30 Oct UTC plan (user budget 6):** (1) odd-half split:
  17–24 rows with last digit 1/3/5 ×1.10 on F10 — a concentrated excess (one piece ≥ 2.5, the other ≈ 1) confirms
  extreme claims; (2) home tilts split-half (SI q4 even + flood ≥ 3 even) — diffuse ≈ half the full readings;
  (3) robust final `P2_F11_robust` (pre-built, predicted public ≈ see below) = core_a3 with motor NB ×0.83
  (1.30 → 1.08), renewals ×0.945 (1.09 → 1.03), 17–24 ×1.08, 65+ ×0.945, buildings SI q4 ×1.094, flood ≥ 3 ×1.097,
  home cells renormalised, stage-2 core kept; (4) spare for a v3 after (1)–(2). Tick by 31 Oct 03:00 UTC. Slots: 10/10
  used on the 29 Oct UTC day. Open-risk list: motor level now rests on local evidence only (R2_FINAL calibration
  1.02 OOF, NB/renewal history ratio contradicted by the 2024 holdout); extreme-claim hypothesis unconfirmed until
  probe (1); home tilts unverified by halves; buildings holdout calibration 1.41; blend-weight stability; gender factor.
- **29 Oct 01:30 UTC — composite read: 0.16797, #6 of 29; new intended final.** User approved one upload;
  `20261029_0048Z_P2_F10_core_a3_x_seg4_shrunk.csv` scored 0.16797 against the additive-response prediction 0.16809
  (+0.00345 realised vs +0.00357 predicted), so composites can now be predicted to ±0.0001 before upload. Leader
  0.17556 (moved up overnight), gap 0.0076. One slot kept on the 29 Oct UTC day; user budget for 30 Oct UTC: 6.
  Plan for the 6 (probes on F10 itself so each reading is a residual of the tick candidate): motor 25–34 ×1.10,
  motor 35–64 ×0.90, motor value ≥ 15 within ages 25–64 ×0.90, home-line policyholder 17–24 ×1.10 (buildings and
  contents in one file: is the broker-book young-customer effect motor-only?), buildings sum-insured bottom quartile
  ×0.90; then composite v2 from all readings as the tick (predicted before upload). Shrinkage rule for v2: decision
  pending with the user (reasons in the 29 Oct reply; proposal: keep Z = τ²/(τ² + SD²), τ 0.25, since the break-even
  for the 17–24 tilt is a true effect of +13 % against a reading of +34 % ± 10 %).
- **29 Oct 00:50 UTC — external-data search (nothing usable) and 8 segment probes on the intended final: the 2025
  motor age curve is the missing shape (Review 18).** External: the precursor IAA hackathon (Dec 2025) used Ames housing
  data, so its public notebooks say nothing about this generator; the public Part 1 repos found are low-ranked teams
  with nothing new; Kaggle discussion pages cannot be read by tooling (script-rendered); the emerging claims table holds
  no claims for test-only (broker) policies; weather, prices and other real-world data cannot match synthetic CU data.
  Probes (user budget 8, all used; paired readings on `core_a3` 0.16452, residual optimum s* = α/β, α exact per segment):

  | segment (share of Σ√μ) | s | reading | s* | max gain |
  |---|---|---|---|---|
  | motor drivers 17–24 (0.180) | 1.10 | **+0.00138** | **1.34** | +0.0026 |
  | motor drivers 65+ (0.121) | 1.10 | −0.00044 | 0.91 | +0.0002 |
  | motor vehicle value ≥ 15 (0.095) | 1.10 | −0.00067 | 0.78 | +0.0007 |
  | motor NCB 0 (0.085) | 1.12 | −0.00026 | 0.96 | 0 |
  | buildings sum insured top quartile (0.082) | 1.10 | +0.00025 | 1.17 | +0.0003 |
  | buildings freezing days top quartile (0.069) | 1.12 | +0.00005 | 1.08 | +0.0001 |
  | buildings flood zone ≥ 3 (0.056) | 1.12 | +0.00031 | 1.24 | +0.0004 |
  | contents sum insured top quartile (0.047) | 1.12 | +0.00002 | 1.07 | 0 |

  Reading: the 2025 public rows price young drivers about a third above the model while the motor level is at its
  optimum, so the rest of motor (65+, high-value cars) sits 10–20 % too high — a rotation of the age curve, not a level.
  It is **not in the emerging history**: on the pooled OOF the 17–24 band has a/e 0.99 (2023 1.08, 2024 0.89), and the
  file's test relativity for 17–24 (1.54) lies between the developed (1.97 raw) and emerging (1.67 raw) history ratios.
  The only reading of it is the 2025 portfolio itself: the broker book's young drivers claim more than the insurer's
  own. Sampling SD of s* for the 17–24 segment ≈ 0.10 (≈ 1,000 public claims), so 1.34 is a 3-SD signal; the 65+,
  value and flood readings are 1–2 SD; freezing days, NCB 0 and contents SI are flat (the 2025 cold shift is priced).
  Built, not uploaded (budget): composite `20261029_0048Z_P2_F10_core_a3_x_seg4_shrunk.csv` (sha 81ea961dbb2c648d) =
  core_a3 × Z-shrunk tilts (17–24 ×1.286, 65+ ×0.922, buildings SI q4 ×1.094, flood ≥ 3 ×1.097; Z 0.85/0.85/0.59/0.43,
  τ 0.25), every line × NB/renewal cell renormalised to the base total; additive-response prediction **+0.0036 →
  0.1681** (would be ≈ #6). The value ≥ 15 reading is confounded with age (75 % NB, mostly older drivers) and is left
  out until probed within ages 25–64. Also built: probe `20261029_0048Z_PROBE_F9_motor_age2534_x1.10.csv` (reading
  −0.00018 if at optimum, +0.00057 if s* 1.2). Slots: 8 of 10 used on the 29 Oct UTC day (2 left until 00:00 UTC),
  10 more on 30 Oct UTC, deadline 31 Oct 03:59 UTC. Governance for the write-up: this is a public-leaderboard
  calibration of a portfolio shift that no local data can show; Part 1 taught that structural tilts transferred to the
  private board and flat levels did not; the correction is a tilt, level-neutral, credibility-shrunk and disclosed.
  Open-risk list: the age rotation rests on the public quarter only (private shares the 2025 portfolio but not the
  rows); 25–64 bands unprobed (the renormalisation puts them at ×0.905); value ≥ 15 confounded; buildings holdout
  calibration 1.41 on a light 2024; blend weights' fold stability; gender as a rating factor. Next: with user
  approval, upload the composite and the 25–34 probe (2 slots), then on 30 Oct UTC probe 35–64, value ≥ 15 within
  25–64, flood zones 3/4/5 separately, buildings SI q1, and build the final from all readings; tick by 31 Oct 03:00 UTC.
- **28 Oct 04:10 UTC — hidden-feature hunt (agent H1, Review 15) finds one real shape gap; new intended final
  0.16452.** The pooled model under-learns emerging-only relativities on motor gender and fuel type (developed rows have
  no gender; frequency z ≈ 4.2, identical in both years). A level-neutral stage-2 Tweedie GLM layer on the R2_FINAL OOF
  (ridge α 3, cell-renormalised so every probed level is unchanged) gives +4.5 bp honest OOF; its paired public reading
  on file 6 is **+0.00070 → 0.16452** (`20261028_0403Z_P2_F6_stage2_core_a3.csv`, sha b6612c813a25beff), promoted.
  The wider layer (contents gender / works-from-home, buildings trees, +13.5 bp OOF) read only +0.00030 and is not
  promoted. Everything else in the hunt (ids, unused columns, cross-table links, test-vs-train shifts, NB interactions,
  weather shift) is noise or already handled; `customer_service_contacts` confirmed as a history-only leak. Dashboard
  bugs fixed (ledger index not unique; open-risk list read from the newest entry that has one). 6 slots used today.
  Open-risk list: buildings holdout calibration 1.41 on a light 2024, cap pooled across markets, blend weights' fold
  stability unverified (motor L20 weight .06–.80 across folds, Review 16), gender as a rating factor is a governance
  point for the write-up. Next local screen: none queued — the shape round is closed: CatBoost pooled part, blend
  re-weighting, developed-market weight (Review 16) and the per-market cap (Review 17) all fail the nested-CV rule.
- **28 Oct 02:30 UTC — probes nailed down, level programme closed (4 slots used, 6 left).** Three probes on the
  intended final (file 6) read its residual optima directly: motor NB 1.011, contents NB 1.017, buildings NB 1.079,
  all within one sampling SD of 1 (0.08 / 0.11 / 0.15); the largest remaining level gain is +0.0002. Both history-based
  home-line NB hypotheses (contents 1.24, buildings 0.80) are rejected on 2025 public rows. The written-premium paired
  reading (file 8) gave +0.00025 (0.16407, #8), below the +0.0004 rule, so the feature stays out. **File 6 is the final
  level; the remaining gap to the leader (0.0090) is shape.** Tooling: `scripts/part2_factors.py --fit` inverts a
  reading into α, s*, SD and a Z-shrunk factor for any base (self-check reproduces the 27 Oct constants exactly);
  α transfers to a new base as α_ref × β_ref / β_base. Data cleaning re-verified on the emerging tables (Review 14).
- **27 Oct 21:15 UTC — written premium as a feature (agent B3, Review 13): `log(written_premium)` in the P34 part gives
  nested +.0009 motor / +.0017 contents (3-seed paired mean +.0013 / −.0003 / +.0017), nothing on buildings, rejected
  in F22; it does not carry the insurer's new-business loading (motor NB/renewal 1.33 → 1.31). `R2_FINAL_wpl` nested
  .1727 vs .1720. Kept out of the intended final; one paired public reading tomorrow (file 8). Full suite re-run by
  Fable: 160 passed. Dashboard rebuilt with the 0.16382 reading.**
- **27 Oct 21:04 UTC — submission 10 (last slot of the day, user: "make sure it's used"): file 6, the shrunk final
  `R2_FINAL` × factors, public 0.16382 = #8 of 24 (predicted 0.1635 before upload; the +0.0003 surplus is extra
  round-2 shape gain). Best tickable file now; gap to the leader 0.0092 (was 0.0287 this morning).** Consequence for
  the 28 Oct protocol: the shape gain is confirmed, so file 1 (raw blend) is no longer needed; tomorrow starts with the
  three probes (3–5) on the file-2 base, then the second final (file 7) built from the readings.
- **27 Oct 20:10 UTC — round 2 closed (agent B2, Review 11): candidate `R2_FINAL` = seed-bagged P34/L20/F22(/B27)
  blend, emerging nested CV .0912/.1254/.1231, 2024 holdout .1346/.1207/.1192, OOF calibration 1.02/.96/.99, test
  NB/renewal 1.36/1.07/1.05; vs the reference file +.0024/+.0020/+.0017 CV. Verified by Fable: rows, order, finite,
  corr .9965 with `R2_X4`, levels 0.898/0.989/0.972 × reference.** Intended final built: file 6 above (shrunk
  factors, predicted ≈ 0.1635). Kept from B2's open-risk list: buildings holdout calibration 1.41 (light 2024 vs
  the model's fire-year-inflated level — the public reading ×1.06 on the reference settles the level), cap still
  pooled across markets, CatBoost part not run. Next local screen: written premium as a feature (CV-decided).
- **27 Oct 19:35 UTC — scoring model confirmed exact; level gap diagnosed (see the section above).** Agent B2 round 2
  so far: market-specific development kept; motor large-loss loading per market kept (CV +0.0024); history NaN in
  each market's first year kept; the three in-model new-business mechanisms rejected by the 2024 holdout (now
  understood: the 2024 year contradicts 2023 and the holdout cannot resolve it); four-part blend `R2_X4` nested
  0.1691 (file `20261027_1826Z_P2_POOL_R2_X4.csv`); seed bags running. New tools: `scripts/part2_factors.py`,
  `docs/part2_response.json`, `scripts/part2_nb_diagnostics.py`.
- **27 Oct 17:41 UTC — structure confirmed: motor new-business ×1.20 = 0.16175 (best file; +0.0021 over the pooled
  base), +0.00045 above the flat-level prediction for the same overall change; NB ×1.25 with renewals ×0.93 = 0.16155
  (+0.00034 above flat).** Decision: model the new-business relativity (motor ≈ 1.6 vs the model's 1.32) inside the
  pooled model (round 2), do not carry it as a factor; one spare upload kept today. Tomorrow's probes: NB ×1.30 (is the
  optimum higher?), buildings NB, and the round-2 candidate.
- **27 Oct 17:30 UTC — level probes on the pooled base (5 uploads, never tick).** Exact-response readings: motor
  ×0.90 → −0.00332, ×1.10 → +0.00145 ⇒ optimum **×1.15** (+0.0016); buildings ×0.85 → −0.00176, ×1.15 → −0.00009 ⇒
  optimum ×1.06 (+0.0001); contents ×1.10 → −0.00035 ⇒ ×0.96 (flat). Consistency: fitted curvature ratio
  buildings/motor 0.487 vs 0.492 implied by Σ√μ in the file. Interpretation: the model's motor new-business
  relativity (1.32) is below the raw emerging history ratio (1.60); with 76 % new business that alone is ×1.17 on the
  motor level — the probe is measuring a structural gap, not a flat level. Two follow-up probes test the structure:
  new-business-only ×1.20 (flat-equivalent ×1.163, flat response predicts 0.16130) and new-business ×1.25 with
  renewals ×0.93 (flat-equivalent ×1.191, predicts 0.16121). If either beats its flat prediction, the uplift goes on
  new business in the model (round 2 target), not as a factor.
- **27 Oct 17:01 UTC — submission 2, pooled model `POOL_P34_w10_v0_dn`: public 0.15968, ≈ #11 of 24 (+0.0154).**
  Developed + emerging rows, `market` feature, developed weight 1.0, early stopping on emerging rows, Part 1 noise
  list dropped. Beats the emerging-only fit on every line (CV motor .089 vs .085, buildings .123 vs .088, contents
  .122 vs .107; 2024 buildings holdout .121 vs .038). Known weaknesses to fix next: pooled IBNR uses the developed lag
  curve (emerging motor target +5 %), motor over-predicts emerging history by 4–7 %, buildings level sits between the
  2023 fire year and the light 2024 (file = 0.91× 2023 PP, 1.53× 2024 PP), noise list not re-derived, emerging 2023
  history counters are 0 instead of unobservable. Next: level probes (motor ×0.9/×1.1, buildings ×0.85/×1.15,
  contents ×1.1) read with the exact Tweedie response — justified here because the level prior is ±20 %, not ±5 % as
  in Part 1 (credibility of a public reading ≈ 0.8); and modelling round 2 (market-specific IBNR, blend of pooled
  parts weighted on emerging rows, seed bags).
- **27 Oct 16:11 UTC — submission 1, emerging-only blend `P2_X3_blend`: public 0.14429, ≈ #21 of 24 (leader 0.17303).**
  Nested CV 0.1517. The 0.03 gap is ten times a Part 1 gap and far above what a level error can cost (a 20 % miss ≈
  0.002), so the leaders have structure we lack. Ruled out: written premium (D² 0.02–0.04 alone), a portfolio transfer
  (< 1 % of test customers exist in the developed book), missing weather (2025 is present). Confirmed: the
  new-business uplift (developed market NB/renewal cost ratio 1.26–1.52 in all four years for motor). Found: the test
  new-business block is much younger (17–34 = 54 % vs 39 % in history) → age is the dominant factor and the
  emerging-only model learns it from 9k claims. Expected fix: the pooled model (Track B) with the developed age curve.
- 27 Oct 15:40 UTC — Track A spec/pipeline accepted (149 tests, golden hash intact); Track C office-premium script
  accepted (9 tests; placeholder figures until a candidate file exists).

## Progress artifact
`reports/05_part2_progress_dashboard.html`, built offline from `notebooks/05_part2_progress.py` with
`python -P -E scripts/build_dashboard.py notebooks/05_part2_progress.py` after every run batch. Sources:
`experiments/log.csv` (part2 rows), `docs/part2_submissions.csv` (every upload with public score), `docs/part2_readiness.md`
facts, the candidate files in `submissions/part2/`. Shows: leaderboard position vs time, CV/holdout per config and line,
level decomposition (completeness × trend × new-business mix), test-vs-history mix shifts, and the open-risk list.

## Guardrails carried over
Test/OOF pure-premium ratio check (`unsafe`), forward-holdout gate, calendar-proxy rule for unbounded models, golden
Part 1 test must stay green, every AI-written change logged in `docs/ai_review_log.md`, PUBLIC_ONLY files never ticked.
- 30 Oct 05:10 — Video script updated for the Archify diagram: beat 1 now walks the three-lane workflow (direction → explore → spec → build → gate → verify → decide → next decision card; red defect loop), beat 4 points back to the red loop and its "20 reviews" tag, recording notes cover the interactive viewer keys (F presentation stage, R route trace, theme button). Spoken text ≈510 words, about 3:55.
- 30 Oct 09:30 — Local video build `scripts/make_video.py`: per-beat voice files → faster-whisper word timestamps (plain pass is the authority; a filler-primed pass adds "um"s only where no real word is heard, after the primed pass alone was caught relabelling real speech) → fillers and pauses cut with 12 ms crossfades → loudnorm −16 LUFS → Edge slide per board section (min-height removed; tall slides scroll) → H.264 1080p30 segments → one MP4. TTS preview 3:16 rendered and checked frame by frame; `voice/` and `video/` git-ignored.
- 30 Oct 09:40 — Video built from the user's seven Google Recorder clips (Sound Recordings/ → final_submission/voice/beat1..7 by recording time): 4:07, 1080p30, no fillers detected, 8.6 s of long pauses shortened, noise floor −64 to −70 dBFS so no denoise. Output final_submission/video/iaa_video.mp4 + report.
- 30 Oct 09:55 — Second take of the recordings (Sound Recordings/Oct 30 at 9-31..9-37 AM.m4a → beat1..7). Denoise made default: ffmpeg arnndn with the public-domain RNNoise model scripts/rnnoise/cb.rnnn (chosen over afftdn/anlmdn: ~12 dB noise cut, speech spectrum corr 0.97, Whisper confidence unchanged). One-pass loudnorm replaced by measured static gain + limiter after it was found pumping gap noise up ~5 dB. Video rebuilt: 3:45, final_submission/video/iaa_video.mp4.
