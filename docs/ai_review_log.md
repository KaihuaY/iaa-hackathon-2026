# AI review log

Independent reviews of AI-written code and analysis: what was checked, what was found, and what changed.
Complements `docs/experiment_log.md` (experiments) and the pytest suite (automated guardrails).

## 2026-10-17 · Review 1: new modelling code (frequency × severity, quick screen, LB predictor, trend backtest)

**How:** a separate AI agent did a read-only review of `src/models.py`, `src/pipeline.py`, `src/lbmodel.py`,
`src/target.py` and `scripts/run_experiment.py`. It looked for correctness bugs only. A human-directed fix and a test
followed for each finding.

| # | Finding | Severity | Fix | Test |
|---|---|---|---|---|
| 1 | 5 % auto-submit rule had no noise check; a tiny or ≤ 0 best score could let a no-better model auto-submit | Bug | Gain must be > 0, ≥ 5 % of \|best\|, and > max(0.002, residual sd) | `test_lb_predictor_and_submit_gate` |
| 2 | Blend parts aligned only by length | Risk | Assert row_id equality for OOF and test rows | (assert in `run_blend`) |
| 3 | Nested blend CV flattered if a part is itself a blend | Risk | Refuse blend-of-blend | (assert in `run_blend`) |
| 4 | Gamma severity crashes on zero-cost claims (count > 0, cost 0) | Bug (latent) | Severity fitted on cost > 0 only | `test_freqsev_calibrated_and_handles_zero_cost_claims` |
| 5 | Claims trend fitted through one point when backtesting to ≤ 2022 | Bug (latent) | Needs ≥ 2 years, else no trend | n/a |
| 6 | Target-construction parameters (cap, loading, IBNR, trend) estimated on all training years, including CV validation rows | Minor leakage | Documented; effect small (aggregate parameters, not row targets). Not changed. | n/a |

The reviewer confirmed as correct: Poisson objective override, severity start level, rebalancing, frequency early
stopping labels, loading only with a cap, the GroupKFold folds, LB-predictor anchoring, per-line trend handling, and
that nothing can be submitted without explicit approval or the 5 % rule.

## 2026-10-17 · Earlier self-caught issues (found from diagnostics, before this log existed)

| Issue | How it was caught | Fix |
|---|---|---|
| LightGBM started boosting from rate 1 CU/yr (init_score disables boost_from_average) | LB below GLM, test mean 10–20 % low | Start from the average rate; calibration test |
| Tweedie GBM drifted low by chasing chance claim-free pockets | One-fold diagnostic: in-sample calibration 0.90 → 0.70 | Large leaves, slow learning, rebalancing |
| Cumulative claim history acted as a calendar-year proxy | Holdout drop and test-mean jump in G5 | 2-year look-back window; stability test |
| Noise features (`previous_insurer` etc.) ranked as important | Year-to-year stability screen | Features removed: CV +0.004 |

## 2026-10-17 · Review 2: credibility correction layer (code written by a smaller model to a spec; reviewed before first run)

| # | Finding | Severity | Fix | Evidence |
|---|---|---|---|---|
| 1 | Bühlmann actual `A` was raw reported claim cost while expected `P` was on the developed, trended scale, so `A/P` was biased 11–24 % low and the test rows (longest history) would have been pushed down hardest | Bug | `A` built from the evaluation target (same scale), then rescaled so the population ratio is 1 | Sanity print of mean `A/P`: 0.80 before, 0.98–0.99 after |
| 2 | Segment design included ~80 `make_model` / `previous_insurer` level columns the stability screen had classed as noise; they forced the ridge penalty to its maximum and shrank the real indicators too | Design | Structural indicators only by default; levels opt-in (`--levels`) | Nested D² falls when levels are included |
| 3 | Unit tests passed on synthetic data with no trend, so neither issue could be caught by tests alone | Process | Real-data sanity prints added to the run | — |

Outcome: correction layer worth +0.0001 nested (in-sample scan had suggested up to +0.006; the rare-segment deviations are
mostly policy-level noise). Kept as a marginal blend member.

## 2026-10-18 · Review 3: smooth log-GLM extrapolation caught by the holdout and test-mean checks

| # | Finding | Severity | How caught | Fix |
|---|---|---|---|---|
| 1 | The new smooth GLM (log1p + squared terms + interactions, written by a smaller model to spec) scored a normal 5-fold CV (buildings 0.1117) but a 2023 time-holdout D² of −0.66 and a 2025 test-prediction mean of 24,016 vs ~400: quadratic log terms and products extrapolate explosively outside the fit range | Bug (would have corrupted any blend containing it) | Not by CV (out-of-fold rows are interpolation). By the 2023 holdout printed alongside every run and by the per-line test-mean print | Winsorise inputs to the fit range, clip standardised terms, bound the linear predictor; plus a new pipeline guard: any model whose test pure premium per exposure is outside 0.5–2× its OOF value is marked unsafe, cannot be blended, cannot be written to a submission |

Lesson for the write-up: nested CV is necessary but not sufficient; a forward holdout and a test-distribution sanity check
are the guardrails that catch extrapolation, and they must be automatic because a blend would otherwise hide a broken part
behind a small weight on one line.

## 2026-10-18 · Review 4: the smooth GLM still failed the 2023 holdout after the extrapolation fix — root cause

After Review 3's fix the smooth GLM (G14) and smooth-boost (B19) scored sane 5-fold CVs (0.1562 / 0.1570), sane test
means (test/OOF 0.93–1.04) and sane test tails (max/median comparable with CatBoost), yet the buildings 2023 holdout was
still −0.59. A targeted diagnostic (script written by a smaller model to a spec; findings verified against the feature
ranges in the real training and test sets) found:

| # | Finding | Evidence | Decision |
|---|---|---|---|
| 1 | Not an outlier problem: the 20 worst rows carry 8 % of the deviance; removing rows with pred > 50× median makes the holdout *worse* (−0.61) | deviance decomposition on the reproduced holdout (−0.587) | Clipping/winsorising cannot fix it |
| 2 | The cumulative-exposure history counters (`cust_expo`, `pol_expo`, `cust_{motor,buildings,contents}_expo`) are calendar-time proxies: mean 0.00 (2021), 1.67 (2022), 2.99 (2023), 2.95 (2024). Trained on ≤2022 only, every 2023 row sits in a region of the quadratic terms the fit never saw; the five largest GLM coefficients are their linear/squared terms (±1.1 to ±3.8, offsetting each other) | 55–69 % of 2023 rows outside the ≤2022 range; dropping the five counters moves the holdout from −0.587 to **+0.108** (≈ the 5-fold CV of 0.111) | Smooth GLM family drops the counters (`EXPO_COUNTERS`); trees keep them (they map out-of-range values to the last leaf; tree holdouts are +0.06–0.07) |
| 3 | The 2025 test set is **in range**: 0 % of test rows above the training max on any history feature; test mean `cust_expo` 2.95 = the 2023–24 level, which is half the training rows. So the test predictions of G14/B19 were not actually extrapolating — the failure is specific to the ≤2022-only fit | feature-range check on the full training vs test matrices | A part must still pass the time-shift test to be in the final blend: B19 is replaced by B20 (no counters), G14 had zero blend weight and is dropped |
| 4 | Bands-mode GLM on the same holdout: +0.085 — quantile banding is inherently shift-robust, which is why the original GLM base never showed the problem | — | Documented as a design trade-off: smooth terms gain CV (0.1562 vs 0.1522) but need shift-safe inputs |

Lesson: random K-fold CV cannot see a feature that drifts with calendar time; only the forward holdout can. Any feature
built from cumulative history must be either windowed and saturating (as the 2-year window is from 2023 on) or excluded
from models with unbounded functional forms.

## 2026-10-18 · Review 5: verification of the round-5 code written by smaller models (no defects found, one risk noted)

| # | Item | Verification | Outcome |
|---|---|---|---|
| 1 | Pipeline refactor for pooled lines (`_units/_load_unit/_fit_unit`; fold, holdout and full-fit logic unified; loading applied after the raw fit) | Golden-hash test on the Part 1 pipeline + 91-test suite after merge | Byte-identical outputs for every config without `pool`; merged |
| 2 | Neural residual learner (`NNBoost`): Tweedie deviance formula, exposure offset through the GLM margin, ±3 residual clamp, early stopping, rebalancing | Line-by-line read against `LGBMTweedie`/`GLMBoost`; synthetic tests (interaction recovery, calibration, unseen category); real quick screen (0.1623, early stop at epoch 1–2) | Correct; weak as a model — retune N2 queued, rejection if < the LightGBM residual |
| 3 | Covariate-shift weights hook (`_fit_predict`): weights computed against the rows each fit predicts; mean 1; no-op without the key | Synthetic tests (same-distribution ≈ 1, shifted target, no-op); golden hash | Correct; **judge only on the 2023 holdout** (CV folds are training-distributed, so CV cannot score it) |
| 4 | Risk: `_concat_frames` drops columns not shared by the pooled lines | Noted at review; checked on the first pooled quick screen | — |
| 5 | `largeloss._num` (smaller model) used `np.issubdtype` on pandas Categorical columns → `TypeError` on the first real run (P27) | Caught by the run, not by the synthetic tests (which used object columns) | Fixed with `pd.api.types.is_numeric_dtype`; regression test with a real `Categorical` added |

## 2026-10-19 · Review 6: three-agent strategy debate (actuary / Kaggle tactician / validation skeptic), judged by Fable

| # | Finding | Evidence | Decision |
|---|---|---|---|
| 1 | Factors fitted on the public rows are substantially sampling noise: SD 0.10–0.13 per SI quartile, 0.05 per line; credibility Z ≈ 0.03–0.5 | X21 OOF bootstrap on public-sized samples; year-shock vs public-noise decomposition | Credibility-shrink every leaderboard-derived factor; build a shrunk final candidate; decide the ticked file on 22 Oct |
| 2 | "CV gains do not transfer" was a misreading: P(public gain ≤ 0 \| CV +0.0007) = 0.17 | paired subset bootstrap | Keep choosing models by nested CV; probes only for the 2025 level/structure |
| 3 | Small-cell probes (< 10 % of a line) are unreadable (σ 0.15–0.25) | noise algebra + bootstrap | Two small-cell probes replaced by large-segment ones (user approved) |
| 4 | No data mechanism supports a 2025 high-SI shift (indexation uniform; no SI-by-year trend; tariff flat by SI; reserves flat by SI) | read-only checks | Treat the public tilt as mostly noise; keep the model-driven SI^0.7 tilt |
Full synthesis: docs/debate_19oct.md.

## 2026-10-19 · Review 7: clean-room rebuild (independent Sonnet agent, no access to `src/`), diffed by Fable

A Sonnet agent was given only `COMPETITION.md`, `docs/kaggle_pages.md` and `data/raw/`, and wrote
`notebooks/03_cleanroom.py` (own cleaning, net-incurred target, count chain-ladder IBNR, LightGBM Tweedie per line,
customer-grouped 5-fold CV). Outputs in `reports/cleanroom/` (`assumptions.md` lists 12 interpretive choices with source sentences).

| # | Check | Result | Decision |
|---|---|---|---|
| 1 | Rows, reported claim counts, reported net cost per line × year (2021–24) | **Identical** in all 12 cells (0.00 %); same 604 duplicate claims and same loss-date-in-period attribution | No join, netting or repudiation bug |
| 2 | Exposure | −0.01 % in every cell (clean-room caps the ~3–4k training rows with exposure > 1) | Immaterial |
| 3 | Developed target, mature years 2021–22 | within 0.7 % | Agrees |
| 4 | Developed target, immature years: motor 2023 +10 %, motor 2024 −5 %, buildings 2024 +4 % (clean-room vs ours) | Explained by two deliberate choices, not bugs: (a) suspect claim `CL30973963` (11.9M BI) valued at paid in our eval target; (b) ours is Bornhuetter-Ferguson with late-severity relativity on every row, the clean-room scales reported cost by a count chain-ladder, so rows with no reported claim get no IBNR (its assumption 12) and late claims are assumed average-sized | Keep ours; BF is the right row-level method |
| 5 | Test-prediction level: clean-room is 17–25 % below our finals per exposure (motor 475 vs 572, buildings 402 vs 539, contents 106 vs 130) | Clean-room has no 2025 trend and predicts below its own training mean; public-board level probes (17 Oct) showed ×0.95 losing on both motor and buildings | Clean-room level would score worse; ours is board-validated |
| 6 | Ranking: correlation with X30_sifix on test, Spearman 0.88 (motor), 0.92 (buildings), 0.95 (contents) | Clean-room OOF, put on our level, scores D² 0.1470 vs ours 0.1600 on our eval target; best blend (5–10 % weight) +0.00008 | Noise level, same as the 80-part grand blend; not submitted |

Conclusion: an independent rebuild agrees with our data handling exactly; differences are documented method choices
in development and trend. No hidden systematic error found, so no score change, but this is independent validation evidence.

## 2026-10-20 · Review 8: "different analysis skills" sweep — four lenses, each with a falsifiable test

User question: would analysts with different skills (instead of another model) raise the score? Fable designed four
lenses with a kill rule each; Sonnet agents wrote `scripts/lens_ceiling.py` and `src/season.py` from specs (unit
tests first); Fable reviewed the diffs, ran every job and judged.

| Lens | Hypothesis | Test | Result | Decision |
|---|---|---|---|---|
| L0 statistician | Missing structure is still large enough to find | Parametric bootstrap of X21 OOF on raw 2021–23 claims (Poisson × empirical severities) | Actual deviance = simulated-truth deviance (overall z ≈ −0.2; resolution ±0.004 D²). 25 % row-subset SD of D² = 0.0036 | **No detectable headroom**; the public gap to the leader (0.0016) is inside sampling noise |
| L1 forensic | Host statements reveal generator structure | Re-read competition pages (target definition, excess, recoveries) | Target definition matches the pipeline (paid is net of excess; repudiated excluded) | Nothing new |
| L2a reserving actuary | Winter peril seasonality is mispriced on partial-year rows | A/E by seasonal weight → seasonal exposure offset (P35) | Real in counts (buildings A/E 1.30–1.48, z ≈ 3–4) but winter claims are smaller; quick screen buildings −0.0004 | Rejected after one negative screen; option kept off by default |
| L2b reserving actuary | Open-claim case reserves are biased | Severity by status × accident year | Smooth AY severity trend, no anomaly | Rejected before modelling |

Lesson for the write-up: a statistically significant pattern in claim **counts** did not survive in **cost**, which is
what the metric scores; the score-relevant test (cost-based quick screen) is the arbiter, not the diagnostic.
The ceiling test turns "plateau" from an impression into a measured statement.

## 2026-10-27 · Review 9 — Part 2 spec and pipeline adaptation

Fable wrote the PART2 spec, the code changes and `experiments/configs_part2.py`, and ran every screen. Each decision was
checked against data before use (details in `docs/part2_decisions.md`).

| AI-written item | Check | Outcome |
|---|---|---|
| `completeness_last_year` derivation | Re-ran the same code under PART1 and compared with Part 1's hard-coded 0.920/0.939/0.960 | Unweighted row mean reproduces Part 1 exactly; the exposure-weighted variant (first guess in the task spec) does not (0.932/0.950/0.970). Part 2 values 0.923/0.944/0.961 also agree with the 2024/2023 reported-frequency ratio for motor (0.921) |
| Ported GLM-based configs (B15, B27) with Part 1 ridge alpha 1e-4 | Quick screen + fitted-rate distribution | **Caught:** buildings D² −0.27..−0.48, fitted rates 2.7–12,000 (median 175). Alpha sweep → 10; B27 became the best single model (CV 0.149, best holdout) |
| Leaf sizes scaled by row count | Quick screen leaf 150/300/600 | 300 kept in the spec; low-reg parts at 100 |
| `submit.run_tests` under another market | Code reading | **Caught:** the shared pass marker meant a part2 pass (Part 1 data tests skipped) could satisfy the Part 1 submit gate on the same commit. Marker is now per market |
| occupation normalisation | Real-data test (one spelling per key) + Part 1 gating (`normalise_labels=()` returns the input frame) | Cleared; 47,924 rows merged; golden Part 1 test passes |
| construction_type map | Real-data test: only the 4 canonical labels, no NaN, history and test | Cleared |
| Motor large-loss segments | 2023 excess/capped ratio by candidate segment | VVG quartiles monotone (0.133 → 0.032); driver-age bands non-monotone and thin (4–13 large claims) |
| Submission file | Row ids identical to sample_submission, 209,063 rows, finite, ≥ 0; per-line PP vs history | Cleared; level ratios explained (development, 2023 fire year, trend, new-business mix) |

Not verified: the new-business relativity on a 1-January bulk book (76 % of test), which history (mid-year new business) cannot test.

## 2026-10-27 · Review 10 — pooled market (developed + emerging history, `market` feature, emerging test)

Fable designed the pooled market from the brief; this agent wrote the code and ran it. Every change is additive; Part 1
behaviour is unchanged (golden test re-run with IAA_MARKET unset).

| # | Change / finding | Check that cleared or caught it | Decision |
|---|---|---|---|
| 1 | `scripts/make_pooled_market.py`: developed ids prefixed "D" (customer_id zero-padded first), ages banded to the emerging labels, telematics/mileage/date_of_birth/credit_band dropped, gender added as missing, construction labels canonical (BRK/STN/WOD/MIX -> brick/stone/timber/non_standard; shares 70/8/12/10 % match the developed brick/stone/timber/non_standard split) | 145 built-in checks: row counts = sum of halves, all ids unique, test row_ids = sample_submission (209,063), customer/address joins 100 %, claims attach to their own market's policies exactly as in the source data, same value kind per column in both halves; `tests/test_pool.py` | accepted |
| 2 | Part 1 licence repair (years_licensed <= driver_age - 17) needs the exact age, which banding removes: applied in the writer before banding (2,962 developed motor rows) | unit test | accepted |
| 3 | **Caught:** the premium-index trend (`target.trend_index`) divides the 2025 mean premium (emerging test) by the pooled mean premium of each training year, so it mixed market price level with time: emerging targets would be off by +4 % motor, +4 % buildings, -10 % contents | hand calculation from the per-market premium tables | developed written_premium rescaled per line (x1.062 / 1.038 / 0.891) to the emerging 2023-24 level (written_premium is not a model feature); resulting trend factors within +-1.5 % of the emerging-only spec |
| 4 | **Open:** pooled BF IBNR uses pooled (developed-dominated) a-priori frequency x severity and the developed 2021-22 lag curve: emerging 2024 IBNR is 1.45-1.61x the emerging-only spec's; emerging motor y_eval +5 %, buildings / contents +1 % | `diag_target` comparison of both specs on the emerging rows | not fixable without editing target.py; reported, and every score also given on the emerging-only y_eval ("ref") |
| 5 | `src/pipeline.py`: cfg `market_weight_developed` (+ `_val` for the early-stopping set) multiplies `w` for developed rows; per-row `holdout_idx` / `holdout_pred` saved per line; `quick()` prints emerging-row D2 when a `market` column exists; blends drop part 0's per-row holdout and store the blended one | default-off (no key / no `market` column -> identical arrays); `tests/test_pool.py`, full suite incl. golden | accepted |
| 6 | `market` is a string column, so `model_matrix` makes it categorical with no code change; but its gain is ~0: region (R.. vs E..) already identifies the market | importance of the fitted models | no action; market effects enter through region |
| 7 | **Caught:** with `noise_features=()` the 40-level `previous_insurer` became the top-gain feature in every pooled line | full run with the Part 1 noise list dropped: emerging CV 0.1793 -> 0.1817, holdout ref +0.002 / +0.003 / -0.001 | `_dn` variant is the candidate |
| 8 | Emerging-only early stopping (`market_weight_developed_val=0`) is unstable for GLM+boost (iters 1 to 1382 across folds; buildings holdout -0.03) | fold iteration counts | rejected for B15; fine for LightGBM |
| 9 | Emerging motor OOF over-predicted by ~7 % (holdout 4 %); a per-line A/E rescale fitted on 2023 OOF worsens the 2024 holdout | holdout check | file written unscaled; residual level risk stated |

## 2026-10-27 · Review 11 — pooled round 2 (market-aware development, loading, NB screens, blend, bags)

Agent-written code, every change additive (Part 1 golden test green; 159 tests with IAA_MARKET unset, 136 + 23 skipped
under part2pool). Scores are on EMERGING rows only (`scripts/score_market.py`, now `--log` / `--nb-cal`), on the new
market-aware target unless stated; log rows in `experiments/log.csv` (market part2pool, cv columns = emerging rows).

| # | Change / finding | Check | Decision |
|---|---|---|---|
| 1 | `target.prepare_target`: with a `market` column, completeness curve, late-severity relativity and BF a-priori freq x sev fitted per market (`_market_specs`, `_develop_by_market`); emerging overrides in `PooledSpec.market_dev` (curve/mature/severity year 2023). `develop_by_market: False` reproduces Review 10 | emerging y_eval / emerging-only PART2 y_eval 0.999 / 1.000 / 1.001 (was 1.050 / 1.011 / 1.009); 2024 completeness 0.923 / 0.944 / 0.961 = PART2; uncapped y_dev identical row by row (`tests/test_target_market.py`, real-data test) | kept (R2_P34_dev: CV +.0000/+.0009/+.0008, holdout -.0002/-.0004/+.0012, motor OOF calib 1.12 -> 1.09) |
| 2 | **Caught:** the large-loss loading was pooled too: at the 0.99 cap the emerging motor excess ratio is 1.046 vs 1.154 applied (+10 % over-prediction), buildings 1.262 vs 1.085 (2023 fire year) | per-market excess / capped on mature claims | `loading_by_market` (opt-in) kept for **motor only**: CV +.0024, holdout +.0021, calib 1.09 -> 1.02; buildings rejected (holdout .1217 -> .1007), contents neutral |
| 3 | NB step (coordinator, public probe): (a) `market_nb` feature, (b) `weight_emerging_nb` x2 / x3, (c) post-hoc NB/renewal factors (`score_market --nb-cal`) | (a) motor +.0002 / -.0005, test NB/ren 1.32 -> 1.28; (b) test NB/ren unchanged (1.317 / 1.328), scores -.0001..-.0017; (c) 2023-fitted motor NB factor 1.64 worsens the 2024 holdout .1344 -> .1242: emerging actual NB/ren 2.24 (2023) vs 1.20 (2024) | all three rejected; history cannot confirm the public NB x1.2 reading (76 % 1-Jan bulk NB in test is untestable in history) |
| 4 | `features.model_matrix` `hist_nan_2021`: first observed year of EACH market NaN when `market` exists | `tests/test_features_market.py` | kept (weak): CV .0000/-.0012/.0000, holdout -.0006/+.0028/+.0021 |
| 5 | Noise list re-derived on the pooled frame (level-frequency stability, emerging 23~24, developed 21-22~23-24, developed~emerging) | previous_insurer -0.05/-0.06/-0.08, -0.10/-0.14/-0.13, -0.09/-0.05/+0.02: no persistent signal; with Review 10 #7 (dropping it +.0024 CV) | PART2POOL.noise_features = Part 1 list; Review 10 configs frozen to `()` so they reproduce |
| 6 | `run_blend`: `weight_market` (weights + nested weights fitted on one market's rows), `external_parts` (other market's results by row_id), `fixed_weights` per line; default path unchanged | `tests/test_blend_market.py` (incl. default-path equality) | accepted. Note: `_simplex` is un-memoised, 7 parts on a 0.1 grid did not finish in 25 min (killed) |
| 7 | Blends on emerging rows: fitted buildings weights unstable across folds (nested .1172 vs single .1231); emerging-only parts / P2_X3_blend get 0 home-line weight, 0.14-0.15 motor, nested -.0001..-.0004 | nested CV + holdout | emerging-only parts rejected; buildings fixed to the P34 bag |
| 8 | Seed bags (3 seeds): P34 CV +0/+.0023/+.0018 (holdout +.0011/-.0038/-.0014), L20 +.0004/-.0012/+.0018, F22 -.0001/+.0007/+.0002 | — | used in R2_FINAL |
| 9 | Final R2_FINAL: nested .0912/.1254/.1231, holdout .1346/.1207/.1192 vs base on the same target .0888/.1234/.1214, .1325/.1221/.1163; ties the single P34 bag on nested | — | candidate file written (no level factors); motor level -10 % vs the base file conflicts with the public motor x1.15 probe: level left to the user |
| 10 | Housekeeping: `submit.write` puts part2pool files in `submissions/part2/` with prefix `P2_POOL_` | code reading + runner output | fixed |

## 2026-10-27 · Review 12 — scoring model confirmed and the Part 2 level gap diagnosed (Fable, read-only checks)

Context: agent B2's round-2 candidate is calibrated to history (OOF A/E 1.02) but sits 10 % below the reference file on
motor, while the public probes want the reference raised 15 % (new business 19.5 %). Fable resolved the conflict with
data checks before any factor was applied (`scripts/part2_nb_diagnostics.py`, `docs/part2_response.json`).

| # | Claim / hypothesis | Check | Outcome |
|---|---|---|---|
| 1 | The public score is the row-level Tweedie D² (p = 1.5) on a random subset and the probe response ΔLB(s) = −α(s^−½−1) − β(s^½−1), β ∝ Σ√μ, is exact | buildings β from two probes 0.1471 vs 0.1485 from the file's Σ√μ; composite probe (NB ×1.25, ren ×0.93) predicted +0.00188, scored +0.00187 | **confirmed**; every later upload gets a predicted score first (`scripts/part2_factors.py`) |
| 2 | The motor shortfall is an age effect (test new business is younger) | emerging OOF A/E by age band × new business | **rejected**: new business under-predicted in every band (0.97–1.41), renewals over-predicted in every band (0.87–0.99) |
| 3 | The 2024 holdout is right to reject a new-business factor | A/E by year: NB/renewal calibration 1.82 (2023) vs 0.95 (2024), ≈ 350 NB claims each; public reading SD ≈ 0.05 on ≈ 3,600 NB claims | the holdout cannot resolve it; the public reading is the better estimate (implied 2025 NB/renewal 1.61 vs model 1.34, raw history 1.60) |
| 4 | 2024 new business is under-developed because it starts mid-year (mean loss month 8.3 vs 6.0) | pipeline uses row-level BF development; lag mean 16–26 days | small (≈ −6 % on NB rows at most), not the cause |
| 5 | The level gap is the insurer's premium trend | renewal premium per policy 604 / 601 / 608 (2023 / 24 / 25) | flat — rejected |
| 6 | The level gap is 2025 weather | hail days 14.0 / 17.1 / 11.4 | milder — rejected |
| 7 | The emerging lag curve (one year) under-develops claims | lags identical to developed (26 vs 25 days; 0.9 % vs 0.9–1.6 % after a year; open 3.7 % vs 3.9 %) | rejected |
| 8 | The pooled trend index mixes market price levels | full-year premium 603 (emerging) vs 594 (developed) after B2's rescale (Review 10 #3) | clean |
| 9 | Decision | credibility Z = τ²/(τ²+σ²) with τ ≈ 0.25 (history years disagree), σ ≈ 0.05 → Z ≈ 0.9; factors applied on the structural subset (new business) to the bagged model, shape untouched; everything predicted before upload | file 6 in `docs/part2_plan.md`; documented as a 2025 measurement, unlike Part 1 where ±5 % readings with SD 5 % were (rightly) distrusted |

Verified on B2's hand-back: candidate file rows/order/finite/min 0.08, corr .9965 with `R2_X4`, per-line levels and
NB/renewal ratios as reported; 159 tests green with IAA_MARKET unset (B2's run, not re-run here). Not verified: the
blend weights' fold stability (B2 reports buildings unstable and fixed it to the P34 bag).

## 2026-10-27 · Review 13 — written premium as a model feature (one CV-decided screen)

Hypothesis: the emerging insurer's own rating encodes risk the model cannot see. Agent-written, additive and default
off (`features.model_matrix` keys `premium_log`, `premium_rate`, `premium_rel`; the existing `premium` key keeps raw
`written_premium`, which was dropped by default). Check before modelling: `written_premium` is the ANNUAL premium (mean
flat across exposure bands in both markets and in test), so log(premium / exposure) is mostly an exposure feature; for
trees log(premium) and premium are the same split. Scores on EMERGING rows (`scripts/score_market.py`), motor /
buildings / contents. 160 tests green with IAA_MARKET unset (golden Part 1 test included; new `tests/test_features_premium.py`).

| # | Change | Check | Decision |
|---|---|---|---|
| 1 | Quick 80/20 screen on P34 (`R2_P34_hn` .1124/.0761/.1227): `_wpl` log premium alone, `_wp` raw + log rate, `_wprel` log(premium / market-year mean) | `_wpl` .1147/.0794/.1305, `_wp` .1154/.0792/.1314, `_wprel` .1167/.0711/.1303 | all pass the +.001 gate -> full CV |
| 2 | Full 5-fold CV + 2024 holdout vs `R2_P34_hn` (CV .0912/.1231/.1222, holdout .1338/.1245/.1196) | `_wpl` CV +.0010/+.0017/+.0003, holdout +.0017/-.0051/-.0021; `_wp` +.0007/-.0010/+.0010, +.0004/-.0044/-.0022; `_wprel` -.0010/+.0001/+.0027, +.0008/-.0066/-.0041 | `_wp`, `_wprel` rejected; `_wpl` seed-checked (single-seed gains are inside the P34 seed range) |
| 3 | 3-seed paired check `_wpl` vs `R2_P34_hn` seeds 0/101/202 | mean CV +.0013/-.0003/+.0017 (motor, contents 3/3 positive), holdout +.0014/+.0003/+.0005; bag vs bag CV +.0012/-.0004/+.0018, holdout +.0015/+.0004/+.0004 | kept for motor and contents; buildings neutral |
| 4 | F22 part with log premium (`R2_F22_wpl`) vs `R2_F22` | CV -.0006/-.0007/+.0002, holdout -.0012/-.0013/-.0020 | rejected (F22 keeps no premium) |
| 5 | Feature importance / level | log premium is the top-gain feature in P34 for motor and buildings, 2nd for contents (gain share 19 / 21 / 13 % of top 25); test level vs base x1.001/1.029/1.010 (bag x1.005/1.021/1.020); motor test NB/renewal 1.33 -> 1.31 | the premium does NOT carry the insurer's new-business loading into the model (NB ratio slightly down); OOF calibration unchanged (1.02/0.96/1.00 on the bag) |
| 6 | Candidate `R2_FINAL_wpl` = R2_FINAL + the premium P34 bag as a fifth part (weights on emerging rows; buildings fixed to the R2_P34_hn bag) | weights motor .70 premium bag / .30 F22 bag, contents .90 / .05 / .05; nested +.0009/.0000/+.0017 (overall .1727 vs .1720), holdout +.0014/.0000/-.0005; test level x1.002/1.000/1.018, corr .9988 with R2_FINAL; file 209,063 rows, row order = R2_FINAL, finite, min 0.08 | candidate file `submissions/part2/20261027_2108Z_P2_POOL_R2_FINAL_wpl.csv` (sha256_16 44c583d42fedde4b, no level factors); gain is small (+.0007 nested overall), on the order of seed noise; whether to swap it into the factor-shrunk final is the orchestrator's call |

## 2026-10-28 · Review 14 — probes on the final candidate close the level programme; data cleaning re-verified (Fable)

Question from the user: was the data cleaned before this analysis, and can the probes be nailed down so the model can be
finalised. Checks run read-only on `data/part2/emerging` and on the public readings.

| # | Check | Evidence | Decision |
|---|---|---|---|
| 1 | Cleaning applied before modelling | `data.build_all` log (part2_decisions §3): exposure-in-days 47/33/34 rows recomputed; sum_insured ×1000 unit errors 53/57 history + 104/135 test rows divided; construction codes BRK/WOD/MIX/STN mapped; occupation case/whitespace variants (47,924 customer rows) normalised; `smoke_alarms` None → NaN; leaky `renewal_decision`, `customer_service_contacts` excluded; repudiated claims excluded; 36 claims with loss dates before any policy period attach to no row | Yes, the Part 1 rules plus two Part 2 rules, each logged in the readiness register |
| 2 | Independent re-check of what the register does not list | policies and test, all lines: 0 duplicate (policy_id, period_start), 0 full-row duplicates, 0 duplicate row_id, 0 overlapping periods within a policy, 0 end < start, 0 exposure ≤ 0 or NaN, 0 sum_insured ≤ 0, 0 premium ≤ 0; claims: 0 duplicate ids, 0 duplicates under a new id, 0 negative incurred, 0 recoveries > incurred, 0 report < loss date, 0 claims attaching to more than one row, 0 claim-line / policy-line mismatches (9,258 claims) | Nothing left to clean; the 12 % missing rate in many columns is uniform (planted MCAR) and handled by NaN-aware models |
| 3 | Still open from readiness §7 (not cleaning) | noise/irrelevant feature re-screen for the emerging schema, gender/band leakage review | Low value with a two-year history (a year-to-year stability screen has one comparison); noted for the write-up as a limitation |
| 4 | Probe design: base = the intended final (file 6), not the file-2 base | β of any subset of any file follows from its Σ√μ with the 27 Oct constant, so one reading fixes α exactly for that base; `--fit` self-check reproduces the 27 Oct constants (motor:nb α 0.2853, s* 1.195; motor 0.3477 / 1.152; contents 0.0852 / 0.964) | Files 3–5 superseded; three probes built on file 6 with predicted readings written first |
| 5 | Readings (02:22 UTC) | motor NB ×1.10 → −0.00045 (predicted −0.00035): residual s* 1.011 ± 0.08; contents NB ×1.20 → −0.00046: s* 1.017 ± 0.11; buildings NB ×0.85 → −0.00157: s* 1.079 ± 0.15 | File 6 is at the public optimum on all three structural subsets; max remaining level gain +0.0002 (inside noise). History hypotheses contents NB 1.24 and buildings NB 0.80 rejected on 2025 rows |
| 6 | Written premium feature, paired reading (02:25 UTC) | `R2_FINAL_wpl` × same factors: 0.16407 vs 0.16382 (+0.00025; predicted 0 to +0.0006; rule ≥ +0.0004) | Stays out of the final; the gain is seed-noise-sized on CV and on the public rows |
| 7 | Sampling SD of a reading | relative row-level variance of the emerging history cost on the same line × NB subset over the public row count (¼ of rows): motor NB 0.08, contents NB 0.11, buildings NB 0.15 (home-line renewals ≈ 0.3) | Credibility per subset Z = τ²/(τ² + σ²), τ 0.25: 0.91 / 0.84 / 0.74; home-line renewals deliberately not probed (a reading could not decide a factor) |

Outcome: the level programme is closed with file 6 (`20261027_2009Z_P2_POOL_R2_FINAL_shrunk_nb130_ren109_b107_c099.csv`,
0.16382) as the final level; the gap to the leader (0.0090) is model shape. 6 upload slots remain for 28 Oct.

## 2026-10-28 · Review 15 — hidden-feature hunt (agent H1) and the stage-2 emerging relativity layer

Question from the user: is there a feature or structure we missed. Agent H1 (read-only on `src/`, emerging-only quick
screens, `scripts/part2_feature_hunt.py`, `docs/part2_feature_hunt.md`) scanned every raw column, the ids, test-vs-train
shifts, cross-table links, target construction and the R2 OOF residuals by segment (225 feature × line tests, each
against random-feature nulls; 14 exceed chi²/df 2.2 where chance gives ≈ 11, so only effects that hold in both years
and beat the null out-of-fold were kept). Fable verified the factor file (209,063 rows, no duplicates, level ratio
1.0000 in every line × new-business cell) and took two paired public readings.

| # | Finding | Evidence | Decision |
|---|---|---|---|
| 1 | The pooled model imposes developed-market relativities on a few emerging-only features. Robust: motor gender (M 1.068 / F 0.925 on 3,943 / 3,105 claims, same in 2023 and 2024, NB and renewals, frequency z ≈ 4.2; developed rows have no gender, so the model sees it on 4.6 % of rows) and motor fuel (petrol .924 / diesel 1.081 / hybrid 1.142, year-to-year corr .83, flat on developed rows). Fragile (severity-only): contents gender, works_from_home, buildings trees | Level-neutral stage-2 Tweedie GLM on the R2_FINAL OOF (5-fold, offset log base, cell-renormalised): core (gender + fuel, α 3) +4.5 bp honest OOF; full (α 10) +13.5 bp; fit-one-year / score-the-other positive on every line | **Paired public readings on file 6: core +0.00070 (0.16452), full +0.00030 (0.16412).** Core confirmed (same sign, larger than OOF) → promoted to intended final. Full not promoted: it reads below core, so the extra features do not confirm |
| 2 | 2025 weather is shifted (colder, drier, less hail), inside historical ranges; all test addresses covered; it moves test predictions ×0.997 / 1.022 / 0.985 | Drop-weather quick screen +16 ± 15 bp over 8 seeds (unresolved) | Not acted on; a pooled CV would be needed and the level is probed anyway |
| 3 | `customer_service_contacts` is a claim-driven leak in history (≥ 3 contacts a/e 4.55) with no signal in test (Poisson 0.40, flat) | in-sample lift +3,500 bp if re-added | Stays excluded (guardrail confirmed) |
| 4-8 | Test NB equals history NB on every rating attribute (PSI ≤ .01; only start month and exposure differ); NB × feature interactions are 2023-only; ids carry nothing (chi²/df .32-1.42); no cross-table link beyond existing features; unused raw columns (previous_insurer, campaign_code, branch_id, email_domain, make_model, written_premium, customer_since_year, year_built) chi²/df .09-1.58; `customer_tenure` duplicates `renewal_count` | | Nothing to add; noise list kept |
| 9 | Quick emerging-only screens (3 seeds, paired SD 11-41 bp): drop gender +4 ± 40, drop noise list −4 ± 23, rich history −20 ± 33, drop weather +16 ± 41, drop start_month −9 ± 30, seasonal offset −1 ± 11 | | All neutral or rejected; the quick screen cannot resolve 5-20 bp effects, which is why the stage-2 layer was measured on the OOF and then on public rows |

Governance note for the write-up: gender is a rating factor in this synthetic emerging market and already a model
feature; it would not be permitted in some jurisdictions (EU). The layer is documented as a measured emerging-market
relativity, applied level-neutrally, and can be removed by dropping one column of
`experiments/logs/part2_feature_hunt/R2_FINAL/stage2_factors.csv`.

Outcome: intended final = `20261028_0403Z_P2_F6_stage2_core_a3.csv` (sha256_16 b6612c813a25beff, public 0.16452, #8).
Slots: 6 of 10 used on 28 Oct; 2 of the user's 4-slot budget for this round remain for the shape agent's candidate.

## 2026-10-28 · Review 16 — model-shape screens on the pooled final (agent S1): clean negative

Four shape ideas run on the pooled model, each decided by nested CV on emerging rows (rule: at least +.0015 overall or
+.002 on one line, 2024 holdout not worse). Reference R2_FINAL nested .0912 / .1254 / .1231, overall .1720; holdout
.1346 / .1207 / .1192. One process at a time; no src/ change; 4 rows in `experiments/log.csv` ("S1 IDEA …").

| # | Idea | Result (emerging) | Decision |
|---|---|---|---|
| 1 | Pooled CatBoost part `R2_C9` (P34 target/features/market weights, A9 defaults; 25.6 min) as a 5th blend part `R2_FINAL_C9` | C9 alone .0910 / .1225 / .1176; in the blend motor weight .42 (P34 bag → 0), contents 0; nested +.0003 / .0000 / −.0003, overall .1720 = .1720. Buildings 50/50 mix +.0002 CV; its +.0068 holdout is level (calibration 1.31 vs 1.41), +.0013 once level-matched | Rejected |
| 2 | Per-market large-loss cap | Diagnostic only: emerging motor q99 19.1k vs pooled cap 34.7k (0.38 % of emerging motor claims above it); buildings 1.35× (16 fire claims); contents 0.98× | Needs `src/target.py`; handed to agent S2 with ownership (Review 17) |
| 3 | Blend-weight fold stability | Unstable (motor L20 weight .06–.80, contents B27 0–.36, buildings P34 .2–1.0, already fixed to the P34 bag); best fixed-weight alternative chosen after the fact +.0005 nested, holdout +.0003; equal weights +.0002 | Rejected; the fitted weights stay, instability noted for the write-up |
| 4 | Developed-market weight 0.5 / 2.0 (`R2_P34_hn_w05` / `_w20`) | 0.5: overall −.0004, contents holdout −.0073. 2.0: overall +.0003; buildings CV +.0021 but buildings holdout −.0075 | Rejected (holdout condition) |

Files written by the pipeline for these screens (`20261028_04xxZ_P2_POOL_R2_C9 / R2_FINAL_C9 / R2_P34_hn_w05 / R2_P34_hn_w20.csv`)
are rejected screens: never tick, never upload.

## 2026-10-28 · Review 17 — per-market large-loss cap (agent S2): clean negative

Gated change in `src/target.py` (`cap_by_market: True`, default off; per-market cap quantile, per-market loading for
every line, per-market IBNR relativity; `peril_matrix` refuses the flag rather than clipping silently). 161 tests pass
with IAA_MARKET unset, golden Part 1 tests included; new `test_cap_by_market`. One run, 5 min 25 s, emerging scores.

| | motor | buildings | contents | overall |
|---|---|---|---|---|
| R2_P34_hn CV → capm | .0912 → .0897 (−.0015) | .1231 → .1245 (+.0014) | .1222 → .1224 | .1710 → .1710 |
| 2024 holdout → capm | .1338 → .1351 | .1245 → .1116 (**−.0129**) | .1196 → .1206 | .1988 → .1956 |
| caps pooled / developed / emerging | 34.7k / 35.7k / 18.5k | 99.5k / 98.8k / 138.5k | 22.1k / 22.1k / 20.0k | |

Decision: rejected under the rule (overall −.0001; buildings +.0014 < +.002 and its holdout falls; motor CV down). The
emerging buildings loading is fitted on 2023, the only mature emerging year and the fire year, so a per-market cap
raises the buildings test level by 11 % on 16 claims — the opposite of credible. Pooled caps stay; the flag stays in the
code, off, as a documented screen. File `20261028_0431Z_P2_POOL_R2_P34_hn_capm.csv` is a screen, never tick.

**Round close (28 Oct 04:40 UTC).** Of the user's 4-slot test budget, 2 were used (stage-2 core / full readings), 2 are
unspent; 6 of 10 daily slots used. Intended final: `20261028_0403Z_P2_F6_stage2_core_a3.csv` (0.16452, #8).

## 2026-10-29 · Review 18 — external-data search and eight segment probes on the intended final (Fable)

**External data (rules §2.7 allow it).** Searched for anything that could inform the synthetic generator or the
emerging market: the precursor IAA Kaggle hackathon (`ai-assisted-hackathon-1`, Dec 2025, 30 public notebooks
including two by teams now above us) used the Ames housing data, so nothing transfers; three public GitHub repos of
Part 1 teams (almac09, gitaukennedy, ISHUKLA) are low-ranked and report nothing beyond what our EDA found (one of them
committed the competition data to GitHub, which we did not download); Kaggle discussion pages are script-rendered and
return nothing to the fetch tool; real-world weather, price or inflation data cannot be matched to fictional CU data.
Local cross-checks done at the same time: the emerging claims table has no claims for policies that exist only in
the 2025 test files (0 of 149,721 test-only policies), so the broker book brings no claims history; the pooled-OOF
residual scan over 1,416 feature levels has no |z| above 2.8 (consistent with noise at ~1,000 home-line claims).

**Probes.** Eight paired readings on `20261028_0403Z_P2_F6_stage2_core_a3.csv` (0.16452), one segment tilt each,
chosen on test weight and prior plausibility (age curve, vehicle value, NCB, sum-insured tilt that transferred in
Part 1, the 2025 cold shift, flood zones). Files `20261029_0043Z_PROBE_F9_*.csv`, uploaded 00:44 UTC, never tick.
Inversion α = (Δ + β(√s − 1)) / (1 − s^−½), s* = α/β, with β from the file's Σ√μ (constant fitted 27 Oct); results
in `docs/part2_response.json` under `fits → core_a3 → segments` and in the plan's 29 Oct entry. Verified: every probe
file has 209,063 rows, changes only the segment rows, leaves the other lines untouched (level ratios printed by
`scripts/part2_factors.py --row-factors`).

**Finding.** Young drivers (17–24, 18 % of Σ√μ) are under-predicted by about a third on the 2025 public rows
(s* 1.34, reading +0.00138, ≈ 3 SD), while 65+ and high-value cars are over-predicted (s* 0.91 / 0.78) and the motor
level is already at its optimum: the age curve rotates. Emerging history does not show it (pooled-OOF a/e for 17–24
is 0.99), so it is a property of the 2025 broker portfolio, visible only on the leaderboard. Buildings: sum-insured
top quartile s* 1.17 and flood zone ≥ 3 s* 1.24 (1–2 SD); freezing days, NCB 0 and contents sum insured are flat.

**Built, awaiting the user's upload approval (the 8-upload budget is spent).** Composite
`20261029_0048Z_P2_F10_core_a3_x_seg4_shrunk.csv` (Z-shrunk tilts on the four unconfounded readings, cells
renormalised, predicted +0.0036 → 0.1681 from the additive response) and probe
`20261029_0048Z_PROBE_F9_motor_age2534_x1.10.csv`. Intended final remains `core_a3` (0.16452) until the composite
is read. Write-up implication: a disclosed, level-neutral, credibility-shrunk leaderboard calibration of a portfolio
shift; Part 1 evidence (tilts transferred, flat levels did not) is the basis for applying it.

**Addendum 01:30 UTC — composite read.** User approved one upload. `20261029_0048Z_P2_F10_core_a3_x_seg4_shrunk.csv`
scored **0.16797** (#6 of 29; leader now 0.17556) against a prediction of 0.16809: the additive response (per-segment
α, remainder α from the cell optimum) is accurate to 0.0001, so further composites can be predicted before upload.
New intended final: this file. The 25–34 probe was not uploaded (1 slot kept). Tomorrow's budget set by the user: 6.

## 2026-10-29 · Review 19 — split-half check overturns the young-driver reading; the motor public signal is a few extreme claims (Fable)

**Probe.** `20261029_0134Z_PROBE_F10_motor_age1724_even_x1.10.csv` (F10 base 0.16797; drivers 17–24 with an even last
digit in the row id ×1.10): **0.16714, −0.00083**. A uniform young-driver effect would have read about +0.00026.
Inversion (α additive over disjoint rows, converted to core_a3 units with the F10/core factor 1.164 NB / 1.225 ren):
even half s* 0.84, odd half s* 1.85 (full reading 1.34). The halves differ by 0.79 in log.

**Noise check from history.** Bootstrapping public-quarter-sized samples of the 9,419 emerging young-driver history
rows (2,114 claims, row-cost CV 4.9, largest row 149k) gives a probe-style s* log SD of 0.10 for the full segment and
0.14 per half; in 4,000 draws two halves never differed by 0.79 (median 0.12). So the split is not compound-Poisson
sampling noise of history-like claims. The odd-half excess α (0.051 in core units) equals one claim of about 2.5m CU
on a young-driver row (or five of 0.5m). The developed history shows the generator makes such claims (motor bodily
injury 11.9m, 1.1m, 0.6m, 0.5m in 113k claims); the emerging history's largest claim is 0.36m.

**Consequence for every motor reading.** The same excess is 38 % of the young α and 19 % of the file-6 motor:NB α.
Removing it: reference-file motor s* 1.15 → 0.98, motor:NB 1.195 → 0.98, file-6 motor:NB 1.011 → 0.815. In other
words the 27 Oct motor level and new-business uplift (file 6: NB ×1.30, renewals ×1.09) is explained by the same few
public claims, which the private rows do not contain. Part 1's post-mortem (flat levels did not transfer, tilts did)
is consistent with the same mechanism. Local evidence for an NB uplift (history NB/renewal ratio 1.6 vs model 1.32)
was itself contradicted by the 2024 holdout, so it cannot carry a ×1.30 factor on its own.

**Decision (pending the user).** Do not tick F10 as it stands. Tomorrow's first slot: split the odd half once more
(digits 1/3/5 vs 7/9); a concentrated excess confirms extreme claims. A robust candidate is pre-built with the
ex-excess optima (motor NB ≈ ×1.08, renewals ≈ ×1.03, young ×1.08, 65+ ×0.945, home-line tilts kept, stage-2 core
kept); it will read lower on the public board and that is expected. Files and predictions in the plan entry.

## 2026-10-30 · Review 20 — split-half programme closes the leaderboard calibration; robust F11 v3 is the tick (Fable)

**Three probes, read in order, each inverted before the next was chosen** (all on the exact Tweedie response, α additive
over disjoint rows; readings and inversions in `docs/part2_response.json` → `fits.splithalf_30oct`).

1. Drivers 17–24, odd-id digits 1/3/5 ×1.10 on F10: 0.16947 (+0.00150; predicted +0.00168 if the odd-half excess were
   entirely in 1/3/5, +0.00080 if spread over the odd half, −0.00050 if in 7/9). About 90 % of the excess sits in 30 % of
   the young rows: residual optima 2.04 (1/3/5), 0.72 (even), 0.90 (7/9). The ex-excess young optimum on core_a3 is 0.90,
   i.e. the model's age curve was right; the public "young-driver effect" is a handful of extreme claims. The excess α
   (0.0445 in core units) is the equivalent of one claim of ≈ 2.5m CU. 84 % of these rows are new business, so the
   excess sits in the motor NB cell (the alternative needs a renewal optimum of 0.28); a further NB/renewal split would
   not have discriminated and was skipped.
2. Home tilts, even half (SI q4 ×1.10, flood ≥ 3 ×1.12) on F10: 0.16784 (−0.00013; predicted −0.00009 if already fully
   priced, +0.00022 if diffuse). The even half shows no residual at all (1.04 on core_a3 for both); the odd half carries
   1.29/1.34 by subtraction. Same pattern as motor: concentrated, not a portfolio shift. History a/e gives a mild prior
   for flood zone 3 only (1.52 on 92 claims; zones 4–5 below 1; SI top-quartile bins mixed).
3. Motor NB excluding 17–24, even half ×1.10 on core_a3: 0.16370 (−0.00082; predicted −0.00019 if the NB level were
   right, −0.00097 at the inferred ex-excess optimum 0.81). Even 0.84, odd 0.77 (subtraction), whole 0.81: the halves
   agree, so this is a diffuse, real over-pricing: the NB ×1.30 of 27 Oct (and core_a3) is ≈ 20 % too high once the
   extreme claims are set aside. The model prior for NB is 1.0 (new_business is in the model; history a/e 1.00).

**F11 v3** = core_a3 × (motor NB 1.05, renewals 1.03, no 17–24 or 65+ tilt, buildings SI q4 1.03, flood ≥ 3 1.05,
buildings cells renormalised, stage-2 gender/fuel layer kept). Predicted public 0.16130, scored **0.16139**; the response
model has now predicted four composites to ±0.0001. Expected change vs core_a3 with the public extreme claims removed:
+0.0016 (NB non-young +0.0018, young NB ≈ 0, home −0.0001). Motor level ×0.829 vs core_a3 (the 27 Oct "2025 motor
level 1.27×" diagnosis was the same few claims seen through the public quarter; the premium-trend and weather checks
that found nothing were right).

**Why tick F11 v3 and not F10 (0.16797) or core_a3 (0.16452).** The public ordering F10 > core_a3 > F11 v3 is produced
entirely by α 0.0445 of extreme-claim excess on ≈ 1,700 public rows. The private rows (3× the public) either do not
contain those claims, in which case F11 v3 is ≈ +0.0016 above core_a3 and ≈ +0.005 above F10, or they contain the
generator's heavy tail at its expected rate, in which case the expected uplift on motor NB is only 2–4 % (≈ 0.00005
either way). There is no scenario with a material private gain for F10's factors (young ×1.29, NB ×1.30): the private
rows are disjoint from the public rows that carry the excess. Part 1's post-mortem (public-probed flat levels did not
transfer; tilts and CV-backed structure did) is the same mechanism and supports the choice. Risk accepted: the public
rank will drop from #5-6 to ≈ #10 on the visible board.

**Operational note.** The account's best public score is now a PROBE file (0.16947). Kaggle's default without a tick is
the best public file, so the final must be ticked explicitly on kaggle.com before 31 Oct 03:59 UTC: exactly one tick,
on `20261030_0145Z_P2_F11v3_robust_nb105_ren103_si103_fl105.csv` (sha 378f7b835d3ace59). 2 of the user's 6 slots unspent;
remaining untested levers are below 0.0001 of maximum gain.
