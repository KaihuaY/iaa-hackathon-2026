# Part 1 assumption inventory (read-only audit, 17 Oct 2026)

Line numbers are the `cat -n` numbers. I read every file listed in the brief except the tests, which I grepped and partly read. I did not read `explog.py`, `metrics.py` or `decisions*.py`.

## 1. Calendar

| file:line | assumption | why it breaks on a new market | parameterisation |
|---|---|---|---|
| data.py:15 | `VALUATION_DATE = 2024-12-31` | Used by `_daily` (target.py:54-55). A different cut-off gives a wrong completeness and IBNR calendar. | `spec.valuation_date` |
| target.py:15 | `MATURE_YEARS = (2021, 2022)` for the completeness and late-severity curves | A different panel length or lag tail makes these years too young or too few. | `spec.curve_years` |
| target.py:17 | `DAY0 = 2021-01-01` | The daily calendar starts at the first training day. | `spec.first_year` |
| target.py:16 | `MAX_LAG = 730` | A longer-tailed market is truncated. | `spec.max_lag_days` |
| target.py:46-47 | Severity ratio needs at least 30 claims in the tail | Thin, small markets give NaN. | `spec.min_tail_claims` |
| target.py:74, 85-89 | Trend years `[2021..2024]` are hard-coded in `trend_index` | The year list and trend-fit window are Part 1 specific. | `spec.train_years` |
| target.py:68, 80-81 | `target_year=2025` default; `idx[2025]` is read from the test set | The test year may differ, or the test set may span several years. | `spec.test_year` |
| target.py:84 | Claims-trend fit uses only `(2021, 2022, 2023)`, "avoid immature 2024" | Which years are mature is a market property. | `spec.mature_years` / `spec.immature_years` |
| target.py:97-99 | `df.year <= 2023` and `loss_date.year <= 2023` for the BF a-priori frequency; severity uses `== 2023` | Same "2024 is immature" logic. | `spec.mature_years`, `spec.ref_sev_year` |
| target.py:141 | Same `<= 2023` for the large-loss loading | Same. | `spec.mature_years` |
| target.py:168 | Same `<= 2023` for the cap loading | Same. | `spec.mature_years` |
| credibility.py:9 | `COMPLETENESS_2024 = {motor: .920, buildings: .939, contents: .960}` | These are Part 1 numbers. They should be computed from the completeness curve, not typed in. | `spec.completeness_last_year[line]` or derive from `target.completeness_curve` |
| credibility.py:31 | `year == 2024` marks the immature year | Same. | `spec.immature_years` |
| credibility.py:54 | Test years must be later than train years (a guard) | An emerging market may have an overlapping test window. | keep, document |
| features.py:10 | `YEARS = [2021..2025]` | Used by `_prior` and `_prior_cols` to reindex columns. Other years are silently dropped, so history features become zero. | `spec.all_years` |
| features.py:17 | `LOOKBACK = 2` | A short panel, or one that is mostly 1-year, changes the right window. | `spec.history_lookback` |
| features.py:127-130 | `hist_nan_2021`: the first year has no history | Hard-coded year. | `spec.first_year` |
| features.py:106-107 | `start_month` and `row_type` | Assumes annual policies. | none needed; check |
| data.py:112-121 | `row_type`: "2024 periods end 30-Dec", `-12-30` cutoff | The calendar-year policy structure is Part 1 specific. A fiscal-year market would label everything "ends early". | `spec.year_end_day` or drop |
| data.py:51-52 | Exposure is `days/365` | Assumes a 365-day year. | `spec.days_per_year` |
| pipeline.py:92 | Holdout is `yr <= 2022` vs `yr == 2023` | Hard-coded years. | `spec.holdout_year` |
| pipeline.py:85, 93 | Time holdout uses the last mature year, not the actual last year | Same. | `spec.holdout_year` |
| backtest.py:16 | `years=(2023, 2024)` | Hard-coded years. | `spec.backtest_years` |
| market.py:497-498 | `yrs != [2021, 2022, 2023, 2024]` | The checklist flags years against the hard-coded list. | `spec.train_years` |
| test_pipeline.py:66-81, 93, 117, 145 | Tests hard-code 2021..2025 | Fail on other panels. | derive from spec |
| test_credibility.py:36-50, 64 | Synthetic years 2021..2025 | Self-contained. Low risk. | none |

## 2. Lines and perils

| file:line | assumption | why it breaks | parameterisation |
|---|---|---|---|
| data.py:12 | `LINES = (motor, buildings, contents)` | Used by about 15 modules (features, pipeline, market, tests, reproduce). A new market may have a different line set. | `spec.lines` |
| data.py:21 | `SI_UNIT_ERROR = {buildings: 5e6, contents: 1e6}` | The currency scale differs, so the thresholds are meaningless. | `spec.si_unit_error` (or a quantile-based test) |
| data.py:23-24 | `CONSTRUCTION_MAP` | Different local material labels would map to NaN. | `spec.construction_map` |
| data.py:18 | `LEAKY_COLS = [renewal_decision, customer_service_contacts]` | Leakage columns may differ. | `spec.leaky_cols` |
| target.py:14 | `SUSPECT_CLAIMS = ["CL30973963"]` | A Part 1 claim id. Harmless but useless. A new outlier needs a rule. | `spec.suspect_claims`, or auto-detect (e.g. > 10x the next claim) |
| target.py:107, 112-113 | Peril pooling: perils with < 1000 claims go to "other" | A smaller market pools everything into "other". | `spec.min_peril_claims`, or a share |
| target.py:124-134 | Segments: motor = `cover_type` x `driver_age` bands (<25, 25-34, 35-59, 60+); other lines = sum-insured quartiles | Needs `driver_age` and `cover_type` (motor) and `sum_insured` (the others). | `spec.segments[line]` |
| target.py:137 | `K=200` credibility constant | Scale-dependent. | `spec.seg_load_k` |
| target.py:22, 165 | Cap quantile (cfg `cap_q`, 0.99 or 0.995) | Heavy-tail markets need a different cap. | `target.cap_q` (already a config key) |
| features.py:11-12 | `WEATHER` list | New or missing climate columns. `weather_pca` selects with `c in X`, so it degrades quietly. | `spec.weather_cols` |
| features.py:13-14 | `DROP` list (`compulsory_excess`, `customer_since_year`, `year_built`, `exposure`, ids) | Column names. | `spec.drop_cols` |
| features.py:47 | rich_history cap fixed at 0.99 per line | Fixed. | `spec.rich_hist_cap_q` |
| configs.py:8 | `NOISE` list | Found by a Part 1 stability screen. May differ. | `spec.noise_cols`, or auto-screen |
| configs.py:11-16 | `IRRELEVANT` per line (property fields for motor, etc.) | Peril relevance is market specific (e.g. flood or cyclone matters more). | `spec.irrelevant[line]` |
| configs.py:21-23 | `MONOTONE` dict | Directions are actuarial priors. `sum_insured`, `flood_zone` and others may not hold, and missing features are skipped. | `spec.monotone` |
| configs.py:51, 81, 96 | `make_model` dropped as "300 levels" | Cardinality is market specific. | `spec.drop_high_card` |
| credibility.py:114 | `("make_model", "previous_insurer")` levels | Column names. | `spec.level_cols` |
| credibility.py:123-147 | Hard-coded segment columns: `pol_claims`, `cust_claims`, `flood_zone==5`, `listed_building`, `energy_rating`, `cover_type`, `sum_insured` deciles, `bedrooms 1 / >=5`, `prior_claims_5y` | Part 1 residual findings. They are quiet no-ops if a column is absent (the `has()` guard), so they would hide a loss of signal. | `spec.cred_segments` |
| credibility.py:12 | `MIN_LEVEL_ROWS=500` | Size dependent. | `spec` |
| credibility.py:183-196 | `_segment_defs`: `MM193/251/186/124`, `third_party_only`, `flood_zone==5`, `bedrooms==1` | Part 1 diagnostics only. Will KeyError if `make_model` is absent, since there is no `has()` guard (L184-185). | `spec.diagnostic_segments` |
| credibility.py:9-10, 20, 31, 59, 94 | `K_GRID`, `ALPHA_GRID`, A_c cap_q=0.995, clip (0.25, 4.0) | Tuned on Part 1. | `spec.cred` |
| models.py:6-10 | `min_data_in_leaf=20000`, `num_leaves=15` | Tuned for roughly 600k rows and 86-96 % zeros. A small dataset will not split at all. | scale by n rows: `spec.min_leaf_frac` |
| models.py:118 | Severity `min_data_in_leaf=2000` | Same. | same |
| configs.py:9 | `TUNED min_data_in_leaf=8000` | Same. | same |
| pipeline.py:132-136 | Tweedie p=1.5 for the metric | The host metric may change. | `spec.tweedie_p` |

## 3. Data-quality fixes

| file:line | assumption | why it breaks | parameterisation |
|---|---|---|---|
| data.py:46-48 | `customer_id.zfill(8)`, and it would crash if the id is not a string | Different id format. A wrong zfill creates false orphan joins. | `spec.customer_id_width` (None to skip) |
| data.py:50-55 | Exposure-in-days fix: if `abs(exposure - days/365) > 0.02`, overwrite | May be right or wrong in the new market. It silently overwrites a genuine part-year exposure. | `spec.fix_exposure` and tolerance |
| data.py:56-59 | `sum_insured > threshold` then `/1000` | Wrong if the new currency scale puts valid values above the threshold. This would divide real data by 1000. | `spec.si_unit_error`, off by default |
| data.py:61-64 | `construction_type` lower, strip and map; unmapped labels become NaN | New labels turn into NaN without warning. | `spec.construction_map` plus a fallback keep-unknown |
| data.py:66-67 | `smoke_alarms` boolean map | Label specific. | `spec.bool_maps` |
| data.py:69-73 | `years_licensed > driver_age - 17`; the minimum licence age 17 is a developed-market value | The minimum licence age differs by market (and the age may be missing for two-wheelers). | `spec.min_licence_age` |
| data.py:78-84 | Claim de-dup by all columns except `claim_id` | Presumes the Part 1 duplicate-re-key pattern. | `spec.dedup` |
| data.py:87-91 | Drop `status == "repudiated"`; net incurred = incurred - recoveries; open status | New status labels. | `spec.excluded_statuses` |
| data.py:129 | Customer age is `days/365.25` | Fixed. | none |
| data.py:127 | `customers.drop(columns=["surname"])` | KeyError if the column is absent. | `spec.pii_cols` |
| data.py:133 | `property_age = year - year_built` | Column specific. | `spec` |
| target.py:164 | `cfg.open_factor`; open claims | Status specific. | existing cfg key |
| market.py:121-129 | `_clean_policy` swallows exceptions, so cleaning is silently skipped | This hides bugs. | log loudly |
| market.py:211 | The generic check `sum_insured > 1e7` | Scale specific. | `spec` |

## 4. Schema

| file:line | assumption | why it breaks | parameterisation |
|---|---|---|---|
| data.py:11, 29 | `DATA_DIR = data/raw/public`; parquet files named `{name}.parquet` | Part 2 may use CSV, different names or a different folder. | `spec.data_dir`, `spec.file_fmt` |
| data.py:13-14 | `ID_COLS`, `DATE_COLS` | New id or date columns are not parsed. | `spec.id_cols`, `spec.date_cols` |
| data.py:101-103 | `attach_claims` joins on `policy_id`, `loss_date` in `[period_start, period_end]`, and `claim_id` | Any change to the claim to policy link breaks the target. | `spec.keys` |
| data.py:127-135 | Join keys `customer_id`, `address_id`, `(address_id, year)`; "same-year weather" | A market without addresses or weather breaks `enrich`. | `spec.lookup_tables` (optional) |
| data.py:140-150 | Table names `claims`, `customers`, `addresses`, `weather`, `policies_{line}`, `test_{line}` | Names may change. | `spec.tables` |
| submit.py:28 | `len(out) == 626_639` | A new test size trips the assertion. | `spec.expected_rows` (None or from the sample file) |
| submit.py:21-24 | `sample_submission` with `row_id` and `prediction` | Format may change. | `spec` |
| submit.py:16 | `COMP = "iaa-ai-assisted-hackathon-2026-part-1"` | Wrong competition. | `spec.competition` |
| submit.py:15, 34 | `SUB_DIR` and the `float_format` | Shared output folder. | `spec` |
| reproduce.py:13-20 | `RAW_FILES`, `CACHE_FILES` list the 3 lines and `COMP` is repeated; the `kaggle download` path | Part 1 specific. | `spec` |
| reproduce.py:99 | Default `--final X10_blend4` | Part 1 experiment. | CLI arg |
| market.py:21-56 | `EXPECTED_SCHEMA` and `_LINE_COLS` hard-coded (this is deliberate, to detect Part 1 vs Part 2) | Becomes the reference rather than an assumption. Keep, but add `spec.ref_schema`. | `spec.ref_schema` |
| market.py:58-69 | `_BOOL` / `_DATE` / `_STR` dtype sets | Column specific. | `spec` |
| market.py:185-189 | `_NONNEG` list | Column specific. | `spec` |
| market.py:17, 183 | `LINES = D.LINES` | Same as `LINES`. | `spec.lines` |
| tests/test_pipeline.py:35-140 | Tests read the real `data/` (`build_base`, `load("sample_submission")`, `626_639`, `customer_id.len == 8`, `train_motor`, `pol_claims`, `pol_expo`, `hx_*`) | Tests gate submission through `run_tests`. They will fail on Part 2 data and block `submit`. | The Part 1 data tests should be marked, or read from `spec`. |
| tests/test_pipeline.py:164-165 | `lbmodel.fit()` on `experiments/logs`; corr > 0.95, n >= 5 | The LB model has Part 1 submissions only. | per-market log |
| tests/test_market.py:15-170 | Synthetic data, line names, columns | Self-contained. Low risk. | none |
| tests/test_level_probe.py, test_credibility.py | Mostly self-contained | Low risk. | none |

## 5. Anything else

| file:line | assumption | why it breaks | parameterisation |
|---|---|---|---|
| pipeline.py:59, 84 | `GroupKFold(5)` grouped by `policy_id` | A new market may have no repeat policies. Folds are policy-disjoint but time-leaky. | `spec.folds`, `spec.group_col` |
| pipeline.py:59 | `GroupShuffleSplit(0.2, seed=11)` | Fixed seed. | `spec.seed` |
| credibility.py:17 | `folds_for` repeats `GroupKFold(5)` | Must match pipeline.py. | one shared function |
| models.py:10, configs.py:122, 140 | `seed=7`; bag seeds `101, 202` | Fixed seeds. | `spec.seeds` |
| features.py:9 | `CACHE` path is shared across markets. A Part 1 cache loads unless `refresh=True`, so **stale Part 1 features may be reused** (features.py:89-92). | High risk. | `spec.cache_dir` keyed by market |
| pipeline.py:16, explog, lbmodel | `RESULTS` dir shared; the LB model fits Part 1 points | Shared state. | `spec.results_dir` |
| scripts/level_probe.py:21-23, 98, 125 | `PROBE_CSV`, CLIP (0.85, 1.15), `--lines` default `motor buildings` | Part 1 lines. | `spec.probe_lines` |
| scripts/run_experiment.py:13 | `EXPERIMENTS` imported from `configs` | Part 1 configs. | per-market configs |
| lbmodel.py:14, 33-40 | `AUTO_REL=0.05` and predicted LB from Part 1 | Calibrated on Part 1. | per-market log |
| submit.py:39-47 | `_code_state` uses git. The repo may not be git. | Not a git repo here. | fall back to hash |
| scripts/reproduce.py:35 | `catboost` detection by model name | Name specific. | none |
| eda.py:5-6, 61-66 | `SKIP` column list; `lag_cdf(years=(2021, 2022))` | Part 1 years and columns. | `spec.curve_years` |
| configs.py:3-6 | Target presets (`weight_power=1.0`, `trend="premium"`) | The premium-index trend assumes `written_premium` tracks claims inflation. Emerging markets may have high inflation or currency swings. | `target.trend` already a config key |
| target.py:80-82 | Trend from mean `written_premium` among rows with `exposure > 0.99` | The 0.99 filter assumes most policies are full year. | `spec.full_year_expo` |

## Proposed design

```python
@dataclass
class MarketSpec:
    name: str = "part1"
    data_dir: str = "data/raw/public"
    cache_dir: str = "experiments/cache/part1"
    results_dir: str = "experiments/results/part1"
    competition: str = "iaa-ai-assisted-hackathon-2026-part-1"
    lines: tuple = ("motor", "buildings", "contents")
    tables: dict = {}                  # lookup tables (customers, addresses, weather); optional
    id_cols: tuple = ("row_id", "policy_id", "customer_id", "address_id", "claim_id")
    date_cols: tuple = ("period_start", "period_end", "loss_date", "report_date", "date_of_birth")
    first_year: int = 2021
    train_years: tuple = (2021, 2022, 2023, 2024)
    test_year: int = 2025
    valuation_date: str = "2024-12-31"
    mature_years: tuple = (2021, 2022, 2023)       # replaces every "<= 2023"
    curve_years: tuple = (2021, 2022)
    ref_sev_year: int = 2023
    holdout_year: int = 2023
    backtest_years: tuple = (2023, 2024)
    completeness_last_year: dict | None = {"motor": .920, "buildings": .939, "contents": .960}  # None = derive
    max_lag_days: int = 730
    history_lookback: int = 2
    days_per_year: int = 365
    expected_rows: int | None = 626_639            # None = take it from sample_submission
    customer_id_width: int | None = 8
    si_unit_error: dict = {"buildings": 5e6, "contents": 1e6}
    construction_map: dict = {...}
    min_licence_age: int = 17
    leaky_cols: tuple = ("renewal_decision", "customer_service_contacts")
    suspect_claims: tuple = ("CL30973963",)
    excluded_statuses: tuple = ("repudiated",)
    cap_q: float = 0.99
    weather_cols: tuple = (...)
    noise_cols: tuple = (...)
    irrelevant: dict = {...}
    monotone: dict = {...}
    segments: dict = {...}                         # target.py segment definitions
    cred_segments: dict = {...}
    diagnostic_segments: dict = {...}
    min_peril_claims: int = 1000
    min_leaf_frac: float = 0.03
    folds: int = 5
    group_col: str = "policy_id"
    seed: int = 7
    tweedie_p: float = 1.5
```

Modules would read a module-level `SPEC` (set from an env var or CLI flag), so the Part 1 defaults reproduce today's behaviour.

### 10 highest-risk items to parameterise first

1. **features.py:9, 89-92 cache path.** The shared `experiments/cache` can load Part 1 features for Part 2 without any warning.
2. **features.py:10 `YEARS`.** Years outside the list are silently dropped, so history features become zero.
3. **data.py:56-59 `SI_UNIT_ERROR`.** A different currency scale means valid values get divided by 1000, which corrupts a rating factor.
4. **credibility.py:9, 31 completeness constants.** Typed-in Part 1 numbers. Derive them from the curve.
5. **target.py:97, 141, 168, 84 `<= 2023` and `(2021, 2022, 2023)`.** Silent mis-development if the panel differs.
6. **data.py:15 and target.py:15-17 valuation date, `DAY0`, `MATURE_YEARS`.** These drive the entire IBNR calendar.
7. **data.py:12 `LINES`.** Used about 15 times across the code base. Also needs per-line column and segment maps (target.py:124-134, credibility.py:180-196).
8. **submit.py:28 `626_639` and test_pipeline.py:135.** A hard assert will block every submission. The pytest gate (`run_tests`) will also fail on Part 1 data tests.
9. **data.py:46-55 `zfill(8)` and the exposure overwrite.** These can corrupt valid data if the new market has other id or exposure conventions.
10. **target.py:68, 80-81 and models.py:6-10 test year, premium trend and leaf sizes.** Level calibration and regularisation are tied to Part 1 size and inflation. The credibility segment code at credibility.py:123-147 (and `_segment_defs`, L183-185, which can KeyError) also needs a `has()` guard.

Limitation: I only grepped `tests/test_market.py`, `test_models.py`, `test_level_probe.py` and `test_credibility.py` for hard-coded values and did not read them in full.
