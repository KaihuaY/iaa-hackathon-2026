# Video outline (5 minutes maximum)

Only the first 5 minutes are graded. Total below: 4:33. Screens: EDA dashboard (`notebooks/01_eda.py`), results dashboard (`notebooks/02_results.py`), repo files. Arc: data fixes, target engineering, models and blend, level probe, non-linear hunt, final pick.

| # | Time | On screen | Say (one sentence) |
|---|---|---|---|
| 1 | 0:00-0:25 | EDA title and section 1 (data map) | We price 626,639 policy rows for 2025 across motor, buildings and contents, scored by Tweedie D², so the target is expected ultimate cost. |
| 2 | 0:25-0:55 | EDA section 2 (data-quality register, leaking columns), `tests/test_pipeline.py` | Planted errors (duplicate claims, exposure in days, sum insured x1000) are fixed in `src/data.py` with row counts logged, and two outcome-revealing columns are excluded, with a test. |
| 3 | 0:55-1:30 | EDA sections 4 and 5 (development, large losses) | We develop to ultimate with a BF IBNR, cap at the 99th percentile with a segment loading, and trend with the premium index that beat the claims trend in a backtest (level error 1.8 % against 3.0 %). |
| 4 | 1:30-2:00 | Results section 3 (what each step was worth) | A GLM at CV 0.140 became a per-line, four-way blend at 0.159, with feature stability screening alone adding 0.004. |
| 5 | 2:00-2:25 | Results sections 2 and 5b (CV vs LB, level probe) | CV tracks the public leaderboard (correlation 0.995, noise 0.002), and a one-parameter-per-line probe corrected the 2025 price level. |
| 6 | 2:25-3:00 | Results section 9 (non-linear hunt) | Following the ln(ab) hint, a smooth log-link GLM lifted quick CV from 0.1608 to 0.1620, but our guardrails, not CV, caught two real defects: an extrapolation blow-up (2023 holdout -0.66, test mean 24,016 against about 400) and calendar-proxy exposure counters (holdout -0.59, fixed to +0.108); the accepted base reaches CV 0.1574. |
| 7 | 3:00-3:30 | `docs/ai_review_log.md`, `docs/experiment_log.md` | AI proposed most experiments and code, but every change passed pytest, CV and a human decision, and AI reviews caught bugs such as a credibility ratio biased 11 to 24 % low. |
| 8 | 3:30-4:08 | Results section 1, then probe table (split-half, tail tests) | We stopped chasing the public board (best 0.12838): probes showed the high sum-insured signal was a few claims, so we shrank the factors and picked the 0.12820 file built for the hidden 80 % of rows. |
| 9 | 4:08-4:33 | Repo tree, Part 2 checklist | For the new market we re-run the register, backtests, level probe, stability screen and guardrails, and [TBD: what changed in Part 2]. |

Notes
- About 400 words; beats 6 and 8 need a steady pace. Fill beat 9 after Part 2; quote only figures in `docs/experiment_log.md`.
- Ten submissions used on 20 Oct; the final tick is made on 22 Oct.
