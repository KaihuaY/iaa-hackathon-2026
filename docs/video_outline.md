# Video outline (5 minutes maximum)

Only the first 5 minutes are graded. Total below: 4:50 so there is margin. Screens refer to the EDA dashboard (`notebooks/01_eda.py`), the results dashboard (`notebooks/02_results.py`) and repo files.

| # | Time | On screen | Say (one sentence) |
|---|---|---|---|
| 1 | 0:00-0:30 | EDA dashboard, title and section 1 (data map) | We price 626,639 policy rows for 2025 across motor, buildings and contents, scored by Tweedie D², so the target is expected ultimate cost, not what has been reported so far. |
| 2 | 0:30-1:10 | EDA section 2, data-quality register | The data carried planted errors (duplicate claims, lost id zeros, exposure in days, sum insured x1000, messy labels, impossible licence years), and each is fixed in `src/data.py` with its row count logged. |
| 3 | 1:10-1:40 | EDA section 2a (leaking columns chart), then `tests/test_pipeline.py` | Two columns, `renewal_decision` and `customer_service_contacts`, reveal the outcome, so they are excluded and a test proves it. |
| 4 | 1:40-2:15 | EDA sections 4 and 5 (claims development, large losses) | 2024 is immature and one 11.9m open claim is ten times the next largest, so we develop to ultimate with a BF IBNR, cap at the 99th percentile with a segment loading, and trend to 2025 using the premium index that beat the claims trend in a backtest (level error 1.8 % against 3.0 %). |
| 5 | 2:15-2:50 | Results dashboard, section 3 (what each step was worth) | Each user decision added accuracy, from a GLM at CV 0.140 through a GBM and a per-line blend to a four-way blend at 0.159, with feature stability screening alone adding 0.004. |
| 6 | 2:50-3:25 | Results dashboard, section 2 (CV against public LB) | Local CV tracks the public leaderboard with correlation about 0.995 and noise of 0.002, so we submit only when the predicted gain is above that noise. |
| 7 | 3:25-4:05 | `docs/ai_review_log.md` and `docs/experiment_log.md` (proposer and verification columns) | AI proposed most experiments and wrote most code, but every change passed pytest, CV and a human decision, and independent AI reviews caught bugs such as the LightGBM start-level error and a credibility ratio biased 11 to 24 % low. |
| 8 | 4:05-4:30 | Results dashboard, section 1 (leaderboard progress) | The best public score is 0.12749, up from 0.10753 for the GLM baseline, and late gains are now inside the noise. |
| 9 | 4:30-4:50 | Repo tree (`src/`, `tests/`, `experiments/configs.py`) and the Part 2 checklist from the write-up | For the new market we re-run the data-quality register, development curves, trend backtest, level probe and stability screen rather than carrying parameters over, and [TBD: what changed in Part 2]. |

Notes for the recording
- Timing is a plan; the nine sentences are about 330 words, which fits in 4:50 at a calm pace.
- Fill beat 9 and any Part 2 numbers once Part 2 is complete. Do not quote figures that are not in `docs/experiment_log.md`.
