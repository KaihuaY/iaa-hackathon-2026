# IAA AI-Assisted Actuarial Hackathon 2026 – Part 1

Source: Kaggle pages pulled via API → [docs/kaggle_pages.md](docs/kaggle_pages.md) (full text incl. rules).

## Key facts
| | |
|---|---|
| Part 1 deadline | **2026-10-23 16:00 UTC** (per `kaggle competitions list`) |
| Part 2 | "Emerging market" dataset, separate Kaggle page, opens when Part 1 closes |
| Team size | max 3, same team for both parts |
| Submissions | 5/day; select 1 final |
| Prizes | CAD 500 / 100 / 50 – decided after Part 2 by judges, **not** Part 1 leaderboard alone |

## Task
For every row in `test_motor`, `test_buildings`, `test_contents` (2025 policy periods), predict the
**expected ultimate claim cost** (risk premium, not charged premium): all claims with `loss_date` inside the
row's period, **net of excess, net of recoveries, excluding repudiated claims**.

- Submission: one CSV, `row_id,prediction`, 626,639 rows (M 254,986 / B 184,155 / C 187,498), prediction ≥ 0.
- Leaderboard metric: **Tweedie D² (power 1.5)** — higher is better.

## Overall scoring (after Part 2)
| Weight | Category |
|---|---|
| 40% | Pricing performance & adaptation (Part 1 + Part 2 private LB equally, plus judges' view) |
| 25% | Effective use of AI across the workflow, with verification of AI outputs |
| 20% | Trust, validation & guardrails (tests, diagnostics, governance, reproducibility) |
| 15% | Communication: ≤5-min video, ≤5-page write-up, version-controlled repo |

## Data (`data/raw/public/`, Parquet)
| Table | Rows | Notes |
|---|---|---|
| policies_motor | 979,968 | 2021–2024, 39 cols |
| policies_buildings | 707,726 | 27 cols |
| policies_contents | 721,617 | 28 cols |
| claims | 163,150 | valued 31-Dec-2024; motor 112.6k, buildings 26.4k, contents 24.1k; 10.1k open, 1.7k repudiated |
| customers | 545,240 | joins on `customer_id` |
| addresses | 575,983 | joins on `address_id` |
| weather | 2,879,915 | `address_id` × year (2021–2025) |
| test_* | 254,986 / 184,155 / 187,498 | 2025, no `renewal_decision` |

## Modelling gotchas spotted so far
- **Building the target**: claims link by `policy_id` only, and a policy has several rows (years, mid-term changes), so assign each claim to a row by `loss_date` ∈ [period_start, period_end].
- **IBNR / development**: 2024 claims are under-reported and open claims may still develop. Training on raw incurred will understate cost for recent periods unless it's adjusted to ultimate.
- `renewal_decision` exists in train only. Don't use it as a feature (leakage).
- **Exposure**: test rows can be partial years (e.g. 0.345). Predictions are per row, so they scale with exposure.
- Data quality: see the register in the EDA dashboard (section 2). Duplicate claims, ids that lost their leading zeros, exposure in days, sum insured ×1000, messy construction labels, impossible licence years. ~2.5% missing in many fields; `telematics_score` is ~87% null by design.
- Leakage: `customer_service_contacts` (includes claim calls) and `renewal_decision`. Never use them as features.
- IDs are text. Read them as strings.
- **Rules**: no redistributing the data, so keep `data/` out of the repo.
