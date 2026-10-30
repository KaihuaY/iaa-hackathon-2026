# IAA AI-Assisted Actuarial Hackathon 2026

Competition summary: [COMPETITION.md](COMPETITION.md). Full Kaggle page text: [docs/kaggle_pages.md](docs/kaggle_pages.md).

All analysis runs locally. Data is not committed (competition rules forbid redistributing it).

## Setup
```
pip install -r requirements.txt
kaggle competitions download -c iaa-ai-assisted-hackathon-2026-part-1 -p data/raw   # then unzip
```

## Layout
| Path | What |
|---|---|
| `src/data.py` | Loading, data-quality fixes (logged), claim→row target construction, joins |
| `src/eda.py` | One-way / summary / drift helpers |
| `src/viz.py` | Shared Plotly theme (offline) |
| `notebooks/01_eda.py` | EDA notebook source (jupytext percent format, the version-controlled file) |
| `scripts/build_dashboard.py` | Runs a notebook and exports a self-contained offline HTML dashboard to `reports/` |

## EDA dashboard
```
python scripts/build_dashboard.py          # ~3 min → reports/01_eda_dashboard.html
```
Or open `notebooks/01_eda.py` as a notebook in VS Code / Jupyter (jupytext) and run it interactively.

## Reproduce the submission
Prerequisites: Python 3.13, `pip install -r requirements.txt`, and a Kaggle API token (`~/.kaggle/kaggle.json`) with the competition rules accepted.
```
python -P -E scripts/reproduce.py --plan     # ordered steps + runtime estimate, runs nothing
python -P -E scripts/reproduce.py --check    # which artefacts already exist / which steps would run
python -P -E scripts/reproduce.py            # full run (download, cache, tests, all experiments, final blend)
```
Options: `--final X10_blend4`, `--skip-download`, `--skip-tests`, `--force`. Steps with an existing artefact are skipped, so an interrupted run resumes.
Expected runtime from scratch: roughly 5.5-6 hours (dominated by the GBM and CatBoost experiments); `--plan` prints the exact estimate.
Output: `submissions/<timestamp>_X10_blend4_reproduced.csv`; per-step logs in `experiments/logs/reproduce.log`; result pickles in `experiments/results/`.

## Part 2 (emerging market)
Part 2 reuses the whole pipeline through a `MarketSpec` (`src/market_spec.py`, selected by the `IAA_MARKET` environment variable; Part 1 output is proven byte-identical by `tests/test_golden_part1.py`). Data goes in `data/part2/{developed,emerging}` (not committed).

Chain as run (27–30 Oct 2026; every step and its result is logged in `docs/part2_plan.md`, `docs/experiment_log.md` and `docs/ai_review_log.md`):
```
python -P -E scripts/new_market_report.py data/part2 --ref data/raw/public --out docs/part2_readiness.md   # schema diff, DQ register, drift
python -P -E scripts/make_pooled_market.py                     # developed history mapped to the emerging schema -> data/part2/pooled
$env:IAA_MARKET="part2pool"                                    # pooled spec (src/market_spec_pool.py); "part2" = emerging-only
python -P -E scripts/run_experiment.py R2_FINAL                # configs in experiments/configs_pool.py (parts, bags, blend)
python -P -E scripts/score_market.py <results>                 # emerging-row CV / holdout / calibration of any result
python -P -E scripts/part2_feature_hunt.py stage2 --ref R2_FINAL   # stage-2 emerging relativities (gender, fuel) -> factor file
python -P -E scripts/part2_factors.py --base <file> --factor motor:nb=1.05 ... --row-factors <csv>:<col>   # level / tilt files with predicted public change
python -P -E scripts/office_premium.py --predictions <final file> --out docs/part2_office_premium.md --csv reports/part2_office_premium.csv
python -P -E scripts/build_dashboard.py notebooks/05_part2_progress.py   # progress dashboard
```
Public-score readings and the exact response model (`docs/part2_response.json`) are documented in `docs/part2_plan.md`; probe uploads are labelled `PROBE` and were never selected as a final. Ticked final: `20261030_0145Z_P2_F11v3_robust_nb105_ren103_si103_fl105.csv` (SHA-256 prefix `378f7b835d3ace59`).

## Final deliverables
`final_submission/`: write-up (`writeup.pdf`, ≤ 5 pages, both parts, office-premium build-up included), the two ticked prediction files, video script, office-premium CSV, and `GITHUB_CHECKLIST.md`. Rebuild the write-up with `python -P -E scripts/build_writeup.py`.

## Licence
MIT (see `LICENSE`).

