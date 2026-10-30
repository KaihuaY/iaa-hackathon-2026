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

## Final files (the two ticked submissions)
Both files are rebuilt from the code here; the data comes from Kaggle (not committed). SHA-256 prefixes are those of the ticked files.

**Part 1 — `20261022_1708_X33_blend4_si07_allbags_shrunk_c102.csv` (`d83ce8cba461830d`)**, public 0.12826, private 0.13777 (5th):
```
python -P -E scripts/reproduce.py --final X33_blend4_si07_allbags        # four families, 3-seed bags, per-line blend (experiments/configs.py)
python -P -E scripts/segment_calibrate.py X33_blend4_si07_allbags --level-factors motor=1.012 contents=1.02     --band-line buildings --band-col sum_insured --band-factors 1.01 1.03 1.05 1.07 --tag shrunk_c102
```
The level factors are the credibility-shrunk public-leaderboard readings (`scripts/lb_response_fit.py`, `docs/final_pick_22oct.md`).

**Part 2 — `20261030_0145Z_P2_F11v3_robust_nb105_ren103_si103_fl105.csv` (`378f7b835d3ace59`)**, public 0.16139:
```
python -P -E scripts/make_pooled_market.py                                   # developed history mapped to the emerging schema
$env:IAA_MARKET="part2pool"
python -P -E scripts/run_experiment.py R2_FINAL                              # pooled four-family blend (experiments/configs_pool.py)
python -P -E scripts/part2_factors.py --base submissions/part2/<R2_FINAL file>.csv --tag R2_FINAL_shrunk_nb130_ren109_b107_c099     --factor motor:nb=1.30 --factor motor:ren=1.09 --factor buildings=1.07 --factor contents=0.99          # "file 6"
python -P -E scripts/part2_feature_hunt.py factors --ref R2_FINAL            # -> experiments/logs/part2_feature_hunt/R2_FINAL/stage2_factors.csv
python -P -E scripts/part2_factors.py --base submissions/part2/<file 6>.csv --tag F6_stage2_core_a3     --row-factors experiments/logs/part2_feature_hunt/R2_FINAL/stage2_factors.csv:core_a3                 # "core_a3"
python -P -E scripts/build_f11v3.py                                          # core_a3 -> F11 v3 (levels, home tilts, renormalisation)
python -P -E scripts/office_premium.py --predictions submissions/part2/<F11 v3 file>.csv --out docs/part2_office_premium.md --csv reports/part2_office_premium.csv
```
Every file in this chain, its public score and the decision taken on it are in `docs/part2_submissions.csv`; the readings behind the F11 v3 factors are in `docs/part2_plan.md` (30 Oct) and `docs/part2_response.json`.

## Other deliverables
The write-up (DOCX, built from its Markdown source by `scripts/build_writeup.py`), the video (built offline from per-beat voice recordings by `scripts/make_video.py`) and the two prediction files are submitted through the hackathon form, not through this repository. The development record is in `docs/experiment_log.md`, `experiments/log.csv`, `docs/ai_review_log.md` (20 second-model reviews) and `docs/part2_plan.md`.
## Licence
MIT (see `LICENSE`).

