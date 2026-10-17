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
