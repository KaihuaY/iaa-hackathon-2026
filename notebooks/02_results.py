# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # IAA Hackathon 2026 · Part 1 — Modelling results
#
# What we tried, what each step was worth, why local testing can be trusted, and the decisions waiting for you
# (section 8). Everything is read from `experiments/log.csv` and `experiments/results/`, so rebuilding this page
# (`python scripts/build_dashboard.py notebooks/02_results.py`) always shows the latest state.

# %%
import pickle
import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display, Markdown, HTML

import lbmodel
from viz import setup, LINE_COLOURS, NEUTRAL, SERIES, GRID, INK2, MUTED

setup()
LINES = ["motor", "buildings", "contents"]
log = pd.read_csv(ROOT / "experiments" / "log.csv")
log["lb"] = pd.to_numeric(log.lb_public, errors="coerce")
log["t"] = pd.to_datetime(log.time)
RES = ROOT / "experiments" / "results"


def load(name):
    with open(RES / f"{name}.pkl", "rb") as f:
        return pickle.load(f)


def fmt_table(df, fmts=None):
    text_cols = [c for c in df.columns if df[c].dtype == object]
    return (df.style.format(fmts or {}, na_rep="–").set_properties(subset=text_cols, **{"text-align": "left"})
            .set_table_styles([{"selector": "th", "props": "text-align:left; font-weight:600;"}]))


sub = log[log.lb.notna()].sort_values("t")
m = lbmodel.fit()
best = sub.loc[sub.lb.idxmax()]
display(Markdown(f"""
| | |
|---|---|
| **Best public LB** | **{best.lb:.5f}** ({best.exp}) |
| Submissions made | {len(sub)} |
| Experiments run locally | {len(log)} |
| CV → LB agreement | corr {m['corr']:.3f} on {m['n']} submissions, typical error ±{m['sd']:.4f} |
"""))

# %% [markdown]
# ## 1 · Leaderboard progress
# Each submission in the order made. The dotted line is the best so far.

# %%
fig = make_subplots(rows=1, cols=2, subplot_titles=("Public leaderboard score", "Local CV score (same submissions)"),
                    horizontal_spacing=0.08)
x = [f"{i + 1}. {e}" for i, e in enumerate(sub.exp)]
fig.add_scatter(x=x, y=sub.lb, mode="lines+markers", line_color=SERIES[0], name="public LB", row=1, col=1,
                hovertemplate="%{x}<br>LB %{y:.5f}<extra></extra>")
fig.add_scatter(x=x, y=sub.lb.cummax(), mode="lines", line=dict(color=MUTED, dash="dot"), name="best so far", row=1, col=1,
                hovertemplate="best so far %{y:.5f}<extra></extra>")
fig.add_scatter(x=x, y=sub.cv, mode="lines+markers", line_color=SERIES[1], name="CV", row=1, col=2,
                hovertemplate="%{x}<br>CV %{y:.4f}<extra></extra>")
fig.update_xaxes(tickangle=-35)
fig.update_layout(height=420)
fig.show()

# %% [markdown]
# ## 2 · Can we trust local testing? Yes
# CV against the public LB for every submission. Points sit on a straight line, and the order is identical. Hollow markers are
# candidates **not** submitted, plotted at their predicted LB. Anything inside the shaded band is
# indistinguishable from the current best, which is why it is not worth a submission.

# %%
d = sub[~sub.decision.fillna("").str.contains("bug|DIAGNOSTIC")]
xs = np.linspace(log.cv.min() - 0.003, log.cv.max() + 0.003, 50)
line_y = m["best_lb"] + m["b"] * (xs - m["best_cv"])
fig = go.Figure()
fig.add_scatter(x=np.r_[xs, xs[::-1]], y=np.r_[line_y + m["sd"], (line_y - m["sd"])[::-1]], fill="toself",
                fillcolor="rgba(42,120,214,0.10)", line=dict(width=0), hoverinfo="skip", name="± noise")
fig.add_scatter(x=xs, y=line_y, mode="lines", line=dict(color=SERIES[0], width=1.5), name="CV → LB fit", hoverinfo="skip")
fig.add_scatter(x=d.cv, y=d.lb, mode="markers+text", text=d.exp, textposition="top left", textfont=dict(size=10, color=INK2),
                marker=dict(color=SERIES[0], size=10), name="submitted",
                hovertemplate="%{text}<br>CV %{x:.4f}<br>LB %{y:.5f}<extra></extra>")
cand = log[log.lb.isna() & log.cv.notna() & (log.cv > 0.15)].copy()
cand["pred"] = [lbmodel.predict(c, m)[0] for c in cand.cv_nested.fillna(cand.cv)]
fig.add_scatter(x=cand.cv, y=cand.pred, mode="markers", marker=dict(color="white", size=9, line=dict(color=SERIES[1], width=2)),
                name="not submitted (predicted)", text=cand.exp,
                hovertemplate="%{text}<br>CV %{x:.4f}<br>predicted LB %{y:.5f}<extra></extra>")
fig.update_layout(height=480, xaxis_title="local CV (Tweedie D²)", yaxis_title="public leaderboard")
fig.show()

# %% [markdown]
# ## 3 · What each step was worth
# CV after each change, overall and per line. Every step was one of your decision-board choices, except where marked.

# %%
steps = [("E1_glm", "GLM, raw labels"), ("G3_dev", "+ develop to ultimate"), ("G4_cap", "+ cap large losses"),
         ("G6_trend", "+ experience, trend, weights"), ("B6_boost", "+ GBM boosting from GLM"),
         ("L6_lgbm", "standalone GBM"), ("X1_blend", "blend B6 + L6"), ("X2_blend", "blend, cap 99%"),
         ("X3_blend", "blend, noise features removed"), ("X4_blend3", "3-way blend + freq x sev"), ("X5_blend3", "3-way blend, tuned GBMs")]
st = log.set_index("exp").reindex([s for s, _ in steps])
lab = [f"{s}<br>{t}" for s, t in steps]
fig = make_subplots(rows=1, cols=2, column_widths=[0.4, 0.6], subplot_titles=("Overall CV", "CV by line"), horizontal_spacing=0.08)
fig.add_bar(x=lab, y=st.cv, marker_color=SERIES[0], showlegend=False, row=1, col=1, hovertemplate="%{x}<br>CV %{y:.4f}<extra></extra>")
for l in LINES:
    fig.add_scatter(x=lab, y=st[f"cv_{l}"], mode="lines+markers", line_color=LINE_COLOURS[l], name=l, row=1, col=2,
                    hovertemplate="%{x}<br>" + l + " CV %{y:.4f}<extra></extra>")
fig.update_yaxes(range=[0.13, st.cv.max() + 0.004], row=1, col=1)
fig.update_xaxes(tickfont=dict(size=9))
fig.update_layout(height=470)
fig.show()

# %% [markdown]
# **Reading it.** Each of your choices (development, cap, experience rating + trend, blending) added accuracy.
# The best model differs by line. The GLM-based model is better for motor (blend weight 0.65 on it). Standalone GBM is
# better for buildings and contents, where risk depends on combinations of conditions (soil × dry summer, flood zone ×
# rainfall) that trees capture and a main-effects GLM cannot.

# %% [markdown]
# ## 4 · Day-2 local experiments (single changes vs L6, CV only, no submissions used)

# %%
d2 = log[log.exp.str.match(r"(L(7|8|9|1[0-7])|F1[57]|P15)_")].copy()
base_cv = log.set_index("exp").loc["L6_lgbm", "cv"]
d2["vs L6"] = d2.cv - base_cv
display(fmt_table(d2[["exp", "description", "cv", "vs L6", "holdout_2023", "decision"]].set_index("exp"),
                  {"cv": "{:.4f}", "vs L6": "{:+.4f}", "holdout_2023": "{:.4f}"}))

# %% [markdown]
# ### 4a · Feature stability screen: why removing features helped
# A genuine rating factor ranks its levels the same way every year. For each categorical feature, we correlate its
# levels' claim frequency in 2021–22 with 2023. `previous_insurer`, `branch_id`, `email_domain` and `campaign_code` show no
# year-to-year consistency (≈ 0), so they are noise the GBM was fitting. `make_model` (300 levels) is only weakly stable, and
# dropping it lifted motor CV by +0.004. Fewer, honest features: better CV, better 2023 holdout, and a simpler model.

# %%
import sys as _s
from features import build_base, model_matrix
_b = build_base()
rows = []
for l in LINES:
    tr = _b[f"train_{l}"]
    for c in ["previous_insurer", "branch_id", "email_domain", "campaign_code", "make_model", "occupation", "region",
              "marketing_segment", "credit_band", "channel"]:
        if c not in tr:
            continue
        s = tr[c].astype(str)
        def rel(mask):
            g = tr[mask].groupby(s[mask]).agg(n=("claim_count", "sum"), e=("exposure", "sum"))
            g = g[g.e > 300]
            return g.n / g.e
        j = pd.concat([rel(tr.year.values <= 2022), rel(tr.year.values == 2023)], axis=1, keys=["a", "z"]).dropna()
        rows.append({"line": l, "feature": c, "levels": s.nunique(), "stability": j.a.corr(j.z)})
stab = pd.DataFrame(rows).pivot_table(index="feature", columns="line", values="stability")
stab = stab.reindex(stab.mean(axis=1).sort_values().index)
fig = go.Figure(go.Heatmap(z=stab.values, x=stab.columns, y=stab.index, zmin=-1, zmax=1,
                           colorscale=[[0, "#e34948"], [0.5, "#f0efec"], [1, "#2a78d6"]],
                           text=stab.round(2).values, texttemplate="%{text}",
                           hovertemplate="%{y} · %{x}<br>year-to-year stability %{z:.2f}<extra></extra>",
                           colorbar=dict(title="stability", thickness=12)))
fig.update_layout(height=420, title="Year-to-year stability of level frequencies (1 = consistent risk signal, 0 = noise)")
fig.show()
del _b

# %% [markdown]
# ## 5 · Trend backtest: which trend projects costs best?
# CV cannot judge the trend, because its label already uses the premium trend. This forward test can: train on years before *T*
# with labels trended to *T*, predict year *T*, and compare the total with year *T*'s developed actual cost.
# **Level ratio 1.00 = perfect.** Above 1 over-prices; below 1 under-prices.

# %%
bt_path = RES / "backtest_level.csv"
if bt_path.exists():
    bt = pd.read_csv(bt_path)
    piv = bt.pivot_table(index=["line", "target_year"], columns="trend", values="level_ratio")
    display(fmt_table(piv, "{:.3f}"))
    fig = make_subplots(rows=1, cols=3, subplot_titles=[f"{l}: level ratio" for l in LINES], horizontal_spacing=0.06)
    for i, l in enumerate(LINES, start=1):
        for k, meth in enumerate(["premium", "claims", "none"]):
            s = bt[(bt.line == l) & (bt.trend == meth)]
            fig.add_bar(x=s.target_year.astype(str), y=s.level_ratio, name=meth, marker_color=SERIES[k],
                        showlegend=(i == 1), legendgroup=meth, row=1, col=i,
                        hovertemplate=f"{meth}" + " · target %{x}<br>level %{y:.3f}<extra></extra>")
        fig.add_hline(y=1, line=dict(color=MUTED, dash="dash", width=1), row=1, col=i)
    fig.update_yaxes(range=[0.85, 1.12])
    fig.update_layout(height=380, barmode="group", bargroupgap=0.08)
    fig.show()
else:
    display(Markdown("*Backtest not run yet.*"))

# %% [markdown]
# ## 6 · Inside the models

# %%
best_parts = {"GLM+boost": "B10_cap99", "GBM": "L10_cap99"}
fig = make_subplots(rows=1, cols=3, subplot_titles=[f"{l}: top 12 features (GBM gain)" for l in LINES], horizontal_spacing=0.14)
gbm = load("L17_tuned")
for i, l in enumerate(LINES, start=1):
    imp = pd.Series(gbm["lines"][l]["importance"]).sort_values(ascending=False).head(12)[::-1]
    fig.add_bar(y=imp.index, x=imp.values / imp.values.sum(), orientation="h", marker_color=LINE_COLOURS[l],
                showlegend=False, row=1, col=i, hovertemplate="%{y}: %{x:.1%} of top-12 gain<extra></extra>")
fig.update_xaxes(tickformat=".0%")
fig.update_layout(height=460)
fig.show()

bl = load("X5_blend3")
w = pd.DataFrame({l: {**{f"w {k}": v for k, v in bl["lines"][l]["weights"].items()},
                      "blend CV": bl["lines"][l]["cv"], "nested CV": bl["lines"][l].get("cv_nested", np.nan)} for l in LINES}).T
display(Markdown("**Blend weights per line** (chosen on CV; the nested CV picks the weight out-of-fold, so it is an honest score):"))
display(fmt_table(w, {c: ("{:.2f}" if c.startswith("w ") else "{:.4f}") for c in w.columns}))

# %% [markdown]
# ## 7 · Full experiment log
# Every run, who proposed it, how it was verified and what was decided. This doubles as the AI-use record for the judges.

# %%
cols = ["exp", "description", "proposer", "cv", "holdout_2023", "lb_pred", "lb_public", "decision"]
display(fmt_table(log[cols].set_index("exp"), {"cv": "{:.4f}", "holdout_2023": "{:.4f}", "lb_pred": "{:.4f}"}))

# %% [markdown]
# ## 8 · Decisions for you (round 2)
# Ranked by impact. Pick options and add notes, then **Download choices (JSON)** and save it as
# `docs/decisions_round2.json`. Nothing is submitted to Kaggle without your go-ahead.

# %%
import importlib
import board
import decisions_round2
importlib.reload(decisions_round2)
html = board.render_board(decisions_round2.DECISIONS).replace("iaa-decision-board-v1", "iaa-decision-board-round2")
HTML(html)
