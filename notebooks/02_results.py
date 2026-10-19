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
m = lbmodel.fit(log[log.cv.notna()])  # rows without a CV (scaled copies) cannot enter the CV -> LB fit
_bcv = sub[sub.cv.notna()]
best = _bcv.loc[_bcv.lb.idxmax()]
_cal = log[log.exp == "X10_levelcal"]
_cal_row = (f"| **Best public LB (level-calibrated)** | **{_cal.lb.iloc[0]:.5f}** (X10_levelcal) |\n"
            if len(_cal) and pd.notna(_cal.lb.iloc[0]) else "")
display(Markdown(f"""
| | |
|---|---|
| **Best public LB (by CV)** | **{best.lb:.5f}** ({best.exp}) |
{_cal_row}| Submissions made | {len(sub)} |
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
         ("X3_blend", "blend, noise features removed"), ("X4_blend3", "3-way blend + freq x sev"), ("X5_blend3", "3-way blend, tuned GBMs"), ("X6_blend3bag", "X5 + seed bagging"), ("L19_segload", "segment loading"),
         ("L20_mono_lowreg", "+ monotone, less reg."), ("X10_blend4", "4-way blend + CatBoost + credibility"),
         ("X10_levelcal", "+ LB-calibrated 2025 level")]
st = log.set_index("exp").reindex([s for s, _ in steps])
st.loc["X10_levelcal", "cv"] = st.loc["X10_blend4", "cv"]  # scaled copy of X10, no CV of its own
lab = [f"{s}<br>{t}" for s, t in steps]
fig = make_subplots(rows=1, cols=2, column_widths=[0.4, 0.6], subplot_titles=("Overall CV", "CV by line"), horizontal_spacing=0.08)
fig.add_bar(x=lab, y=st.cv, marker_color=SERIES[0], showlegend=False, row=1, col=1, hovertemplate="%{x}<br>CV %{y:.4f}<extra></extra>")
fig.add_annotation(x=lab[-1], y=st.cv.iloc[-1], text="LB only", showarrow=True, arrowhead=2, ax=0, ay=-28,
                   font=dict(size=10, color=INK2), xref="x", yref="y")
for l in LINES:
    fig.add_scatter(x=lab[:-1], y=st[f"cv_{l}"].iloc[:-1], mode="lines+markers", line_color=LINE_COLOURS[l], name=l, row=1, col=2,
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
d2 = log[log.exp.str.match(r"(L(7|8|9|1[0-9]|20)|F1[57]|P15|A[123]|S1|X[3-9]|X10|OPT|P20)_")].copy()
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
# ## 5b · Level probe (2025 price level)
# The public leaderboard scores a fixed set of rows, so scaled copies of one file differ with **zero sampling noise**: any
# score change is the effect of the scale alone. A parabola through the probe points of each line gives the LB-optimal 2025
# price level for that line. Dashed vertical line = fitted optimum; dotted horizontal line = the calibrated submission.

# %%
from metrics import d2_tweedie

probe_path = RES / "level_probe.csv"
if probe_path.exists():
    pr = pd.read_csv(probe_path)
    pr = pr[pr.base == "X10_blend4"]
    base_lb = float(log.loc[log.exp == "X10_blend4", "lb"].iloc[0])
    cal_lb = log.loc[log.exp == "X10_levelcal", "lb"]
    cal_lb = float(cal_lb.iloc[0]) if len(cal_lb) and pd.notna(cal_lb.iloc[0]) else None
    x10 = load("X10_blend4")
    pl = [l for l in ["motor", "buildings"] if (pr.line == l).any()]
    fig = make_subplots(rows=1, cols=len(pl), subplot_titles=[f"{l}: public LB vs scale" for l in pl], horizontal_spacing=0.1)
    rows = []
    for i, l in enumerate(pl, start=1):
        p = pr[pr.line == l]
        sc = np.r_[p.scale.values, 1.0]
        lbs = np.r_[p.lb_public.values, base_lb]
        order = np.argsort(sc)
        sc, lbs = sc[order], lbs[order]
        a, b, c = np.polyfit(sc, lbs, 2)
        opt = -b / (2 * a)
        grid = np.linspace(min(sc.min(), opt) - 0.02, max(sc.max(), opt) + 0.02, 100)
        fig.add_scatter(x=grid, y=np.polyval([a, b, c], grid), mode="lines", line=dict(color=LINE_COLOURS[l], width=1.5),
                        name="quadratic fit", showlegend=(i == 1), hoverinfo="skip", row=1, col=i)
        fig.add_scatter(x=sc, y=lbs, mode="markers", marker=dict(color=LINE_COLOURS[l], size=10), name="probe points",
                        showlegend=(i == 1), hovertemplate="scale %{x:.2f}<br>LB %{y:.5f}<extra></extra>", row=1, col=i)
        fig.add_vline(x=opt, line=dict(color=MUTED, dash="dash", width=1), row=1, col=i,
                      annotation_text=f"optimum {opt:.3f}", annotation_font_size=10)
        if cal_lb is not None:
            fig.add_hline(y=cal_lb, line=dict(color=NEUTRAL, dash="dot", width=1), row=1, col=i,
                          annotation_text=f"calibrated {cal_lb:.5f}", annotation_font_size=10, annotation_position="bottom right")
        d = x10["lines"][l]
        y_ev, oof = np.asarray(d["y_eval"], float), np.asarray(d["oof"], float)
        ss = np.linspace(0.9, 1.1, 21)
        oc = np.polyfit(ss, [d2_tweedie(y_ev, oof * s_) for s_ in ss], 2)[0]
        rows.append({"line": l, "points": len(sc), "optimal scale": opt, "LB curvature": a, "local OOF curvature": oc})
    fig.update_xaxes(title_text="scale on 2025 prices")
    fig.update_yaxes(title_text="public LB", row=1, col=1)
    fig.update_layout(height=420)
    fig.show()
    display(fmt_table(pd.DataFrame(rows).set_index("line"),
                      {"optimal scale": "{:.3f}", "LB curvature": "{:.3f}", "local OOF curvature": "{:.3f}"}))
else:
    print("probe not run")

# %% [markdown]
# ## 6 · Inside the models

# %%
best_parts = {"GLM+boost": "B10_cap99", "GBM": "L10_cap99"}
fig = make_subplots(rows=1, cols=3, subplot_titles=[f"{l}: top 12 features (GBM gain)" for l in LINES], horizontal_spacing=0.14)
gbm = load("L20_mono_lowreg")
for i, l in enumerate(LINES, start=1):
    imp = pd.Series(gbm["lines"][l]["importance"]).sort_values(ascending=False).head(12)[::-1]
    fig.add_bar(y=imp.index, x=imp.values / imp.values.sum(), orientation="h", marker_color=LINE_COLOURS[l],
                showlegend=False, row=1, col=i, hovertemplate="%{y}: %{x:.1%} of top-12 gain<extra></extra>")
fig.update_xaxes(tickformat=".0%")
fig.update_layout(height=460)
fig.show()

bl = load("X10_blend4")
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
# ## 8 · Decisions for you (round 3)
# Ranked by impact. Pick options and add notes, then **Download choices (JSON)** and save it as
# `docs/decisions_round3.json`. Nothing is submitted to Kaggle without your go-ahead.

# %%
import importlib
import board
import decisions_round2
importlib.reload(decisions_round2)
html = board.render_board(decisions_round2.DECISIONS).replace("iaa-decision-board-v1", "iaa-decision-board-round3")
HTML(html)

# %% [markdown]
# ## 9 · Non-linear hunt and novel techniques, 18 Oct
# All numbers are local nested 5-fold CV (Tweedie D²). **No Kaggle submissions since 17 Oct.** Leaderboard now: #1 0.12908, #2 0.12893,
# #3 0.12855, us #4 0.12815 (FINAL_avg3_levelcal, submitted 17 Oct). Local candidate X16 is predicted at about 0.1290 ± 0.002 with level
# factors; written to `submissions/20261018_1300_FINAL_part1.csv`, **not uploaded**.
#
# - **Smooth log-link GLM** (power-law / squared log terms, log-product interactions): quick 0.1608 → 0.1620 with engineered ratios; banded GLM 0.1522 (full CV).
# - **Review 3 caught an extrapolation blow-up** (2023 holdout −0.66, test mean 24,016 vs ~400) → winsorise, clip, bound, plus an automatic "unsafe" guard (test pure premium per exposure outside 0.5–2× OOF cannot blend or be submitted).
# - **Review 4: holdout still −0.59.** Cumulative-exposure counters are calendar-time proxies (mean 0.00 / 1.67 / 2.99 / 2.95 in 2021–24); dropped → holdout +0.108. Test set verified in range.
# - **Accepted B20** (smooth GLM base + LightGBM residual): CV 0.1574, holdout +0.097; blend X16 nested 0.1597 vs 0.1593; B20 weights 0.16 / 0.38 / 0.39 motor / buildings / contents.
# - **Private-LB bootstrap:** X16 beats X12 / X14 / X10 / X11 with P 0.96–1.00 on 30 % subsets (paired sd 0.0002); k = 1 best.
# - **No gain:** G15 peril interactions (0.1622 vs 0.1620), B20 seed bag (X17 0.1597), B21 smooth + CatBoost (0.1572, X18 0.1597), smooth-GLM grid G16–G19 (0.1621–0.1622), recency weights (0.1635 / 0.1634 vs 0.1639), exposure feature (0.1636), neural residual N1 (0.1623, early stop at epoch 1–2; retune N2 pending).
# - **In progress:** pooled home-lines model (P25 / B23), NN retune (N2), covariate-shift importance weights (P24, judged on the 2023 holdout only).

# %%
import decisions_round4
importlib.reload(decisions_round4)
html4 = (board.render_board(decisions_round4.DECISIONS[:1]).replace("iaa-decision-board-v1", "iaa-decision-board-round4")
         .replace("dboard", "dboard4"))
HTML(html4)

# %% [markdown]
# ## 10 · 19 Oct: why the leaders are ahead, and what the probes found
# **Leaderboard:** #1 0.12996, #2 0.12910, us #3 0.12815 (file of 17 Oct). 10 submissions were used on 19 Oct (user-approved batch).
#
# **Ruled out:** 2025 claims in the data (none), trend method (premium vs claims 1 % apart), weather range (in range), reserve adequacy,
# data re-audit (nothing material), training-years sum-insured miscalibration (nested calibration +0.00003).
#
# **Probe table** (same rows, noise-free differences; base X21 raw 0.12744):
#
# | Probe | Public LB |
# |---|---|
# | Buildings ×1.09 / ×1.15 | 0.12783 / 0.12772 (optimum ×1.098) |
# | Contents ×1.03 / ×1.06 | 0.12748 / 0.12748 (flat) |
# | Motor new business ×1.06 | 0.12744 |
# | Motor full-year rows ×1.04 | 0.12751 |
# | Buildings SI top quartile ×1.08 / ×1.16 | 0.12785 / 0.12808 (optimum ≈ ×1.22, ≈ +0.0007) |
# | Buildings SI bottom quartile ×1.08 | 0.12734 |
#
# **Reading.** The 2025 buildings shift is a sum-insured structure (high-SI homes under-priced), not in the training years. The local
# mechanism that helps is sum_insured^0.7 in the buildings exposure offset (+0.0003–0.0005 per family on buildings; X30 nested 0.1598 = X21).
#
# **Round-6 verdicts:** segment blends −, recalibration −, grid 0, CatBoost bag +0.0001 (X21), smooth frequency 0, per-line cap 0,
# model-based loading −, low-lr +0.0001, unconstrained −, mileage sentinel 0. The model loop closes after X28/X29.
#
# **Next steps (UTC):**
# - X25 / X28 / X29 blends finish ~03:20 / ~03:50 / ~05:30 on 19 Oct; accept only if nested ≥ 0.1600.
# - 20 Oct 00:03: submit the two candidate files `20261018_2021_X21_blend4_a1bag_sifix.csv` and `20261018_2152_X30_blend4_si07_sifix.csv` (expected ≈ 0.1284 public).
# - 20 Oct: your decision on up to 4 extra probes (card below).
# - 22 Oct: final pick and tick on kaggle.com. 23 Oct 16:00: close.

# %%
html5 = (board.render_board(decisions_round4.DECISIONS[1:]).replace("iaa-decision-board-v1", "iaa-decision-board-round5")
         .replace("dboard", "dboard5"))
HTML(html5)

# %% [markdown]
# ## 11 · 19 Oct 17:00 UTC — status and plan
#
# 1. **Nothing is running.** The local model search is closed at nested CV 0.1598 ± 0.0002 (about 40 ideas since 17 Oct). Latest: tail models
#    for large outliers (T3 0.1579, motor power 1.7 0.1082 — below baseline; the large-loss excess is not predictable from features, OOF corr 0.012);
#    written premium as a blend part X27 nested 0.1599 (+0.0001, not adopted).
# 2. **Leaderboard (public, 11:07 UTC):** #1 0.12996, #2 0.12940, #3 0.12923, #4 0.12817, us #5 0.12815.
# 3. **Debate outcome:** public-row factors are mostly sampling noise (SD 0.10–0.13 per sum-insured quartile, 0.05 per line); a +0.0007 CV gain
#    shows ≤ 0 on the public slice 1 time in 6; the final (private) ranking favours a credibility-shrunk file.
# 4. **Scheduled (UTC, 20 Oct):**
#    - 00:03 — two candidates (X21_sifix, X30_sifix) + four large-segment probes (contents SI Q4 ×1.10; motor value groups 15–20 ×1.06;
#      buildings wind-jump regions ×1.08; buildings Q3 split-half ×1.20).
#    - 00:37 — three tail-test files (T3 tail model; X21 × large-loss propensity uplift; X21 with power-1.7 motor). Slot 10 in reserve.
# 5. **Next:** 20 Oct read the 9 scores with the exact Tweedie response and credibility Z; 21 Oct up to 4 follow-ups; 22 Oct submit the shrunk file
#    (`submissions/20261019_0825_X30_blend4_si07_shrunk.csv`) and the probe-fitted file, user ticks the final; 23 Oct 16:00 Part 1 closes.
# 6. **Under discussion:** combining existing submission files into a better view (see the chat).
