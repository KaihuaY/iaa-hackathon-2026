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
# # IAA Hackathon 2026 · Part 1 — Exploratory data analysis
#
# **Goal of the competition:** predict, for every 2025 policy row, the *expected ultimate claim cost*
# (net of excess and recoveries, excluding repudiated claims). Scored by Tweedie D² (power 1.5).
#
# **How to read this dashboard.** Sections run from "what data do we have" → "what is wrong with it" →
# "what drives claims" → "how is 2025 different". All charts are interactive: hover for values,
# click legend entries to hide series, use the dropdowns in section 6 to switch rating factors.
#
# Everything is computed locally from `data/raw/public`. Cleaning rules live in `src/data.py`
# so modelling reuses exactly what is shown here.

# %%
import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display, Markdown

from data import build_all, enrich, load, row_type, LINES, LEAKY_COLS, VALUATION_DATE
from eda import summary, oneway, features, lag_cdf, psi
from viz import setup, LINE_COLOURS, NEUTRAL, SERIES, GRID, INK2, MUTED

setup()
pd.set_option("display.float_format", lambda x: f"{x:,.3f}")

log = []
D = build_all(log)
claims = D["claims"]
TR = {l: enrich(D[f"train_{l}"], D["customers"], D["addresses"], D["weather"]) for l in LINES}
TE = {l: enrich(D[f"test_{l}"], D["customers"], D["addresses"], D["weather"]) for l in LINES}
ALL = pd.concat([TR[l].assign(line=l) for l in LINES], ignore_index=True)

def fmt_table(df, fmts=None):
    text_cols = [c for c in df.columns if df[c].dtype == object]
    return (df.style.format(fmts or "{:,.3f}", na_rep="–")
            .set_properties(subset=text_cols, **{"text-align": "left"})
            .set_table_styles([{"selector": "th", "props": "text-align:left; font-weight:600;"}]))

# %% [markdown]
# ## 1 · Data map
# Eleven tables. Policy tables are one row per **policy period** (a calendar year or part of one).
# Claims link to policies by `policy_id` **and** `loss_date` falling inside the row's period.

# %%
raw_counts = {n: load(n).shape for n in ["customers", "addresses", "weather", "claims",
              "policies_motor", "policies_buildings", "policies_contents",
              "test_motor", "test_buildings", "test_contents", "sample_submission"]}
data_map = pd.DataFrame(raw_counts, index=["rows", "columns"]).T
data_map["grain"] = ["customer", "address", "address × year (2021–2025)", "claim (valued 31-Dec-2024)",
                     "policy period 2021–24", "policy period 2021–24", "policy period 2021–24",
                     "policy period 2025", "policy period 2025", "policy period 2025", "test row"]
display(fmt_table(data_map, {"rows": "{:,}", "columns": "{:,}"}))

m = claims.row_id.notna()
display(Markdown(f"**Claim → policy-row matching:** {m.mean():.2%} of non-repudiated claims match a training row "
                 f"by policy and loss date. The {(~m).sum():,} unmatched claims mostly have loss dates in Dec-2020, "
                 f"before the first policy period."))

# %% [markdown]
# ## 2 · Data-quality register
# Problems found and how `src/data.py` fixes them. Several look deliberately planted. Each is
# a candidate talking point for the **Trust, Validation & Guardrails** criterion.

# %%
explain = {
    "duplicate claims under new claim_id": "Identical claim re-keyed under a new id. Dropped (keep first).",
    "customer_id lost leading zeros": "Ids stored as numbers lost leading zeros, which breaks the customer join. Left-padded to 8 characters.",
    "exposure recorded in days, not years": "Exposure = 365 instead of 1.0. Recomputed from the period dates.",
    "years_licensed impossible for driver_age": "Licence held longer than age − 17. Capped and flagged.",
    "sum_insured x1000 unit error": "Values in the hundreds of millions, entered in units instead of thousands. Divided by 1000.",
    "construction_type inconsistent labels": "Brick/brick/BRICK, plus single-letter codes T/W (= timber). Normalised to 4 levels.",
}
reg = pd.DataFrame(log)
reg = reg[reg.rows > 0].assign(fix=lambda d: d.issue.map(explain))
display(fmt_table(reg, {"rows": "{:,}"}))

# %% [markdown]
# ### 2a · Two leaking columns must not be features
# * `renewal_decision` exists only in training data, and **declined_by_insurer** rows claim at 10× the normal rate:
#   the insurer declines *because* of claims.
# * `customer_service_contacts` is recorded over the period, so it includes claim-related calls.
#   Rows with 0 contacts have **zero** claims, and test values look like the no-claim rows.

# %%
fig = make_subplots(rows=1, cols=2, subplot_titles=("Claim frequency by service contacts (train)",
                                                    "Mean contacts: train claim vs no-claim vs test"))
for l in LINES:
    t = TR[l].assign(c=TR[l].customer_service_contacts.clip(upper=6))
    g = t.groupby("c").apply(lambda x: x.claim_count.sum() / x.exposure.sum(), include_groups=False)
    fig.add_scatter(x=g.index, y=g.values, name=l, mode="lines+markers", line_color=LINE_COLOURS[l],
                    legendgroup=l, row=1, col=1, hovertemplate="%{x} contacts<br>freq %{y:.3f}<extra>" + l + "</extra>")
    vals = [TR[l].loc[TR[l].claim_count == 0, "customer_service_contacts"].mean(),
            TR[l].loc[TR[l].claim_count > 0, "customer_service_contacts"].mean(),
            TE[l].customer_service_contacts.mean()]
    fig.add_bar(x=["train, no claim", "train, ≥1 claim", "test"], y=vals, name=l, marker_color=LINE_COLOURS[l],
                legendgroup=l, showlegend=False, row=1, col=2, hovertemplate="%{x}: %{y:.2f}<extra>" + l + "</extra>")
fig.update_xaxes(title_text="customer_service_contacts (6 = 6+)", row=1, col=1)
fig.update_yaxes(title_text="claims per exposure year", row=1, col=1)
fig.update_layout(height=380, bargroupgap=0.1)
fig.show()

# %% [markdown]
# ## 3 · Portfolio and target overview
# Target = claims cost per row, **as reported at 31-Dec-2024** (not yet developed to ultimate).

# %%
yr = summary(ALL, ["line", "year"])
display(fmt_table(yr[["rows", "exposure", "claims", "frequency", "severity", "pure_premium", "loss_ratio"]],
                  {"rows": "{:,.0f}", "exposure": "{:,.0f}", "claims": "{:,.0f}", "frequency": "{:.4f}",
                   "severity": "{:,.0f}", "pure_premium": "{:,.1f}", "loss_ratio": "{:.1%}"}))

# %%
metrics = [("frequency", "Claim frequency (per exposure year)"), ("severity", "Average severity (CU)"),
           ("pure_premium", "Pure premium (CU per exposure year)"), ("loss_ratio", "Loss ratio (cost / written premium)")]
fig = make_subplots(rows=1, cols=4, subplot_titles=[t for _, t in metrics], horizontal_spacing=0.06)
for i, (mcol, _) in enumerate(metrics, start=1):
    for l in LINES:
        s = yr.loc[l, mcol]
        fig.add_scatter(x=s.index, y=s.values, name=l, legendgroup=l, showlegend=(i == 1),
                        mode="lines+markers", line_color=LINE_COLOURS[l], row=1, col=i,
                        hovertemplate="%{x}: %{y:,.3f}<extra>" + l + "</extra>")
fig.update_xaxes(dtick=1)
fig.update_yaxes(tickformat=".0%", row=1, col=4)
fig.update_layout(height=380, title="Trends by policy year. Motor 2024 dips: unreported and undeveloped claims, not better risk")
fig.show()

# %% [markdown]
# **Reading it.** Buildings and contents frequency is flat, but buildings severity grows ~5–6 % a year
# (claims inflation). Motor 2024 frequency *and* severity fall, which is what immature data looks like.
# Motor also has the 11.9M open claim in 2023 (section 5), which inflates 2023 severity.
# Buildings runs at a loss ratio above 100 %, so the current price is not adequate there.

# %% [markdown]
# ## 4 · Claims development (IBNR and open claims)
# Claims are valued at 31-Dec-2024. Late losses are not yet reported and open claims carry case
# estimates that may move. If the raw target is used for training, recent years are understated.

# %%
c = claims.copy()
c["loss_month"] = c.loss_date.dt.to_period("M").dt.to_timestamp()
mon = c[c.loss_date >= "2021-01-01"].groupby(["line", "loss_month"]).size()
cdf = lag_cdf(c)
age = (VALUATION_DATE - pd.date_range("2024-01-01", "2024-12-01", freq="MS") - pd.Timedelta(days=15)).days
completeness = pd.DataFrame({l: cdf[l].reindex(age).values for l in cdf}, index=pd.date_range("2024-01-01", "2024-12-01", freq="MS"))

fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.07, subplot_titles=(
    "Reported claim count by loss month", "Report lag: share reported within N days (2021–22 losses)",
    "Estimated reporting completeness of 2024 losses"))
for l in LINES:
    s = mon.loc[l]
    fig.add_scatter(x=s.index, y=s.values, name=l, legendgroup=l, mode="lines", line_color=LINE_COLOURS[l],
                    row=1, col=1, hovertemplate="%{x|%b %Y}: %{y:,} claims<extra>" + l + "</extra>")
    fig.add_scatter(x=cdf.index, y=cdf[l], name=l, legendgroup=l, showlegend=False, mode="lines",
                    line_color=LINE_COLOURS[l], row=1, col=2, hovertemplate="%{x} days: %{y:.1%}<extra>" + l + "</extra>")
    fig.add_scatter(x=completeness.index, y=completeness[l], name=l, legendgroup=l, showlegend=False,
                    mode="lines+markers", line_color=LINE_COLOURS[l], row=1, col=3,
                    hovertemplate="%{x|%b %Y}: %{y:.1%} reported<extra>" + l + "</extra>")
fig.update_yaxes(tickformat=".0%", row=1, col=2); fig.update_yaxes(tickformat=".0%", row=1, col=3)
fig.update_xaxes(title_text="days from loss to report", range=[0, 365], row=1, col=2)
fig.update_layout(height=400)
fig.show()

# %%
st = c.assign(loss_year=c.loss_date.dt.year, os=c.incurred_to_date - c.paid_to_date)
st = st[st.loss_year >= 2021]
dev = st.groupby(["line", "loss_year"]).apply(lambda g: pd.Series({
    "open_share": (g.status == "open").mean(),
    "outstanding_share_of_incurred": g.os.sum() / g.incurred_to_date.sum()}), include_groups=False)
fig = make_subplots(rows=1, cols=2, subplot_titles=("Share of claims still open", "Outstanding case reserve as share of incurred"))
for l in LINES:
    s = dev.loc[l]
    fig.add_bar(x=s.index, y=s.open_share, name=l, legendgroup=l, marker_color=LINE_COLOURS[l], row=1, col=1,
                hovertemplate="%{x}: %{y:.1%}<extra>" + l + "</extra>")
    fig.add_bar(x=s.index, y=s.outstanding_share_of_incurred, name=l, legendgroup=l, showlegend=False,
                marker_color=LINE_COLOURS[l], row=1, col=2, hovertemplate="%{x}: %{y:.1%}<extra>" + l + "</extra>")
fig.update_yaxes(tickformat=".0%"); fig.update_xaxes(dtick=1, title_text="loss year")
fig.update_layout(height=360, bargroupgap=0.1)
fig.show()

# %% [markdown]
# **Reading it.** Most claims are reported within a month. Motor has a longer tail (5.5 % arrive after 90 days,
# mostly bodily injury), buildings 3 %, contents under 1 %. December-2024 losses are only ~67–80 % reported, but
# **2024 as a whole is ~95 % complete by count**. Motor 2024 cost is down ~23 % against frequency down ~10 %, so most
# of the gap is *severity*: late-reported large claims and open case estimates that have not developed yet. About a quarter
# of 2024 claims are still open. Motor outstanding is high even for 2023: that is mostly the one 11.9M case estimate.
# **Implication:** either develop 2024 (and late 2023) to ultimate before training, down-weight immature
# periods, or include a "development" offset. The decision has to be made explicitly and justified.

# %% [markdown]
# ## 5 · Severity distribution and large losses

# %%
fig = make_subplots(rows=1, cols=3, subplot_titles=[f"{l}: net claim size (log scale)" for l in LINES], horizontal_spacing=0.05)
for i, l in enumerate(LINES, start=1):
    x = np.log10(claims.loc[(claims.line == l) & (claims.net_incurred > 0), "net_incurred"])
    fig.add_histogram(x=x, nbinsx=60, marker_color=LINE_COLOURS[l], name=l, showlegend=False, row=1, col=i,
                      hovertemplate="10^%{x:.2f} CU: %{y:,} claims<extra></extra>")
    fig.update_xaxes(tickvals=[1, 2, 3, 4, 5, 6, 7], ticktext=["10", "100", "1k", "10k", "100k", "1M", "10M"], row=1, col=i)
fig.update_layout(height=320, bargap=0.05)
fig.show()

# %%
per = claims.groupby(["line", "peril"]).agg(claims=("claim_id", "size"), mean_cost=("net_incurred", "mean"),
                                            total_cost=("net_incurred", "sum")).reset_index()
per["cost_share"] = per.total_cost / per.groupby("line").total_cost.transform("sum")
fig = make_subplots(rows=1, cols=3, subplot_titles=[f"{l}: share of cost by peril" for l in LINES], horizontal_spacing=0.12)
for i, l in enumerate(LINES, start=1):
    s = per[per.line == l].sort_values("cost_share")
    fig.add_bar(y=s.peril, x=s.cost_share, orientation="h", marker_color=LINE_COLOURS[l], showlegend=False, row=1, col=i,
                customdata=s[["claims", "mean_cost"]], hovertemplate="%{y}<br>%{x:.1%} of cost<br>%{customdata[0]:,} claims"
                "<br>mean %{customdata[1]:,.0f} CU<extra></extra>")
fig.update_xaxes(tickformat=".0%")
fig.update_layout(height=340)
fig.show()

# %%
big = claims.nlargest(12, "net_incurred")[["claim_id", "line", "peril", "status", "loss_date", "paid_to_date",
                                           "incurred_to_date", "net_incurred"]]
display(Markdown("**Largest claims.** The top motor bodily-injury claim is open with an 11.9M CU case estimate against "
                 "0.74M paid, 10× the next largest claim. Is it genuine or a keying error? Either way it needs capping or "
                 "large-loss treatment, or it will distort any model."))
display(fmt_table(big.set_index("claim_id"), {"paid_to_date": "{:,.0f}", "incurred_to_date": "{:,.0f}",
                                              "net_incurred": "{:,.0f}", "loss_date": "{:%Y-%m-%d}"}))
for l in LINES:
    x = claims.loc[claims.line == l, "net_incurred"]
    display(Markdown(f"* **{l}:** median {x.median():,.0f} CU · 99th pct {x.quantile(.99):,.0f} · "
                     f"top 0.1 % of claims = {x[x > x.quantile(.999)].sum() / x.sum():.1%} of cost"))

# %% [markdown]
# ## 6 · Rating-factor one-ways (interactive)
# For each line, pick a factor from the dropdown. **Top:** frequency and pure-premium relativity to the
# line average (1.0 = average). **Bottom:** exposure in each level, so you can see which differences have
# enough volume to trust. Uses policy years **2021–2023** only (2024 is immature). Numeric factors are
# split into deciles, and `missing` is kept as its own level.

# %%
def oneway_dashboard(line):
    df = TR[line][TR[line].year < 2024]
    feats = features(df, exclude=LEAKY_COLS)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.05)
    n_tr = 3
    for k, f in enumerate(feats):
        t = oneway(df, f)
        x = t.level.astype(str)
        vis = k == 0
        cd = np.c_[t.exposure, t.claims, t.frequency, t.severity, t.pure_premium]
        hov = ("%{x}<br>exposure %{customdata[0]:,.0f} · claims %{customdata[1]:,.0f}<br>freq %{customdata[2]:.4f}"
               " · sev %{customdata[3]:,.0f} · PP %{customdata[4]:,.1f}<br>")
        fig.add_scatter(x=x, y=t.pp_rel, name="pure premium relativity", mode="lines+markers", visible=vis,
                        line_color=LINE_COLOURS[line], customdata=cd, hovertemplate=hov + "PP rel %{y:.2f}<extra></extra>", row=1, col=1)
        fig.add_scatter(x=x, y=t.freq_rel, name="frequency relativity", mode="lines+markers", visible=vis,
                        line=dict(color=INK2, dash="dot"), marker=dict(color=INK2), customdata=cd,
                        hovertemplate=hov + "freq rel %{y:.2f}<extra></extra>", row=1, col=1)
        fig.add_bar(x=x, y=t.exposure, name="exposure", marker_color=NEUTRAL, visible=vis, showlegend=False,
                    hovertemplate="%{x}: %{y:,.0f} exposure years<extra></extra>", row=2, col=1)
    buttons = [dict(label=f, method="update",
                    args=[{"visible": [j // n_tr == k for j in range(len(feats) * n_tr)]},
                          {"title.text": f"{line.title()} · {f}"}]) for k, f in enumerate(feats)]
    fig.update_layout(updatemenus=[dict(buttons=buttons, x=1, xanchor="right", y=1.32, yanchor="top",
                                        bgcolor="white", bordercolor=GRID, font=dict(color=INK2))],
                      title=f"{line.title()} · {feats[0]}", height=540)
    fig.add_hline(y=1, line=dict(color=MUTED, width=1, dash="dash"), row=1, col=1)
    fig.update_yaxes(title_text="relativity", row=1, col=1)
    fig.update_yaxes(title_text="exposure", row=2, col=1)
    fig.show()

# %% [markdown]
# ### Motor

# %%
oneway_dashboard("motor")

# %% [markdown]
# ### Buildings

# %%
oneway_dashboard("buildings")

# %% [markdown]
# ### Contents

# %%
oneway_dashboard("contents")

# %% [markdown]
# ### 6a · Factor strength ranking
# A quick screen of which factors separate risk most. It is the exposure-weighted spread of pure-premium relativities
# across levels, so it is a univariate view only: correlated factors will each look strong.

# %%
rank = []
for l in LINES:
    df = TR[l][TR[l].year < 2024]
    for f in features(df, exclude=LEAKY_COLS):
        t = oneway(df, f)
        w = t.exposure / t.exposure.sum()
        rank.append({"line": l, "factor": f, "levels": len(t),
                     "pp_spread": float(np.sqrt((w * (t.pp_rel - 1) ** 2).sum())),
                     "freq_spread": float(np.sqrt((w * (t.freq_rel - 1) ** 2).sum()))})
rank = pd.DataFrame(rank)
fig = make_subplots(rows=1, cols=3, subplot_titles=[f"{l}: top 15 factors" for l in LINES], horizontal_spacing=0.14)
for i, l in enumerate(LINES, start=1):
    s = rank[rank.line == l].nlargest(15, "pp_spread").sort_values("pp_spread")
    fig.add_bar(y=s.factor, x=s.pp_spread, orientation="h", marker_color=LINE_COLOURS[l], showlegend=False, row=1, col=i,
                customdata=s[["freq_spread", "levels"]], hovertemplate="%{y}<br>PP spread %{x:.3f}<br>freq spread "
                "%{customdata[0]:.3f}<br>%{customdata[1]} levels<extra></extra>")
fig.update_layout(height=520)
fig.show()

# %% [markdown]
# ## 7 · Geography and weather
# Region relativities by line, then whether the weather table explains the perils it should.

# %%
reg_rel = []
for l in LINES:
    t = oneway(TR[l][TR[l].year < 2024], "region").set_index("level")
    reg_rel.append(t.pp_rel.rename(l))
reg_rel = pd.concat(reg_rel, axis=1)
fig = go.Figure(go.Heatmap(z=reg_rel.T.values, x=reg_rel.index, y=reg_rel.columns, zmid=1,
                           colorscale=[[0, "#2a78d6"], [0.5, "#f0efec"], [1, "#e34948"]],
                           text=reg_rel.T.round(2).values, texttemplate="%{text}", textfont=dict(size=11),
                           hovertemplate="%{y} · %{x}<br>PP relativity %{z:.2f}<extra></extra>",
                           colorbar=dict(title="PP rel", thickness=12)))
fig.update_layout(height=280, title="Pure-premium relativity by region (2021–23)", yaxis=dict(autorange="reversed"))
fig.show()

# %%
def peril_freq(line, peril, feature, bins=8):
    df = TR[line][TR[line].year < 2024]
    pc = claims[(claims.line == line) & (claims.peril == peril)].groupby("row_id").size()
    df = df.assign(pc=df.row_id.map(pc).fillna(0))
    s = df[feature]
    lvl = pd.qcut(s, bins, duplicates="drop") if s.nunique() > 12 else s.astype(str)
    g = df.groupby(lvl, observed=True).agg(n=("pc", "sum"), e=("exposure", "sum"))
    x = [f"{iv.mid:,.2f}" for iv in g.index] if s.nunique() > 12 else g.index.astype(str)
    return x, g.n / g.e, g.e

pairs = [("motor", "glass_hail", "hail_days"), ("buildings", "storm", "wind_exposure_index"),
         ("buildings", "flood", "flood_zone"), ("buildings", "subsidence", "dry_summer_index"),
         ("buildings", "subsidence", "soil_type"), ("buildings", "escape_of_water", "freezing_days"),
         ("contents", "flood", "elevation_m"), ("contents", "theft", "urban_density")]
fig = make_subplots(rows=2, cols=4, subplot_titles=[f"{l}: {p} freq by {f}" for l, p, f in pairs],
                    horizontal_spacing=0.06, vertical_spacing=0.22)
for k, (l, p, f) in enumerate(pairs):
    x, y, e = peril_freq(l, p, f)
    fig.add_scatter(x=x, y=y, mode="lines+markers", line_color=LINE_COLOURS[l], showlegend=False,
                    customdata=e, row=k // 4 + 1, col=k % 4 + 1,
                    hovertemplate=f + " %{x}<br>freq %{y:.4f}<br>exposure %{customdata:,.0f}<extra></extra>")
fig.update_annotations(font_size=11)
fig.update_layout(height=560, title="Do weather and location explain the perils they should?")
fig.show()

# %%
wy = D["weather"].groupby("year").mean(numeric_only=True)
wy_idx = wy / wy.loc[2021:2024].mean()
fig = go.Figure()
for i, col in enumerate(wy.columns):
    fig.add_scatter(x=wy_idx.index, y=wy_idx[col], name=col, mode="lines+markers", line_color=SERIES[i],
                    customdata=wy[col], hovertemplate="%{x}: %{customdata:.2f} (index %{y:.2f})<extra>" + col + "</extra>")
fig.add_hline(y=1, line=dict(color=MUTED, width=1, dash="dash"))
fig.add_vrect(x0=2024.5, x1=2025.5, fillcolor=GRID, opacity=0.5, line_width=0, annotation_text="test year",
              annotation_position="top left")
fig.update_layout(height=380, title="Average weather by year, indexed to the 2021–24 mean (2025 = the test year)",
                  xaxis=dict(dtick=1), yaxis_title="index")
fig.show()

# %% [markdown]
# **Reading it.** Weather for **2025 is provided**, so test-year conditions are known. 2025 is windier than
# any training year (wind index about 10 % above average), which points to higher storm cost in buildings. Per-peril
# models (or peril-weighted features) let the 2025 weather flow through properly.

# %% [markdown]
# ## 8 · How does the 2025 test set differ from training?
#
# **What the test set is.** The `test_*` files are the **2025 policy rows we must price**. They have every rating
# factor but no claims; their real 2025 claims are hidden and used to score us. We **do** use their features
# (that is what we predict from, and comparing them with training data is a sanity check). We **cannot** train on them,
# because they have no target.
#
# **Why so many rows have exposure 1.0.** About 60 % of rows are full-year policies (exposure = 1). The rest are partial:
# new customers joining mid-year, policies that end early (cancelled or lapsed), or a policy split into several rows
# by a mid-term change (new car, new address). Both years have the same pattern. In the chart below the bars sit side
# by side and are almost identical.

# %%
bands = [0, 0.1, 0.25, 0.5, 0.75, 0.99, 1.0001]
band_lbl = ["≤0.1", "0.1–0.25", "0.25–0.5", "0.5–0.75", "0.75–0.99", "1.0 (full year)"]
fig = make_subplots(rows=1, cols=3, subplot_titles=[f"{l}: share of rows by exposure" for l in LINES], horizontal_spacing=0.06)
for i, l in enumerate(LINES, start=1):
    for name, d, colour in [("train 2024", TR[l][TR[l].year == 2024], NEUTRAL), ("test 2025", TE[l], LINE_COLOURS[l])]:
        sh = pd.cut(d.exposure, bands, labels=band_lbl).value_counts(normalize=True).reindex(band_lbl)
        fig.add_bar(x=band_lbl, y=sh.values, name=name + (" (line colour)" if name == "test 2025" else ""),
                    marker_color=colour, legendgroup=name, showlegend=(i == 1),
                    row=1, col=i, hovertemplate="exposure %{x}: %{y:.1%} of rows<extra>" + f"{l} {name}" + "</extra>")
fig.update_yaxes(tickformat=".0%")
fig.update_layout(height=380, barmode="group", bargroupgap=0.08)
fig.show()

# %%
rt = []
for l in LINES:
    for name, d in [("train 2024", TR[l][TR[l].year == 2024]), ("test 2025", TE[l])]:
        rt.append(row_type(d).value_counts(normalize=True).rename((l, name)))
rt = pd.concat(rt, axis=1).fillna(0)
display(Markdown("**Why rows are shorter than a year: share of rows by type.** Train 2024 and test 2025 match to the decimal:"))
display(fmt_table(rt, "{:.1%}"))

# %% [markdown]
# ### 8a · Are short rows riskier? No: new business is
# Claims per exposure-year by row type (2021–23). Rows that start mid-year have a higher claim rate, and those rows
# are new customers (`new_business` = True). Full-year, split and early-ending rows claim at the same rate per year
# of exposure. So `log(exposure)` as an offset is fine, provided `new_business` is a feature.

# %%
fig = make_subplots(rows=1, cols=3, subplot_titles=[f"{l}: claim frequency by row type" for l in LINES], horizontal_spacing=0.06)
order = ["full year", "split by mid-term change", "ends early", "starts mid-year"]
for i, l in enumerate(LINES, start=1):
    d = TR[l][TR[l].year < 2024]
    g = d.assign(rt=row_type(d)).groupby("rt").agg(n=("claim_count", "sum"), e=("exposure", "sum"),
                                                     nb=("new_business", "mean")).reindex(order)
    fig.add_bar(x=order, y=g.n / g.e, marker_color=LINE_COLOURS[l], showlegend=False, row=1, col=i,
                customdata=np.c_[g.e, g.nb], hovertemplate="%{x}<br>freq %{y:.3f}<br>exposure %{customdata[0]:,.0f}"
                "<br>new business %{customdata[1]:.0%}<extra></extra>")
fig.update_layout(height=380)
fig.show()

# %%
drift = []
for l in LINES:
    a, b = TR[l][TR[l].year == 2024], TE[l]
    for f in features(b, exclude=LEAKY_COLS + list(D["weather"].columns)):
        if f in a:
            drift.append({"line": l, "factor": f, "psi": psi(a[f], b[f])})
drift = pd.DataFrame(drift).sort_values("psi", ascending=False)
display(Markdown("**Population stability index, train-2024 vs test** (rule of thumb: < 0.1 stable, 0.1–0.25 watch, > 0.25 shift). "
                 "Top 15, with weather left out because it differs by year by design (see section 7):"))
display(fmt_table(drift.head(15).set_index(["line", "factor"]), {"psi": "{:.3f}"}))
num_shift = []
for l in LINES:
    a, b = TR[l][TR[l].year == 2024], TE[l]
    for f in ["written_premium", "sum_insured", "vehicle_age", "driver_age", "policyholder_age", "exposure"]:
        if f in b:
            num_shift.append({"line": l, "factor": f, "train_2024_mean": a[f].mean(), "test_mean": b[f].mean(),
                              "change": b[f].mean() / a[f].mean() - 1})
display(fmt_table(pd.DataFrame(num_shift).set_index(["line", "factor"]),
                  {"train_2024_mean": "{:,.2f}", "test_mean": "{:,.2f}", "change": "{:+.1%}"}))

# %% [markdown]
# **Reading it.** The 2025 portfolio looks like 2024: same row-type mix, same exposure profile, stable
# rating-factor mixes (low PSI). Premiums and sums insured are up ~3–6 % (rate / inflation), consistent with the
# severity trend. The market shift comes in Part 2, not here.

# %% [markdown]
# ## 9 · Takeaways to brainstorm
#
# | # | Finding | Modelling decision it forces |
# |---|---|---|
# | 1 | Target is per row, Tweedie D² p=1.5 | Tweedie GLM/GBM on cost with `log(exposure)` offset, or frequency × severity |
# | 2 | 2024 immature: motor cost −23 %, mostly severity (open / late large claims); Dec-24 count ~70 % reported | IBNR / development adjustment vs. down-weighting vs. year effect |
# | 3 | Buildings severity +5–6 %/yr, premiums +3–6 % into 2025 | Trend projection to 2025 |
# | 4 | 11.9M open motor BI claim, heavy tails | Large-loss cap + loading, or robust treatment |
# | 5 | Planted DQ issues (ids, exposure, SI units, labels, duplicates) | Cleaning + automated tests, part of the guardrails score |
# | 6 | Leakage: `customer_service_contacts`, `renewal_decision` | Exclude, with a test to prove it |
# | 7 | 2025 weather provided, 2025 windier | Weather / peril features; possibly per-peril models |
# | 8 | Customers hold multiple lines; 85 % of test policies seen in training | Prior-claims-history features (experience rating) |
# | 9 | Part 2 = emerging market | Keep the pipeline modular and documented so it can adapt |

# %% [markdown]
# ## 10 · Decision board
#
# **Competition mechanics that shape these choices** (from the Kaggle rules):
# * Up to **5 submissions per day** to the leaderboard, but only **one Final Submission** is judged (if none is
#   selected, the best public-leaderboard one is used).
# * One submission is one prediction file. **Inside it you may combine as many models as you like**: per-line
#   models, GLM + GBM blends, per-peril models summed. Only the file is scored.
# * The public leaderboard scores a sample of the test rows. The **private** leaderboard (the rest) decides, so choose
#   the final submission on our own validation and not on public-LB rank.
# * Part 1's final submission counts 50 % of the accuracy score; Part 2 counts the other 50 %.
#
# Decisions are ranked by expected impact on the leaderboard score. The last card is about judging, not the model.
# Click an option (several where marked), add your own option or notes, then **Download choices (JSON)**
# and drop the file into `docs/`.

# %%
from IPython.display import HTML
from board import render_board
HTML(render_board())
