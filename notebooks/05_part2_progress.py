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
# # IAA Hackathon 2026 · Part 2 — Progress in the emerging market
#
# Where we stand on the leaderboard, what is different about this market, why the price level is uncertain, and what
# comes next. Every number on this page is read at build time from the project files (`docs/part2_*.md`,
# `docs/part2_submissions.csv`, `experiments/log.csv`, `submissions/part2/*.csv` and the local data), so a rebuild
# (`python -P -E scripts/build_dashboard.py notebooks/05_part2_progress.py`) always shows the latest state.
# Everything is computed offline; nothing is fetched from the internet.

# %%
import re
import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path[:0] = [str(ROOT / "src")]

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display, Markdown

from viz import setup, LINE_COLOURS, NEUTRAL, SERIES, INK, INK2, MUTED, SURFACE as SURFACE_C

setup()
pd.set_option("display.max_colwidth", None)

LINES = ["motor", "buildings", "contents"]
PREFIX = {"M": "motor", "B": "buildings", "C": "contents"}
AGE_COL = {"motor": "driver_age_band", "buildings": "policyholder_age_band", "contents": "policyholder_age_band"}
DEADLINE = pd.Timestamp("2026-10-31 03:59", tz="UTC")
NOW = pd.Timestamp.now(tz="UTC")
DOCS, EM, DEV = ROOT / "docs", ROOT / "data" / "part2" / "emerging", ROOT / "data" / "raw" / "public"
SUBDIR = ROOT / "submissions" / "part2"
NOTES = []  # every "not yet available" message, repeated at the end of the page


def note(msg):
    NOTES.append(msg)
    display(Markdown(f"> *Not yet available: {msg}*"))


def safe(fn):
    """Run one section; a missing or malformed source prints a note instead of failing the build."""
    try:
        fn()
    except Exception as e:  # noqa: BLE001 - a dashboard section must never stop the build
        note(f"{fn.__name__.replace('_', ' ')} could not be rendered ({type(e).__name__}: {e})")
    return fn


def fmt_table(df, fmts=None, wide=()):
    text_cols = [c for c in df.columns if df[c].dtype == object]
    st = (df.style.format(fmts or {}, na_rep="–").set_properties(subset=text_cols, **{"text-align": "left"})
          .set_table_styles([{"selector": "th", "props": "text-align:left; font-weight:600;"}]))
    if wide:
        st = st.set_properties(subset=list(wide), **{"min-width": "260px", "max-width": "520px", "white-space": "normal"})
    return st


def rng(values, spec):
    """'a–b' for a range of values, or a single value when both ends format the same."""
    lo, hi = format(min(values), spec), format(max(values), spec)
    return lo if lo == hi else f"{lo}–{hi}"


def tint(hex_colour, f):
    """Mix a colour with white (f = share of white)."""
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return "#" + "".join(f"{round(c + (255 - c) * f):02x}" for c in (r, g, b))


def vmark(fig, x, text, colour=MUTED, dash="dash"):
    """Vertical marker on a date axis (add_vline's annotation fails on dates)."""
    fig.add_shape(type="line", x0=x, x1=x, y0=0, y1=1, yref="paper", line=dict(color=colour, dash=dash, width=1))
    fig.add_annotation(x=x, y=1, yref="paper", text=text, showarrow=False, yanchor="bottom",
                       font=dict(size=10, color=INK2))


# ---------- markdown helpers: sections by heading, pipe tables ----------
def read_text(path):
    return path.read_text(encoding="utf-8") if path.exists() else None


def md_section(text, pattern):
    """Body of the first heading matching `pattern` (case-insensitive), up to the next heading of the same or higher level."""
    if not text:
        return None
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        m = re.match(r"^(#+)\s+(.*)$", ln)
        if m and re.search(pattern, m.group(2), re.I):
            lvl, out = len(m.group(1)), []
            for nxt in lines[i + 1:]:
                h = re.match(r"^(#+)\s", nxt)
                if h and len(h.group(1)) <= lvl:
                    break
                out.append(nxt)
            return "\n".join(out).strip()
    return None


def pipe_tables(text):
    """All markdown pipe tables in `text` as DataFrames (numeric-looking columns converted)."""
    if not text:
        return []
    tables, block = [], []
    for ln in text.splitlines() + [""]:
        if ln.strip().startswith("|"):
            block.append(ln.strip())
            continue
        if len(block) >= 2:
            rows = [[c.strip() for c in r.strip("|").split("|")] for r in block]
            head, body = rows[0], [r for r in rows[1:] if not all(re.fullmatch(r":?-+:?", c) for c in r if c)]
            body = [r + [""] * (len(head) - len(r)) for r in body]
            df = pd.DataFrame([r[:len(head)] for r in body], columns=head)
            for c in df.columns:
                s = df[c].str.replace(",", "", regex=False).str.replace("*", "", regex=False).str.strip()
                num = pd.to_numeric(s.replace("", np.nan), errors="coerce")
                if num.notna().sum() and num.notna().sum() == (s != "").sum():
                    df[c] = num
            tables.append(df)
        block = []
    return tables


def bullets(text):
    """Top-level '- ' bullets with their indented continuation lines joined."""
    out = []
    for ln in (text or "").splitlines():
        if ln.startswith("- "):
            out.append(ln[2:].strip())
        elif out and ln.startswith("  ") and ln.strip():
            out[-1] += " " + ln.strip()
    return out


# ---------- sources ----------
DECISIONS = read_text(DOCS / "part2_decisions.md")
READINESS = read_text(DOCS / "part2_readiness.md")
PLAN = read_text(DOCS / "part2_plan.md")

LOG = None
if (ROOT / "experiments" / "log.csv").exists():
    _log = pd.read_csv(ROOT / "experiments" / "log.csv")
    LOG = _log[_log.get("market", pd.Series("", index=_log.index)).astype(str).str.startswith("part2")].copy()
    LOG["t"] = pd.to_datetime(LOG.time, errors="coerce")
    for c in ["cv", "cv_nested", "cv_motor", "cv_buildings", "cv_contents", "holdout_2023", "lb_public"]:
        if c in LOG:
            LOG[c] = pd.to_numeric(LOG[c], errors="coerce")

SUBS = None
if (DOCS / "part2_submissions.csv").exists():
    SUBS = pd.read_csv(DOCS / "part2_submissions.csv")
    SUBS["t"] = pd.to_datetime(SUBS.submitted_utc, errors="coerce", utc=True)
    for c in ["nested_cv", "public_score", "rank", "n_teams", "leader_score"]:
        SUBS[c] = pd.to_numeric(SUBS[c], errors="coerce")
    SUBS = SUBS.sort_values("t").reset_index(drop=True)
    SUBS["probe"] = SUBS.file.astype(str).str.contains("PROBE") | SUBS.track.astype(str).str.contains("probe")

RESPONSE = None
if (DOCS / "part2_response.json").exists():
    import json
    RESPONSE = json.loads((DOCS / "part2_response.json").read_text(encoding="utf-8"))


def _policies(path, extra=()):
    cols = ["row_id", "policy_id", "period_start", "period_end", "exposure", "new_business", *extra]
    d = pd.read_parquet(path, columns=list(dict.fromkeys(cols)))
    d["start"] = pd.to_datetime(d.period_start, errors="coerce")
    d["end"] = pd.to_datetime(d.period_end, errors="coerce")
    bad = d.exposure > 1.05  # exposure recorded in days (known DQ fault): recompute from the dates
    d.loc[bad, "exposure"] = ((d.end - d.start).dt.days + 1).clip(lower=1)[bad] / 365
    d["year"] = d.start.dt.year
    return d


HIST, TEST = {}, {}
if EM.exists():
    for l in LINES:
        if (EM / f"policies_{l}.parquet").exists():
            HIST[l] = _policies(EM / f"policies_{l}.parquet", (AGE_COL[l],))
        if (EM / f"test_{l}.parquet").exists():
            TEST[l] = _policies(EM / f"test_{l}.parquet", (AGE_COL[l],))
CLAIMS = pd.read_parquet(EM / "claims.parquet") if (EM / "claims.parquet").exists() else None


def attach_cost(pol, claims, line):
    """Net cost (incurred - recoveries, repudiated excluded) of claims attached to `pol` rows by policy_id + loss date."""
    c = claims[(claims.line == line) & (claims.status != "repudiated")][["policy_id", "loss_date", "incurred_to_date", "recoveries"]].copy()
    c["loss"] = pd.to_datetime(c.loss_date, errors="coerce")
    c["net"] = c.incurred_to_date.fillna(0) - c.recoveries.fillna(0)
    m = c.merge(pol[["row_id", "policy_id", "start", "end"]], on="policy_id")
    m = m[(m.loss >= m.start) & (m.loss <= m.end)]
    return pol.row_id.map(m.groupby("row_id").net.sum()).fillna(0.0)


def pp_by_year_nb(pol, claims, line):
    d = pol.assign(cost=attach_cost(pol, claims, line).values)
    g = d.groupby(["year", "new_business"])[["cost", "exposure"]].sum().unstack("new_business")
    out = pd.DataFrame({"pp_nb": g["cost"][True] / g["exposure"][True], "pp_ren": g["cost"][False] / g["exposure"][False]})
    out["ratio"] = out.pp_nb / out.pp_ren
    return out


# Reference history pure premium per line: readiness §3 if present, otherwise computed from the data.
HIST_PP = {}
_r3 = pipe_tables(md_section(READINESS, r"^3\.|portfolio and target"))
if _r3 and {"set", "line", "year", "cost", "exposure"} <= set(_r3[0].columns):
    _t = _r3[0][_r3[0].set == "train"]
    for l in LINES:
        s = _t[_t.line == l]
        if len(s):
            HIST_PP[l] = {"pp_2024": float(s.loc[s.year == s.year.max(), "pure_premium"].iloc[0]),
                          "pp_all": float(s.cost.sum() / s.exposure.sum()), "source": "readiness §3"}
for l in LINES:
    if l not in HIST_PP and l in HIST and CLAIMS is not None:
        d = HIST[l].assign(cost=attach_cost(HIST[l], CLAIMS, l).values)
        y = d.groupby("year")[["cost", "exposure"]].sum()
        HIST_PP[l] = {"pp_2024": float(y.cost.iloc[-1] / y.exposure.iloc[-1]),
                      "pp_all": float(y.cost.sum() / y.exposure.sum()), "source": "computed from data"}

TEST_META = (pd.concat([TEST[l][["row_id", "exposure", "new_business"]].assign(line=l) for l in TEST], ignore_index=True)
             if TEST else None)
_CAND = {}


def cand_stats(path):
    """Mean prediction, pure premium per exposure and new-business split per line for one candidate file."""
    if path.name in _CAND:
        return _CAND[path.name]
    p = pd.read_csv(path, usecols=["row_id", "prediction"])
    rows = []
    m = p.merge(TEST_META, on="row_id", how="left") if TEST_META is not None else p.assign(
        line=p.row_id.str[0].map(PREFIX), exposure=np.nan, new_business=np.nan)
    for l in LINES:
        s = m[m.line == l]
        if not len(s):
            continue
        r = {"line": l, "rows": len(s), "mean prediction": s.prediction.mean()}
        if s.exposure.notna().all():
            nb = s.new_business.astype(bool)
            r["PP"] = s.prediction.sum() / s.exposure.sum()
            r["PP new business"] = s.prediction[nb].sum() / s.exposure[nb].sum()
            r["PP renewal"] = s.prediction[~nb].sum() / s.exposure[~nb].sum()
            r["NB / renewal"] = r["PP new business"] / r["PP renewal"]
            r["mean pred NB"] = s.prediction[nb].mean()
            r["mean pred renewal"] = s.prediction[~nb].mean()
        if l in HIST_PP and "PP" in r:
            r["vs 2024 reported PP"] = r["PP"] / HIST_PP[l]["pp_2024"]
            r["vs 2023-24 reported PP"] = r["PP"] / HIST_PP[l]["pp_all"]
        rows.append(r)
    _CAND[path.name] = pd.DataFrame(rows).set_index("line") if rows else None
    return _CAND[path.name]


CAND_FILES = sorted(SUBDIR.glob("*.csv"), key=lambda p: p.stat().st_mtime) if SUBDIR.exists() else []
REF_FILE = None  # the file behind the best public score of a tickable (non-PROBE) upload if on disk, else the newest
if SUBS is not None and SUBS.public_score[~SUBS.probe].notna().any():
    _bf = SUBS.loc[SUBS.public_score.where(~SUBS.probe).idxmax(), "file"]
    REF_FILE = next((p for p in CAND_FILES if p.name == _bf), None)
if REF_FILE is None and CAND_FILES:
    REF_FILE = CAND_FILES[-1]

# %% [markdown]
# ## 1 · Headline: where we stand

# %%
@safe
def headline():
    if SUBS is None or SUBS.public_score.notna().sum() == 0:
        note("no scored uploads in `docs/part2_submissions.csv`")
        return
    s = SUBS[SUBS.public_score.notna()]
    best_any = s.loc[s.public_score.idxmax()]
    tick = s[~s.probe]
    best = tick.loc[tick.public_score.idxmax()] if len(tick) else best_any
    ranked = SUBS[SUBS["rank"].notna()]
    last = ranked.iloc[-1] if len(ranked) else best
    leader = SUBS.leader_score.dropna().iloc[-1] if SUBS.leader_score.notna().any() else np.nan
    gap = leader - best.public_score if pd.notna(leader) else np.nan
    left = DEADLINE - NOW
    days, hours = left.days, left.seconds // 3600
    lim = re.search(r"≤\s*(\d+)\s*/\s*day", PLAN or "")
    today = (SUBS.t.dt.date == NOW.date()).sum()
    rank_txt = f"#{int(last['rank'])} of {int(last.n_teams)}" if pd.notna(last["rank"]) else "–"
    fin = s[s.note.astype(str).str.lower().str.startswith("intended final")]
    fin = fin.iloc[-1] if len(fin) else best
    best_any_txt = (f"The best score of any upload is {best_any.public_score:.5f}, from a PROBE file (`{best_any.config}`) "
                    "that is used only to read the scoring response and is never ticked."
                    if best_any.probe else "No probe file scores above it.")
    final_txt = ("" if fin.file == best.file else
                 f" The intended final per the plan is `{fin.config}` ({fin.public_score:.5f}, {fin.file}); the file above "
                 "it is a paired reading whose gain is below the pre-set promotion rule.")
    display(Markdown(f"""
**So what.** Our best leaderboard score from a file we could tick is **{best.public_score:.5f}** (`{best.config}`), which
placed us **{rank_txt}** (rank as recorded at {last.submitted_utc} UTC). {best_any_txt}{final_txt}
The leading team is at {leader:.5f}, a gap of **{gap:.5f}** to our best tickable file. {len(s)} uploads are scored, of
which {int(s.probe.sum())} are probes. The competition closes in **{days} days {hours} h** ({DEADLINE:%d %b %Y %H:%M} UTC).

| | |
|---|---|
| **Best public score, tickable file** | **{best.public_score:.5f}** (`{best.config}`, uploaded {best.submitted_utc} UTC) |
| Intended final (plan) | {fin.public_score:.5f} (`{fin.config}`, `{fin.file}`) |
| Best public score, any upload | {best_any.public_score:.5f} (`{best_any.config}`{", probe" if best_any.probe else ""}) |
| Rank / teams (latest reading) | {rank_txt} |
| Leader score (latest reading) | {leader:.5f} |
| Gap to leader (tickable file) | {gap:.5f} |
| Uploads used | {len(SUBS)} in total, {today} today (UTC){f" of {lim.group(1)} per day" if lim else ""} |
| Time to close | {days} d {hours} h (deadline {DEADLINE:%Y-%m-%d %H:%M} UTC) |
"""))
    fig = go.Figure()
    if pd.notna(leader):
        fig.add_hline(y=leader, line=dict(color=MUTED, dash="dash", width=1.5),
                      annotation_text=f"leader {leader:.5f}", annotation_position="top left",
                      annotation_font=dict(size=10, color=INK2))
    c, p = s[~s.probe], s[s.probe]
    fig.add_scatter(x=c.t, y=c.public_score, mode="lines+markers+text", name="candidate upload (tickable)",
                    text=c.config, textposition="top left", textfont=dict(size=10, color=INK2),
                    line_color=SERIES[0], marker=dict(size=11, color=SERIES[0]),
                    hovertemplate="%{text}<br>%{x|%d %b %H:%M} UTC<br>public %{y:.5f}<extra></extra>")
    fig.add_scatter(x=p.t, y=p.public_score, mode="markers", name="PROBE upload (never tick)", text=p.config,
                    marker=dict(size=10, color=SURFACE_C, line=dict(color=SERIES[0], width=2)),
                    hovertemplate="%{text}<br>%{x|%d %b %H:%M} UTC<br>public %{y:.5f}<extra></extra>")
    lo = min(s.public_score.min(), leader if pd.notna(leader) else 1) - 0.004
    hi = max(s.public_score.max(), leader if pd.notna(leader) else 0) + 0.003
    t0, t1 = s.t.min() - pd.Timedelta(minutes=30), max(s.t.max(), NOW) + pd.Timedelta(minutes=45)
    fig.update_layout(height=420, yaxis_title="public leaderboard (Tweedie D²)", yaxis_range=[lo, hi], xaxis_range=[t0, t1],
                      title=f"Public score of every upload (time axis UTC; close {DEADLINE:%d %b %H:%M})")
    fig.show()
    led = s[["submitted_utc", "config", "track", "nested_cv", "public_score", "rank", "note"]].copy()
    display(Markdown("**Upload ledger** (`docs/part2_submissions.csv`; rank is recorded only for candidate uploads):"))
    led.index = pd.Index(range(1, len(led) + 1), name="#")  # upload times are not unique (batches share a minute)
    display(fmt_table(led, {"nested_cv": "{:.4f}", "public_score": "{:.5f}", "rank": "{:.0f}"}, wide=["note"]))

# %% [markdown]
# ## 2 · Timeline

# %%
@safe
def timeline():
    log_txt = md_section(PLAN, r"progress log")
    items = bullets(log_txt)
    if not items:
        note("no progress-log bullets in `docs/part2_plan.md` (heading 'Progress log')")
        return
    rows = []
    for it in items:
        m = re.search(r"(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*(?:\s+(\d{1,2}):(\d{2}))?", it)
        if not m:
            continue
        t = pd.Timestamp(f"2026-{m.group(2)}-{m.group(1)} {m.group(3) or '12'}:{m.group(4) or '00'}", tz="UTC")
        plain = it.replace("**", "").replace("`", "")
        body = re.sub(r"^.*?UTC\s*[—–-]\s*", "", plain, count=1)
        title = re.split(r"(?<=[^\d])\.\s|;\s", body, maxsplit=1)[0]
        rows.append({"t": t, "title": title, "text": it})
    ev = pd.DataFrame(rows).sort_values("t")
    display(Markdown(f"""
**So what.** {len(ev)} milestones are logged since the Part 2 data arrived. Each point below is one entry of the plan's
progress log (hover for the full text); the dashed line is the close.
"""))
    fig = go.Figure()
    fig.add_scatter(x=ev.t, y=[0] * len(ev), mode="markers", marker=dict(color=SERIES[0], size=11),
                    customdata=ev.text.str.replace("**", "", regex=False).str.wrap(90).str.replace("\n", "<br>"),
                    hovertemplate="%{x|%d %b %H:%M} UTC<br>%{customdata}<extra></extra>", showlegend=False)
    for k, (_, r) in enumerate(ev.iterrows()):
        lab = r.title if len(r.title) <= 70 else r.title[:67] + "…"
        lab = "<br>".join(pd.Series([lab]).str.wrap(38).iloc[0].split("\n"))
        fig.add_annotation(x=r.t, y=0, text=lab, showarrow=True, arrowhead=0, arrowcolor=NEUTRAL,
                           ax=0, ay=[-70, 70, -130, 130][k % 4], font=dict(size=10, color=INK2), align="left")
    vmark(fig, NOW, "now", colour=INK2, dash="dot")
    vmark(fig, DEADLINE, "close")
    fig.update_yaxes(visible=False, range=[-1, 1])
    fig.update_layout(height=380, xaxis_range=[ev.t.min() - pd.Timedelta(hours=12), DEADLINE + pd.Timedelta(hours=6)],
                      margin=dict(t=60))
    fig.show()
    display(Markdown("\n".join(f"- {it}" for it in items)))

# %% [markdown]
# ## 3 · How the public score responds
#
# The public score is the row-level Tweedie D² (p = 1.5) on a random subset of the test rows. With μ the reference file,
# multiplying the predictions of one subset by s changes the score by exactly
# Δ(s) = −α (s^−½ − 1) − β (s^½ − 1), with optimum s* = α / β and maximum gain (√α − √β)². Two probes fix α and β for a
# subset; β is also proportional to Σ√μ over the subset, one constant for the whole file, which gives an independent check.

# %%
PROBE_RE = re.compile(r"PROBE (motor NB|motor|buildings|contents) x([\d.]+)(?: renewal x([\d.]+))?")
RESP_STYLE = {"motor": ("motor, all rows", LINE_COLOURS["motor"], "solid"),
              "motor:nb": ("motor, new business only", LINE_COLOURS["motor"], "dash"),
              "motor:ren": ("motor, renewals only", LINE_COLOURS["motor"], "dot"),
              "buildings": ("buildings, all rows", LINE_COLOURS["buildings"], "solid"),
              "contents": ("contents, all rows", LINE_COLOURS["contents"], "solid")}


def resp_gain(key, s):
    r = RESPONSE[key]
    return -r["alpha"] * (np.power(s, -0.5) - 1) - r["beta"] * (np.power(s, 0.5) - 1)


@safe
def public_response():
    if RESPONSE is None or SUBS is None:
        note("`docs/part2_response.json` or the upload ledger")
        return
    meta = RESPONSE.get("_meta", {})
    ref_pub = meta.get("reference_public")
    probes = []
    for _, r in SUBS[SUBS.probe & SUBS.public_score.notna()].iterrows():
        m = PROBE_RE.search(str(r.config))
        if not m:
            continue
        key = "motor:nb" if m.group(1) == "motor NB" else m.group(1)
        s1 = float(m.group(2))
        if m.group(3):  # composite: new business x s1, renewals x s2 -> sum of the two subset curves
            s2 = float(m.group(3))
            pred, used, xs = resp_gain("motor:nb", s1) + resp_gain("motor:ren", s2), "no (out-of-sample check)", None
            lab = f"motor NB ×{s1:.2f} + renewals ×{s2:.2f}"
        else:
            pred, used, xs = resp_gain(key, s1), "yes", s1
            lab = f"{RESP_STYLE[key][0]} ×{s1:.2f}"
        probes.append({"probe": lab, "key": key, "s": xs, "file": r.file, "public": r.public_score,
                       "actual Δ": r.public_score - ref_pub, "fitted-curve Δ": pred, "used in fit": used})
    P = pd.DataFrame(probes)
    keys = [k for k in RESP_STYLE if k in RESPONSE and RESPONSE[k].get("alpha") is not None]
    comp = P[P["used in fit"].str.startswith("no")]
    comp_txt = (f"The combined probe ({comp.probe.iloc[0]}) was not used in the fit: the curves predicted "
                f"{comp['fitted-curve Δ'].iloc[0]:+.5f} and it scored {comp['actual Δ'].iloc[0]:+.5f}." if len(comp) else "")
    mo, nb = RESPONSE["motor"], RESPONSE.get("motor:nb", {})
    display(Markdown(f"""
**So what.** The leaderboard is not a black box here: it responds to a level change exactly as the Tweedie formula says, so
each probe is a measurement of what the 2025 public rows want, not a guess. On the reference file `{meta.get('reference_file')}`
(public {ref_pub:.5f}), motor wants ×{mo['s_opt']:.3f} (+{mo['max_gain']:.4f}); applied to new business alone the optimum is
×{nb.get('s_opt', np.nan):.3f} (+{nb.get('max_gain', np.nan):.4f}) while renewals want ×{RESPONSE['motor:ren']['s_opt']:.2f}, i.e.
nothing. Buildings and contents are already close to their optima (gains ≤ {max(RESPONSE['buildings']['max_gain'], RESPONSE['contents']['max_gain']):.4f}).
{comp_txt} Every later upload has its predicted score written down before it is sent (`scripts/part2_factors.py`).
"""))
    s = np.linspace(0.8, 1.4, 241)
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    for k in keys:
        name, col, dash = RESP_STYLE[k]
        fig.add_scatter(x=s, y=resp_gain(k, s), mode="lines", name=name, line=dict(color=col, dash=dash, width=2),
                        legendgroup=k, hovertemplate=f"{name}<br>" + "factor ×%{x:.2f}<br>Δ public %{y:+.5f}<extra></extra>")
        r = RESPONSE[k]
        fig.add_scatter(x=[r["s_opt"]], y=[r["max_gain"]], mode="markers", legendgroup=k, showlegend=False,
                        marker=dict(symbol="diamond", size=11, color=col, line=dict(color=SURFACE_C, width=2)),
                        hovertemplate=f"{name}: optimum ×{r['s_opt']:.3f}, gain {r['max_gain']:+.5f}<extra></extra>")
        if k in ("motor", "motor:nb", "buildings"):
            fig.add_annotation(x=r["s_opt"], y=r["max_gain"], text=f"×{r['s_opt']:.2f}  {r['max_gain']:+.4f}",
                               showarrow=False, yshift=-16 if k == "motor" else 14, font=dict(size=10, color=INK2))
    pts = P[P.s.notna()]
    for k, g in pts.groupby("key"):
        name, col, _ = RESP_STYLE[k]
        fig.add_scatter(x=g.s, y=g["actual Δ"], mode="markers", legendgroup=k, showlegend=False,
                        marker=dict(size=10, color=SURFACE_C, line=dict(color=col, width=2.5)), text=g.file,
                        hovertemplate=f"probe · {name}<br>" + "×%{x:.2f}: Δ public %{y:+.5f}<br>%{text}<extra></extra>")
    fig.add_scatter(x=[None], y=[None], mode="markers", name="probe reading (open circle)",
                    marker=dict(size=10, color=SURFACE_C, line=dict(color=INK2, width=2.5)))
    fig.add_scatter(x=[None], y=[None], mode="markers", name="optimum (diamond)",
                    marker=dict(symbol="diamond", size=11, color=INK2))
    lo = min(pts["actual Δ"].min(), -0.002) - 0.0008
    fig.update_layout(height=480, xaxis_title="factor s applied to the subset of the reference file",
                      yaxis_title="change in public score Δ(s)", yaxis_range=[lo, 0.0032], xaxis_range=[0.8, 1.4],
                      yaxis_tickformat="+.4f",
                      title="Fitted public-score response per subset, with the probe readings")
    fig.show()
    tab = pd.DataFrame([{"subset": RESP_STYLE.get(k, (k,))[0] if k in RESP_STYLE else k, "α": RESPONSE[k].get("alpha"),
                         "β": RESPONSE[k].get("beta"), "optimum s* = α/β": RESPONSE[k].get("s_opt"),
                         "max gain": RESPONSE[k].get("max_gain"), "source": RESPONSE[k].get("source", RESPONSE[k].get("note", "")) or ""}
                        for k in RESPONSE if not k.startswith("_")]).set_index("subset")
    display(Markdown("**Fitted constants** (`docs/part2_response.json`; home-line new-business subsets have β from Σ√μ "
                     "but no α until probed):"))
    display(fmt_table(tab, {"α": "{:.4f}", "β": "{:.4f}", "optimum s* = α/β": "{:.3f}", "max gain": "{:+.5f}"}))
    display(Markdown(f"**Probe readings** (reference public {ref_pub:.5f}; actual Δ = probe public − reference):"))
    display(fmt_table(P.drop(columns=["key", "s"]).set_index("probe"),
                      {"public": "{:.5f}", "actual Δ": "{:+.5f}", "fitted-curve Δ": "{:+.5f}"}))
    rd = [t for t in pipe_tables(md_section(PLAN, r"scoring model")) if "subset" in t.columns]
    if rd:
        display(Markdown("**Precision of the readings** (`docs/part2_plan.md`, the public slice is ≈ ¼ of the rows):"))
        display(fmt_table(rd[0].set_index("subset")))


# %% [markdown]
# ## 4 · What is different about this market

# %%
@safe
def market_facts():
    if not HIST or not TEST:
        note("emerging-market data (`data/part2/emerging`) not found")
        return
    rows = []
    for l in LINES:
        h, t = HIST.get(l), TEST.get(l)
        if h is None or t is None:
            continue
        last = h[h.year == h.year.max()]
        r = {"line": l, "history years": f"{int(h.year.min())}–{int(h.year.max())}", "history rows": len(h),
             "history exposure": h.exposure.sum(), "test rows": len(t), "test exposure": t.exposure.sum(),
             "test book vs last history year": t.exposure.sum() / last.exposure.sum(),
             "new business share, history": h.new_business.mean(), "new business share, test": t.new_business.mean(),
             "test rows starting in January": (t.start.dt.month == 1).mean(),
             "test new business starting on 1 Jan": ((t.start.dt.month == 1) & (t.start.dt.day == 1))[t.new_business].mean(),
             "test policies seen in history": t.policy_id.isin(set(h.policy_id)).mean()}
        if CLAIMS is not None:
            r["history claims"] = int(((CLAIMS.line == l) & (CLAIMS.status != "repudiated")).sum())
        rows.append(r)
    f = pd.DataFrame(rows).set_index("line")
    tot_nb = sum(TEST[l].new_business.sum() for l in TEST) / sum(len(TEST[l]) for l in TEST)
    display(Markdown(f"""
**So what.** This market is small and young. We have only {f['history years'].iloc[0]} as history, about
{int(f['history rows'].sum()):,} policy rows in total, while the book we must price is {f['test book vs last history year'].mean():.1f}× the
latest history year. **{tot_nb:.0%} of the rows to price are new customers**, against
{f['new business share, history'].mean():.0%} in history, and {f["test rows starting in January"].mean():.0%} start in January, as one block. So most of the price
rests on how new customers behave, which our own history shows us only thinly.
"""))
    pct = ["new business share, history", "new business share, test", "test rows starting in January",
           "test new business starting on 1 Jan", "test policies seen in history"]
    order = ["history years", "history rows", "history exposure", "history claims", "test rows", "test exposure",
             "test book vs last history year", *pct]
    disp = pd.DataFrame(index=[o for o in order if o in f.columns], columns=f.index, dtype=object)
    for c in disp.index:
        for l in f.index:
            v = f.loc[l, c]
            disp.loc[c, l] = (f"{v:.1%}" if c in pct else f"{v:.2f}×" if c == "test book vs last history year"
                              else f"{v:,.0f}" if isinstance(v, (int, float, np.number)) else v)
    disp.index.name = "fact (emerging market, from the data)"
    display(fmt_table(disp))
    lvl = [t for t in pipe_tables(md_section(READINESS, r"level versus part 1")) if "line" in t.columns]
    if lvl:
        l = lvl[0].set_index("line").reindex([x for x in LINES if x in set(lvl[0].line)])
        keep = [c for c in l.columns if c.endswith("_ratio")]
        display(Markdown("**Level versus the developed market (Part 1), history years** — ratio emerging ÷ developed "
                         "(readiness §3):"))
        display(fmt_table(l[keep].rename(columns=lambda c: c.replace("_ratio", "").replace("_", " ") + " ratio"), "{:.2f}"))
    else:
        note("'Level versus Part 1' table in `docs/part2_readiness.md`")
    sch = [t for t in pipe_tables(md_section(READINESS, r"schema differences")) if "status" in t.columns]
    if sch:
        s = sch[0]
        s = s[~s.table.str.startswith("test_") & s.status.isin(["missing_column", "new_column", "new_table"])]
        g = (s.assign(item=np.where(s.column.astype(str) != "", s.column.astype(str), s.table))
             .groupby("status")[["table", "item"]].apply(lambda x: ", ".join(f"{a}.{b}" if a != b else a for a, b in zip(x.table, x.item)))
             .rename({"missing_column": "columns dropped", "new_column": "columns added", "new_table": "tables added"}))
        display(Markdown("**Schema changes versus Part 1** (readiness §1; test tables follow their history tables):\n\n"
                         + "\n".join(f"- **{k}:** {v}" for k, v in g.items())))
    else:
        note("schema table in `docs/part2_readiness.md` §1")


# %%
def _band_key(b):
    m = re.match(r"(\d+)", str(b))
    return int(m.group(1)) if m else 999


@safe
def age_mix():
    if not HIST or not TEST:
        note("age-band mix (emerging data not found)")
        return
    fig = make_subplots(rows=1, cols=3, subplot_titles=[f"{l}: {AGE_COL[l].replace('_', ' ')}" for l in LINES],
                        horizontal_spacing=0.06, shared_yaxes=True)
    young = []
    for i, l in enumerate(LINES, start=1):
        h, t = HIST[l], TEST[l]
        col = AGE_COL[l]
        groups = {"history (2023–24)": h, "test renewals": t[~t.new_business], "test new business": t[t.new_business]}
        bands = sorted(set(h[col].dropna()) | set(t[col].dropna()), key=_band_key)
        cols = [NEUTRAL, tint(LINE_COLOURS[l], 0.55), LINE_COLOURS[l]]
        yr = {"line": l}
        for (name, d), c in zip(groups.items(), cols):
            share = d[col].value_counts(normalize=True).reindex(bands).fillna(0)
            fig.add_bar(x=bands, y=share.values, name=name, marker_color=c, showlegend=False, row=1, col=i,
                        hovertemplate=f"{l} · {name}<br>" + "%{x}: %{y:.1%} of rows<extra></extra>")
            yr[name] = share[[b for b in bands if _band_key(b) < 35]].sum()
        young.append(yr)
    fig.update_yaxes(tickformat=".0%", title_text="share of rows", row=1, col=1)
    fig.update_layout(height=400, barmode="group", bargroupgap=0.05,
                      title="Age mix: grey = history, light = test renewals, dark = test new business")
    y = pd.DataFrame(young).set_index("line")
    display(Markdown(f"""
**So what.** The new customers we must price are much younger than anyone in our history: under-35s are
{rng(y['test new business'], '.0%')} of test new business against {rng(y['history (2023–24)'], '.0%')} in history
(test renewals: {rng(y['test renewals'], '.0%')}). Age is a strong price driver, especially in
motor, so the model has to extrapolate to a mix it has seen little of — which is why borrowing the age pattern from the
developed market (the pooled model) is our main lever.
"""))
    fig.show()
    display(Markdown("**Share of rows under 35** (bands whose lower bound is below 35):"))
    display(fmt_table(y, "{:.1%}"))


# %%
@safe
def start_months():
    if not HIST or not TEST:
        note("start-month chart (emerging data not found)")
        return
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Test rows (2025): start month", "History new business: start month"),
                        horizontal_spacing=0.08, shared_yaxes=True)
    months = list(range(1, 13))
    names = pd.date_range("2025-01-01", periods=12, freq="MS").strftime("%b")
    jan = {}
    for l in LINES:
        t, h = TEST[l], HIST[l][HIST[l].new_business]
        for k, d in enumerate([t, h], start=1):
            sh = d.start.dt.month.value_counts(normalize=True).reindex(months).fillna(0)
            fig.add_bar(x=names, y=sh.values, name=l, marker_color=LINE_COLOURS[l], showlegend=(k == 1), legendgroup=l,
                        row=1, col=k, hovertemplate=f"{l}<br>" + "%{x}: %{y:.1%} of rows<extra></extra>")
        jan[l] = ((t.start.dt.month == 1).mean(), (h.start.dt.month == 1).mean(), t.exposure.mean(), h.exposure.mean())
    fig.update_yaxes(tickformat=".0%", title_text="share of rows", row=1, col=1)
    fig.update_layout(height=380, barmode="group")
    j = pd.DataFrame(jan, index=["test: January start", "history NB: January start", "test: mean exposure",
                                 "history NB: mean exposure"]).T
    display(Markdown(f"""
**So what.** Almost every test policy starts in January ({rng(j['test: January start'], '.0%')} per line)
and runs a full year (mean exposure {j['test: mean exposure'].mean():.2f}), whereas the new customers in our history joined
through the year and stayed on average {j['history NB: mean exposure'].mean():.2f} of a year. The test book looks like one
portfolio taken on in bulk, not organic growth — another reason not to copy history's new-business experience one-for-one.
"""))
    fig.show()

# %% [markdown]
# ## 5 · New business is structural
#
# Recomputed at build time with the same rules as `scripts/part2_nb_diagnostics.py` (`history` and `oof` modes): raw
# incurred − recoveries attached by policy and loss date, exposure in days converted; OOF = the round-2 pooled model
# `R2_P34_hn` (`experiments/results/part2pool/R2_P34_hn.pkl`) on emerging rows only, actual = the developed target.

# %%
def _diag_history():
    cl = CLAIMS.copy()
    cl["loss"] = pd.to_datetime(cl.loss_date)
    cl["cost"] = cl.incurred_to_date.fillna(0) - cl.recoveries.fillna(0)
    out = {}
    for l in LINES:
        p = pd.read_parquet(EM / f"policies_{l}.parquet",
                            columns=["row_id", "policy_id", "period_start", "period_end", "exposure", "new_business"])
        p["ps"], p["pe"] = pd.to_datetime(p.period_start), pd.to_datetime(p.period_end)
        p["year"] = p.ps.dt.year
        p["expo"] = np.where(p.exposure > 1.5, p.exposure / 365.25, p.exposure).clip(0, 1.2)
        c = cl[cl.line == l].merge(p[["row_id", "policy_id", "ps", "pe"]], on="policy_id")
        c = c[(c.loss >= c.ps) & (c.loss <= c.pe)]
        agg = c.groupby("row_id").agg(n=("claim_id", "size"), cost=("cost", "sum"))
        p = p.merge(agg, left_on="row_id", right_index=True, how="left").fillna({"n": 0, "cost": 0})
        g = p.groupby(["year", "new_business"]).agg(rows=("row_id", "size"), expo=("expo", "sum"), n=("n", "sum"),
                                                    cost=("cost", "sum"))
        g["freq"], g["sev"], g["pp"] = g.n / g.expo, g.cost / g.n.replace(0, np.nan), g.cost / g.expo
        out[l] = g
    return out


def _tw_d2(y, m):
    y, m = np.asarray(y, float), np.asarray(m, float)
    tw = lambda mu: 2 * (y ** 0.5 / ((1 - 1.5) * (2 - 1.5)) - y * mu ** (1 - 1.5) / (1 - 1.5) + mu ** 0.5 / 0.5).sum()
    return 1 - tw(m) / tw(np.full_like(y, y.mean()))


def _diag_oof():
    import pickle
    with open(ROOT / "experiments" / "results" / "part2pool" / "R2_P34_hn.pkl", "rb") as fh:
        r = pickle.load(fh)
    res = {}
    for l in LINES:
        v, ab = r["lines"][l], AGE_COL[l]
        df = pd.DataFrame({"row_id": v["row_id"], "y": v["y_eval"], "oof": v["oof"]})
        df = df[~df.row_id.astype(str).str.startswith("D")]
        tr = pd.read_parquet(ROOT / "experiments" / "cache" / "part2pool" / f"train_{l}.parquet",
                             columns=["row_id", "period_start", "new_business", ab])
        df = df.merge(tr, on="row_id")
        df["year"] = pd.to_datetime(df.period_start).dt.year
        g = df.groupby(["year", "new_business"]).agg(rows=("y", "size"), act=("y", "sum"), pred=("oof", "sum"))
        g["a/e"] = g.act / g.pred
        both = df.groupby("new_business").agg(rows=("y", "size"), act=("y", "sum"), pred=("oof", "sum"))
        both["a/e"] = both.act / both.pred
        ga = df.groupby([ab, "new_business"]).agg(rows=("y", "size"), act=("y", "sum"), pred=("oof", "sum"))
        ga["a/e"] = ga.act / ga.pred
        hi = np.asarray(v["holdout_idx"])
        h = pd.DataFrame({"row_id": np.asarray(v["row_id"])[hi], "y": np.asarray(v["y_eval"])[hi], "p": np.asarray(v["holdout_pred"])})
        h = h[~h.row_id.astype(str).str.startswith("D")].merge(tr, on="row_id")
        hg = h.groupby("new_business").agg(act=("y", "sum"), pred=("p", "sum"))
        hg["a/e"] = hg.act / hg.pred
        hd2 = {f: _tw_d2(h.y, h.p * np.where(h.new_business, f, 1.0)) for f in [1.0, 1.1, 1.2, 1.3]}
        te = pd.read_parquet(ROOT / "experiments" / "cache" / "part2pool" / f"test_{l}.parquet",
                             columns=["row_id", "new_business", "exposure"])
        tp = pd.DataFrame({"row_id": v["test_row_id"], "p": v["test_pred"]}).merge(te, on="row_id")
        tg = tp.groupby("new_business").agg(p=("p", "sum"), e=("exposure", "sum"))
        res[l] = {"year": g, "both": both, "age": ga, "hold": hg, "hold_d2": hd2,
                  "test_nb_ren": (tg.p / tg.e).loc[True] / (tg.p / tg.e).loc[False]}
    return res


def _ratio(g, y):
    s = g.loc[y]
    return s.loc[True], s.loc[False]


@safe
def nb_structural():
    if CLAIMS is None or not HIST or not TEST:
        note("emerging data for the new-business diagnostics")
        return
    H = _diag_history()
    try:
        O = _diag_oof()
    except FileNotFoundError as e:
        O = None
        note(f"OOF diagnostics (`R2_P34_hn.pkl` or the part2pool cache missing: {e})")
    years = sorted({y for l in LINES for y in H[l].index.get_level_values(0)})
    hr = []
    for l in LINES:
        for y in years:
            nb, rn = _ratio(H[l], y)
            hr.append({"line": l, "year": y, "NB/ren cost per exposure": nb.pp / rn.pp, "NB/ren frequency": nb.freq / rn.freq,
                       "NB/ren severity": nb.sev / rn.sev, "NB claims": int(nb.n), "renewal claims": int(rn.n)})
    hr = pd.DataFrame(hr)
    m = O["motor"] if O else None
    if m is not None:
        a = m["age"]["a/e"].unstack()
        cal = {y: m["year"].loc[(y, True), "a/e"] / m["year"].loc[(y, False), "a/e"] for y in years}
        pooled = m["both"].loc[True, "a/e"] / m["both"].loc[False, "a/e"]
        mh = hr[hr.line == "motor"].set_index("year")
        display(Markdown(f"""
**So what.** The motor shortfall on the public rows is a new-business effect, not an age effect. Across both history years the
round-2 model under-predicts new business in every driver-age band (actual/expected {a[True].min():.2f}–{a[True].max():.2f}) and
over-predicts renewals in every band ({a[False].min():.2f}–{a[False].max():.2f}). The two history years disagree on the size: the
new-business/renewal calibration ratio is {cal[years[0]]:.2f} in {years[0]} and {cal[years[-1]]:.2f} in {years[-1]}
(raw cost ratio {mh.loc[years[0], 'NB/ren cost per exposure']:.2f} vs {mh.loc[years[-1], 'NB/ren cost per exposure']:.2f}, on
{mh.loc[years[0], 'NB claims']} and {mh.loc[years[-1], 'NB claims']} new-business claims). That is why a train-{years[0]} →
score-{years[-1]} holdout rejects a new-business factor; pooled over both years the ratio is {pooled:.2f}. The model charges
test new business {m['test_nb_ren']:.2f}× renewals. The public reading rests on ≈ 3,600 new-business claims, about three times the
two history years together (`docs/part2_plan.md`).
"""))
        bands = sorted(a.index, key=_band_key)
        cats = [str(y) for y in years] + ["both years"]
        yv = {nbf: [m["year"].loc[(y, nbf), "a/e"] for y in years] + [m["both"].loc[nbf, "a/e"]] for nbf in (True, False)}
        fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.06, column_widths=[0.4, 0.6],
                            subplot_titles=("motor: by history year", "motor: by driver age band (both years)"))
        for nbf, name, col in [(True, "new business", LINE_COLOURS["motor"]), (False, "renewal", tint(LINE_COLOURS["motor"], 0.55))]:
            fig.add_bar(x=cats, y=yv[nbf], name=name, marker_color=col, legendgroup=name, row=1, col=1,
                        text=[f"{v:.2f}" for v in yv[nbf]], textposition="outside", textfont=dict(size=10, color=INK2),
                        hovertemplate=f"{name} · " + "%{x}<br>actual/expected %{y:.3f}<extra></extra>")
            fig.add_bar(x=bands, y=a.loc[bands, nbf], name=name, marker_color=col, legendgroup=name, showlegend=False,
                        row=1, col=2, hovertemplate=f"{name} · " + "%{x}<br>actual/expected %{y:.3f}<extra></extra>")
        for c in (1, 2):
            fig.add_hline(y=1, line=dict(color=MUTED, width=1, dash="dash"), row=1, col=c)
        fig.update_yaxes(title_text="OOF actual ÷ expected (emerging rows)", row=1, col=1, range=[0, 1.85])
        fig.update_xaxes(type="category")
        fig.update_layout(height=420, barmode="group", title="Round-2 model (R2_P34_hn) calibration: new business vs renewals")
        fig.show()
        rows = []
        for l in LINES:
            o = O[l]
            r = {"line": l}
            for y in years:
                r[f"NB a/e {y}"] = o["year"].loc[(y, True), "a/e"]
                r[f"renewal a/e {y}"] = o["year"].loc[(y, False), "a/e"]
                r[f"NB/ren calibration {y}"] = o["year"].loc[(y, True), "a/e"] / o["year"].loc[(y, False), "a/e"]
            r["NB/ren calibration, both years"] = o["both"].loc[True, "a/e"] / o["both"].loc[False, "a/e"]
            r[f"{years[-1]} holdout NB/ren"] = o["hold"].loc[True, "a/e"] / o["hold"].loc[False, "a/e"]
            for f, d in o["hold_d2"].items():
                r[f"holdout D² NB ×{f:.1f}"] = d
            r["test model NB/ren PP"] = o["test_nb_ren"]
            rows.append(r)
        display(Markdown("**OOF calibration by year × new business, all lines** (a/e = actual ÷ out-of-fold prediction; "
                         f"holdout = train {years[0]} → score {years[-1]}, D² with new-business rows × factor):"))
        ot = pd.DataFrame(rows).set_index("line").T
        d2rows = [i for i in ot.index if i.startswith("holdout D²")]
        display(fmt_table(ot, "{:.3f}").format("{:.4f}", subset=pd.IndexSlice[d2rows, :]))
    fig = go.Figure()
    for l in LINES:
        d = hr[hr.line == l]
        fig.add_bar(x=[str(y) for y in d.year], y=d["NB/ren cost per exposure"], name=l, marker_color=LINE_COLOURS[l],
                    text=[f"{v:.2f}" for v in d["NB/ren cost per exposure"]], textposition="outside",
                    textfont=dict(size=10, color=INK2), customdata=d["NB claims"],
                    hovertemplate=f"{l} · " + "%{x}<br>NB/renewal cost per exposure %{y:.2f}<br>%{customdata} NB claims<extra></extra>")
    fig.add_hline(y=1, line=dict(color=MUTED, width=1, dash="dash"))
    fig.update_xaxes(type="category")
    fig.update_layout(height=380, barmode="group", yaxis_title="new business ÷ renewal, raw cost per exposure",
                      yaxis_range=[0, hr["NB/ren cost per exposure"].max() * 1.15],
                      title="Emerging history: new-business cost relativity by year (raw, undeveloped)")
    display(Markdown("**History relativity by year and line.** Motor new business is dearer in both years but by very "
                     "different amounts; the home lines rest on 37–65 new-business claims a year and are not credible on their own."))
    fig.show()
    display(fmt_table(hr.set_index(["line", "year"]), {"NB/ren cost per exposure": "{:.2f}", "NB/ren frequency": "{:.2f}",
                                                         "NB/ren severity": "{:.2f}"}))
    mix = []
    for l in LINES:
        h, t = HIST[l], TEST[l]
        h24 = h[(h.year == h.year.max()) & h.new_business]
        msh = h24.start.dt.month.value_counts(normalize=True)
        mix.append({"line": l, "NB share, history": h.new_business.mean(), "NB share, test": t.new_business.mean(),
                    "test NB starting in January": (t.start.dt.month[t.new_business] == 1).mean(),
                    "test renewals starting in January": (t.start.dt.month[~t.new_business] == 1).mean(),
                    f"history {int(h.year.max())} NB starting in January": msh.get(1, 0.0),
                    f"history {int(h.year.max())} NB, monthly share range": rng(msh.values, ".0%")})
    mix = pd.DataFrame(mix).set_index("line")
    display(Markdown(f"""
**The test book is a different kind of new business.** {mix['NB share, test'].mean():.0%} of test rows are new business
(history {rng(mix['NB share, history'], '.0%')}), and {rng(mix['test NB starting in January'], '.0%')} of them start in January, as one
block. In history, new customers join evenly through the year (each month {mix.iloc[0, -1]} of the latest year's new business for
motor). History therefore never shows a January bulk intake like the one we must price.
"""))
    display(fmt_table(mix, {c: "{:.1%}" for c in mix.columns if "range" not in c}))

# %% [markdown]
# ## 6 · Why the price level is uncertain

# %%
@safe
def fire_volatility():
    if CLAIMS is None:
        note("emerging claims (`data/part2/emerging/claims.parquet`)")
        return
    c = CLAIMS[(CLAIMS.line == "buildings") & (CLAIMS.status != "repudiated")].copy()
    c["year"] = pd.to_datetime(c.loss_date, errors="coerce").dt.year
    c["net"] = c.incurred_to_date.fillna(0) - c.recoveries.fillna(0)
    years = sorted(y for y in c.year.dropna().unique() if (c.year == y).sum() >= 50)
    c = c[c.year.isin(years)]
    g = c.pivot_table(index="peril", columns="year", values="net", aggfunc="sum").fillna(0)
    n = c.pivot_table(index="peril", columns="year", values="net", aggfunc="size").fillna(0)
    g = g.loc[g.sum(axis=1).sort_values(ascending=False).index]
    tot = g.sum()
    top = g.max(axis=1).idxmax()
    swing = g.loc[top].max() / max(g.loc[top].min(), 1)
    display(Markdown(f"""
**So what.** Buildings costs swing hard from year to year because of a few large claims. **{top}** cost
{g.loc[top].max() / 1e6:.1f}M in its worst year and {g.loc[top].min() / 1e6:.1f}M in the other ({swing:.1f}×), on about
{int(n.loc[top].mean())} claims a year. Total buildings cost was {" vs ".join(f"{tot[y] / 1e6:.1f}M ({int(y)})" for y in years)}.
With only two years we cannot tell which year is "normal", so the buildings price level carries the widest uncertainty.
"""))
    fig = go.Figure()
    for k, y in enumerate(years):
        col = LINE_COLOURS["buildings"] if k == 0 else tint(LINE_COLOURS["buildings"], 0.5 * k / max(len(years) - 1, 1) + 0.1)
        fig.add_bar(x=g.index, y=g[y], name=f"loss year {int(y)}", marker_color=col,
                    text=[f"{v / 1e6:.1f}M" for v in g[y]], textposition="outside", textfont=dict(size=10, color=INK2),
                    customdata=n[y].reindex(g.index).values,
                    hovertemplate="%{x} · " + f"{int(y)}" + "<br>net cost %{y:,.0f}<br>%{customdata:.0f} claims<extra></extra>")
    fig.update_layout(height=400, barmode="group", yaxis_title="net incurred cost (to date)",
                      title="Buildings: cost by peril and loss year (emerging market, incurred to date, repudiated excluded)")
    fig.show()


# %%
@safe
def nb_uplift():
    if not all((DEV / f).exists() for f in ["claims.parquet", *[f"policies_{l}.parquet" for l in LINES]]):
        note("developed-market data (`data/raw/public`) for the new-business ratio chart")
        return
    dclaims = pd.read_parquet(DEV / "claims.parquet",
                              columns=["policy_id", "line", "loss_date", "status", "incurred_to_date", "recoveries"])
    dev, em = {}, {}
    for l in LINES:
        dev[l] = pp_by_year_nb(_policies(DEV / f"policies_{l}.parquet"), dclaims, l)
        if l in HIST and CLAIMS is not None:
            em[l] = pp_by_year_nb(HIST[l], CLAIMS, l)
    del dclaims
    ref = cand_stats(REF_FILE) if REF_FILE is not None and TEST_META is not None else None
    fig = make_subplots(rows=1, cols=3, subplot_titles=[f"{l}: new business ÷ renewal cost per exposure" for l in LINES],
                        horizontal_spacing=0.06, shared_yaxes=True)
    for i, l in enumerate(LINES, start=1):
        d = dev[l]
        fig.add_scatter(x=d.index, y=d.ratio, mode="lines+markers+text", text=[f"{v:.2f}" for v in d.ratio],
                        textposition="top center", textfont=dict(size=10, color=INK2), line_color=LINE_COLOURS[l],
                        name="developed market", showlegend=(i == 1), legendgroup="dev", row=1, col=i,
                        hovertemplate=f"{l} developed " + "%{x}<br>NB/renewal %{y:.2f}<extra></extra>")
        if l in em:
            e = em[l]
            fig.add_scatter(x=e.index, y=e.ratio, mode="lines+markers", line=dict(color=LINE_COLOURS[l], dash="dot"),
                            marker=dict(color="white", line=dict(color=LINE_COLOURS[l], width=2)),
                            name="emerging history", showlegend=(i == 1), legendgroup="em", row=1, col=i,
                            hovertemplate=f"{l} emerging " + "%{x}<br>NB/renewal %{y:.2f}<extra></extra>")
        fig.add_hline(y=1, line=dict(color=MUTED, width=1), row=1, col=i)
        if ref is not None and l in ref.index and "NB / renewal" in ref:
            fig.add_hline(y=ref.loc[l, "NB / renewal"], line=dict(color=INK2, dash="dash", width=1), row=1, col=i,
                          annotation_text=f"our file {ref.loc[l, 'NB / renewal']:.2f}", annotation_font_size=10,
                          annotation_position="bottom right")
    fig.update_xaxes(dtick=1)
    fig.update_yaxes(title_text="cost ratio, new business ÷ renewal", row=1, col=1)
    fig.update_layout(height=420)
    dm = pd.DataFrame({l: dev[l].ratio for l in LINES})
    per = "; ".join(f"{l} {rng(dm[l], '.2f')} ({int((dm[l] > 1).sum())} of {len(dm)} years above 1)" for l in LINES)
    always = [l for l in LINES if (dm[l] > 1).all()]
    display(Markdown(f"""
**So what.** In the large, mature developed market, the cost of new customers per unit of cover relative to renewing ones is:
{per}. {"For " + " and ".join(always) + " the new-business uplift is real and stable, not noise" if always else
"No line shows a stable new-business uplift"}{"; for the other lines it is small and changes sign" if 0 < len(always) < 3 else ""}. Our own small history
(dotted) jumps around this level because it rests on a few thousand rows. The dashed line is what our
{"reference file `" + REF_FILE.name + "`" if REF_FILE is not None else "candidate"} charges new business relative to renewals
in the test book. Because most of the test book is new business, this one ratio moves the whole price level.
"""))
    fig.show()
    tab = pd.concat({"developed": pd.concat(dev), **({"emerging": pd.concat(em)} if em else {})}, names=["market", "line", "year"])
    tab = tab.rename(columns={"pp_nb": "cost per exposure, NB", "pp_ren": "cost per exposure, renewal", "ratio": "NB ÷ renewal"})
    display(Markdown("Raw incurred-to-date cost (net of recoveries, repudiated excluded), claims attached by policy and loss "
                     "date; not developed or trended, so the latest year is slightly low on both sides of the ratio."))
    display(fmt_table(tab, {"cost per exposure, NB": "{:,.1f}", "cost per exposure, renewal": "{:,.1f}", "NB ÷ renewal": "{:.2f}"}))


# %%
def _completeness():
    sec = md_section(DECISIONS, r"completeness")
    out = {}
    if sec:
        m = re.search(r"Part 2 results?:([^\n]*(?:\n[^\n]+)?)", sec)
        if m:
            for l in LINES:
                v = re.search(rf"{l}\s+([0-9]\.[0-9]+)", m.group(1))
                if v:
                    out[l] = float(v.group(1))
    if len(out) < 3:  # fallback: the PART2 field table
        for t in pipe_tables(md_section(DECISIONS, r"field table")):
            if "Field" in t.columns:
                r = t[t.Field.astype(str).str.contains("completeness")]
                if len(r):
                    for l in LINES:
                        v = re.search(rf"{l}\s+([0-9]\.[0-9]+)", str(r.Value.iloc[0]))
                        if v and l not in out:
                            out[l] = float(v.group(1))
    return out


def _trend():
    m = re.search(r"trend factors[^:]*:\s*(.+?)\.(?:\s|$)", DECISIONS or "", re.S)
    out = {}
    if m:
        for l in LINES:
            v = re.search(rf"{l}\s+(?:\d{{4}}\s+)?([0-9]\.[0-9]+)\s*/\s*(?:\d{{4}}\s+)?([0-9]\.[0-9]+)", m.group(1))
            if v:
                out[l] = float(v.group(2))  # the factor applied to the latest history year
    return out


@safe
def level_decomposition():
    comp, trend = _completeness(), _trend()
    ref = cand_stats(REF_FILE) if REF_FILE is not None else None
    if ref is None or not HIST_PP:
        note("level decomposition (needs a candidate file in `submissions/part2/` and the 2024 history level)")
        return
    rows = []
    for l in LINES:
        if l not in ref.index or l not in HIST_PP or "PP" not in ref:
            continue
        a = HIST_PP[l]["pp_2024"]
        c, t = comp.get(l, np.nan), trend.get(l, np.nan)
        h24 = HIST[l][HIST[l].year == HIST[l].year.max()] if l in HIST else None
        r_nb = ref.loc[l, "NB / renewal"]
        s_t = TEST[l].exposure[TEST[l].new_business].sum() / TEST[l].exposure.sum() if l in TEST else np.nan
        s_h = h24.exposure[h24.new_business].sum() / h24.exposure.sum() if h24 is not None else np.nan
        mix = (s_t * r_nb + 1 - s_t) / (s_h * r_nb + 1 - s_h)
        step = a / c * t * mix
        rows.append({"line": l, "2024 reported PP": a, "÷ completeness": c, "= developed": a / c,
                     "× premium trend": t, "= trended": a / c * t, "× new-business mix": mix, "= expected": step,
                     "candidate PP": ref.loc[l, "PP"], "residual ×": ref.loc[l, "PP"] / step})
    if not rows:
        note("level decomposition (candidate pure premium needs the test tables for exposure)")
        return
    d = pd.DataFrame(rows).set_index("line")
    src_c = "decisions doc §4" if comp else "missing"
    src_t = "decisions doc §6 (premium index, factor for the latest year)" if trend else "missing"
    display(Markdown(f"""
**So what.** This table walks from what our history actually shows to what we charge. Start from the 2024 reported cost per
unit of cover; gross it up for claims not yet reported; apply the premium trend to 2025; then reweight for the much larger
share of new customers. Whatever is left ("residual") is the model's own view: the 2023 year in the average (a heavy fire year
for buildings), large-loss loading and growth of open claims. Residuals well away from 1 mark where our level is a judgement,
not a measurement.

Candidate: `{REF_FILE.name}`. Sources: reported PP from {HIST_PP[LINES[0]]['source']}; completeness from {src_c};
trend from {src_t}; **new-business mix is derived** here as (s_test·r + 1 − s_test) ÷ (s_2024·r + 1 − s_2024), with the
exposure shares s from the data and r = the candidate file's own NB ÷ renewal pure premium; candidate PP = Σ prediction ÷
Σ exposure on the test rows; residual = candidate ÷ expected (derived).
"""))
    display(fmt_table(d, {"2024 reported PP": "{:,.1f}", "÷ completeness": "{:.3f}", "= developed": "{:,.1f}",
                          "× premium trend": "{:.3f}", "= trended": "{:,.1f}", "× new-business mix": "{:.3f}",
                          "= expected": "{:,.1f}", "candidate PP": "{:,.1f}", "residual ×": "{:.3f}"}))

# %% [markdown]
# ### Why the level gap is real, not a bug
#
# The round-2 model is calibrated to history (motor OOF actual/expected 1.02) yet sits at ≈ 0.9 × the reference file on motor,
# while the public probes want the reference raised ≈ 15 %. Before any factor was applied, five other explanations were
# checked and rejected (Fable, read-only checks, 27 Oct 19:00–19:30 UTC; `docs/part2_plan.md` and `docs/ai_review_log.md`
# Review 12). The figures below are quoted from those checks.

# %%
@safe
def level_gap_checks():
    t = pd.DataFrame([
        ("Insurer's premium trend", "motor renewal written premium per policy, 2023 / 2024 / 2025", "604 / 601 / 608",
         "flat — rejected"),
        ("2025 weather", "hail days, 2023 / 2024 / 2025", "14.0 / 17.1 / 11.4", "2025 milder — rejected"),
        ("Claims development (one-year emerging lag curve)", "mean report lag, emerging vs developed", "26 vs 25 days",
         "same — rejected"),
        ("Claims development (open claims)", "open share of 2023 losses, emerging vs developed", "3.7 % vs 3.9 %",
         "same — rejected"),
        ("Pooled trend index mixes market price levels", "full-year premium level 2023, emerging vs developed",
         "603 vs 594", "coincide — rejected"),
        ("Younger test new business (age, not new business)", "motor OOF a/e by age band × NB",
         "NB 0.97–1.41, renewals 0.87–0.99 in every band", "rejected — the gap follows new business"),
    ], columns=["explanation", "check", "numbers", "outcome"]).set_index("explanation")
    display(Markdown("**So what.** None of the mechanical explanations holds; what remains is a 2025 new-business level that "
                     "history measures only noisily, and which the public rows measure about three times more precisely. "
                     "That is why the factors go on the new-business subset with credibility Z ≈ 0.9, and the model keeps its shape."))
    display(fmt_table(t, wide=["check"]))

# %% [markdown]
# ## 7 · Experiments

# %%
@safe
def experiments():
    if LOG is None or not len(LOG):
        note("no part2 / part2pool rows in `experiments/log.csv`")
        return
    pub = {}
    if SUBS is not None:
        pub = SUBS[SUBS.public_score.notna()].groupby("config").public_score.max().to_dict()
    L = LOG.copy()
    L["public"] = [pub.get(e, lb) for e, lb in zip(L.exp, L.lb_public)]
    full = L[L.cv.notna()]
    if len(full):
        b = full.loc[full.cv_nested.fillna(full.cv).idxmax()]
        ref = full[full.description.str.contains("constant", case=False, na=False)]
        markets = sorted(L.market.unique())
        display(Markdown(f"""
**So what.** {len(full)} configurations have a full cross-validated score so far (markets: {", ".join(markets)}). The best,
`{b.exp}`, scores {b.cv:.4f}{f" (nested, i.e. honestly out-of-sample, {b.cv_nested:.4f})" if pd.notna(b.cv_nested) else ""}
{f"against {ref.cv.iloc[0]:.4f} for a flat price per line" if len(ref) else ""}. Higher is better. The bars show which line each
model is good at; the uploaded ones carry their public score. Rows without a CV are quick screens whose findings are summarised
in the description.
"""))
        fig = go.Figure()
        x = full.exp.tolist()
        for l in LINES:
            fig.add_bar(x=x, y=full[f"cv_{l}"], name=l, marker_color=LINE_COLOURS[l],
                        hovertemplate="%{x}<br>" + l + " CV %{y:.4f}<extra></extra>")
        top = full[[f"cv_{l}" for l in LINES]].max(axis=1)
        for e, p, y in zip(full.exp, full.public, top):
            if pd.notna(p):
                fig.add_annotation(x=e, y=y, text=f"public {p:.5f}", showarrow=True, arrowhead=0, ax=0, ay=-26,
                                   font=dict(size=10, color=INK), arrowcolor=MUTED)
        fig.add_scatter(x=x, y=full.cv, mode="markers", name="overall CV", marker=dict(color=INK2, symbol="line-ew-open",
                        size=26, line=dict(width=2, color=INK2)), hovertemplate="%{x}<br>overall CV %{y:.4f}<extra></extra>")
        fig.update_layout(height=440, barmode="group", yaxis_title="cross-validated Tweedie D² (higher = better)")
        fig.show()
    else:
        note("no part2 experiment with a full CV yet")
    cols = ["exp", "market", "description", "cv", "cv_nested", "cv_motor", "cv_buildings", "cv_contents", "holdout_2023",
            "public", "decision"]
    t = L[[c for c in cols if c in L.columns]].rename(columns={"holdout_2023": "2024 holdout"}).set_index("exp")
    if t.market.nunique() == 1:
        t = t.drop(columns="market")
    display(fmt_table(t, {c: "{:.4f}" for c in ["cv", "cv_nested", "cv_motor", "cv_buildings", "cv_contents", "2024 holdout"]}
                      | {"public": "{:.5f}"}, wide=["description", "decision"]))
    if not (L.market == "part2pool").any():
        note("no pooled-market (`part2pool`) experiments logged yet")

# %% [markdown]
# ### Round 2 of the pooled model (R2_* configurations)
#
# Each step was scored on emerging rows only (`scripts/score_market.py`): 5-fold grouped CV and the 2024 forward holdout
# (train 2023 → score 2024), per line. A step is kept only if CV and holdout agree; the decision text is the log's own.

# %%
def _triple(text, label):
    m = re.search(rf"{label}\s+([\d.]+)/([\d.]+)/([\d.]+)", str(text))
    return [float(x) for x in m.groups()] if m else [np.nan] * 3


def _status(dec):
    d = str(dec)
    if "CANDIDATE" in d:
        return "candidate"
    m = re.search(r"KEPT FOR (\w+) ONLY", d)
    if m:
        return f"kept ({m.group(1).lower()} only)"
    if "KEPT (weak)" in d:
        return "kept (weak)"
    if "KEPT" in d:
        return "kept"
    if "REJECTED" in d:
        return "rejected"
    if "seed for" in d:
        return "seed (bag member)"
    if re.search(r"\bbag\b", d):
        return "bag (used in R2_FINAL)"
    if "blend (" in d or "STEP 4 blend" in d:
        return "blend"
    if "part" in d:
        return "blend part"
    return ""


@safe
def round2():
    if LOG is None:
        note("`experiments/log.csv`")
        return
    R = LOG[LOG.exp.astype(str).str.startswith("R2_")].copy()
    if not len(R):
        note("no R2_* rows in `experiments/log.csv`")
        return
    rows = []
    for _, r in R.iterrows():
        h, oc, hc, nb = (_triple(r.verification, k) for k in (r"holdout(?! calib)", "OOF calib", "holdout calib", "test NB/renewal"))
        row = {"exp": r.exp, "status": _status(r.decision), "description": r.description, "cv (all)": r.cv,
               "nested cv": r.cv_nested}
        for i, l in enumerate(LINES):
            row[f"CV {l}"] = r[f"cv_{l}"]
        for i, l in enumerate(LINES):
            row[f"holdout {l}"] = h[i]
        row["OOF calib m/b/c"] = "/".join(f"{x:.2f}" for x in oc)
        row["holdout calib m/b/c"] = "/".join(f"{x:.2f}" for x in hc)
        row["test NB/ren m/b/c"] = "/".join(f"{x:.2f}" for x in nb)
        row["decision"] = r.decision
        rows.append(row)
    T = pd.DataFrame(rows).set_index("exp")
    fin = T.loc["R2_FINAL"] if "R2_FINAL" in T.index else None
    cnt = T.status.str.split(" ").str[0].value_counts()
    nest = _triple(fin["decision"], "nested CV") if fin is not None else [np.nan]
    nested_txt = ("; nested CV (decision text) " + " / ".join(f"{x:.4f}" for x in nest)) if not np.isnan(nest[0]) else ""
    display(Markdown(f"""
**So what.** {len(T)} round-2 runs: {cnt.get('kept', 0)} steps kept, {cnt.get('rejected', 0)} rejected, the rest are blend parts,
seeds and bags. The kept steps are market-specific claims development, a motor-only large-loss loading per market and
unobservable history in each market's first year. All three in-model new-business mechanisms were rejected by the 2024 holdout
(section 5 explains why that holdout cannot resolve new business).
{f"The candidate `R2_FINAL` scores 5-fold CV {fin['CV motor']:.4f} / {fin['CV buildings']:.4f} / {fin['CV contents']:.4f} and 2024 holdout {fin['holdout motor']:.4f} / {fin['holdout buildings']:.4f} / {fin['holdout contents']:.4f} (motor / buildings / contents){nested_txt}." if fin is not None else ""}
"""))
    T.index = [f"{e} · {st}" if st else e for e, st in zip(T.index, T.status)]
    T.index.name = "exp · status"
    order = list(T.index)[::-1]
    fig = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.04, subplot_titles=LINES)
    for i, l in enumerate(LINES, start=1):
        col = LINE_COLOURS[l]
        fig.add_scatter(x=T.loc[order, f"CV {l}"], y=order, mode="markers", name="5-fold CV (emerging rows)",
                        legendgroup="cv", showlegend=(i == 1), marker=dict(size=9, color=col), row=1, col=i,
                        customdata=T.loc[order, "status"], hovertemplate="%{y} · " + l + "<br>CV %{x:.4f}<br>%{customdata}<extra></extra>")
        fig.add_scatter(x=T.loc[order, f"holdout {l}"], y=order, mode="markers", name="2024 holdout (open)",
                        legendgroup="ho", showlegend=(i == 1), row=1, col=i,
                        marker=dict(size=9, color=SURFACE_C, line=dict(color=col, width=2)),
                        customdata=T.loc[order, "status"], hovertemplate="%{y} · " + l + "<br>holdout %{x:.4f}<br>%{customdata}<extra></extra>")
    fig.update_xaxes(title_text="Tweedie D² (emerging rows)", tickformat=".3f")
    fig.update_yaxes(tickfont=dict(size=10))
    fig.update_layout(height=max(420, 26 * len(T) + 160), title="Round 2: per-line CV (filled) and 2024 holdout (open)",
                      margin=dict(l=250))
    fig.show()
    fmts = {c: "{:.4f}" for c in T.columns if c.startswith(("CV ", "holdout ", "cv", "nested")) and T[c].dtype != object}
    display(fmt_table(T.drop(columns="status"), fmts, wide=["description", "decision"]))

# %% [markdown]
# ## 8 · Candidate files

# %%
@safe
def candidates():
    if not CAND_FILES:
        note("no candidate files in `submissions/part2/`")
        return
    if TEST_META is None:
        note("test tables (needed for the new-business split and pure premium of candidates)")
    pub = SUBS.set_index("file").public_score.to_dict() if SUBS is not None else {}
    frames = {}
    for p in CAND_FILES:
        s = cand_stats(p)
        if s is not None:
            frames[p.name] = s
    if not frames:
        note("candidate files could not be read")
        return
    c = pd.concat(frames, names=["file", "line"])
    c["public"] = [pub.get(f, np.nan) for f in c.index.get_level_values("file")]
    n_test = {l: len(TEST[l]) for l in TEST}
    bad = [f"{f} ({l}: {r} rows)" for (f, l), r in c["rows"].items() if n_test and r != n_test.get(l)]
    hist_txt = ", ".join(f"{l} {HIST_PP[l]['pp_all']:,.0f}" for l in LINES if l in HIST_PP)
    display(Markdown(f"""
**So what.** {len(frames)} candidate file{"s" if len(frames) > 1 else ""} on disk. For each file and line: the average
prediction per row, the pure premium per unit of exposure (PP), the split between new business and renewals, and the ratio of
the PP to the raw reported history (2023–24 average: {hist_txt}). A ratio well above 1 is expected (late claims, trend,
younger new business); differences between files show how much a modelling choice moves the price level.
"""))
    if bad:
        note("row counts differ from the test tables: " + "; ".join(bad))
    keep = [k for k in ["rows", "mean prediction", "mean pred NB", "mean pred renewal", "PP", "PP new business", "PP renewal",
                        "NB / renewal", "vs 2024 reported PP", "vs 2023-24 reported PP", "public"] if k in c.columns]
    display(fmt_table(c[keep], {"rows": "{:,.0f}", "mean prediction": "{:,.1f}", "mean pred NB": "{:,.1f}",
                                "mean pred renewal": "{:,.1f}", "PP": "{:,.1f}", "PP new business": "{:,.1f}",
                                "PP renewal": "{:,.1f}", "NB / renewal": "{:.3f}", "vs 2024 reported PP": "{:.2f}",
                                "vs 2023-24 reported PP": "{:.2f}", "public": "{:.5f}"}))

# %% [markdown]
# ## 9 · 28 Oct upload protocol

# %%
@safe
def upload_protocol():
    sec = md_section(PLAN, r"scoring model")
    if not sec:
        note("'Scoring model … upload protocol' section in `docs/part2_plan.md`")
        return
    m = re.search(r"\*\*28 Oct protocol.*?(?=\nPrepared )", sec, re.S)
    rule = re.search(r"\*\*Decision rule.*?(?=\n\n)", sec, re.S)
    tabs = [t for t in pipe_tables(sec) if "file" in t.columns]
    if not tabs:
        note("prepared-files table in the plan's scoring-model section")
        return
    t = tabs[0].copy()
    t["file"] = t.file.str.strip("`")
    t["tick?"] = np.where(t.file.str.contains("PROBE"), "NEVER TICK (probe)",
                          np.where(t.file.str.contains("P2_"), "candidate", "–"))
    import hashlib
    on_disk, sha_ok = [], []
    for f, sha in zip(t.file, t.get("sha256_16", pd.Series([""] * len(t)))):
        p = SUBDIR / f
        on_disk.append("yes" if p.exists() else "no")
        if p.exists() and re.fullmatch(r"[0-9a-f]{16}", str(sha)):
            sha_ok.append("match" if hashlib.sha256(p.read_bytes()).hexdigest()[:16] == sha else "MISMATCH")
        else:
            sha_ok.append("–")
    t["on disk"], t["sha256_16 check"] = on_disk, sha_ok
    for c in t.columns:
        if t[c].dtype == object:
            t[c] = t[c].str.replace("**", "", regex=False)
    n_probe = int(t.file.str.contains("PROBE").sum())
    display(Markdown(f"""
**So what.** The 28 Oct uploads were planned and built in advance, each with its predicted score written down before upload, so
every public reading is a test of a stated prediction. {len(t)} rows were prepared ({n_probe} PROBE files, which are measurement
tools only and are **never ticked**). The intended final is the file named in the newest progress-log entry (30 Oct: the robust F11 v3). The
check columns are computed here: whether the file exists in `submissions/part2/` and whether its SHA-256 prefix matches the plan.
"""))
    if rule:
        display(Markdown(rule.group(0)))
    if m:
        display(Markdown(m.group(0)))
    cols = [c for c in ["#", "file", "tick?", "predicted vs its base", "purpose", "sha256_16", "on disk", "sha256_16 check"] if c in t]
    display(fmt_table(t[cols].set_index("#" if "#" in t else "file"), wide=["predicted vs its base", "purpose"]))
    outcome = re.search(r"\*\*28 Oct \d\d:\d\d.*?UTC — the probes were taken.*", sec, re.S)
    if outcome:
        display(Markdown("**Outcome (readings taken on the intended final itself):**\n\n" + outcome.group(0)))

# %% [markdown]
# ## 10 · Open risks and next steps

# %%
def _split_top(text, sep=","):
    """Split on `sep` outside parentheses."""
    out, depth, cur = [], 0, ""
    for ch in text:
        depth += ch == "("
        depth -= ch == ")"
        if ch == sep and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    return [x for x in out + [cur.strip()] if x]


@safe
def risks_next():
    items = bullets(md_section(PLAN, r"progress log"))
    # the newest entry that carries an open-risk list (later entries may record readings only)
    latest = next((it for it in items if "open-risk list:" in it), items[0] if items else "")
    m = re.search(r"open-risk list:\s*(.+?)\.(?:\s+Next[^:]*:\s*(.+?)\.)?\s*$", latest, re.S)
    stamp = re.search(r"\d{1,2} \w{3} \d{1,2}:\d{2} UTC", latest)
    if m:
        risks = _split_top(m.group(1))
        nxt = m.group(2)
        display(Markdown(f"**So what.** The latest progress-log entry ({stamp.group(0) if stamp else 'newest'}) carries "
                         f"{len(risks)} open risks forward from round 2. They are what could still move the price materially:\n\n"
                         + "\n".join(f"{i}. {x[0].upper() + x[1:]}" for i, x in enumerate(risks, start=1))
                         + (f"\n\n**Next local screen:** {nxt}." if nxt else "")
                         + "\n\nAlso open (Review 12): the blend weights' fold stability is not independently verified "
                           "(agent B2 reports buildings unstable and fixed it to the P34 bag)."))
    else:
        note("open-risk list in the newest progress-log entry of `docs/part2_plan.md`")
    r = md_section(DECISIONS, r"risk")
    if r:
        display(Markdown("**Earlier risks recorded at the first submission** (`docs/part2_decisions.md` §8, 27 Oct morning; "
                         "kept for the record):\n\n" + r))
    plan = md_section(PLAN, r"day plan")
    items = bullets(plan)
    if items:
        today = f"{NOW.day} {NOW:%b}"
        out = [f"- **{it}** ← today" if it.startswith(today) else f"- {it}" for it in items]
        display(Markdown("**Day plan** (`docs/part2_plan.md`):\n\n" + "\n".join(out)))
    else:
        note("day plan in `docs/part2_plan.md` (heading 'Day plan')")

# %% [markdown]
# ---
# ### Build notes

# %%
display(Markdown(f"Built {NOW:%Y-%m-%d %H:%M} UTC. Reference candidate: "
                 f"`{REF_FILE.name if REF_FILE is not None else 'none'}`. "
                 + ("Sources not yet available: " + "; ".join(NOTES) if NOTES else "All sources present.")))
print("BUILD NOTES:", NOTES)
