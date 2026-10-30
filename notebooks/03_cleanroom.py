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
# # Clean-room reimplementation (IAA Hackathon 2026, Part 1)
#
# Independent build of target, exposure and a per-line LightGBM Tweedie model, using only
# `COMPETITION.md`, `docs/kaggle_pages.md` and `data/raw/public/`.
# Every interpretive choice is recorded in `reports/cleanroom/assumptions.md`.

# %%
import time
import warnings
import zlib
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_tweedie_deviance

warnings.filterwarnings("ignore")
T0 = time.time()
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
RAW = ROOT / "data" / "raw" / "public"
OUT = ROOT / "reports" / "cleanroom"
OUT.mkdir(parents=True, exist_ok=True)

LINES = ["motor", "buildings", "contents"]
VALUATION = pd.Timestamp("2024-12-31")  # kaggle_pages: "valued at 31 December 2024"
POWER = 1.5
N_FOLDS = 5
SEED = 42

clean_log = []  # (rule, rows_affected)
assumptions = []  # (title, text, source)


def log(rule, n):
    clean_log.append((rule, int(n)))
    print(f"[clean] {rule}: {int(n):,}")


def assume(title, text, source):
    assumptions.append((title, text, source))


# %% [markdown]
# ## 1. Load and clean
# Identifiers are text (data-description: "Read them as text, not numbers"). Parquet dates arrive as
# `datetime.date`; they are parsed with `errors="coerce"` so invalid dates would be counted.

# %%
claims = pd.read_parquet(RAW / "claims.parquet")
customers = pd.read_parquet(RAW / "customers.parquet")
addresses = pd.read_parquet(RAW / "addresses.parquet")
weather = pd.read_parquet(RAW / "weather.parquet")
pol = {l: pd.read_parquet(RAW / f"policies_{l}.parquet") for l in LINES}
test = {l: pd.read_parquet(RAW / f"test_{l}.parquet") for l in LINES}
print({l: pol[l].shape for l in LINES}, claims.shape)

# %% [markdown]
# ### 1a. Key hygiene
# Motor `customer_id` values have lost their leading zeros (the register in COMPETITION.md warns
# "ids that lost their leading zeros"). `customers.customer_id` is always 8 characters, so we left-pad
# every policy `customer_id` to 8 and count the orphans before/after.

# %%
for l in LINES:
    for df, nm in ((pol[l], "train"), (test[l], "test")):
        before = (~df.customer_id.isin(customers.customer_id)).sum()
        df["customer_id"] = df.customer_id.str.zfill(8)
        after = (~df.customer_id.isin(customers.customer_id)).sum()
        log(f"{l} {nm}: customer_id left-padded to 8 chars (orphans before={before:,}, after={after:,})",
            before)
        log(f"{l} {nm}: address_id not found in addresses (orphan)", (~df.address_id.isin(addresses.address_id)).sum())
        log(f"{l} {nm}: duplicate row_id", df.row_id.duplicated().sum())
    log(f"{l} train: orphan claims (policy_id not in {l} policies)",
        (~claims[claims.line == l].policy_id.isin(pol[l].policy_id)).sum())

for l in LINES:
    for df, nm in ((pol[l], "train"), (test[l], "test")):
        df["ps"] = pd.to_datetime(df.period_start, errors="coerce")
        df["pe"] = pd.to_datetime(df.period_end, errors="coerce")
        log(f"{l} {nm}: invalid/unparseable period dates", df.ps.isna().sum() + df.pe.isna().sum())
        log(f"{l} {nm}: period_end < period_start", (df.pe < df.ps).sum())

claims["ld"] = pd.to_datetime(claims.loss_date, errors="coerce")
claims["rd"] = pd.to_datetime(claims.report_date, errors="coerce")
log("claims: invalid loss/report dates", claims.ld.isna().sum() + claims.rd.isna().sum())
log("claims: report_date < loss_date", (claims.rd < claims.ld).sum())
log("claims: report_date after valuation date", (claims.rd > VALUATION).sum())
log("claims: negative paid/incurred/recoveries",
    ((claims.paid_to_date < 0) | (claims.incurred_to_date < 0) | (claims.recoveries < 0)).sum())
log("claims: incurred < paid", (claims.incurred_to_date < claims.paid_to_date).sum())
log("claims: recoveries > incurred", (claims.recoveries > claims.incurred_to_date).sum())

# %% [markdown]
# ### 1b. Duplicate claims
# The register warns of "duplicate claims". There are no repeated `claim_id`s, but 604 rows are identical
# in every field other than `claim_id` (same policy, peril, both dates, status, amounts, repairer, third
# party). With 2-decimal amounts this cannot be coincidence, so the first row of each such group is kept.

# %%
log("claims: duplicate claim_id", claims.claim_id.duplicated().sum())
key_cols = [c for c in claims.columns if c not in ("claim_id", "ld", "rd")]
dup = claims.duplicated(subset=key_cols, keep="first")
log("claims: exact duplicates ignoring claim_id (dropped)", dup.sum())
claims = claims[~dup].copy()

# %% [markdown]
# ### 1c. Exposure
# `exposure` is "the length of the row in years" (data-description). In every line ~3-4k rows have
# exposure > 1 (values like 1.0139 or 327.0, i.e. days not years, or a 365/360 convention) which is
# impossible for a row covering at most a calendar year. For rows with exposure <= 1 the given value equals
# (period_end - period_start + 1)/365 to 4 dp (a full 2024 row runs 1-Jan to 30-Dec = 365 days).
# We therefore keep the supplied exposure when it is in (0, 1] and replace invalid values by
# min(1, days/365).

# %%
for l in LINES:
    for df, nm in ((pol[l], "train"), (test[l], "test")):
        days = (df.pe - df.ps).dt.days + 1
        bad = (df.exposure > 1.0001) | (df.exposure <= 0)
        log(f"{l} {nm}: exposure outside (0,1] replaced by min(1,days/365)", bad.sum())
        df["exposure_raw"] = df.exposure
        df["exposure"] = np.where(bad, np.minimum(1.0, days / 365.0), df.exposure)
        df["year"] = df.ps.dt.year

# %% [markdown]
# ### 1d. Feature-level cleaning
# * `sum_insured` x1000 in a small subset (register: "sum insured x1000"). Normal ranges are
#   buildings <= ~1M, contents <= ~0.3M; inflated values start at >= 3.6M / 3.8M. Divide by 1000 above
#   5M (buildings) or 1M (contents).
# * `construction_type`: mixed case plus the codes `W`/`T`. Documented classes are brick, stone, timber,
#   non-standard; lower-casing gives brick/stone/non_standard/w/t and `w`,`t` are mapped to timber.
# * `years_licensed` > driver_age - 16 is impossible (register: "impossible licence years"); set to missing.

# %%
for l, thr in (("buildings", 5e6), ("contents", 1e6)):
    for df, nm in ((pol[l], "train"), (test[l], "test")):
        m = df.sum_insured > thr
        log(f"{l} {nm}: sum_insured divided by 1000", m.sum())
        df.loc[m, "sum_insured"] = df.loc[m, "sum_insured"] / 1000.0
        assert df.sum_insured.max() < 1.5e6, "sum insured still has a scaling problem"

cmap = {"brick": "brick", "stone": "stone", "non_standard": "non_standard", "w": "timber", "t": "timber", "timber": "timber"}
for df, nm in ((pol["buildings"], "train"), (test["buildings"], "test")):
    raw = df.construction_type
    df["construction_type"] = raw.str.strip().str.lower().map(cmap)
    log(f"buildings {nm}: construction_type label normalised (not already canonical)",
        (raw != df.construction_type).sum())
    assert df.construction_type.notna().all()

for df, nm in ((pol["motor"], "train"), (test["motor"], "test")):
    m = df.years_licensed > df.driver_age - 16
    log(f"motor {nm}: impossible years_licensed (> age-16) set to NaN", m.sum())
    df.loc[m, "years_licensed"] = np.nan

# %% [markdown]
# ## 2. Target
# Wording relied on (Description page): "the final cost of all claims with a loss date in the row's period,
# net of the policy excess and net of recoveries, excluding declined claims".
# * Claim cost = `incurred_to_date - recoveries` (`paid_to_date` is already net of excess; `incurred` is paid
#   plus case reserve; for closed claims incurred == paid). Repudiated claims are excluded (their amounts are 0 anyway).
# * Attribution: claim -> the row of the same `policy_id` (in the table of the claim's `line`) with
#   `period_start <= loss_date <= period_end`. Rows of one policy never overlap so the match is unique.
#   Claims with no covering row (2020 losses, gaps after early cancellation) are dropped.
# * Development: claims reported after 31-Dec-2024 are missing (IBNR). We run a **chain-ladder on the
#   reporting-delay triangle** (loss month x months-to-report claim counts, per line x peril), take the
#   cumulative development factor for each loss month, and weight each reported claim's cost by that factor.
#   Open claims: no case-reserve history is provided, so the case estimate is taken as the best estimate
#   (cannot be developed further from a single snapshot).

# %%
cl = claims.copy()
log("claims: loss_date before 2021 (outside any policy period)", (cl.ld < "2021-01-01").sum())
n_rep = (cl.status == "repudiated").sum()
log("claims: repudiated (excluded from cost and count)", n_rep)

# --- chain-ladder on reporting-delay triangle (all non-duplicate claims incl. repudiated: they are reported too)
NM = 48  # Jan-2021 .. Dec-2024
KMAX = 36
cl["lm"] = (cl.ld.dt.year - 2021) * 12 + cl.ld.dt.month - 1
cl["rm"] = (cl.rd.dt.year - 2021) * 12 + cl.rd.dt.month - 1
cl["k"] = cl.rm - cl.lm
tri_src = cl[(cl.lm >= 0) & (cl.lm < NM)]
dev_rows = []
cl["dev_factor"] = 1.0
for (l, per), g in tri_src.groupby(["line", "peril"]):
    N = np.zeros((NM, NM))
    np.add.at(N, (g.lm.values, g.k.clip(upper=NM - 1).values), 1)
    C = N.cumsum(axis=1)
    f = np.ones(KMAX)
    for k in range(KMAX):
        ms = np.arange(0, NM - (k + 1))  # loss months observable at age k+1
        if C[ms, k].sum() > 0:
            f[k] = C[ms, k + 1].sum() / C[ms, k].sum()
    # cumulative factor from age a to ultimate; ages >= KMAX -> 1 (tail assumed fully reported)
    cdf = np.ones(NM + 1)
    for a in range(KMAX - 1, -1, -1):
        cdf[a] = cdf[a + 1] * f[a]
    ages = np.arange(NM)
    for a in ages:
        dev_rows.append((l, per, a, cdf[min(a, KMAX)]))
    sel = (cl.line == l) & (cl.peril == per)
    age = (NM - 1 - cl.loc[sel, "lm"]).clip(lower=0).astype(int)
    cl.loc[sel, "dev_factor"] = np.where(cl.loc[sel, "lm"] >= 0, cdf[np.minimum(age, KMAX)], 1.0)
pd.DataFrame(dev_rows, columns=["line", "peril", "age_months_at_valuation", "factor_to_ultimate"]).to_csv(
    OUT / "dev_factors.csv", index=False)
print(cl[cl.ld >= "2024-12-01"].groupby("line").dev_factor.mean().round(2).to_dict(), "(Dec-2024 mean factor)")
print("share of claims with report delay > %d months: %.5f" % (KMAX, (tri_src.k > KMAX).mean()))
print(cl.groupby(cl.ld.dt.year).dev_factor.mean().round(3).to_dict())

cl["cost_reported"] = (cl.incurred_to_date - cl.recoveries).clip(lower=0)
cl = cl[cl.status != "repudiated"].copy()
cl["cost_ult"] = cl.cost_reported * cl.dev_factor

# --- attribute to rows
parts = []
unmatched = 0
for l in LINES:
    p = pol[l][["row_id", "policy_id", "ps", "pe"]]
    c = cl[cl.line == l].merge(p, on="policy_id", how="left")
    ok = (c.ld >= c.ps) & (c.ld <= c.pe)
    n_match = c[ok].groupby("claim_id").size()
    assert n_match.max() == 1, "claim matched to >1 row"
    unmatched += c.claim_id.nunique() - len(n_match)
    parts.append(c[ok][["row_id", "claim_id", "cost_reported", "cost_ult", "dev_factor"]])
log("claims: no policy row covers the loss date (dropped)", unmatched)
cm = pd.concat(parts)
agg = cm.groupby("row_id").agg(target_reported=("cost_reported", "sum"), target=("cost_ult", "sum"),
                               n_claims=("claim_id", "size"), n_claims_ult=("dev_factor", "sum"))

frames = {}
for l in LINES:
    d = pol[l][["row_id", "policy_id", "customer_id", "address_id", "year", "exposure"]].merge(
        agg, left_on="row_id", right_index=True, how="left")
    for c_ in ["target_reported", "target", "n_claims", "n_claims_ult"]:
        d[c_] = d[c_].fillna(0.0)
    d["line"] = l
    frames[l] = d
ts = pd.concat(frames.values())
tsum = ts.groupby(["line", "year"]).agg(rows=("row_id", "size"), exposure_sum=("exposure", "sum"),
                                         target_sum=("target", "sum"), target_reported_sum=("target_reported", "sum"),
                                         claim_count=("n_claims", "sum"), claim_count_ultimate=("n_claims_ult", "sum")).reset_index()
tsum["target_per_exposure"] = tsum.target_sum / tsum.exposure_sum
tsum.to_csv(OUT / "target_summary.csv", index=False)
print(tsum.round(2).to_string())
print("largest single-claim ultimate costs:", cl.cost_ult.nlargest(3).round(0).tolist())

# %% [markdown]
# ## 3. Features
# Row features + customer attributes (age at period start computed from date of birth) + address +
# weather of the row's calendar year (`address_id` x `year`). Excluded: IDs, surname, e-mail domain,
# marketing flags (no causal story), `period_*`/calendar year (cannot extrapolate to 2025), and the two
# leakage columns named in COMPETITION.md: `customer_service_contacts` and `renewal_decision`.
# `written_premium` is kept: it is supplied for test rows and is not a post-event quantity.

# %%
cust_keep = ["customer_id", "date_of_birth", "occupation", "marital_status", "household_size", "credit_band",
             "prior_claims_5y", "licence_points", "years_at_address", "customer_since_year", "previous_insurer",
             "marketing_segment"]
cust = customers[cust_keep].copy()
cust["dob"] = pd.to_datetime(cust.date_of_birth, errors="coerce")
cust = cust.drop(columns="date_of_birth")
addr = addresses.copy()
wthr = weather.copy()

DROP = ["row_id", "policy_id", "customer_id", "address_id", "period_start", "period_end", "ps", "pe", "year",
        "exposure", "exposure_raw", "customer_service_contacts", "renewal_decision", "dob",
        "target", "target_reported", "n_claims", "n_claims_ult", "line"]


def build_features(df):
    x = df.merge(cust, on="customer_id", how="left").merge(addr, on="address_id", how="left")
    x = x.merge(wthr, on=["address_id", "year"], how="left")
    x["cust_age"] = (x.ps - x.dob).dt.days / 365.25
    x["building_age"] = x.year - x.year_built
    x["years_since_joined"] = x.year - x.customer_since_year
    if "driver_age" in x:
        x["licence_age_gap"] = x.driver_age - x.years_licensed
    cols = [c for c in x.columns if c not in DROP]
    return x, cols


# %% [markdown]
# ## 4. Model
# One LightGBM per line, `objective=tweedie`, `tweedie_variance_power=1.5`. Label = target/exposure (a rate),
# weight = exposure, so the prediction for a row is rate x exposure. Folds: 5-fold by `customer_id`
# (deterministic crc32 hash, so one customer - who may hold several policies, years and lines, and shares
# addresses - is never split across train/validation). Early stopping on the validation fold (slightly
# optimistic; the final model uses mean best iteration x 1.1). Metric: Tweedie D^2 (power 1.5) on
# row-level total cost vs prediction = rate x exposure, relative to a constant-mean null model.

# %%
PARAMS = dict(objective="tweedie", tweedie_variance_power=POWER, learning_rate=0.05, num_leaves=31,
              min_data_in_leaf=300, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=10.0, max_bin=127, num_threads=4, seed=SEED, verbose=-1)


def d2(y, p, w=None):
    p = np.clip(p, 1e-9, None)
    null = np.full_like(y, np.average(y, weights=w), dtype=float)
    return 1.0 - mean_tweedie_deviance(y, p, power=POWER, sample_weight=w) / mean_tweedie_deviance(
        y, null, power=POWER, sample_weight=w)


def fold_of(cid):
    return np.array([zlib.crc32(c.encode()) % N_FOLDS for c in cid])


oof_parts, test_parts, imp_parts, cv_rows = [], [], [], []
for l in LINES:
    t1 = time.time()
    tr = frames[l].merge(pol[l].drop(columns=["policy_id", "customer_id", "address_id", "year", "exposure"]),
                         on="row_id")
    tr, cols = build_features(tr)
    te = test[l].copy()
    te, _ = build_features(te)
    cat_cols = [c for c in cols if tr[c].dtype == object or str(tr[c].dtype) == "category"]
    for c in cat_cols:
        cats = pd.Index(pd.concat([tr[c], te[c]]).dropna().astype(str).unique())
        tr[c] = pd.Categorical(tr[c].astype(object).where(tr[c].notna(), None), categories=cats)
        te[c] = pd.Categorical(te[c].astype(object).where(te[c].notna(), None), categories=cats)
    for c in cols:  # bools -> int8 so LightGBM accepts them with NaN-free handling
        if tr[c].dtype == bool:
            tr[c] = tr[c].astype("int8"); te[c] = te[c].astype("int8")
    print(f"\n=== {l}: {len(cols)} features, {len(tr):,} train rows, {len(te):,} test rows")
    X = tr[cols]
    y = (tr.target / tr.exposure).values
    w = tr.exposure.values
    folds = fold_of(tr.customer_id.values)
    oof = np.zeros(len(tr))
    best_its = []
    for k in range(N_FOLDS):
        trn, val = folds != k, folds == k
        dtr = lgb.Dataset(X[trn], y[trn], weight=w[trn], categorical_feature=cat_cols, free_raw_data=False)
        dva = lgb.Dataset(X[val], y[val], weight=w[val], reference=dtr)
        m = lgb.train(PARAMS, dtr, num_boost_round=1500, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        best_its.append(m.best_iteration)
        oof[val] = m.predict(X[val], num_iteration=m.best_iteration) * tr.exposure.values[val]
        s = d2(tr.target.values[val], oof[val])
        cv_rows.append((l, f"fold{k}", s, m.best_iteration, int(val.sum())))
        print(f"  fold {k}: D2={s:.5f} best_iter={m.best_iteration}  ({time.time()-t1:.0f}s)")
    s_all = d2(tr.target.values, oof)
    cv_rows.append((l, "oof_all", s_all, int(np.mean(best_its)), len(tr)))
    top_idx = tr.target.values.argmax()
    keep = np.ones(len(tr), bool); keep[top_idx] = False
    cv_rows.append((l, "oof_excl_largest_row", d2(tr.target.values[keep], oof[keep]), 0, int(keep.sum())))
    cv_rows.append((l, "oof_reported_target", d2(tr.target_reported.values, oof), 0, len(tr)))
    for yr in sorted(tr.year.unique()):
        mk = (tr.year == yr).values
        cv_rows.append((l, f"oof_year{yr}", d2(tr.target.values[mk], oof[mk]), 0, int(mk.sum())))
    print(f"  {l} OOF D2 = {s_all:.5f}")
    oof_parts.append(pd.DataFrame({"row_id": tr.row_id.values, "line": l, "year": tr.year.values,
                                   "target": tr.target.values, "exposure": tr.exposure.values,
                                   "oof_pred": oof, "fold": folds}))
    # final fit
    n_final = int(np.mean(best_its) * 1.1)
    dall = lgb.Dataset(X, y, weight=w, categorical_feature=cat_cols)
    mf = lgb.train(PARAMS, dall, num_boost_round=n_final)
    pred = mf.predict(te[cols]) * te.exposure.values
    test_parts.append(pd.DataFrame({"row_id": te.row_id.values, "line": l, "prediction": np.clip(pred, 0, None)}))
    imp = pd.DataFrame({"line": l, "feature": cols, "gain": mf.feature_importance("gain")})
    imp["gain_share"] = imp.gain / imp.gain.sum()
    imp_parts.append(imp.sort_values("gain", ascending=False).head(20))
    print(f"  final rounds={n_final}; test pred sum={pred.sum():,.0f}, mean per exposure={pred.sum()/te.exposure.sum():.2f}"
          f"; train target per exposure={tr.target.sum()/tr.exposure.sum():.2f}  ({time.time()-t1:.0f}s)")
    del tr, te, X, dtr, dva, dall
    import gc; gc.collect()

# %% [markdown]
# ## 5. Save artefacts

# %%
oof_df = pd.concat(oof_parts)
oof_df.to_parquet(OUT / "oof.parquet", index=False)
pd.concat(test_parts).to_parquet(OUT / "test_pred.parquet", index=False)
pd.concat(imp_parts).to_csv(OUT / "feature_importance.csv", index=False)
cv = pd.DataFrame(cv_rows, columns=["line", "scope", "d2", "best_iter", "n_rows"])
tot = {"line": "ALL", "scope": "oof_all", "d2": d2(oof_df.target.values, oof_df.oof_pred.values),
       "best_iter": 0, "n_rows": len(oof_df)}
cv = pd.concat([cv, pd.DataFrame([tot])])
cv.to_csv(OUT / "cv_scores.csv", index=False)
pd.DataFrame(clean_log, columns=["rule", "rows_affected"]).to_csv(OUT / "cleaning_log.csv", index=False)
print(cv[cv.scope.isin(["oof_all", "oof_excl_largest_row", "oof_reported_target"])].round(5).to_string())

# %% [markdown]
# ## 6. Assumptions file

# %%
assume("Claim cost definition",
       "cost = incurred_to_date - recoveries (floored at 0); paid_to_date is already net of excess; for closed claims "
       "incurred == paid; for open claims the case estimate is included (incurred = paid + case reserve). No capping "
       "of large losses (largest single claim ~11.9M incurred is kept).",
       "\"the final cost of all claims with a loss date in the row's period, net of the policy excess and net of recoveries\" (Description); "
       "\"paid_to_date: Amount paid so far, net of excess\"; \"incurred_to_date: Paid to date plus the current case estimate for open claims\" (data-description)")
assume("Repudiated claims excluded",
       "status == 'repudiated' claims are removed from both cost and claim counts.",
       "\"excluding declined claims\" (Description); \"repudiated (declined)\" (data-description)")
assume("Attribution by loss date",
       "A claim belongs to the row with the same policy_id (in the table of the claim's line) whose [period_start, period_end] (both inclusive) contains loss_date. "
       "Report date is irrelevant to attribution. Claims with no covering row (2020 losses, post-cancellation gaps) are dropped.",
       "\"A policy keeps the same policy_id across years, so match each claim to a policy row using its loss_date\" (data-description); \"last day covered\" for period_end")
assume("Duplicate claims removed",
       "Claims identical in all fields except claim_id are treated as duplicates and the first is kept (604).",
       "\"Duplicate claims\" in the data-quality register (COMPETITION.md)")
assume("IBNR / development method",
       "Chain-ladder on a monthly reporting-delay count triangle (loss month x months to report), per line x peril, age-to-age factors "
       "from the volume-weighted pooled history of Jan-2021..Dec-2024 loss months, tail = 1 beyond 36 months. Each reported claim's cost "
       "is multiplied by the cumulative factor of its loss-month age at 31-Dec-2024. Count development is applied to cost (assumes severity "
       "of late-reported claims equals severity of reported ones). Valuation date 31-Dec-2024 (end of December). "
       "Open-claim case reserves are NOT developed (no transaction history available), i.e. case estimates assumed unbiased.",
       "\"Some 2024 claims had not been reported by then, and open claims may still develop\" (data-description); "
       "\"Ultimate: The final cost of claims once all are reported and settled\" (glossary)")
assume("Exposure",
       "Supplied exposure kept when in (0,1]; otherwise replaced by min(1, (period_end-period_start+1)/365). Row-level model: label = target/exposure, weight = exposure; "
       "prediction = rate x exposure. Test exposure used as supplied (all in (0,1]).",
       "\"exposure is the length of the row in years\" (data-description); \"exposure in days\" in the register (COMPETITION.md); "
       "\"Predictions are per row, so they scale with exposure\"")
assume("Customer-id repair",
       "Policy customer_id left-padded with zeros to 8 characters before joining customers (motor only needed it).",
       "\"ids that lost their leading zeros\" (COMPETITION.md); \"Identifiers ... are text\" (data-description)")
assume("Weather join year",
       "Weather joined on address_id and the calendar year of period_start (same-year weather; 2025 weather exists for test).",
       "\"weather: address_id x year\" (data-description)")
assume("Feature exclusions (leakage)",
       "customer_service_contacts and renewal_decision are never used as features; IDs, surname, e-mail domain, marketing consent, document preference, campaign code, "
       "branch and calendar year also excluded. written_premium kept (present in test).",
       "\"Leakage: customer_service_contacts (includes claim calls) and renewal_decision. Never use them as features.\" (COMPETITION.md)")
assume("Sum-insured and construction repairs",
       "sum_insured divided by 1000 when > 5M (buildings) / > 1M (contents); construction_type lower-cased and codes w,t mapped to timber; "
       "impossible years_licensed (> age-16) set to NaN.",
       "\"sum insured x1000, messy construction labels, impossible licence years\" (COMPETITION.md); construction classes \"brick, stone, timber, non-standard\" (data-description)")
assume("Modelling and CV",
       "One LightGBM Tweedie (power 1.5) model per line; 5 folds by crc32(customer_id) % 5 so customers (and thus their policies across years and lines) never straddle folds; "
       "early stopping on validation fold (mildly optimistic OOF); D2 computed on row-level total cost vs rate x exposure.",
       "\"Leaderboard metric: Tweedie D2 (power 1.5)\" (COMPETITION.md)")
assume("Unadjusted items",
       "No trend/inflation adjustment across 2021-2024 to 2025; no cross-line features; no adjustment of 2024 test-year differences. Rows without "
       "a claim have target 0 even if IBNR exists for them (IBNR is carried only by reported claims' weights).",
       "Task statement: predict 2025 rows (COMPETITION.md)")
lines_md = ["# Assumptions (clean-room build)", ""]
for i, (t, txt, src) in enumerate(assumptions, 1):
    lines_md += [f"{i}. **{t}.** {txt}", f"   - Source: {src}", ""]
(OUT / "assumptions.md").write_text("\n".join(lines_md), encoding="utf-8")
print(f"done in {time.time()-T0:.0f}s")
