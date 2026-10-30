"""L0 headroom ceiling: parametric bootstrap of our own predictions mu as if they were the true expected cost.
Excess actual Tweedie deviance over simulated (share of null deviance) ~ upper bound on remaining D2 gain. Writes reports/lens/."""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path[:0] = [str(ROOT / "src")]

from metrics import tweedie_deviance  # noqa: E402

LINES = ("motor", "buildings", "contents")
VARIANTS = ("global", "binned", "gamma")


def simulate_compound(mu, pool_amounts, sbar, rng, gamma_cv=None):
    """One replicate y*: N_i ~ Poisson(mu_i/sbar), amounts bootstrapped from the pool (or Gamma(mean sbar, cv) if gamma_cv)."""
    mu = np.asarray(mu, float)
    n = rng.poisson(np.clip(mu, 0, None) / sbar)
    total = int(n.sum())
    if total == 0:
        return np.zeros(len(mu))
    if gamma_cv is not None:
        shape = 1.0 / gamma_cv ** 2
        amt = rng.gamma(shape, sbar / shape, size=total)
    else:
        amt = rng.choice(np.asarray(pool_amounts, float), size=total, replace=True)
    return np.bincount(np.repeat(np.arange(len(mu)), n), weights=amt, minlength=len(mu))


def simulate_binned(mu, bin_idx, pools, glob_pool, rng, min_claims=200):
    """Per mu-decile severity pool (fallback to the global pool when thin)."""
    y = np.zeros(len(mu))
    for b in np.unique(bin_idx):
        m = bin_idx == b
        pool = pools.get(b)
        if pool is None or len(pool) < min_claims:
            pool = glob_pool
        y[m] = simulate_compound(mu[m], pool, pool.mean(), rng)
    return y


def ceiling_stats(y, mu, sims):
    """sims: iterable of simulated y* arrays. Returns actual/null deviance and per-replicate simulated ones."""
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    d_act = tweedie_deviance(y, mu).sum()
    d_null = tweedie_deviance(y, np.full_like(y, y.mean())).sum()
    ds, dn = [], []
    for ys in sims:
        ds.append(tweedie_deviance(ys, mu).sum())
        dn.append(tweedie_deviance(ys, np.full_like(ys, ys.mean())).sum())
    return dict(d_act=d_act, d_null=d_null, d_sim=np.array(ds), d_null_sim=np.array(dn))


def summarize(st):
    d2_sim = 1 - st["d_sim"] / st["d_null_sim"]
    sd_d = st["d_sim"].std(ddof=1) if len(d2_sim) > 1 else np.nan
    return dict(d2_act=1 - st["d_act"] / st["d_null"], d2_sim_mean=d2_sim.mean(),
                d2_sim_sd=d2_sim.std(ddof=1) if len(d2_sim) > 1 else np.nan,
                z=(st["d_act"] - st["d_sim"].mean()) / sd_d,
                excess_share=(st["d_act"] - st["d_sim"].mean()) / st["d_null"])


def combine(stats):
    return dict(d_act=sum(s["d_act"] for s in stats), d_null=sum(s["d_null"] for s in stats),
                d_sim=sum(s["d_sim"] for s in stats), d_null_sim=sum(s["d_null_sim"] for s in stats))


def subset_sd(y, mu, rng, reps=200, frac=0.25):
    """sd of d2 over random row subsets (public-board noise proxy)."""
    da, dn = tweedie_deviance(y, mu), tweedie_deviance(y, np.full_like(y, y.mean()))
    out = []
    for _ in range(reps):
        m = rng.random(len(y)) < frac
        out.append(1 - da[m].sum() / dn[m].sum())
    return float(np.std(out, ddof=1))


def relevel(oof, y, year):
    mu = np.asarray(oof, float).copy()
    fac = {}
    for yr in np.unique(year):
        m = year == yr
        f = y[m].sum() / mu[m].sum()
        mu[m] *= f
        fac[int(yr)] = float(f)
    return mu, fac


def load_line(res, line, years):
    r = res["lines"][line]
    oof = pd.DataFrame({"row_id": np.asarray(r["row_id"]).astype(str), "oof": np.asarray(r["oof"], float)})
    tr = pd.read_parquet(ROOT / "experiments" / "cache" / f"train_{line}.parquet", columns=["row_id", "year", "claim_cost"])
    tr["row_id"] = tr["row_id"].astype(str)
    d = oof.merge(tr, on="row_id")
    d = d[d["year"].isin(years)].reset_index(drop=True)
    d["y"] = d["claim_cost"].fillna(0).clip(lower=0)
    return d


def run_line(d, claims, reps, seed, sub_reps=200):
    rng = np.random.default_rng(seed)
    y = d["y"].to_numpy(float)
    mu, fac = relevel(d["oof"].to_numpy(float), y, d["year"].to_numpy())
    mu = np.clip(mu, 1e-9, None)
    c = claims[claims["row_id"].isin(set(d["row_id"]))]
    pool = c.loc[c["net_incurred"] > 0, ["row_id", "net_incurred"]]
    glob = pool["net_incurred"].to_numpy(float)
    sbar, cv = glob.mean(), glob.std() / glob.mean()
    # decile bins of mu and the claims attached to each
    bins = pd.qcut(pd.Series(mu).rank(method="first"), 10, labels=False).to_numpy()
    rbin = pd.Series(bins, index=d["row_id"].to_numpy())
    rbin = rbin[~rbin.index.duplicated()]
    pb = pool.assign(b=rbin.reindex(pool["row_id"]).to_numpy())
    pools = {b: g["net_incurred"].to_numpy(float) for b, g in pb.groupby("b")}
    gens = {
        "global": lambda: simulate_compound(mu, glob, sbar, rng),
        "binned": lambda: simulate_binned(mu, bins, pools, glob, rng),
        "gamma": lambda: simulate_compound(mu, None, sbar, rng, gamma_cv=cv),
    }
    stats = {v: ceiling_stats(y, mu, (gens[v]() for _ in range(reps))) for v in VARIANTS}
    sd25 = subset_sd(y, mu, np.random.default_rng(seed + 1), sub_reps)
    return dict(stats=stats, n_rows=len(y), n_claims=len(glob), sub25_sd=sd25, fac=fac, y=y, mu=mu)


def reading(r):
    if abs(r.excess_share) < 2 * r.d2_sim_sd:
        return "consistent with truth, no detectable headroom"
    if r.excess_share > 0:
        return "excess beyond 2 sd of simulated D2 (possible headroom)"
    return "actual fits better than simulated truth (severity pool too noisy or mu too tight)"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default="X21_blend4_a1bag")
    ap.add_argument("--years", default="2021,2022,2023")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    years = [int(x) for x in a.years.split(",")]
    with open(ROOT / "experiments" / "results" / f"{a.name}.pkl", "rb") as f:
        res = pickle.load(f)
    claims = pd.read_parquet(ROOT / "experiments" / "cache" / "claims.parquet", columns=["row_id", "line", "net_incurred"])
    claims["row_id"] = claims["row_id"].astype(str)
    rows, per = [], {}
    for li, line in enumerate(LINES):
        d = load_line(res, line, years)
        per[line] = run_line(d, claims[claims["line"] == line], a.reps, a.seed + 10 * li)
        del d
    y_all = np.concatenate([per[l]["y"] for l in LINES])
    mu_all = np.concatenate([per[l]["mu"] for l in LINES])
    sub_all = subset_sd(y_all, mu_all, np.random.default_rng(a.seed + 99))
    del y_all, mu_all
    for line in LINES + ("overall",):
        for v in VARIANTS:
            if line == "overall":
                st = combine([per[l]["stats"][v] for l in LINES])
                nr = sum(per[l]["n_rows"] for l in LINES)
                nc = sum(per[l]["n_claims"] for l in LINES)
                sd25, fac = sub_all, ""
            else:
                st, nr, nc = per[line]["stats"][v], per[line]["n_rows"], per[line]["n_claims"]
                sd25 = per[line]["sub25_sd"]
                fac = ";".join(f"{k}:{x:.3f}" for k, x in per[line]["fac"].items())
            rows.append(dict(name=a.name, line=line, variant=v, n_rows=nr, n_claims=nc, **summarize(st), sub25_sd=sd25, relevel=fac))
    df = pd.DataFrame(rows)
    out = ROOT / "reports" / "lens"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"ceiling_{a.name}.csv", index=False)
    show = df.drop(columns=["name", "relevel"]).round(5)
    table = show.to_string(index=False)
    print(table)
    cons = [f"{r.line}: {reading(r)}" for r in df[df["variant"] == "global"].itertuples()]
    print("\n" + "\n".join(cons))
    md = [f"# L0 headroom ceiling: {a.name}", "", "```", table, "```", "",
          "Reading rule: excess_share = (D_act - mean D_sim) / D_null; positive = actual fits worse than if mu were the true mean "
          "= possible headroom in D2 units. If |excess_share| < 2*d2_sim_sd the predictions are consistent with truth and no headroom "
          "is detectable. sub25_sd is the sd of d2 over 25% row subsets (public-board noise proxy). Global variant: " + "; ".join(cons) + "."]
    (out / f"ceiling_{a.name}.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
