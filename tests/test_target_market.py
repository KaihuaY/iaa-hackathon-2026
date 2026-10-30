"""Review 11: market-aware claims development and large-loss loading in target.prepare_target (pooled market).

Synthetic two-market book: developed rows 2021-24 with slow reporting, emerging rows 2023-24 with fast reporting.
The pooled per-market result for a market's rows must equal prepare_target run on that market alone under its own
development spec; without a `market` column (or with develop_by_market False) nothing changes."""
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]

import market_spec as ms  # noqa: E402
import market_spec_pool as msp  # noqa: E402
import target  # noqa: E402

SPEC = dataclasses.replace(msp.PART2POOL, name="t_pool_dev", segments={})
EM_DEV = {"curve_years": (2023,), "mature_years": (2023,), "ref_sev_year": 2023}


def _book(seed=0):
    rng = np.random.default_rng(seed)
    pol, cl = [], []
    for mk, years, lag_mean, sev in (("developed", (2021, 2022, 2023, 2024), 90.0, 1000.0),
                                     ("emerging", (2023, 2024), 6.0, 500.0)):
        for y in years:
            n = 1500
            start = pd.Timestamp(f"{y}-01-01") + pd.to_timedelta(rng.integers(0, 180, n), unit="D")
            end = start + pd.Timedelta(days=364)
            p = "D" if mk == "developed" else ""
            ids = [f"{p}M{y}{i:05d}" for i in range(n)]
            pol.append(pd.DataFrame({"row_id": ids, "policy_id": [f"{p}P{i:05d}" for i in range(n)], "period_start": start,
                                     "period_end": end, "exposure": 1.0, "written_premium": 100.0, "market": mk}))
            k = rng.random(n) < 0.3
            loss = start[k] + pd.to_timedelta(rng.integers(0, 365, k.sum()), unit="D")
            rep = loss + pd.to_timedelta(rng.exponential(lag_mean, k.sum()).astype(int), unit="D")
            keep = rep <= pd.Timestamp("2024-12-31")
            cost = rng.gamma(0.5, sev / 0.5, k.sum())
            cl.append(pd.DataFrame({"claim_id": [f"{p}C{y}{i}" for i in range(k.sum())], "policy_id": np.array(ids)[k],
                                    "row_id": np.array(ids)[k], "line": "motor", "loss_date": loss, "report_date": rep,
                                    "status": "closed", "paid_to_date": cost, "net_incurred": cost})[keep])
    pol = pd.concat(pol, ignore_index=True)
    test = pol[pol.market == "emerging"].head(200).assign(
        row_id=lambda d: "T" + d.row_id, period_start=pd.Timestamp("2025-01-01"), period_end=pd.Timestamp("2025-12-31"))
    return pol, test.reset_index(drop=True), pd.concat(cl, ignore_index=True)


CFG = {"develop": True, "cap_q": 0.99, "trend": "none", "weight_power": 1.0}
COLS = ["completeness", "ibnr", "y_dev"]


@pytest.fixture(autouse=True)
def _spec():
    prev = ms.get_spec()
    ms.use_spec(dataclasses.replace(SPEC, market_dev={"emerging": EM_DEV}))
    yield
    ms.use_spec(prev)


def test_no_market_column_or_switch_off_is_unchanged():
    pol, test, cl = _book()
    a, _ = target.prepare_target(pol.drop(columns="market"), test.drop(columns="market"), cl, "motor", CFG)
    b, _ = target.prepare_target(pol, test, cl, "motor", {**CFG, "develop_by_market": False})
    pd.testing.assert_frame_equal(a, b)


def test_each_market_develops_as_if_alone():
    pol, test, cl = _book()
    got, _ = target.prepare_target(pol, test, cl, "motor", CFG)
    s = ms.get_spec()
    for mk, over in (("developed", {}), ("emerging", EM_DEV)):
        rows = (pol.market == mk).values
        cm = ~cl.claim_id.str.startswith("D") if mk == "emerging" else cl.claim_id.str.startswith("D")
        yrs = {"train_years": (2023, 2024)} if mk == "emerging" else {}     # alone: its own years (trend index)
        with ms.using_spec(dataclasses.replace(s, **over, **yrs)):
            alone, _ = target.prepare_target(pol[rows].drop(columns="market"), test.drop(columns="market"), cl[cm],
                                             "motor", {**CFG, "cap_q": None})
        g = got[rows].reset_index(drop=True)
        np.testing.assert_allclose(g.completeness.values, alone.completeness.values, rtol=1e-12)
        np.testing.assert_allclose(g.y_dev.values, alone.y_dev.values, rtol=1e-9)    # uncapped (eval) development
    # fast-reporting emerging market: its rows are more complete than under the pooled (developed) curve
    pooled, _ = target.prepare_target(pol, test, cl, "motor", {**CFG, "develop_by_market": False})
    em24 = ((pol.market == "emerging") & (pol.period_start.dt.year == 2024)).values
    assert got.completeness.values[em24].mean() > pooled.completeness.values[em24].mean() + 0.02
    assert got.ibnr.values[em24].sum() < pooled.ibnr.values[em24].sum()


def test_loading_by_market():
    pol, test, cl = _book()
    _, info0 = target.prepare_target(pol, test, cl, "motor", CFG)
    _, info = target.prepare_target(pol, test, cl, "motor", {**CFG, "loading_by_market": True})
    assert "loading_rows" not in info0                          # off by default (no segment loading in CFG)
    lm = info["loading_by_market"]
    assert set(lm) == {"developed", "emerging"} and all(v >= 1 for v in lm.values())
    em = (pol.market == "emerging").values
    assert np.allclose(info["loading_rows"][em], lm["emerging"]) and np.allclose(info["loading_rows"][~em], lm["developed"])
    assert np.allclose(info["loading_test"], lm["emerging"])   # test rows are emerging
    # same cap for both markets: the emerging ratio is the emerging claims' own excess / capped (mature 2023)
    cap = info["cap"]
    c = cl[~cl.claim_id.str.startswith("D") & (cl.loss_date.dt.year <= 2023)].net_incurred
    assert lm["emerging"] == pytest.approx(c.sum() / c.clip(upper=cap).sum())


def test_segment_loading_by_market():
    pol, test, cl = _book()
    pol["cover_type"] = np.where(np.arange(len(pol)) % 2, "comp", "tpft")
    pol["years_licensed"] = np.arange(len(pol)) % 50
    test["cover_type"], test["years_licensed"] = "comp", 10
    ms.use_spec(dataclasses.replace(ms.get_spec(), segments=msp.PART2POOL.segments))
    _, info = target.prepare_target(pol, test, cl, "motor", {**CFG, "loading": "segment", "loading_by_market": True})
    t = info["loading_table"]
    assert set(t.market) == {"developed", "emerging"} and len(info["loading_test"]) == len(test)
    em_tab = t[t.market == "emerging"].set_index("segment").loading
    assert np.allclose(info["loading_test"], em_tab["comp|8-17"])


@pytest.mark.skipif(not (ROOT / "experiments" / "cache" / "part2pool" / "claims.parquet").exists()
                    or not (ROOT / "experiments" / "cache" / "part2" / "claims.parquet").exists(),
                    reason="part2 / part2pool bases not built")
@pytest.mark.parametrize("line", ["motor", "buildings", "contents"])
def test_real_pooled_emerging_development_matches_emerging_spec(line):
    """Emerging rows of the pooled market develop exactly as in the emerging-only PART2 spec (uncapped y_dev);
    their 2024 mean completeness is PART2's completeness_last_year."""
    out = {}
    for name in ("part2", "part2pool"):
        with ms.using_spec(name) as s:
            cl = pd.read_parquet(s.cache_path / "claims.parquet")
            tr = pd.read_parquet(s.cache_path / f"train_{line}.parquet")
            te = pd.read_parquet(s.cache_path / f"test_{line}.parquet")
            t, _ = target.prepare_target(tr, te, cl, line, {"develop": True, "cap_q": None, "trend": "none"})
            if "market" in tr:
                t = t[(tr.market == "emerging").values]
            out[name] = t.set_index("row_id")
    p2, pp = out["part2"], out["part2pool"].reindex(out["part2"].index)
    np.testing.assert_allclose(pp.y_dev.values, p2.y_dev.values, rtol=1e-6, atol=1e-6)
    em24 = pd.read_parquet(ms.REGISTRY["part2"].cache_path / f"train_{line}.parquet", columns=["row_id", "period_start"])
    ids = em24.row_id[em24.period_start.dt.year == 2024]
    assert pp.completeness.reindex(ids).mean() == pytest.approx(
        msp.PART2POOL.market_dev["emerging"]["completeness_last_year"][line], abs=6e-4)


def test_cap_by_market():
    """Agent S2: cfg `cap_by_market` caps each market's claims at its own quantile and loads each market back to its
    own level; with the flag off the pooled cap and loading are unchanged."""
    pol, test, cl = _book()
    t0, info0 = target.prepare_target(pol, test, cl, "motor", CFG)
    t1, info1 = target.prepare_target(pol, test, cl, "motor", {**CFG, "cap_by_market": False})
    pd.testing.assert_frame_equal(t0, t1)
    assert "cap_by_market" not in info0 and "loading_rows" not in info0
    _, info = target.prepare_target(pol, test, cl, "motor", {**CFG, "cap_by_market": True})
    cm, lm = info["cap_by_market"], info["loading_by_market"]
    em_cl = ~cl.claim_id.str.startswith("D")
    assert cm["emerging"] == pytest.approx(cl[em_cl].net_incurred.quantile(0.99))
    assert cm["developed"] == pytest.approx(cl[~em_cl].net_incurred.quantile(0.99))
    assert cm["emerging"] < info["cap"] < cm["developed"]       # severities 500 vs 1000
    c = cl[em_cl & (cl.loss_date.dt.year <= 2023)].net_incurred
    assert lm["emerging"] == pytest.approx(c.sum() / c.clip(upper=cm["emerging"]).sum())
    assert np.allclose(info["loading_test"], lm["emerging"])
    # per-peril models are not wired for per-market caps (fail loudly rather than clip at the pooled cap)
    with pytest.raises(NotImplementedError):
        target.peril_matrix(pol, cl, "motor", t0, info)
