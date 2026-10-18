"""Infrastructure for the local model-improvement loop (synthetic data, fast)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_params_by_line_merges_over_params(monkeypatch):
    import pipeline
    seen = []

    class Fake:
        def fit(self, *a, **k):
            return self

        def predict(self, X, e):
            return np.zeros(len(X))

    monkeypatch.setattr(pipeline, "make", lambda kind, params: (seen.append(params), Fake())[1])
    cfg = {"model": "lgbm", "params": {"a": 1, "tweedie_variance_power": 1.35},
           "params_by_line": {"motor": {"tweedie_variance_power": 1.2}}}
    X, e = pd.DataFrame({"x": [1.0, 2.0]}), np.ones(2)
    for line in ("motor", "buildings", None):
        pipeline._fit_predict(cfg, X, np.ones(2), np.ones(2), e, X, e, n_iter=5, line=line)
    assert seen[0] == {"a": 1, "tweedie_variance_power": 1.2}
    assert seen[1] == seen[2] == {"a": 1, "tweedie_variance_power": 1.35}
    assert cfg["params"]["tweedie_variance_power"] == 1.35          # input not mutated


def _freqsev_data(n=30000, seed=2):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n)})
    expo = rng.uniform(0.1, 1.0, n)
    counts = rng.poisson(0.15 * expo * np.exp(0.3 * X.a))
    y = counts * rng.gamma(2, 1500, n) * np.exp(0.3 * X.a)
    return X, expo, counts, y


def test_freqsev_sev_params_use_early_stopping_with_validation():
    from models import make
    X, expo, counts, y = _freqsev_data()
    a, b = np.arange(20000), np.arange(20000, 30000)
    sev = {"num_leaves": 7, "min_data_in_leaf": 100, "learning_rate": 0.1}
    m = make("freqsev", {"freq": {"learning_rate": 0.1}, "sev": sev})
    m.fit(X.iloc[a], y[a], np.ones(len(a)), expo[a], X.iloc[b], y[b], np.ones(len(b)), expo[b],
          counts=counts[a], counts_val=counts[b])
    assert m.sev_iter is not None and 0 < m.sev_iter < 2000
    assert 0.9 < m.predict(X.iloc[a], expo[a]).sum() / y[a].sum() < 1.1
    # no validation / n_iter given: fixed rounds from params["sev"]["n_iter"] (default 300)
    m2 = make("freqsev", {"freq": {"learning_rate": 0.1}, "sev": {**sev, "n_iter": 40}})
    m2.fit(X, y, np.ones(len(X)), expo, n_iter=30, counts=counts)
    assert m2.sev_iter == 40


def test_lbmodel_predict_calibrated_adds_level_gain():
    import lbmodel
    from market_spec import get_spec
    n = 6
    log = pd.DataFrame({"market": get_spec().name, "exp": list("abcdef"), "cv": np.linspace(0.10, 0.125, n),
                        "lb_public": np.linspace(0.10, 0.125, n) + 0.002, "decision": ""})
    m = lbmodel.fit(log)
    p, sd = lbmodel.predict(0.13, m)
    pc, sdc = lbmodel.predict_calibrated(0.13, m)
    assert lbmodel.LEVEL_GAIN == 0.0005
    assert np.isclose(pc, p + 0.0005) and sdc == sd
    assert f"with level factors: {p + 0.0005:.4f}" in lbmodel.verdict(0.13, m)


SAMPLE = """  teamId  teamName                             submissionDate              score    \r
--------  -----------------------------------  --------------------------  -------  \r
17136917  kaiY                                 2026-10-18 01:44:01.666000  0.12815  \r
17141948  applause_bin                         2026-10-17 23:01:09.660000  0.12753  \r
17137071  Simone Lee                           2026-10-17 17:57:12.513000  0.12697  \r
"""


def test_lb_watch_parser_and_margin():
    import lb_watch
    rows = lb_watch.parse(SAMPLE)
    assert [(r["rank"], r["team"], r["score"]) for r in rows] == [(1, "kaiY", 0.12815), (2, "applause_bin", 0.12753),
                                                                  (3, "Simone Lee", 0.12697)]
    s = lb_watch.summarise(rows)
    assert s["our_rank"] == 1 and np.isclose(s["margin_to_next"], 0.00062) and not lb_watch.alert(s)
    s2 = lb_watch.summarise([{**rows[1], "rank": 1}, {**rows[0], "rank": 2}, rows[2]])
    assert s2["our_rank"] == 2 and np.isclose(s2["margin_to_next"], 0.12753 - 0.12815) and lb_watch.alert(s2)
    s3 = lb_watch.summarise([{**rows[0], "score": 0.12760}, rows[1]])
    assert lb_watch.alert(s3)                                       # margin 0.00007 < 0.0003


def _fake_part(scale):
    n, m = 6, 4
    oof = np.full(n, 10.0)
    return {"cfg": {"model": "lgbm"}, "lines": {"motor": {
        "row_id": np.arange(n), "y_eval": np.array([0, 5, 0, 20, 0, 15.0]), "oof": oof,
        "test_row_id": np.arange(m), "test_pred": np.full(m, 10.0 * scale), "holdout": np.nan}}}


def _patch_blend(monkeypatch, parts):
    import pipeline

    class Spec:
        lines = ["motor"]
        n_folds = 2
        group_col = "policy_id"
        cache_path = Path("unused")
        results_path = Path("unused")

    monkeypatch.setattr(pipeline, "get_spec", lambda: Spec)
    monkeypatch.setattr(pipeline, "load", lambda n: parts[n])
    monkeypatch.setattr(pipeline, "_save", lambda r: None)
    monkeypatch.setattr(pipeline.pd, "read_parquet",
                        lambda *a, **k: pd.DataFrame({"row_id": np.arange(6), "policy_id": [0, 0, 1, 1, 2, 2]}))


def test_run_blend_rejects_exploding_part(monkeypatch):
    import pipeline
    import pytest
    _patch_blend(monkeypatch, {"good": _fake_part(1.0), "bad": _fake_part(60.0)})
    with pytest.raises(ValueError, match="part bad unsafe on motor"):
        pipeline.run_blend("b", {"model": "blend", "parts": ["good", "bad"]})


def test_run_blend_accepts_normal_parts(monkeypatch):
    import pipeline
    _patch_blend(monkeypatch, {"a": _fake_part(1.0), "b": _fake_part(1.2)})
    res = pipeline.run_blend("b", {"model": "blend", "parts": ["a", "b"], "step": 0.5})
    assert "motor" in res["lines"] and not res.get("unsafe")


def test_test_predictions_refuses_unsafe():
    import pipeline
    import pytest
    res = {"unsafe": True, "lines": {"motor": {"test_row_id": np.arange(2), "test_pred": np.ones(2)}}}
    with pytest.raises(ValueError, match="unsafe result"):
        pipeline.test_predictions(res)
    del res["unsafe"]
    assert len(pipeline.test_predictions(res)) == 2


def test_recency_weights(monkeypatch):
    sys.path.insert(0, str(ROOT / "src"))
    import target
    n = 40
    year = np.repeat([2021, 2022, 2023, 2024], n // 4)
    ps = pd.to_datetime(pd.Series(year).astype(str) + "-03-01")
    pol = pd.DataFrame({"row_id": np.arange(n), "period_start": ps, "period_end": ps + pd.Timedelta(days=364),
                        "exposure": 1.0, "written_premium": 100.0})
    claims = pd.DataFrame({"line": pd.Series([], dtype=object), "row_id": pd.Series([], dtype=float),
                           "claim_id": pd.Series([], dtype=object), "net_incurred": pd.Series([], dtype=float),
                           "paid_to_date": pd.Series([], dtype=float), "status": pd.Series([], dtype=object),
                           "loss_date": pd.Series([], dtype="datetime64[ns]")})
    monkeypatch.setattr(target, "completeness_curve", lambda c, l: np.ones(10))
    monkeypatch.setattr(target, "_daily", lambda curve: np.ones(4000))
    monkeypatch.setattr(target, "_row_integral", lambda p, d: np.ones(len(p)))
    monkeypatch.setattr(target, "_ibnr", lambda *a, **k: np.zeros(n))
    monkeypatch.setattr(target, "trend_index", lambda *a, **k: {y: 1.0 for y in range(2021, 2025)})
    base = {"weight_power": 1.0, "develop": False, "trend": "none"}
    d0, _ = target.prepare_target(pol, pol, claims, "motor", base)
    dk, _ = target.prepare_target(pol, pol, claims, "motor", {**base, "recency": 0.0})
    d5, _ = target.prepare_target(pol, pol, claims, "motor", {**base, "recency": 0.5})
    assert np.array_equal(d0.w.values, dk.w.values)
    assert np.isclose(d5.w.mean(), d0.w.mean())
    m = d5.w.groupby(year).mean()
    assert m.is_monotonic_increasing and m.iloc[-1] > m.iloc[0]


def test_exposure_feature_adds_one_column():
    sys.path.insert(0, str(ROOT / "src"))
    import features
    tr = pd.DataFrame({"row_id": [1, 2, 3, 4], "year": 2021, "exposure": [1.0, 0.5, 0.25, 1.0], "a": [1., 2., 3., 4.]})
    te = tr.copy()
    X0, Xt0, c0 = features.model_matrix(tr, te, {"history": False})
    X1, Xt1, c1 = features.model_matrix(tr, te, {"history": False, "exposure_feature": True})
    assert list(X1.columns) == list(X0.columns) + ["expo_feat"] and list(Xt1.columns) == list(Xt0.columns) + ["expo_feat"]
    assert X1[list(X0.columns)].equals(X0) and c0 == c1
    assert np.allclose(X1["expo_feat"], tr["exposure"])


# ---- pooled lines (cfg["pool"]) ----
def _patch_pool(monkeypatch, fits):
    import pipeline

    class Spec:
        lines = ["motor", "buildings", "contents"]
        n_folds = 2
        group_col = "policy_id"
        holdout_year = 2023
        seed = 1
        quick_test_size = 0.25
        results_path = Path("unused")

    sizes = {"motor": 8, "buildings": 12, "contents": 10}

    def line_data(base, line, cfg):
        n = sizes[line]
        tr = pd.DataFrame({"row_id": np.arange(n) + 1000 * len(line), "policy_id": np.arange(n) // 2, "exposure": 1.0,
                           "period_start": pd.to_datetime(pd.Series([2022, 2023] * (n // 2)).astype(str) + "-03-01"),
                           "claim_count": 0})
        te = tr.iloc[:4].copy()
        v = np.arange(n, dtype=float)
        tgt = pd.DataFrame({"y_train": v, "y_eval": v, "w": 1.0})
        return tr, te, tgt, {"loading": 1.0}, pd.DataFrame({"f": v}), pd.DataFrame({"f": v[:4]})

    class Stub:
        best_iter = None

        def fit(self, X, y, w, e, *a, **k):
            fits.append(X)
            return self

        def predict(self, X, e):
            return X["f"].values * 2.0 + 1.0

    monkeypatch.setattr(pipeline, "get_spec", lambda: Spec)
    monkeypatch.setattr(pipeline, "build_base", lambda: {})
    monkeypatch.setattr(pipeline, "_line_data", line_data)
    monkeypatch.setattr(pipeline, "_save", lambda r: None)
    monkeypatch.setattr(pipeline, "_summarise", lambda r: None)
    monkeypatch.setattr(pipeline, "make", lambda kind, params: Stub())
    return sizes


def test_pool_absent_fits_once_per_line_per_fold(monkeypatch):
    import pipeline
    fits = []
    sizes = _patch_pool(monkeypatch, fits)
    res = pipeline.run("t", {"model": "lgbm"}, folds=2, holdout=False, verbose=False)
    assert len(fits) == 3 * (2 + 1)                                  # 2 folds + full fit, per line
    assert all("line_id" not in X.columns for X in fits)
    assert {l: len(r["oof"]) for l, r in res["lines"].items()} == sizes


def test_pool_fits_home_lines_together_and_splits_back(monkeypatch):
    import pipeline
    fits = []
    sizes = _patch_pool(monkeypatch, fits)
    res = pipeline.run("t", {"model": "lgbm", "pool": ["buildings", "contents", "boat"]}, folds=2, holdout=True, verbose=False)
    pooled = [X for X in fits if "line_id" in X.columns]
    assert len(pooled) == 2 + 1 + 1                                  # 2 folds + holdout + full fit
    assert len(fits) - len(pooled) == 4                              # motor: 2 folds + holdout + full
    full = pooled[-1]
    assert len(full) == sizes["buildings"] + sizes["contents"]
    assert list(full["line_id"].cat.categories) == ["buildings", "contents"]
    assert set(full["line_id"].unique()) == {"buildings", "contents"}
    assert list(res["lines"]) == ["motor", "buildings", "contents"]
    for l in sizes:
        r = res["lines"][l]
        assert len(r["oof"]) == len(r["row_id"]) == sizes[l] and len(r["test_pred"]) == 4 and np.isfinite(r["holdout"])


def test_pool_oof_in_original_row_order(monkeypatch):
    import pipeline
    fits = []
    sizes = _patch_pool(monkeypatch, fits)
    res = pipeline.run("t", {"model": "lgbm", "pool": ["buildings", "contents"]}, folds=2, holdout=False, verbose=False)
    for l, n in sizes.items():
        assert np.allclose(res["lines"][l]["oof"], np.arange(n) * 2.0 + 1.0)
        assert np.allclose(res["lines"][l]["test_pred"], np.arange(4) * 2.0 + 1.0)


def test_pool_quick_pools_split(monkeypatch):
    import pipeline
    fits = []
    _patch_pool(monkeypatch, fits)
    out = pipeline.quick("t", {"model": "lgbm", "pool": ["buildings", "contents"]})
    assert len(fits) == 2 and any("line_id" in X.columns for X in fits)
    assert {"motor", "buildings", "contents", "overall"} <= set(out)


# ---- per-segment blend weights ----
def _seg_world(monkeypatch, n=4000, m=1000, seed=0):
    import pipeline
    rng = np.random.default_rng(seed)

    def make(k):
        g = (np.arange(k) % 2).astype(int)
        mu = np.exp(rng.normal(0, 0.5, k))
        noise = lambda: np.exp(rng.normal(0, 0.9, k))
        a = mu * np.where(g == 0, 1.0, noise())      # A exact on g=0, noisy on g=1
        b = mu * np.where(g == 1, 1.0, noise())      # B the opposite
        return g, mu, a, b

    g, mu, a, b = make(n)
    gt, mut, at, bt = make(m)
    y = rng.gamma(2.0, mu / 2.0) * (rng.random(n) < 0.7)
    mk = lambda o, t: {"cfg": {"model": "lgbm"}, "lines": {"motor": {
        "row_id": np.arange(n), "y_eval": y, "oof": o, "test_row_id": np.arange(m), "test_pred": t, "holdout": np.nan}}}
    parts = {"a": mk(a, at), "b": mk(b, bt)}

    class Spec:
        lines = ["motor"]
        n_folds = 3
        group_col = "policy_id"
        cache_path = Path("unused")
        results_path = Path("unused")

    monkeypatch.setattr(pipeline, "get_spec", lambda: Spec)
    monkeypatch.setattr(pipeline, "load", lambda nm: parts[nm])
    monkeypatch.setattr(pipeline, "_save", lambda r: None)
    monkeypatch.setattr(pipeline.pd, "read_parquet",
                        lambda *aa, **k: pd.DataFrame({"row_id": np.arange(n), "policy_id": np.arange(n) // 2}))
    cat = np.where(np.arange(n) < 10, "tiny", np.where(g == 0, "lo", "hi"))     # rare level for the min_rows test
    base = {"train_motor": pd.DataFrame({"row_id": np.arange(n), "exposure": np.ones(n), "g": g, "cat": cat}),
            "test_motor": pd.DataFrame({"row_id": np.arange(m), "exposure": np.ones(m), "g": gt,
                                        "cat": np.where(gt == 0, "lo", "hi")})}
    monkeypatch.setattr(pipeline, "build_base", lambda: base)
    return pipeline


def test_blend_segments_absent_is_unchanged(monkeypatch):
    pipeline = _seg_world(monkeypatch)
    cfg = {"model": "blend", "parts": ["a", "b"], "step": 0.1}
    l0 = pipeline.run_blend("x", cfg)["lines"]["motor"]
    l1 = pipeline.run_blend("x", dict(cfg))["lines"]["motor"]
    assert l0["cv"] == l1["cv"] and l0["cv_nested"] == l1["cv_nested"]
    assert (l0["test_pred"] == l1["test_pred"]).all() and "segments" not in l0
    assert set(l0["weights"]) == {"a", "b"}


def test_blend_segments_beat_unsegmented_and_weights_differ(monkeypatch):
    pipeline = _seg_world(monkeypatch)
    cfg = {"model": "blend", "parts": ["a", "b"], "step": 0.1}
    plain = pipeline.run_blend("x", cfg)["lines"]["motor"]
    seg = pipeline.run_blend("x", {**cfg, "segments": {"by": "g", "bins": 2, "min_rows": 100}})["lines"]["motor"]
    assert seg["cv_nested"] > plain["cv_nested"]
    w = seg["weights"]
    assert set(w) == {"q0", "q1"} and seg["segments"]["by"] == "g"
    assert w["q0"]["a"] > 0.7 and w["q1"]["b"] > 0.7           # A wins on g=0, B on g=1
    assert len(seg["test_pred"]) == 1000 and np.isfinite(seg["test_pred"]).all()


def test_blend_segments_oof_decile_runs(monkeypatch):
    pipeline = _seg_world(monkeypatch)
    res = pipeline.run_blend("x", {"model": "blend", "parts": ["a", "b"], "step": 0.5,
                                   "segments": {"by": "oof_decile", "bins": 4, "min_rows": 100}})
    assert len(res["lines"]["motor"]["weights"]) == 4 and np.isfinite(res["cv_nested"])


def test_blend_segments_min_rows_merges_into_rest(monkeypatch):
    pipeline = _seg_world(monkeypatch)
    res = pipeline.run_blend("x", {"model": "blend", "parts": ["a", "b"], "step": 0.5,
                                   "segments": {"by": "cat", "min_rows": 100}})
    s = res["lines"]["motor"]["segments"]
    assert s["merged_into_rest"] == ["tiny"] and s["sizes"]["rest"] == 10
    assert set(res["lines"]["motor"]["weights"]) == {"hi", "lo", "rest"}
