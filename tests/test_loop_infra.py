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
