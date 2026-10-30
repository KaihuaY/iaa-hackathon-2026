"""Review 11: run_blend `weight_market` (weights fitted on one market's rows) and `external_parts` (another market's
results aligned by row_id), and the `weight_emerging_nb` fit-weight hook. Synthetic results in a temporary market."""
import dataclasses
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]

import market_spec as ms  # noqa: E402
import pipeline  # noqa: E402


def _res(name, rows, test_ids, oof, y, test, holdout=None):
    lines = {}
    for l in ("motor", "buildings", "contents"):
        d = {"row_id": rows, "y_eval": y, "oof": oof, "test_row_id": test_ids, "test_pred": test, "cv": 0.0,
             "holdout": np.nan}
        if holdout is not None:
            d["holdout_idx"], d["holdout_pred"] = holdout
        lines[l] = d
    return {"name": name, "cfg": {"model": "lgbm"}, "lines": lines}


def test_weight_market_and_external_parts(tmp_path):
    prev = ms.get_spec()
    rng = np.random.default_rng(1)
    n_d, n_e = 600, 400
    rows = np.array([f"D{i}" for i in range(n_d)] + [f"E{i}" for i in range(n_e)], dtype=object)
    mk = np.array(["developed"] * n_d + ["emerging"] * n_e)
    mu = np.where(mk == "emerging", 2.0, 1.0) * rng.uniform(0.5, 2, n_d + n_e)
    y = rng.gamma(0.3, mu / 0.3)
    good_on_em = np.where(mk == "emerging", mu, 1.3)            # part B is right on emerging rows only
    good_on_dev = np.where(mk == "emerging", 1.3, mu)           # part A is right on developed rows only
    tid = np.array([f"T{i}" for i in range(50)], dtype=object)
    s = ms.use_spec(dataclasses.replace(ms.PART1, name="t_blend", cache_dir=str(tmp_path / "c"),
                                        results_dir=str(tmp_path / "r"), n_folds=4))
    try:
        s.cache_path.mkdir(parents=True)
        s.results_path.mkdir(parents=True)
        for l in s.lines:
            pd.DataFrame({"row_id": rows, "policy_id": [f"P{i}" for i in range(len(rows))], "market": mk}).to_parquet(
                s.cache_path / f"train_{l}.parquet")
        hb = np.arange(n_d + n_e)[::3]
        pipeline._save(_res("A", rows, tid, good_on_dev, y, np.ones(50), (hb, good_on_dev[hb])))
        pipeline._save(_res("B", rows, tid, good_on_em, y, np.full(50, 2.0), (hb, good_on_em[hb])))
        r_all = pipeline.run_blend("X_all", {"model": "blend", "parts": ["A", "B"], "step": 0.1})
        r_em = pipeline.run_blend("X_em", {"model": "blend", "parts": ["A", "B"], "step": 0.1, "weight_market": "emerging"})
        assert r_em["lines"]["motor"]["weights"]["B"] > r_all["lines"]["motor"]["weights"]["B"]
        assert r_em["lines"]["motor"]["weights"]["B"] >= 0.9
        assert "cv_nested_emerging" in r_em and len(r_em["lines"]["motor"]["oof_nested"]) == len(rows)
        b = r_em["lines"]["motor"]["holdout_idx"]
        assert (mk[b] == "emerging").all() and len(b) == (mk[hb] == "emerging").sum()
        # external part: emerging rows only, stored in another market's results folder, shuffled order
        ext = ms.register(dataclasses.replace(s, name="t_blend_ext", results_dir=str(tmp_path / "r_ext")))
        ext.results_path.mkdir(parents=True)
        e_idx = np.where(mk == "emerging")[0][::-1]
        perm = np.arange(50)[::-1]
        with ms.using_spec(ext):
            pipeline._save(_res("E", rows[e_idx], tid[perm], good_on_em[e_idx], y[e_idx], np.full(50, 3.0)))
        r_x = pipeline.run_blend("X_ext", {"model": "blend", "parts": ["A"], "step": 0.1, "weight_market": "emerging",
                                           "external_parts": {"E": "t_blend_ext"}})
        w = r_x["lines"]["motor"]["weights"]
        assert set(w) == {"A", "E"} and w["E"] >= 0.9
        tp = r_x["lines"]["motor"]["test_pred"]
        assert np.allclose(tp, w["A"] * 1.0 + w["E"] * 3.0)
        # fixed weights on one line: not fitted, nested = in-sample
        r_f = pipeline.run_blend("X_fix", {"model": "blend", "parts": ["A", "B"], "step": 0.1, "weight_market": "emerging",
                                           "fixed_weights": {"buildings": {"A": 1.0}}})
        assert r_f["lines"]["buildings"]["weights"] == {"A": 1.0, "B": 0.0}
        assert np.allclose(r_f["lines"]["buildings"]["oof_nested"], good_on_dev)
        assert r_f["lines"]["motor"]["weights"] == r_em["lines"]["motor"]["weights"]
        # default (no keys) is unchanged by the additive code path
        r_all2 = pipeline.run_blend("X_all2", {"model": "blend", "parts": ["A", "B"], "step": 0.1})
        assert np.array_equal(r_all["lines"]["motor"]["oof"], r_all2["lines"]["motor"]["oof"])
        assert "oof_nested" not in r_all["lines"]["motor"]
    finally:
        ms.REGISTRY.pop("t_blend_ext", None)
        ms.use_spec(prev)


def test_weight_emerging_nb():
    tr = pd.DataFrame({"market": ["developed", "developed", "emerging", "emerging"],
                       "new_business": [True, False, True, False]})
    fit, val = pipeline._market_weights({"market_weight_developed": 0.5, "weight_emerging_nb": 3}, tr)
    assert fit.tolist() == [0.5, 0.5, 3.0, 1.0] and val.tolist() == [0.5, 0.5, 1.0, 1.0]
    fit0, _ = pipeline._market_weights({"market_weight_developed": 0.5}, tr)
    assert fit0.tolist() == [0.5, 0.5, 1.0, 1.0]
