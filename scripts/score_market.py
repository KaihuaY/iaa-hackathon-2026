"""Score a pooled-market experiment on the EMERGING rows only (developed rows carry the row_id prefix "D").

  $env:IAA_MARKET="part2pool"; python -P -E scripts/score_market.py POOL_L20 [POOL_B15 ...] [--ref part2:L20_mono_lowreg]

Per line and overall, on emerging rows: CV OOF Tweedie D2 (p=1.5, metrics.d2_tweedie on per-row cost, as the pipeline),
calibration sum(pred)/sum(actual) overall and by year, 2024 holdout D2 / calibration (needs `holdout_pred`, saved by
pipeline.run since Review 10), the new-business / renewal mean prediction-per-exposure ratio on the OOF, on the actuals
and on the 2025 test, and the test mean prediction per exposure vs the emerging 2024 / 2023 pure premium.
--ref MARKET:EXP also scores the same predictions against that result's y_eval (row_id-aligned), e.g. the emerging-only
spec's target, so pooled and emerging-only runs can be compared on one yardstick.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]

from metrics import d2_tweedie  # noqa: E402

PREFIX = "D"
PP_2024 = {"motor": 382.6, "buildings": 306.7, "contents": 86.8}   # emerging history, per unit exposure
PP_2023 = {"motor": 432.7, "buildings": 514.2, "contents": 91.1}


def emerging_mask(row_ids) -> np.ndarray:
    return ~pd.Series(np.asarray(row_ids, dtype=object)).astype(str).str.startswith(PREFIX).to_numpy()


def _ratio(num, den):
    return float(np.sum(num) / np.sum(den)) if np.sum(den) > 0 else np.nan


def score_line(L: dict, tr: pd.DataFrame, te: pd.DataFrame, line: str, ref_y: pd.Series | None = None) -> dict:
    """L: one line of a pipeline result. tr/te: row_id-indexed frames with exposure, year, new_business."""
    rid = np.asarray(L["row_id"], dtype=object)
    em = emerging_mask(rid)
    y, p = np.asarray(L["y_eval"], float)[em], np.asarray(L["oof"], float)[em]
    T = tr.reindex(rid[em])
    out = {"line": line, "rows": int(em.sum()), "cv": d2_tweedie(y, p), "calib": _ratio(p, y)}
    if "oof_nested" in L:                         # blend with weights fitted out-of-fold (pipeline.run_blend)
        pn = np.asarray(L["oof_nested"], float)[em]
        out["cv_nested"] = d2_tweedie(y, pn)
        out["_cvn"] = (y, pn)
    for yr in sorted(T["year"].dropna().unique()):
        s = (T["year"] == yr).to_numpy()
        out[f"calib_{int(yr)}"] = _ratio(p[s], y[s])
    nb = T["new_business"].astype(bool).to_numpy()
    e = T["exposure"].to_numpy(float)
    out["nb_ratio_oof"] = (_ratio(p[nb], e[nb]) / _ratio(p[~nb], e[~nb])) if nb.any() and (~nb).any() else np.nan
    out["nb_ratio_actual"] = (_ratio(y[nb], e[nb]) / _ratio(y[~nb], e[~nb])) if nb.any() and (~nb).any() else np.nan
    if ref_y is not None:
        yr_ = ref_y.reindex(rid[em]).to_numpy(float)
        ok = ~np.isnan(yr_)
        out["cv_ref"] = d2_tweedie(yr_[ok], p[ok]) if ok.any() else np.nan
        out["calib_ref"] = _ratio(p[ok], yr_[ok])
        out["ref_rows"] = int(ok.sum())
    if "holdout_pred" in L:
        b = np.asarray(L["holdout_idx"])
        hb = emerging_mask(rid[b])
        yh, ph = np.asarray(L["y_eval"], float)[b][hb], np.asarray(L["holdout_pred"], float)[hb]
        out["holdout"] = d2_tweedie(yh, ph)
        out["holdout_calib"] = _ratio(ph, yh)
        if ref_y is not None:
            yrh = ref_y.reindex(rid[b][hb]).to_numpy(float)
            ok = ~np.isnan(yrh)
            out["holdout_ref"] = d2_tweedie(yrh[ok], ph[ok]) if ok.any() else np.nan
        out["_ho"] = (yh, ph)
    trid = np.asarray(L["test_row_id"], dtype=object)
    tem = emerging_mask(trid)
    pt = np.asarray(L["test_pred"], float)[tem]
    TT = te.reindex(trid[tem])
    et, nbt = TT["exposure"].to_numpy(float), TT["new_business"].astype(bool).to_numpy()
    out["test_rows"] = int(tem.sum())
    out["test_mean"] = float(pt.mean())
    out["test_pp"] = _ratio(pt, et)
    if line in PP_2024:
        out["test_pp_vs_2024"] = out["test_pp"] / PP_2024[line]
        out["test_pp_vs_2023"] = out["test_pp"] / PP_2023[line]
    out["nb_ratio_test"] = (_ratio(pt[nbt], et[nbt]) / _ratio(pt[~nbt], et[~nbt])) if nbt.any() and (~nbt).any() else np.nan
    out["_cv"] = (y, p)
    return out


def score(res: dict, frames: dict, ref: dict | None = None) -> pd.DataFrame:
    rows, cv_all, ho_all, cvn_all = [], [], [], []
    for line, L in res["lines"].items():
        ref_y = None
        if ref is not None and line in ref["lines"]:
            ref_y = pd.Series(np.asarray(ref["lines"][line]["y_eval"], float),
                              index=np.asarray(ref["lines"][line]["row_id"], dtype=object))
        r = score_line(L, frames[f"train_{line}"], frames[f"test_{line}"], line, ref_y)
        cv_all.append(r.pop("_cv"))
        if "_cvn" in r:
            cvn_all.append(r.pop("_cvn"))
        if "_ho" in r:
            ho_all.append(r.pop("_ho"))
        rows.append(r)
    tot = {"line": "overall", "rows": sum(r["rows"] for r in rows),
           "cv": d2_tweedie(np.concatenate([a for a, _ in cv_all]), np.concatenate([b for _, b in cv_all])),
           "calib": _ratio(np.concatenate([b for _, b in cv_all]), np.concatenate([a for a, _ in cv_all]))}
    if cvn_all and len(cvn_all) == len(rows):
        tot["cv_nested"] = d2_tweedie(np.concatenate([a for a, _ in cvn_all]), np.concatenate([b for _, b in cvn_all]))
    if len(ho_all) == len(rows):
        tot["holdout"] = d2_tweedie(np.concatenate([a for a, _ in ho_all]), np.concatenate([b for _, b in ho_all]))
        tot["holdout_calib"] = _ratio(np.concatenate([b for _, b in ho_all]), np.concatenate([a for a, _ in ho_all]))
    rows.append(tot)
    return pd.DataFrame(rows).set_index("line")


def nb_calibration(res: dict, frames: dict) -> pd.DataFrame:
    """Post-hoc new-business relativity check per line on EMERGING rows (Review 11, item c).

    Factors c_nb, c_ren = (actual / predicted) of new-business / renewal rows divided by (actual / predicted) of all
    rows, so the line total is unchanged and only the NB/renewal relativity moves. 'oof' factors use every emerging
    OOF row; 'y23' factors use the 2023 OOF rows only and are tested on the 2024 holdout predictions (a model fitted
    on years < 2024), the honest check. Also: nested over 5 random policy-free row folds is not needed - the holdout
    test is the gate."""
    out = []
    for line, L in res["lines"].items():
        rid = np.asarray(L["row_id"], dtype=object)
        em = emerging_mask(rid)
        T = frames[f"train_{line}"].reindex(rid)
        nb = T["new_business"].astype(bool).to_numpy()
        yr = T["year"].to_numpy()
        y, p = np.asarray(L["y_eval"], float), np.asarray(L["oof"], float)

        def fac(sel):
            ae = _ratio(y[sel], p[sel])
            return _ratio(y[sel & nb], p[sel & nb]) / ae, _ratio(y[sel & ~nb], p[sel & ~nb]) / ae
        r = {"line": line}
        r["c_nb_oof"], r["c_ren_oof"] = fac(em)
        r["c_nb_y23"], r["c_ren_y23"] = fac(em & (yr == 2023))
        r["cv"] = d2_tweedie(y[em], p[em])
        r["cv_cal_oof_insample"] = d2_tweedie(y[em], p[em] * np.where(nb[em], r["c_nb_oof"], r["c_ren_oof"]))
        if "holdout_pred" in L:
            b = np.asarray(L["holdout_idx"])
            hb = b[em[b]]
            ph = np.asarray(L["holdout_pred"], float)[em[b]]
            f = np.where(nb[hb], r["c_nb_y23"], r["c_ren_y23"])
            r["holdout"] = d2_tweedie(y[hb], ph)
            r["holdout_cal_y23"] = d2_tweedie(y[hb], ph * f)
            r["holdout_nb_ae"] = _ratio(y[hb][nb[hb]], ph[nb[hb]]) / _ratio(y[hb], ph)
        trid = np.asarray(L["test_row_id"], dtype=object)
        TT = frames[f"test_{line}"].reindex(trid)
        nbt, et = TT["new_business"].astype(bool).to_numpy(), TT["exposure"].to_numpy(float)
        pt = np.asarray(L["test_pred"], float)
        r["nb_ratio_test"] = _ratio(pt[nbt], et[nbt]) / _ratio(pt[~nbt], et[~nbt])
        r["nb_ratio_test_cal"] = r["nb_ratio_test"] * r["c_nb_oof"] / r["c_ren_oof"]
        r["nb_ratio_oof"] = _ratio(p[em & nb], T["exposure"].to_numpy(float)[em & nb]) / _ratio(p[em & ~nb], T["exposure"].to_numpy(float)[em & ~nb])
        r["nb_ratio_actual"] = _ratio(y[em & nb], T["exposure"].to_numpy(float)[em & nb]) / _ratio(y[em & ~nb], T["exposure"].to_numpy(float)[em & ~nb])
        out.append(r)
    return pd.DataFrame(out).set_index("line")


def load_frames(cache: Path, lines) -> dict:
    cols = ["row_id", "exposure", "period_start", "new_business"]
    out = {}
    for k in ("train", "test"):
        for l in lines:
            d = pd.read_parquet(cache / f"{k}_{l}.parquet", columns=cols)
            d["year"] = pd.to_datetime(d.period_start).dt.year
            out[f"{k}_{l}"] = d.drop(columns="period_start").set_index("row_id")
    return out


def ref_target(market: str) -> dict:
    """y_eval of every training row under another market spec (y_eval does not depend on the experiment cfg)."""
    import features
    import market_spec
    from target import prepare_target
    with market_spec.using_spec(market) as s:
        b = features.build_base()
        out = {"lines": {}}
        for l in s.lines:
            tgt, _ = prepare_target(b[f"train_{l}"], b[f"test_{l}"], b["claims"], l, {})
            out["lines"][l] = {"row_id": b[f"train_{l}"].row_id.values, "y_eval": tgt.y_eval.values}
    return out


def main():
    import market_spec
    try:
        import market_spec_pool  # noqa: F401  (registers part2pool if market_spec's hook has not)
    except ImportError:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("exps", nargs="+")
    ap.add_argument("--ref", help="MARKET:EXP whose y_eval is a second yardstick (e.g. part2:L20_mono_lowreg)")
    ap.add_argument("--ref-target", help="MARKET whose y_eval (computed from its cached base) is a second yardstick")
    ap.add_argument("--nb-cal", action="store_true", help="also print the post-hoc NB/renewal calibration check")
    ap.add_argument("--log", metavar="DECISION", help="write the EMERGING-row scores of each exp into experiments/log.csv "
                    "(cv, cv_nested, cv_<line>, holdout_2023 = emerging 2024 holdout) with this decision text")
    a = ap.parse_args()
    spec = market_spec.get_spec()
    frames = load_frames(spec.cache_path, spec.lines)
    ref = ref_target(a.ref_target) if a.ref_target else None
    if a.ref:
        mk, _, ex = a.ref.partition(":")
        with open(market_spec._resolve(mk).results_path / f"{ex}.pkl", "rb") as f:
            ref = pickle.load(f)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    for exp in a.exps:
        with open(spec.results_path / f"{exp}.pkl", "rb") as f:
            res = pickle.load(f)
        t = score(res, frames, ref)
        print(f"== {exp} [{spec.name}] emerging rows only (CV D2 all rows {res.get('cv', np.nan):.4f})")
        print(t.round(4).to_string())
        if a.nb_cal:
            print(f"-- {exp}: post-hoc NB/renewal calibration (emerging rows)")
            print(nb_calibration(res, frames).round(4).to_string())
        if a.log is not None:
            log_scores(exp, t, a.log)


def log_scores(exp: str, t: pd.DataFrame, decision: str) -> None:
    """Overwrite the score columns of exp's log row (written by run_experiment with all-row CV) with emerging-row scores."""
    import explog
    g = lambda line, c: round(float(t.loc[line, c]), 5) if c in t and t.loc[line, c] == t.loc[line, c] else ""
    lines = [l for l in t.index if l != "overall"]
    calib = "/".join(f"{t.loc[l, 'calib']:.2f}" for l in lines)
    hcal = "/".join(f"{t.loc[l, 'holdout_calib']:.2f}" for l in lines) if "holdout_calib" in t else ""
    nb = "/".join(f"{t.loc[l, 'nb_ratio_test']:.2f}" for l in lines)
    ho = "/".join(f"{t.loc[l, 'holdout']:.4f}" for l in lines) if "holdout" in t else ""
    explog.update(exp, cv=g("overall", "cv"), cv_nested=g("overall", "cv_nested"),
                  **{f"cv_{l}": g(l, "cv") for l in lines}, holdout_2023=g("overall", "holdout"),
                  verification="EMERGING rows only (scripts/score_market.py): 5-fold grouped CV + 2024 holdout; "
                               f"holdout {ho}; OOF calib {calib}; holdout calib {hcal}; test NB/renewal {nb}",
                  decision=decision)


if __name__ == "__main__":
    main()
