"""Pooled-market (part2pool) experiments: developed + emerging history, `market` feature, emerging test only.

Constants come from the PART2POOL spec directly (not from the active spec), so this module gives the same configs
whichever market is active; run them with IAA_MARKET=part2pool. Merged into configs.EXPERIMENTS by the guarded hook
in experiments/configs.py, or imported directly. Weight key: `market_weight_developed` (src/pipeline._market_weights).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from market_spec_pool import PART2POOL as _S  # noqa: E402

_RAW = {"develop": False, "cap_q": None, "trend": "none", "weight_power": 0.0}
_FULL_T = {**_RAW, "develop": True, "cap_q": 0.995, "trend": "premium", "weight_power": 1.0}
_TUNED = dict(_S.tuned)
_MONO = dict(_S.monotone)
_HIST_F = {"history": True, "drop": ["make_model"]}   # Review 10 ran with noise_features=(); frozen so its configs reproduce
_P135 = {**_TUNED, "min_data_in_leaf": 2000, "monotone": _MONO, "tweedie_variance_power": 1.35}
_SI = lambda kb, kc=None: {k: {"si_offset_power": v} for k, v in (("buildings", kb), ("contents", kc)) if v}
_NOISE_P1 = ["previous_insurer", "branch_id", "email_domain", "campaign_code"]   # PART1.noise_features
_W = 0.3
_P = "AI (pooled market, Review 10)"

_BASE = {
    "POOL_B15": dict(desc="Pooled market: GLM + boost (B15_denoise_mm), developed weight 0.3", proposer=_P, model="glmboost",
                     target={**_FULL_T, "cap_q": 0.99}, features=_HIST_F),
    "POOL_L20": dict(desc="Pooled market: monotone LightGBM (L20_mono_lowreg), developed weight 0.3", proposer=_P, model="lgbm",
                     params={**_TUNED, "min_data_in_leaf": 2000, "monotone": _MONO},
                     target={**_FULL_T, "cap_q": 0.99, "loading": "segment"}, features=_HIST_F),
    "POOL_P34": dict(desc="Pooled market: P34_b07 (LightGBM p=1.35, segment loading, buildings SI^0.7 offset), developed weight 0.3",
                     proposer=_P, model="lgbm", params=_P135, target={**_FULL_T, "cap_q": 0.99, "loading": "segment"},
                     target_by_line=_SI(0.7), features=_HIST_F),
}
EXPERIMENTS = {k: {**v, "market_weight_developed": _W} for k, v in _BASE.items()}

# developed-weight variants and SI-offset variants for every base (cheap to define; run only the best)
for k, v in _BASE.items():
    for w in (0.1, 1.0):
        EXPERIMENTS[f"{k}_w{int(w * 10):02d}"] = {**v, "market_weight_developed": w,
                                                  "desc": v["desc"].replace("weight 0.3", f"weight {w}")}
    # early stopping judged on emerging rows only (developed validation weight 0)
    EXPERIMENTS[f"{k}_v0"] = {**v, "market_weight_developed": _W, "market_weight_developed_val": 0.0,
                              "desc": v["desc"] + ", early stopping on emerging rows only"}
    EXPERIMENTS[f"{k}_w10_v0"] = {**v, "market_weight_developed": 1.0, "market_weight_developed_val": 0.0,
                                  "desc": v["desc"].replace("weight 0.3", "weight 1.0") + ", early stopping on emerging rows only"}
    # Part 1 noise list dropped (previous_insurer is the top-gain feature in the pooled fits; it was noise in Part 1)
    _dn = {**v.get("features", {}), "drop": list(v.get("features", {}).get("drop", [])) + _NOISE_P1}
    EXPERIMENTS[f"{k}_w10_v0_dn"] = {**EXPERIMENTS[f"{k}_w10_v0"], "features": _dn,
                                     "desc": EXPERIMENTS[f"{k}_w10_v0"]["desc"] + ", Part 1 noise features dropped"}
    for kb in (0.0, 0.5):
        EXPERIMENTS[f"{k}_si{int(kb * 10):02d}"] = {**v, "market_weight_developed": _W, "target_by_line": _SI(kb, kb),
                                                    "desc": v["desc"] + f", buildings/contents SI^{kb} offset"}

# ---- Review 11, pooled round 2 (market-aware development is the default in target.prepare_target whenever a `market`
# column exists; `develop_by_market: False` reproduces the Review 10 pooled development).
_P11 = "AI (pooled round 2, Review 11)"
_R2_BASE = EXPERIMENTS["POOL_P34_w10_v0_dn"]
EXPERIMENTS["R2_P34_dev"] = {**_R2_BASE, "proposer": _P11,
                             "desc": "POOL_P34_w10_v0_dn with claims development fitted per market (emerging: 2023 lag curve)"}
EXPERIMENTS["R2_P34_devload"] = {**_R2_BASE, "proposer": _P11, "target": {**_R2_BASE["target"], "loading_by_market": True},
                                 "desc": "R2_P34_dev + large-loss (segment) loading fitted per market"}

# Step 1 decision (set after the screen): the kept development / loading variant
# (R2_P34_devload: per-market loading kept for motor only; buildings holdout .1217 -> .1007, contents neutral)
_R2_KEEP = {**EXPERIMENTS["R2_P34_dev"],
            "target_by_line": {**EXPERIMENTS["R2_P34_dev"].get("target_by_line", {}), "motor": {"loading_by_market": True}},
            "desc": "R2_P34_dev + motor large-loss loading per market"}
# Step NB (coordinator, 27 Oct 17:41 UTC public probes): emerging new-business relativity
EXPERIMENTS["R2_P34_mnb"] = {**_R2_KEEP, "features": {**_R2_KEEP["features"], "market_nb": True},
                             "desc": _R2_KEEP["desc"] + " + market x new_business feature"}
for _k in (2, 3):
    EXPERIMENTS[f"R2_P34_wnb{_k}"] = {**_R2_KEEP, "weight_emerging_nb": float(_k),
                                      "desc": _R2_KEEP["desc"] + f" + emerging new-business rows x{_k} fit weight"}
# Step 3: noise list = PART2POOL.noise_features (Part 1 list, re-derived on the pooled frame), as in _dn
assert set(_S.noise_features) == set(_NOISE_P1)
# Step 2: first observed year of each market -> history NaN
EXPERIMENTS["R2_P34_hn"] = {**_R2_KEEP, "features": {**_R2_KEEP["features"], "hist_nan_2021": True},
                            "desc": _R2_KEEP["desc"] + " + history NaN in each market's first year"}

# ---- Step 4: pooled parts with every kept round-2 fix (per-market development [default], motor loading per market,
# history NaN in each market's first year, noise list dropped), weights w10 / emerging-only early stopping for trees.
_FIX_F = {"history": True, "drop": list(_S.noise_features) + ["make_model"], "hist_nan_2021": True}
_MOTOR_LBM = {"motor": {"loading_by_market": True}}
_W10V0 = {"market_weight_developed": 1.0, "market_weight_developed_val": 0.0}
_EXPO_COUNTERS = ["cust_expo", "pol_expo", "cust_motor_expo", "cust_buildings_expo", "cust_contents_expo"]
_SMOOTH_GLM_POOL = {"mode": "smooth", "alpha": 1.0,
                    "interactions": [("sum_insured", "flood_zone"), ("ncb_years", "driver_age_band"),
                                     ("vehicle_value_group", "driver_age_band"), ("sum_insured", "property_age")]}
EXPERIMENTS.update({
    "R2_L20": dict(desc="Round 2: monotone LightGBM (L20_mono_lowreg) on the pooled market, all round-2 fixes", proposer=_P11,
                   model="lgbm", params={**_TUNED, "min_data_in_leaf": 2000, "monotone": _MONO},
                   target={**_FULL_T, "cap_q": 0.99, "loading": "segment"}, target_by_line=_MOTOR_LBM, features=_FIX_F, **_W10V0),
    "R2_F22": dict(desc="Round 2: freq x sev GBMs (F22_b07_fs) on the pooled market, all round-2 fixes", proposer=_P11,
                   model="freqsev", params={"freq": {**_TUNED, "min_data_in_leaf": 2000},
                                            "sev": {"num_leaves": 15, "min_data_in_leaf": 500, "learning_rate": 0.03,
                                                    "lambda_l2": 50.0, "n_iter": 600}},
                   target={**_FULL_T, "cap_q": 0.99}, target_by_line={**_MOTOR_LBM, **_SI(0.7)}, features=_FIX_F, **_W10V0),
    **{f"R2_B27_a{a:g}": dict(desc=f"Round 2: smooth GLM (ridge {a:g}) + monotone boost (B27_b07) on the pooled market, all "
                                   "round-2 fixes, early stopping on all rows (emerging-only unstable for GLM+boost)",
                              proposer=_P11, model="glmboost",
                              params={"glm": {**_SMOOTH_GLM_POOL, "alpha": a}, **_TUNED, "min_data_in_leaf": 2000,
                                      "monotone": _MONO, "tweedie_variance_power": 1.35},
                              target={**_FULL_T, "cap_q": 0.99}, target_by_line={**_MOTOR_LBM, **_SI(0.7)},
                              features={**_FIX_F, "engineered": True, "drop": _FIX_F["drop"] + _EXPO_COUNTERS},
                              market_weight_developed=1.0) for a in (0.1, 1.0, 10.0)},
})

# ---- Step 4 blends: weights fitted on EMERGING OOF rows only (pipeline.run_blend weight_market), nested CV reported
_R2_PARTS = ["R2_P34_hn", "R2_L20", "R2_F22", "R2_B27_a10"]   # B27 ridge 10 chosen on the quick split
_P2_PARTS = {"P2_L20_si05": "part2", "P2_F22_si05": "part2", "P2_B27_a10_si05": "part2"}
EXPERIMENTS.update({
    "R2_X4": dict(desc="Round 2 blend of the 4 pooled parts (P34, L20, F22, B27), weights on emerging rows", proposer=_P11, model="blend", step=0.1,
                  parts=_R2_PARTS, weight_market="emerging"),
    # 7 parts on a 0.1 grid is too slow for pipeline._simplex (un-memoised); P34 + F22 carry most of the pooled weight
    "R2_X4_p2": dict(desc="P34 + F22 pooled parts + the three emerging-only parts (P2_L20_si05, P2_F22_si05, P2_B27_a10_si05)",
                     proposer=_P11, model="blend", step=0.1, parts=["R2_P34_hn", "R2_F22"], weight_market="emerging", external_parts=_P2_PARTS),
    "R2_X4_x3": dict(desc="R2_X4 + the emerging-only blend P2_X3_blend as one part", proposer=_P11, model="blend", step=0.1,
                     parts=_R2_PARTS, weight_market="emerging", external_parts={"P2_X3_blend": "part2"}),
})

# ---- Step 5: 3-seed bags of the kept parts (equal weights, nothing fitted)
def _seeded(cfg, sd):
    p = dict(cfg["params"])
    if cfg["model"] == "freqsev":
        p = {"freq": {**p["freq"], "seed": sd}, "sev": {**p["sev"], "seed": sd}}
    else:
        p["seed"] = sd
    return {**cfg, "params": p, "desc": cfg["desc"] + f", seed {sd}"}


for _b in ("R2_P34_hn", "R2_L20", "R2_F22"):
    for _sd in (101, 202):
        EXPERIMENTS[f"{_b}_s{_sd}"] = _seeded(EXPERIMENTS[_b], _sd)
    EXPERIMENTS[f"{_b}_bag"] = dict(desc=f"{_b} averaged over 3 seeds", proposer=_P11, model="blend", average=True,
                                    parts=[_b, f"{_b}_s101", f"{_b}_s202"])

# ---- Final round-2 candidate: seed bags; weights fitted on emerging OOF for motor / contents; buildings = P34 bag alone
# (fitted buildings weights unstable across folds: nested .1172 vs single .1231; P34 has the best buildings holdout)
EXPERIMENTS["R2_FINAL"] = dict(
    desc="Round 2 final: bags of P34/L20/F22 + B27, weights on emerging rows (motor, contents), buildings P34 bag only",
    proposer=_P11, model="blend", step=0.1, weight_market="emerging",
    parts=["R2_P34_hn_bag", "R2_L20_bag", "R2_F22_bag", "R2_B27_a10"],
    fixed_weights={"buildings": {"R2_P34_hn_bag": 1.0}})

# ---- Review 13: the insurer's written premium as a feature (cfg-gated in features.model_matrix, default off).
# written_premium is the ANNUAL premium (flat across exposure bands), so log(premium / exposure) is mostly exposure;
# for trees log(premium) == premium (monotone transform), so `_wpl` is the "premium alone" screen; `_wprel` deflates by
# the (market, year) mean premium (premium trend 4-5 %/yr would otherwise act as a calendar proxy).
_P13 = "AI (written-premium screen, Review 13)"
for _sfx, _f, _d in (("wp", {"premium": True, "premium_rate": True}, "raw written_premium + log(premium / exposure)"),
                     ("wpl", {"premium_log": True}, "log(written_premium) alone"),
                     ("wprel", {"premium_rel": True}, "log(written_premium / mean premium of the market-year)")):
    for _b in ("R2_P34_hn", "R2_F22"):
        EXPERIMENTS[f"{_b}_{_sfx}"] = {**EXPERIMENTS[_b], "proposer": _P13,
                                       "features": {**EXPERIMENTS[_b]["features"], **_f},
                                       "desc": EXPERIMENTS[_b]["desc"] + f" + premium feature: {_d}"}
# seed check of the premium-alone variant against the existing seeds 101 / 202 (paired 3-seed comparison) and bags
for _b in ("R2_P34_hn_wpl", "R2_F22_wpl"):
    for _sd in (101, 202):
        EXPERIMENTS[f"{_b}_s{_sd}"] = _seeded(EXPERIMENTS[_b], _sd)
    EXPERIMENTS[f"{_b}_bag"] = dict(desc=f"{_b} averaged over 3 seeds", proposer=_P13, model="blend", average=True,
                                    parts=[_b, f"{_b}_s101", f"{_b}_s202"])
# candidate: R2_FINAL with the premium-feature P34 bag as an extra part (motor, contents weights fitted on emerging
# rows); F22 keeps no premium (R2_F22_wpl: CV -.0006/-.0007/+.0002, holdout -.0012/-.0013/-.0020); buildings stays the
# R2_P34_hn bag (premium neutral there over 3 seeds)
EXPERIMENTS["R2_FINAL_wpl"] = {**EXPERIMENTS["R2_FINAL"], "proposer": _P13,
                               "parts": ["R2_P34_hn_wpl_bag", "R2_L20_bag", "R2_F22_bag", "R2_B27_a10", "R2_P34_hn_bag"],
                               "fixed_weights": {"buildings": {"R2_P34_hn_bag": 1.0}},
                               "desc": "R2_FINAL + the log(written_premium) P34 bag as a fifth part (weights on emerging rows, "
                                       "motor, contents); buildings = R2_P34_hn bag"}

# ---- Agent S1 (28 Oct, shape screens on the R2_FINAL candidate; decided on EMERGING nested CV + 2024 holdout)
# 1. CatBoost pooled part: R2_P34_hn's target / features / market weights (all round-2 fixes, emerging-only early
#    stopping), model = Part 1 A9_b07_cat (CatBoostTweedie defaults: p 1.5, depth 6, lr .06, l2 10; no monotone).
_PS1 = "AI (shape screens, agent S1)"
EXPERIMENTS["R2_C9"] = {**EXPERIMENTS["R2_P34_hn"], "proposer": _PS1, "model": "catboost", "params": {},
                        "desc": "Round 2: CatBoost Tweedie (A9_b07_cat) on the pooled market, R2_P34_hn target/features/"
                                "weights (all round-2 fixes), early stopping on emerging rows"}
EXPERIMENTS["R2_FINAL_C9"] = {**EXPERIMENTS["R2_FINAL"], "proposer": _PS1,
                              "parts": EXPERIMENTS["R2_FINAL"]["parts"] + ["R2_C9"],
                              "desc": "R2_FINAL + the pooled CatBoost part R2_C9 (weights on emerging rows, motor, "
                                      "contents); buildings = R2_P34_hn bag"}
# 4. Developed-row fit weight screen on the P34 part (single seed 0, paired with R2_P34_hn at weight 1.0; validation
#    stays emerging-only)
for _w in (0.5, 2.0):
    EXPERIMENTS[f"R2_P34_hn_w{int(_w * 10):02d}"] = {**EXPERIMENTS["R2_P34_hn"], "proposer": _PS1, "market_weight_developed": _w,
                                                    "desc": EXPERIMENTS["R2_P34_hn"]["desc"] + f", developed fit weight {_w}"}

# ---- Agent S2 (28 Oct): large-loss cap quantile per market (target cfg `cap_by_market`, src/target._cap_by_market):
# each market's claims capped at its own q99 of the line, loading fitted per market (each market back to its own level;
# segment loading per market for every line). Single seed 0, paired with R2_P34_hn.
_PS2 = "AI (per-market cap screen, agent S2)"
EXPERIMENTS["R2_P34_hn_capm"] = {**EXPERIMENTS["R2_P34_hn"], "proposer": _PS2,
                                 "target": {**EXPERIMENTS["R2_P34_hn"]["target"], "cap_by_market": True},
                                 "desc": EXPERIMENTS["R2_P34_hn"]["desc"] + " + large-loss cap q99 per market"}
