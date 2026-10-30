"""Part 2 (emerging market) experiment definitions, merged into configs.EXPERIMENTS by the hook at the end of
experiments/configs.py. Run them with IAA_MARKET=part2 (src/market_spec.py PART2); evidence in docs/part2_decisions.md.

Every Part 1 reference config is copied with its leaf sizes scaled to the emerging market's row count
(Part 1 ~0.7-1.0M rows per line, Part 2 27-47k: factor ~1/20) and the Part 1-only columns removed.
This module does not import configs.py (it is imported from there), and reads the PART2 spec explicitly so the
definitions are the same whichever market is active when the registry is built.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from market_spec import REGISTRY  # noqa: E402

_S = REGISTRY["part2"]
NOISE = list(_S.noise_features)                 # () until the Part 2 stability screen re-derives it
TUNED = dict(_S.tuned)                          # min_data_in_leaf 300 (Part 1: 8000), feature_fraction 0.5
MONOTONE = dict(_S.monotone)                    # Part 1 directions minus annual_mileage (absent)
EXPO_COUNTERS = ["cust_expo", "pol_expo", "cust_motor_expo", "cust_buildings_expo", "cust_contents_expo"]

FULL_T = {"develop": True, "cap_q": 0.99, "trend": "premium", "weight_power": 1.0}
SEG_T = {**FULL_T, "loading": "segment"}
HIST_F = {"history": True, "drop": NOISE + ["make_model"]}
LEAF_LOW = 100                                  # Part 1 "low regularisation" 2000 x 1/20
LEAF_B15 = 1000                                 # Part 1 LightGBM default 20000 x 1/20 (B15 used the default)
_SI = lambda k: {l: {"si_offset_power": k} for l in ("buildings", "contents")} if k else {}
_P135 = {**TUNED, "min_data_in_leaf": LEAF_LOW, "monotone": MONOTONE, "tweedie_variance_power": 1.35}
# Part 1 smooth-GLM interactions re-expressed without driver_age / annual_mileage (absent in Part 2)
_SMOOTH_GLM = {"mode": "smooth", "alpha": 1e-4,
               "interactions": [("sum_insured", "flood_zone"), ("ncb_years", "driver_age_band"),
                                ("vehicle_value_group", "driver_age_band"), ("sum_insured", "property_age")]}
_PROP = "AI (Part 2 adaptation, 27 Oct)"

EXPERIMENTS = {
    # B15_denoise_mm: GLM (banded) + Tweedie GBM boosting from it
    "P2_B15": dict(desc="Part 2: B15_denoise_mm (GLM + GBM boost), leaf 1000", proposer=_PROP, model="glmboost",
                   params={"min_data_in_leaf": LEAF_B15}, target=FULL_T, features=HIST_F),
    # L17_tuned: tuned standalone Tweedie GBM; leaf-size sensitivity around the scaled value
    **{f"P2_L17_m{m}": dict(desc=f"Part 2: L17_tuned standalone GBM, min leaf {m}", proposer=_PROP, model="lgbm",
                            params={**TUNED, "min_data_in_leaf": m}, target=FULL_T, features=HIST_F)
       for m in (150, 300, 600)},
    # L20_mono_lowreg: low regularisation + monotone constraints + segment large-loss loading
    "P2_L20": dict(desc="Part 2: L20_mono_lowreg (GBM, leaf 100, monotone, segment loading)", proposer=_PROP, model="lgbm",
                   params={**TUNED, "min_data_in_leaf": LEAF_LOW, "monotone": MONOTONE}, target=SEG_T, features=HIST_F),
    "P2_L20_nohist": dict(desc="P2_L20 without experience rating (76 % of test rows are new business)", proposer=_PROP,
                          model="lgbm", params={**TUNED, "min_data_in_leaf": LEAF_LOW, "monotone": MONOTONE},
                          target=SEG_T, features={**HIST_F, "history": False}),
    "P2_L20_nan1": dict(desc="P2_L20 with 2023 (first-year) history unknown, not zero", proposer=_PROP, model="lgbm",
                        params={**TUNED, "min_data_in_leaf": LEAF_LOW, "monotone": MONOTONE}, target=SEG_T,
                        features={**HIST_F, "hist_nan_2021": True}),
    # P34 family (P20_135: power 1.35, monotone, segment loading) with SI elasticity k in the home offsets
    **{f"P2_P34_si{int(k * 10):02d}": dict(desc=f"Part 2: P20_135 GBM with sum_insured^{k} in the buildings+contents offset",
                                          proposer=_PROP, model="lgbm", params=_P135, target=SEG_T,
                                          target_by_line=_SI(k), features=HIST_F) for k in (0.0, 0.5)},
    # B20/B27 family: smooth GLM base (no cumulative-exposure counters) + monotone boost
    **{f"P2_B27_si{int(k * 10):02d}": dict(desc=f"Part 2: B27 smooth GLM + boost, sum_insured^{k} in the home offsets",
                                          proposer=_PROP, model="glmboost",
                                          params={"glm": _SMOOTH_GLM, **TUNED, "min_data_in_leaf": LEAF_LOW,
                                                  "monotone": MONOTONE, "tweedie_variance_power": 1.35},
                                          target=FULL_T, target_by_line=_SI(k),
                                          features={**HIST_F, "engineered": True,
                                                    "drop": HIST_F["drop"] + EXPO_COUNTERS}) for k in (0.0, 0.5)},
    # F22: frequency x severity GBMs (sev leaf 500 -> 25 on ~1-7k claims per line)
    "P2_F22": dict(desc="Part 2: F22 freq x sev GBMs (freq leaf 100, sev leaf 25)", proposer=_PROP, model="freqsev",
                   params={"freq": {**TUNED, "min_data_in_leaf": LEAF_LOW},
                           "sev": {"num_leaves": 15, "min_data_in_leaf": 25, "learning_rate": 0.03, "lambda_l2": 50.0,
                                   "n_iter": 600}},
                   target=FULL_T, features=HIST_F),
}

# Screen 1 (27 Oct): GLM-based parts collapse (B15 bld -0.48, B27 bld -0.27): the Part 1 ridge alpha 1e-4 was set on
# ~20x more rows. Reference constant + GLM ridge sensitivity, then the GLM bases with the chosen alpha.
EXPERIMENTS.update({
    "P2_E0": dict(desc="Part 2: constant pure premium per line x exposure (reference)", proposer=_PROP, model="const",
                  target=FULL_T, features=HIST_F),
    **{f"P2_G_a{a:g}": dict(desc=f"Part 2: banded Tweedie GLM, ridge alpha {a:g}", proposer=_PROP, model="glm",
                            params={"alpha": a}, target=FULL_T, features=HIST_F) for a in (1e-3, 1e-2, 1e-1)},
    **{f"P2_B15_a{a:g}": dict(desc=f"P2_B15 with GLM ridge alpha {a:g}", proposer=_PROP, model="glmboost",
                              params={"glm": {"alpha": a}, "min_data_in_leaf": LEAF_B15}, target=FULL_T, features=HIST_F)
       for a in (1e-2, 1e-1)},
    **{f"P2_B27_a{a:g}": dict(desc=f"P2_B27_si05 with smooth-GLM ridge alpha {a:g}", proposer=_PROP, model="glmboost",
                              params={"glm": {**_SMOOTH_GLM, "alpha": a}, **TUNED, "min_data_in_leaf": LEAF_LOW,
                                      "monotone": MONOTONE, "tweedie_variance_power": 1.35},
                              target=FULL_T, target_by_line=_SI(0.5),
                              features={**HIST_F, "engineered": True, "drop": HIST_F["drop"] + EXPO_COUNTERS})
       for a in (1e-2, 1e-1)},
})

# Screen 2 (27 Oct): per-line GLM alpha sweep on the quick split (scratch glmalpha.py): banded GLM best at alpha 10
# (motor 0.062, bld -0.081, cnt 0.038), smooth GLM at 10-30 (0.053-0.058 / -0.010-0.002 / 0.102-0.085) vs alpha 1e-4
# collapsing. Tweedie deviance per row is O(sqrt(cost)), so the ridge must be large; 1k-7k claims per line.
GLM_ALPHA = 10.0
EXPERIMENTS.update({
    "P2_B15_a10": dict(desc="P2_B15 with GLM ridge alpha 10", proposer=_PROP, model="glmboost",
                       params={"glm": {"alpha": GLM_ALPHA}, "min_data_in_leaf": LEAF_B15}, target=FULL_T, features=HIST_F),
    **{f"P2_B27_a10_si{int(k * 10):02d}": dict(desc=f"P2_B27 smooth GLM (alpha 10) + boost, sum_insured^{k} in the home offsets",
                                              proposer=_PROP, model="glmboost",
                                              params={"glm": {**_SMOOTH_GLM, "alpha": GLM_ALPHA}, **TUNED,
                                                      "min_data_in_leaf": LEAF_LOW, "monotone": MONOTONE,
                                                      "tweedie_variance_power": 1.35},
                                              target=FULL_T, target_by_line=_SI(k),
                                              features={**HIST_F, "engineered": True, "drop": HIST_F["drop"] + EXPO_COUNTERS})
       for k in (0.0, 0.5)},
    "P2_F22_si05": dict(desc="P2_F22 with sum_insured^0.5 in the home offsets", proposer=_PROP, model="freqsev",
                        params={"freq": {**TUNED, "min_data_in_leaf": LEAF_LOW},
                                "sev": {"num_leaves": 15, "min_data_in_leaf": 25, "learning_rate": 0.03, "lambda_l2": 50.0,
                                        "n_iter": 600}},
                        target=FULL_T, target_by_line=_SI(0.5), features=HIST_F),
    "P2_L20_si05": dict(desc="P2_L20 with sum_insured^0.5 in the home offsets", proposer=_PROP, model="lgbm",
                        params={**TUNED, "min_data_in_leaf": LEAF_LOW, "monotone": MONOTONE}, target=SEG_T,
                        target_by_line=_SI(0.5), features=HIST_F),
})

# First Part 2 blend: the three best single families after the full 5-fold CV
EXPERIMENTS.update({
    "P2_X3_blend": dict(desc="Part 2 per-line blend: P2_L20_si05 (GBM), P2_F22_si05 (freq x sev), P2_B27_a10_si05 (smooth GLM + boost)",
                        proposer=_PROP, model="blend", step=0.05, parts=["P2_L20_si05", "P2_F22_si05", "P2_B27_a10_si05"]),
})

# ---- Agent H1 feature hunt (28 Oct, docs/part2_feature_hunt.md): emerging-only quick screens vs P2_P34_si05.
# Run: $env:IAA_MARKET="part2"; python -P -E scripts/part2_feature_hunt.py screen P2_P34_si05 H1_... --seeds 11 12 13
_H1 = "AI (feature hunt H1, 28 Oct)"
_H1B = EXPERIMENTS["P2_P34_si05"]
_H1_WX = ["hail_days", "wind_exposure_index", "freezing_days", "dry_summer_index", "wildfire_index", "rainfall_mm",
          "mean_temperature_c"]
_H1_NOISE = ["previous_insurer", "branch_id", "email_domain", "campaign_code"]       # PART2POOL.noise_features


def _h1(desc, features=None, target_by_line=None):
    c = {**_H1B, "proposer": _H1, "desc": "P2_P34_si05 " + desc}
    if features is not None:
        c["features"] = {**_H1B["features"], **features}
    if target_by_line is not None:
        c["target_by_line"] = target_by_line
    return c


EXPERIMENTS.update({
    "H1_P34_nogender": _h1("without gender (value of the emerging-only attribute the pooled model under-uses)",
                           {"drop": HIST_F["drop"] + ["gender"]}),
    "H1_P34_dn": _h1("with the pooled noise list dropped (previous_insurer, branch_id, email_domain, campaign_code)",
                     {"drop": HIST_F["drop"] + _H1_NOISE}),
    "H1_P34_rich": _h1("+ rich history (hx_*: prior per-peril counts and capped prior cost, policy and customer)",
                       {"rich_history": True}),
    "H1_P34_nowx": _h1("without the 7 same-year weather columns (2025 weather PSI 0.45-0.96 vs 2024)",
                       {"drop": HIST_F["drop"] + _H1_WX}),
    "H1_P34_nomonth": _h1("without start_month and row_type (test NB start in January like renewals)",
                          {"drop": HIST_F["drop"] + ["start_month", "row_type"]}),
    "H1_P34_season": _h1("+ seasonal exposure offset on buildings/contents (monthly peril-cost index)",
                         target_by_line={l: {"si_offset_power": 0.5, "season_offset": 1.0} for l in ("buildings", "contents")}),
})
