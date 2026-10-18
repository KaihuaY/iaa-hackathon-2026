"""Experiment definitions. Each builds on the previous accepted one (see docs/experiment_log.md)."""

RAW = {"develop": False, "cap_q": None, "trend": "none", "weight_power": 0.0}
DEV = {**RAW, "develop": True}
DEV_CAP = {**DEV, "cap_q": 0.995}
FULL_T = {**DEV_CAP, "trend": "premium", "weight_power": 1.0}

# Feature lists / tuned params are market assumptions: they come from the active MarketSpec (src/market_spec.py),
# whose Part 1 defaults are the original constants. A new market overrides them in its spec.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from market_spec import get_spec  # noqa: E402

_S = get_spec()
NOISE = list(_S.noise_features)
# cumulative-exposure history counters: calendar-time proxies (0 in 2021, ~1.7 in 2022, ~3 from 2023); unsafe in the
# smooth GLM's quadratic terms under a time shift (review 4), harmless in trees
EXPO_COUNTERS = ["cust_expo", "pol_expo", "cust_motor_expo", "cust_buildings_expo", "cust_contents_expo"]
TUNED = dict(_S.tuned)               # quick-screen winner after de-noising (17 Oct)
# Features with no plausible link to the line's perils and/or unstable year to year (17 Oct screen)
IRRELEVANT = {k: list(v) for k, v in _S.irrelevant.items()}

BASE_F = {"history": False}
HIST_F = {"history": True}
# Directions any pricing actuary would expect (+1 = cost rises with the feature); missing features are ignored.
MONOTONE = dict(_S.monotone)

# P20_135 base settings, reused by the local-loop experiments below
_P135 = {**TUNED, "min_data_in_leaf": 2000, "monotone": MONOTONE, "tweedie_variance_power": 1.35}
_T135 = {**FULL_T, "cap_q": 0.99, "loading": "segment"}
_F135 = {**HIST_F, "drop": NOISE + ["make_model"]}

EXPERIMENTS = {
    "E0_const": dict(desc="Average cost per line x exposure (pipeline / metric check vs host's 0.072)", proposer="AI",
                     model="const", target=RAW, features=BASE_F),
    "E1_glm": dict(desc="Tweedie GLM per line, banded factors, raw labels", proposer="user (decision board #1/#9)",
                   model="glm", target=RAW, features=BASE_F),
    "E2_lgbm": dict(desc="Tweedie LightGBM per line, raw labels", proposer="user (decision board #1)",
                    model="lgbm", target=RAW, features=BASE_F),
    "E3_dev": dict(desc="E2 + develop 2024 to ultimate (BF IBNR)", proposer="user (decision board #3)",
                   model="lgbm", target=DEV, features=BASE_F),
    "E4_cap": dict(desc="E3 + cap at 99.5th pct with large-loss loading", proposer="user (decision board #5)",
                   model="lgbm", target=DEV_CAP, features=BASE_F),
    "E5_hist": dict(desc="E4 + experience rating (prior claims of policy/customer)", proposer="user (decision board #2)",
                    model="lgbm", target=DEV_CAP, features=HIST_F),
    "E6_trend": dict(desc="E5 + premium-index trend to 2025 + maturity weights", proposer="user (decision board #6/#7)",
                     model="lgbm", target=FULL_T, features=HIST_F),
    "E7_glm_full": dict(desc="GLM with E6 target and features (blend partner)", proposer="AI",
                        model="glm", target=FULL_T, features=HIST_F),
    "E8_blend": dict(desc="Blend E6 LightGBM + E7 GLM, weights per line on CV", proposer="user (decision board #9)",
                     model="blend", parts=["E6_trend", "E7_glm_full"]),
    # After the 17 Oct diagnostic: standalone GBM only matched the GLM, so the GLM is the base and the GBM boosts from it.
    "G3_dev": dict(desc="GLM + develop to ultimate (BF IBNR)", proposer="user (#3), AI re-sequenced onto GLM",
                   model="glm", target=DEV, features=BASE_F),
    "G4_cap": dict(desc="G3 + cap 99.5th pct with loading", proposer="user (#5)", model="glm", target=DEV_CAP, features=BASE_F),
    "G5_hist": dict(desc="G4 + experience rating", proposer="user (#2)", model="glm", target=DEV_CAP, features=HIST_F),
    "G6_trend": dict(desc="G5 + premium-index trend to 2025 + maturity weights", proposer="user (#6/#7)",
                     model="glm", target=FULL_T, features=HIST_F),
    "B6_boost": dict(desc="G6 GLM + regularised Tweedie GBM boosting from the GLM", proposer="AI (after GBM diagnostic)",
                     model="glmboost", target=FULL_T, features=HIST_F),
    "L6_lgbm": dict(desc="Standalone regularised Tweedie GBM, G6 target/features (challenger)", proposer="user (#1)",
                    model="lgbm", target=FULL_T, features=HIST_F),
    "X1_blend": dict(desc="Per-line blend of B6 (GLM+boost) and L6 (standalone GBM), weights on CV",
                     proposer="user (decision board #9) + AI (per-line evidence)", model="blend", parts=["B6_boost", "L6_lgbm"]),
    # Day 2: single changes vs L6 (standalone GBM, LB 0.12345), the strongest single model
    "L7_nan": dict(desc="L6 + history unknown (not zero) for 2021 rows", proposer="AI (from G5 diagnosis)",
                   model="lgbm", target=FULL_T, features={**HIST_F, "hist_nan_2021": True}),
    "L8_trendB": dict(desc="L6 with claims-based trend", proposer="user (trend: start D, test B)",
                      model="lgbm", target={**FULL_T, "trend": "claims"}, features=HIST_F),
    "L9_tuned": dict(desc="L6 with 31 leaves, min leaf 10k, lr 0.02", proposer="AI (tuning)",
                     model="lgbm", params={"num_leaves": 31, "min_data_in_leaf": 10000, "learning_rate": 0.02},
                     target=FULL_T, features=HIST_F),
    "L10_cap99": dict(desc="L6 with cap at 99th pct", proposer="AI (cap sensitivity)",
                      model="lgbm", target={**FULL_T, "cap_q": 0.99}, features=HIST_F),
    "L11_cap999": dict(desc="L6 with cap at 99.9th pct", proposer="AI (cap sensitivity)",
                       model="lgbm", target={**FULL_T, "cap_q": 0.999}, features=HIST_F),
    "L12_wpca": dict(desc="L6 with weather replaced by 3 principal components", proposer="user (PCA question)",
                     model="lgbm", target=FULL_T, features={**HIST_F, "weather_pca": 3}),
    "L13_prem": dict(desc="DIAGNOSTIC: L6 + written_premium (user approval needed before any final use)",
                     proposer="user (test, decide on results)", model="lgbm", target=FULL_T, features={**HIST_F, "premium": True}),
    "B10_cap99": dict(desc="B6 (GLM+boost) with cap at 99th pct", proposer="AI (cap sensitivity from L10)",
                      model="glmboost", target={**FULL_T, "cap_q": 0.99}, features=HIST_F),
    "X2_blend": dict(desc="Per-line blend of B10 and L10 (both cap 99%)", proposer="user (#9) + AI", model="blend",
                     parts=["B10_cap99", "L10_cap99"]),
    # Noise-feature removal (stability screen 17 Oct: level frequencies uncorrelated year to year)
    "L14_denoise": dict(desc="L10 minus noise features (previous_insurer, branch_id, email_domain, campaign_code)",
                        proposer="AI (feature stability screen)", model="lgbm", target={**FULL_T, "cap_q": 0.99},
                        features={**HIST_F, "drop": NOISE}),
    "L15_denoise_mm": dict(desc="L14 also without make_model (300 levels, stability 0.23)",
                           proposer="AI (feature stability screen)", model="lgbm", target={**FULL_T, "cap_q": 0.99},
                           features={**HIST_F, "drop": NOISE + ["make_model"]}),
    "B14_denoise": dict(desc="B10 (GLM+boost) minus noise features", proposer="AI (feature stability screen)",
                        model="glmboost", target={**FULL_T, "cap_q": 0.99}, features={**HIST_F, "drop": NOISE}),
    "B15_denoise_mm": dict(desc="B14 also without make_model", proposer="AI (feature stability screen)",
                           model="glmboost", target={**FULL_T, "cap_q": 0.99}, features={**HIST_F, "drop": NOISE + ["make_model"]}),
    "X3_blend": dict(desc="Per-line blend of B15 and L15 (de-noised, cap 99%)", proposer="user (#9) + AI", model="blend",
                     parts=["B15_denoise_mm", "L15_denoise_mm"]),
    "L16_relevant": dict(desc="L15 minus line-irrelevant / unstable features (property fields for motor, colour, roof/soil for contents)",
                         proposer="AI (stability screen + domain review)", model="lgbm", target={**FULL_T, "cap_q": 0.99},
                         features={**HIST_F, "drop": NOISE + ["make_model"], "drop_line": IRRELEVANT}),
    "F15_freqsev": dict(desc="Frequency (Poisson GBM) x severity (Gamma GBM), L15 features and target",
                        proposer="AI (round-2 card #1B)", model="freqsev", target={**FULL_T, "cap_q": 0.99},
                        features={**HIST_F, "drop": NOISE + ["make_model"]}),
    "X4_blend3": dict(desc="3-way per-line blend: GLM+boost (B15), Tweedie GBM (L15), freq x sev GBM (F15)",
                      proposer="AI (model diversity)", model="blend", parts=["B15_denoise_mm", "L15_denoise_mm", "F15_freqsev"]),
    "P15_perperil": dict(desc="Per-peril Tweedie GBMs summed (perils < 1k claims pooled), L15 features and target",
                         proposer="AI (round-2 card #1A)", model="perperil", target={**FULL_T, "cap_q": 0.99},
                         features={**HIST_F, "drop": NOISE + ["make_model"]}),
    "L17_tuned": dict(desc="L15 with tuned GBM (min leaf 8k, feature fraction 0.5); quick screen +0.0012",
                      proposer="AI (quick-screen tuning)", model="lgbm", params=TUNED,
                      target={**FULL_T, "cap_q": 0.99}, features={**HIST_F, "drop": NOISE + ["make_model"]}),
    "L19_segload": dict(desc="L17 with segment-specific large-loss loading (cover x age for motor, sum-insured quartile for home)",
                        proposer="AI (residual scan: third_party_only 1.09x)", model="lgbm", params=TUNED,
                        target={**FULL_T, "cap_q": 0.99, "loading": "segment"},
                        features={**HIST_F, "drop": NOISE + ["make_model"]}),
    "L20_mono_lowreg": dict(desc="L17 features/target, min leaf 2000 + monotone constraints (less regularisation, drift guarded)",
                            proposer="AI (residual scan: tails flattened by regularisation)", model="lgbm",
                            params={**TUNED, "min_data_in_leaf": 2000, "monotone": MONOTONE},
                            target={**FULL_T, "cap_q": 0.99, "loading": "segment"}, features={**HIST_F, "drop": NOISE + ["make_model"]}),
    "F17_freqsev": dict(desc="F15 with the same tuned frequency GBM", proposer="AI (quick-screen tuning)",
                        model="freqsev", params={"freq": TUNED}, target={**FULL_T, "cap_q": 0.99},
                        features={**HIST_F, "drop": NOISE + ["make_model"]}),
    "X5_blend3": dict(desc="3-way blend: B15 (GLM+boost), L17 (tuned GBM), F17 (tuned freq x sev)",
                      proposer="AI", model="blend", parts=["B15_denoise_mm", "L17_tuned", "F17_freqsev"]),
    "L18_richhist": dict(desc="L17 + prior claim cost and prior claims by peril (policy and customer, 2-yr window)",
                         proposer="AI (new signal: richer experience rating)", model="lgbm", params=TUNED,
                         target={**FULL_T, "cap_q": 0.99},
                         features={**HIST_F, "rich_history": True, "drop": NOISE + ["make_model"]}),
    # Seed bagging: same models, different random seeds, averaged (variance reduction)
    **{f"L17_s{s}": dict(desc=f"L17 with seed {s}", proposer="AI (seed bagging)", model="lgbm", params={**TUNED, "seed": s},
                         target={**FULL_T, "cap_q": 0.99}, features={**HIST_F, "drop": NOISE + ["make_model"]}) for s in (101, 202)},
    **{f"F17_s{s}": dict(desc=f"F17 with seed {s}", proposer="AI (seed bagging)", model="freqsev",
                         params={"freq": {**TUNED, "seed": s}, "sev": {"seed": s}},
                         target={**FULL_T, "cap_q": 0.99}, features={**HIST_F, "drop": NOISE + ["make_model"]}) for s in (101, 202)},
    "L17_bag": dict(desc="L17 averaged over 3 seeds", proposer="AI (seed bagging)", model="blend", average=True,
                    parts=["L17_tuned", "L17_s101", "L17_s202"]),
    "F17_bag": dict(desc="F17 averaged over 3 seeds", proposer="AI (seed bagging)", model="blend", average=True,
                    parts=["F17_freqsev", "F17_s101", "F17_s202"]),
    "X6_blend3bag": dict(desc="3-way blend: B15 + seed-bagged L17 + seed-bagged F17", proposer="AI (seed bagging)",
                         model="blend", parts=["B15_denoise_mm", "L17_bag", "F17_bag"]),
    "X8_blend_cred": dict(desc="3-way blend: B15 (GLM+boost), L20 corrected (credibility layer), F17_bag (freq x sev)",
                          proposer="AI (global plan steps 1-4)", model="blend", parts=["B15_denoise_mm", "L20_mono_lowreg_cred", "F17_bag"]),
    "X9_blend_l20": dict(desc="3-way blend: B15, L20 (uncorrected), F17_bag", proposer="AI (ablation of the correction layer)",
                         model="blend", parts=["B15_denoise_mm", "L20_mono_lowreg", "F17_bag"]),
    "X10_blend4": dict(desc="4-way blend: B15, L20 corrected, F17_bag, A1 CatBoost (cached predictions from branch advanced-ml)",
                       proposer="AI", model="blend", step=0.1, parts=["B15_denoise_mm", "L20_mono_lowreg_cred", "F17_bag", "A1_catboost"]),
    # Tweedie training power (metric stays p=1.5; the best training objective may differ by line)
    **{f"P20_{str(pw).replace('.', '')}": dict(desc=f"L20 with Tweedie training power {pw}", proposer="AI (plan: training power)",
                                               model="lgbm", params={**TUNED, "min_data_in_leaf": 2000, "monotone": MONOTONE,
                                                                     "tweedie_variance_power": pw},
                                               target={**FULL_T, "cap_q": 0.99, "loading": "segment"},
                                               features={**HIST_F, "drop": NOISE + ["make_model"]}) for pw in (1.2, 1.35, 1.65, 1.8)},
    # ---- local model-improvement loop (backlog) ----
    **{f"PL_{k}": dict(desc=f"P20_135 with per-line Tweedie power motor {a} / buildings {b} / contents {c}",
                       proposer="AI (local loop, backlog #1)", model="lgbm", params=_P135,
                       params_by_line={"motor": {"tweedie_variance_power": a}, "buildings": {"tweedie_variance_power": b},
                                       "contents": {"tweedie_variance_power": c}},
                       target=_T135, features=_F135)
       for k, (a, b, c) in {"a": (1.2, 1.5, 1.35), "b": (1.35, 1.65, 1.5), "c": (1.2, 1.65, 1.35), "d": (1.35, 1.5, 1.35)}.items()},
    "F18_sevtune": dict(desc="F17 with tuned gamma severity (15 leaves, min leaf 500, lr 0.03, l2 50, early stopping)",
                        proposer="AI (local loop, backlog #2)", model="freqsev",
                        params={"freq": {**TUNED, "min_data_in_leaf": 2000},
                                "sev": {"num_leaves": 15, "min_data_in_leaf": 500, "learning_rate": 0.03, "lambda_l2": 50.0, "n_iter": 600}},
                        target={**FULL_T, "cap_q": 0.99}, features={**HIST_F, "drop": NOISE + ["make_model"]}),
    "F19_sevtune2": dict(desc="F17 with tuned gamma severity (31 leaves, min leaf 200, lr 0.02, l2 100, early stopping)",
                         proposer="AI (local loop, backlog #3)", model="freqsev",
                         params={"freq": {**TUNED, "min_data_in_leaf": 2000},
                                 "sev": {"num_leaves": 31, "min_data_in_leaf": 200, "learning_rate": 0.02, "lambda_l2": 100.0, "n_iter": 900}},
                         target={**FULL_T, "cap_q": 0.99}, features={**HIST_F, "drop": NOISE + ["make_model"]}),
    "A4_cat_mono": dict(desc="A1 CatBoost with monotone constraints", proposer="AI (local loop, backlog #4)",
                        model="catboost", params={"monotone": MONOTONE}, target={**FULL_T, "cap_q": 0.99},
                        features={**HIST_F, "drop": NOISE + ["make_model"]}),
    "A5_cat_deep": dict(desc="A4 CatBoost depth 8, l2 20, lr 0.04", proposer="AI (local loop, backlog #5)",
                        model="catboost", params={"depth": 8, "l2_leaf_reg": 20, "learning_rate": 0.04, "monotone": MONOTONE},
                        target={**FULL_T, "cap_q": 0.99}, features={**HIST_F, "drop": NOISE + ["make_model"]}),
    "A6_cat_shallow": dict(desc="A4 CatBoost depth 5, l2 5, lr 0.08", proposer="AI (local loop, backlog #6)",
                           model="catboost", params={"depth": 5, "l2_leaf_reg": 5, "learning_rate": 0.08, "monotone": MONOTONE},
                           target={**FULL_T, "cap_q": 0.99}, features={**HIST_F, "drop": NOISE + ["make_model"]}),
    "A7_cat_shallow_free": dict(desc="A6 CatBoost depth 5, l2 5, lr 0.08, NO monotone constraints", proposer="AI (local loop, backlog #3)",
                                model="catboost", params={"depth": 5, "l2_leaf_reg": 5, "learning_rate": 0.08},
                                target={**FULL_T, "cap_q": 0.99}, features={**HIST_F, "drop": NOISE + ["make_model"]}),
    "X12_blend4_f18": dict(desc="4-way blend: B15, P20_135_cred, F18 (tuned severity), A1 CatBoost", proposer="AI (local loop)",
                           model="blend", step=0.1, parts=["B15_denoise_mm", "P20_135_cred", "F18_sevtune", "A1_catboost"]),
    "X13_blend5": dict(desc="5-way blend: X12 parts + E1 extra-trees", proposer="AI (local loop)",
                       model="blend", step=0.1, parts=["B15_denoise_mm", "P20_135_cred", "F18_sevtune", "A1_catboost", "E1_extra"]),
    # Round 3: seed bags of the best single models; finer GLM base for GLM+boost
    **{f"P20_135_s{sd}": dict(desc=f"P20_135 with seed {sd}", proposer="AI (local loop, seed bag)", model="lgbm",
                              params={**TUNED, "min_data_in_leaf": 2000, "monotone": MONOTONE, "tweedie_variance_power": 1.35, "seed": sd},
                              target={**FULL_T, "cap_q": 0.99, "loading": "segment"},
                              features={**HIST_F, "drop": NOISE + ["make_model"]}) for sd in (101, 202)},
    "P20_135_bag": dict(desc="P20_135 averaged over 3 seeds", proposer="AI (local loop, seed bag)", model="blend", average=True,
                        parts=["P20_135", "P20_135_s101", "P20_135_s202"]),
    **{f"F18_s{sd}": dict(desc=f"F18 with seed {sd}", proposer="AI (local loop, seed bag)", model="freqsev",
                          params={"freq": {**TUNED, "min_data_in_leaf": 2000, "seed": sd},
                                  "sev": {"num_leaves": 15, "min_data_in_leaf": 500, "learning_rate": 0.03, "lambda_l2": 50.0, "n_iter": 600, "seed": sd}},
                          target={**FULL_T, "cap_q": 0.99}, features={**HIST_F, "drop": NOISE + ["make_model"]}) for sd in (101, 202)},
    "F18_bag": dict(desc="F18 averaged over 3 seeds", proposer="AI (local loop, seed bag)", model="blend", average=True,
                    parts=["F18_sevtune", "F18_s101", "F18_s202"]),
    "B17_glmfine_boost": dict(desc="GLM+boost with a finer GLM base (20 bands, 60 levels), de-noised, no make_model, cap 99%",
                              proposer="AI (local loop, backlog #5)", model="glmboost",
                              params={"glm": {"n_bins": 20, "top": 60}}, target={**FULL_T, "cap_q": 0.99},
                              features={**HIST_F, "drop": NOISE + ["make_model"]}),
    "X14_blend4_bags": dict(desc="4-way blend: B15, P20_135_bag corrected, F18_bag, A1 CatBoost", proposer="AI (local loop)",
                            model="blend", step=0.1, parts=["B15_denoise_mm", "P20_135_bag_cred", "F18_bag", "A1_catboost"]),
    "G13_glm_smooth": dict(desc="Smooth GLM: log1p + squared numerics (power laws on the log link), selected interactions, cap 99%",
                           proposer="AI (non-linear structure)", model="glm", params={"mode": "smooth", "alpha": 1e-4, "interactions": [("sum_insured", "flood_zone"), ("driver_age", "cover_type"), ("ncb_years", "driver_age"), ("annual_mileage", "vehicle_value_group"), ("sum_insured", "property_age")]},
                           target={**FULL_T, "cap_q": 0.99}, features={**HIST_F, "drop": NOISE + ["make_model"]}),
    "G14_glm_smooth_eng": dict(desc="G13 plus engineered ratio/difference features",
                               proposer="AI (non-linear structure)", model="glm", params={"mode": "smooth", "alpha": 1e-4, "interactions": [("sum_insured", "flood_zone"), ("driver_age", "cover_type"), ("ncb_years", "driver_age"), ("annual_mileage", "vehicle_value_group"), ("sum_insured", "property_age")]},
                               target={**FULL_T, "cap_q": 0.99}, features={**HIST_F, "engineered": True, "drop": NOISE + ["make_model"]}),
    "B18_smoothboost": dict(desc="Smooth GLM (with interactions, engineered features) as the base, GBM boosted on top",
                            proposer="AI (non-linear structure)", model="glmboost", params={"glm": {"mode": "smooth", "alpha": 1e-4, "interactions": [("sum_insured", "flood_zone"), ("driver_age", "cover_type"), ("ncb_years", "driver_age"), ("annual_mileage", "vehicle_value_group"), ("sum_insured", "property_age")]}},
                            target={**FULL_T, "cap_q": 0.99}, features={**HIST_F, "engineered": True, "drop": NOISE + ["make_model"]}),
    "P21_eng": dict(desc="P20_135 with engineered ratio/difference features", proposer="AI (non-linear structure)", model="lgbm",
                    params={**TUNED, "min_data_in_leaf": 2000, "monotone": MONOTONE, "tweedie_variance_power": 1.35},
                    target={**FULL_T, "cap_q": 0.99, "loading": "segment"},
                    features={**HIST_F, "engineered": True, "drop": NOISE + ["make_model"]}),
    "F20_eng": dict(desc="F18 with engineered ratio/difference features", proposer="AI (non-linear structure)", model="freqsev",
                    params={"freq": {**TUNED, "min_data_in_leaf": 2000},
                            "sev": {"num_leaves": 15, "min_data_in_leaf": 500, "learning_rate": 0.03, "lambda_l2": 50.0, "n_iter": 600}},
                    target={**FULL_T, "cap_q": 0.99}, features={**HIST_F, "engineered": True, "drop": NOISE + ["make_model"]}),
    "B19_smoothboost_tuned": dict(desc="B18 with the tuned boost stage (min leaf 2000, ff 0.5, monotone, power 1.35)",
                                  proposer="AI (non-linear hunt)", model="glmboost",
                                  params={"glm": {"mode": "smooth", "alpha": 1e-4, "interactions": [("sum_insured","flood_zone"),("driver_age","cover_type"),("ncb_years","driver_age"),("annual_mileage","vehicle_value_group"),("sum_insured","property_age")]},
                                          **TUNED, "min_data_in_leaf": 2000, "monotone": MONOTONE, "tweedie_variance_power": 1.35},
                                  target={**FULL_T, "cap_q": 0.99}, features={**HIST_F, "engineered": True, "drop": NOISE + ["make_model"]}),
    "X15_blend5_smooth": dict(desc="5-way blend: P20_135_bag_cred, F18_bag, A1 CatBoost, B19 smooth-boost, G14 smooth GLM", proposer="AI (non-linear hunt)",
                              model="blend", step=0.1, parts=["P20_135_bag_cred", "F18_bag", "A1_catboost", "B19_smoothboost_tuned", "G14_glm_smooth_eng"]),
    # Review 4 (18 Oct): the cumulative-exposure counters are calendar-time proxies (0 in 2021, ~1.7 in 2022, ~3 from 2023);
    # the smooth GLM's quadratic terms extrapolate them badly in the <=2022 -> 2023 holdout (-0.59). Test 2025 is in range,
    # but a part must pass the time-shift test, so the smooth family drops them (trees in the other parts keep them).
    "B20_smoothboost_noexpo": dict(desc="B19 without the cumulative-exposure counters (time-shift robust smooth GLM base)",
                                   proposer="AI (review 4: holdout diagnosis)", model="glmboost",
                                   params={"glm": {"mode": "smooth", "alpha": 1e-4, "interactions": [("sum_insured","flood_zone"),("driver_age","cover_type"),("ncb_years","driver_age"),("annual_mileage","vehicle_value_group"),("sum_insured","property_age")]},
                                           **TUNED, "min_data_in_leaf": 2000, "monotone": MONOTONE, "tweedie_variance_power": 1.35},
                                   target={**FULL_T, "cap_q": 0.99}, features={**HIST_F, "engineered": True, "drop": NOISE + ["make_model"] + EXPO_COUNTERS}),
    "X16_blend4_noexpo": dict(desc="4-way blend: P20_135_bag_cred, F18_bag, A1 CatBoost, B20 smooth-boost (time-shift robust)", proposer="AI (non-linear hunt)",
                              model="blend", step=0.1, parts=["P20_135_bag_cred", "F18_bag", "A1_catboost", "B20_smoothboost_noexpo"]),
    # B20's boost stage stops early at very different rounds per fold (18-178): seed-bag it (the GLM base is deterministic)
    **{f"B20_s{sd}": dict(desc=f"B20 with boost seed {sd}", proposer="AI (non-linear hunt, seed bag)", model="glmboost",
                          params={"glm": {"mode": "smooth", "alpha": 1e-4, "interactions": [("sum_insured","flood_zone"),("driver_age","cover_type"),("ncb_years","driver_age"),("annual_mileage","vehicle_value_group"),("sum_insured","property_age")]},
                                  **TUNED, "min_data_in_leaf": 2000, "monotone": MONOTONE, "tweedie_variance_power": 1.35, "seed": sd},
                          target={**FULL_T, "cap_q": 0.99}, features={**HIST_F, "engineered": True, "drop": NOISE + ["make_model"] + EXPO_COUNTERS})
       for sd in (101, 202)},
    "B20_bag": dict(desc="B20 averaged over 3 boost seeds", proposer="AI (non-linear hunt, seed bag)", model="blend", average=True,
                    parts=["B20_smoothboost_noexpo", "B20_s101", "B20_s202"]),
    "X17_blend4_b20bag": dict(desc="X16 with the B20 seed bag", proposer="AI (non-linear hunt)",
                              model="blend", step=0.1, parts=["P20_135_bag_cred", "F18_bag", "A1_catboost", "B20_bag"]),
    # Same smooth GLM base, boosted by CatBoost (the best single buildings learner) instead of LightGBM: a second
    # smooth-base part with a different residual learner for blend diversity
    "B21_smoothcat": dict(desc="Smooth GLM base (no counters) boosted by CatBoost (A1 defaults + monotone)",
                          proposer="AI (non-linear hunt)", model="glmboost",
                          params={"boost": "catboost", "glm": {"mode": "smooth", "alpha": 1e-4, "interactions": [("sum_insured","flood_zone"),("driver_age","cover_type"),("ncb_years","driver_age"),("annual_mileage","vehicle_value_group"),("sum_insured","property_age")]},
                                  "monotone": MONOTONE},
                          target={**FULL_T, "cap_q": 0.99}, features={**HIST_F, "engineered": True, "drop": NOISE + ["make_model"] + EXPO_COUNTERS}),
    "X18_blend5_smoothcat": dict(desc="X17 plus B21 smooth-CatBoost", proposer="AI (non-linear hunt)",
                                 model="blend", step=0.1, parts=["P20_135_bag_cred", "F18_bag", "A1_catboost", "B20_bag", "B21_smoothcat"]),
    "G15_glm_peril": dict(desc="G14 smooth GLM plus physical peril interactions (soil x dry summer, flood x rainfall, wind x roof, freezing x property age, hail x parking, flood x basement)",
                          proposer="AI (non-linear hunt: GLM lacks peril interactions on buildings)", model="glm",
                          params={"mode": "smooth", "alpha": 1e-4, "interactions": [
                              ("sum_insured","flood_zone"),("driver_age","cover_type"),("ncb_years","driver_age"),("annual_mileage","vehicle_value_group"),("sum_insured","property_age"),
                              ("dry_summer_index","soil_type"),("rainfall_mm","flood_zone"),("wind_exposure_index","roof_type"),("freezing_days","property_age"),
                              ("hail_days","overnight_parking"),("rainfall_mm","basement"),("wind_exposure_index","trees_within_5m"),("elevation_m","flood_zone")]},
                          target={**FULL_T, "cap_q": 0.99}, features={**HIST_F, "engineered": True, "drop": NOISE + ["make_model"] + EXPO_COUNTERS}),
    "D1_dart": dict(desc="P20_135 with DART boosting (drop 0.1, skip 0.5); early stopping is not available in dart",
                    proposer="AI (local loop, backlog #7)", model="lgbm",
                    params={**_P135, "boosting": "dart", "drop_rate": 0.1, "skip_drop": 0.5}, target=_T135, features=_F135),
    "E1_extra": dict(desc="P20_135 with extremely randomised trees", proposer="AI (local loop, backlog #8)",
                     model="lgbm", params={**_P135, "extra_trees": True}, target=_T135, features=_F135),
    "X11_blend4": dict(desc="4-way blend: B15, P20_135 corrected (power 1.35 + credibility layer), F17_bag, A1 CatBoost",
                       proposer="AI", model="blend", step=0.1, parts=["B15_denoise_mm", "P20_135_cred", "F17_bag", "A1_catboost"]),
    # ---- branch advanced-ml ----
    "A1_catboost": dict(desc="CatBoost Tweedie (ordered target stats for categoricals), L17 features/target",
                        proposer="AI (advanced ML)", model="catboost", target={**FULL_T, "cap_q": 0.99},
                        features={**HIST_F, "drop": NOISE + ["make_model"]}),
    "A2_xgb": dict(desc="XGBoost Tweedie (hist, native categoricals), L17 features/target", proposer="AI (advanced ML)",
                   model="xgb", target={**FULL_T, "cap_q": 0.99}, features={**HIST_F, "drop": NOISE + ["make_model"]}),
    "A3_mono": dict(desc="L17 with actuarial monotone constraints (excess -, NCB -, prior claims +, sum insured + ...)",
                    proposer="AI (advanced ML; trust criterion)", model="lgbm", params={**TUNED, "monotone": MONOTONE},
                    target={**FULL_T, "cap_q": 0.99}, features={**HIST_F, "drop": NOISE + ["make_model"]}),
    "S1_stack": dict(desc="Stacking meta-learner (Tweedie GLM on log predictions) over B15, L17_bag, F17_bag",
                     proposer="AI (advanced ML)", model="blend", stack=True, parts=["B15_denoise_mm", "L17_bag", "F17_bag"]),
    "X7_blend4": dict(desc="4-way blend: B15 (GLM+boost), A3 (monotone GBM), F17_bag (freq x sev), A1 (CatBoost)",
                      proposer="AI (advanced ML)", model="blend", step=0.1,
                      parts=["B15_denoise_mm", "A3_mono", "F17_bag", "A1_catboost"]),
    # Quick-screen tuning grid around L15 (de-noised features changed the trade-offs)
    **{f"T15_{k}": dict(desc=f"L15 tuning: {k}", proposer="AI (quick-screen tuning)", model="lgbm",
                        params=p, target={**FULL_T, "cap_q": 0.99}, features={**HIST_F, "drop": NOISE + ["make_model"]})
       for k, p in {"leaves31": {"num_leaves": 31, "min_data_in_leaf": 10000},
                    "minleaf40k": {"min_data_in_leaf": 40000},
                    "minleaf8k": {"min_data_in_leaf": 8000},
                    "ff05": {"feature_fraction": 0.5},
                    "l2_200": {"lambda_l2": 200.0},
                    "combo": {"min_data_in_leaf": 8000, "feature_fraction": 0.5},
                    "combo31": {"min_data_in_leaf": 8000, "feature_fraction": 0.5, "num_leaves": 31}}.items()},
    # Day 2 GLM variants (each a single change vs G6, LB 0.11716)
    "G7_nan": dict(desc="G6 + history unknown (not zero) for 2021 rows", proposer="AI (from G5 diagnosis)",
                   model="glm", target=FULL_T, features={**HIST_F, "hist_nan_2021": True}),
    "G8_trendB": dict(desc="G6 with claims-based trend instead of premium index", proposer="user (trend: start D, test B)",
                      model="glm", target={**FULL_T, "trend": "claims"}, features=HIST_F),
    "G9_bins": dict(desc="G6 with finer GLM bands (20) and more category levels (60)", proposer="AI",
                    model="glm", params={"n_bins": 20, "top": 60}, target=FULL_T, features=HIST_F),
    "G10_cap99": dict(desc="G6 with cap at 99th pct", proposer="AI (cap level sensitivity)",
                      model="glm", target={**FULL_T, "cap_q": 0.99}, features=HIST_F),
    "G11_cap999": dict(desc="G6 with cap at 99.9th pct", proposer="AI (cap level sensitivity)",
                       model="glm", target={**FULL_T, "cap_q": 0.999}, features=HIST_F),
}


# ---- Track A1 / D5 / D6 (local loop): smooth-GLM base grid, recency weighting, exposure as a feature ----
_G14_INT = [("sum_insured", "flood_zone"), ("driver_age", "cover_type"), ("ncb_years", "driver_age"),
            ("annual_mileage", "vehicle_value_group"), ("sum_insured", "property_age")]
_INT_MOTOR = [("vehicle_age", "annual_mileage"), ("driver_age", "annual_mileage"),
              ("vehicle_value_group", "vehicle_age"), ("ncb_years", "cover_type")]
_INT_HOME = [("sum_insured", "bedrooms"), ("property_age", "roof_type"), ("flood_zone", "basement"),
             ("sum_insured", "construction_type"), ("elevation_m", "distance_to_river_km")]
_G_T = {**FULL_T, "cap_q": 0.99}
_G_F = {**HIST_F, "engineered": True, "drop": NOISE + ["make_model"] + EXPO_COUNTERS}


def _g(desc, alpha, interactions):
    return dict(desc=desc, proposer="AI (local loop, track A1)", model="glm",
                params={"mode": "smooth", "alpha": alpha, "interactions": interactions}, target=_G_T, features=_G_F)


EXPERIMENTS.update({
    "G16_alpha_lo": _g("G14 smooth GLM (no counters), alpha 1e-5", 1e-5, _G14_INT),
    "G17_alpha_hi": _g("G14 smooth GLM (no counters), alpha 1e-3", 1e-3, _G14_INT),
    "G18_int_motor": _g("G14 smooth GLM (no counters) plus motor interactions", 1e-4, _G14_INT + _INT_MOTOR),
    "G19_int_home": _g("G14 smooth GLM (no counters) plus home interactions", 1e-4, _G14_INT + _INT_HOME),
    "B22_smoothboost_best": {**EXPERIMENTS["B20_smoothboost_noexpo"],
                             "desc": "B20 with the G18+G19 interaction set in the smooth GLM base",
                             "params": {**EXPERIMENTS["B20_smoothboost_noexpo"]["params"],
                                        "glm": {"mode": "smooth", "alpha": 1e-4,
                                                "interactions": _G14_INT + _INT_MOTOR + _INT_HOME}}},
    "P22_recency25": dict(desc="P20_135 with recency weights 1+0.25*(year-2021), mean-normalised", proposer="AI (local loop, D5)",
                          model="lgbm", params=_P135, target={**_T135, "recency": 0.25}, features=_F135),
    "P22_recency50": dict(desc="P20_135 with recency weights 1+0.5*(year-2021), mean-normalised", proposer="AI (local loop, D5)",
                          model="lgbm", params=_P135, target={**_T135, "recency": 0.5}, features=_F135),
    "P23_expofeat": dict(desc="P20_135 with row exposure as a numeric feature", proposer="AI (local loop, D6)",
                         model="lgbm", params=_P135, target=_T135, features={**_F135, "exposure_feature": True}),
})

# ---- Track D1 (18 Oct): CANN-style neural residual on the smooth GLM base (user-approved PyTorch CPU) ----
_SMOOTH_GLM = {"mode": "smooth", "alpha": 1e-4, "interactions": [("sum_insured","flood_zone"),("driver_age","cover_type"),("ncb_years","driver_age"),("annual_mileage","vehicle_value_group"),("sum_insured","property_age")]}
EXPERIMENTS.update({
    "N1_nnboost": dict(desc="Smooth GLM base (no counters) + neural residual (MLP 2x128, embeddings, Tweedie loss, residual clamp +-3)",
                       proposer="AI (novel techniques track, CANN pattern)", model="nnboost",
                       params={"glm": _SMOOTH_GLM, "hidden": 128, "dropout": 0.1, "lr": 1e-3, "batch": 4096, "epochs": 30,
                               "patience": 3, "weight_decay": 1e-5, "tweedie_variance_power": 1.5, "seed": 7, "threads": 12},
                       target={**FULL_T, "cap_q": 0.99}, features={**HIST_F, "engineered": True, "drop": NOISE + ["make_model"] + EXPO_COUNTERS}),
    "X19_blend5_nn": dict(desc="X16 plus the neural residual part", proposer="AI (novel techniques track)",
                          model="blend", step=0.1, parts=["P20_135_bag_cred", "F18_bag", "A1_catboost", "B20_smoothboost_noexpo", "N1_nnboost"]),
})

# ---- Track D2 (18 Oct): covariate-shift importance weights (train-vs-predicted-rows classifier); gate = 2023 forward holdout ----
EXPERIMENTS.update({
    "P24_shift": dict(desc="P20_135 with covariate-shift density-ratio weights, clip 0.2-5", proposer="AI (novel techniques track, D2)",
                      model="lgbm", params=_P135, target={**_T135, "shift_weights": {"clip": [0.2, 5.0]}}, features=_F135),
    "P24_shift_tight": dict(desc="P20_135 with covariate-shift density-ratio weights, clip 0.5-2", proposer="AI (novel techniques track, D2)",
                            model="lgbm", params=_P135, target={**_T135, "shift_weights": {"clip": [0.5, 2.0]}}, features=_F135),
})

# ---- Track D4 (18 Oct): one model on buildings + contents pooled (line_id categorical); motor stays separate ----
EXPERIMENTS.update({
    "P25_pool_home": dict(desc="P20_135 fitted once on buildings+contents pooled (line_id categorical)", proposer="AI (novel techniques track, D4)",
                          model="lgbm", params=_P135, target=_T135, features=_F135, pool=["buildings", "contents"]),
    "B23_pool_home_smooth": {**EXPERIMENTS["B20_smoothboost_noexpo"], "pool": ["buildings", "contents"],
                             "desc": "B20 smooth-boost fitted once on buildings+contents pooled (line_id one-hot)"},
})

# N1 overfit within 1-2 epochs (val deviance rose immediately): smaller, slower, more regularised retune before judging D1
EXPERIMENTS.update({
    "N2_nnboost_reg": dict(desc="N1 retune: hidden 64, dropout 0.3, lr 3e-4, batch 8192, weight decay 1e-4, patience 4",
                           proposer="AI (novel techniques track, CANN pattern)", model="nnboost",
                           params={"glm": _SMOOTH_GLM, "hidden": 64, "dropout": 0.3, "lr": 3e-4, "batch": 8192, "epochs": 30,
                                   "patience": 4, "weight_decay": 1e-4, "tweedie_variance_power": 1.5, "seed": 7, "threads": 12},
                           target={**FULL_T, "cap_q": 0.99}, features={**HIST_F, "engineered": True, "drop": NOISE + ["make_model"] + EXPO_COUNTERS}),
})

# ---- Round 6 (18 Oct, user: keep hunting) — cheap compute-only screens ----
EXPERIMENTS.update({
    "P26_cap985": dict(desc="P20_135 with cap at the 98.5th percentile (loading absorbs more tail)", proposer="AI (round 6)", model="lgbm",
                       params=_P135, target={**_T135, "cap_q": 0.985}, features=_F135),
    "P26_cap995": dict(desc="P20_135 with cap at the 99.5th percentile", proposer="AI (round 6)", model="lgbm",
                       params=_P135, target={**_T135, "cap_q": 0.995}, features=_F135),
    **{f"A1_s{sd}": dict(desc=f"A1 CatBoost with random_seed {sd}", proposer="AI (round 6, seed bag)", model="catboost",
                         params={"random_seed": sd}, target={**FULL_T, "cap_q": 0.99}, features={**HIST_F, "drop": NOISE + ["make_model"]})
       for sd in (101, 202)},
    "A1_bag": dict(desc="A1 CatBoost averaged over 3 seeds", proposer="AI (round 6, seed bag)", model="blend", average=True,
                   parts=["A1_catboost", "A1_s101", "A1_s202"]),
    "X20_blend4_step05": dict(desc="X16 parts with a 0.05 blend grid", proposer="AI (round 6)", model="blend", step=0.05,
                              parts=["P20_135_bag_cred", "F18_bag", "A1_catboost", "B20_smoothboost_noexpo"]),
    "X21_blend4_a1bag": dict(desc="X16 with the CatBoost seed bag", proposer="AI (round 6)", model="blend", step=0.1,
                             parts=["P20_135_bag_cred", "F18_bag", "A1_bag", "B20_smoothboost_noexpo"]),
})

# ---- Round 6: code-backed ideas (segment blend weights, model-based large-loss loading, smooth frequency base) ----
EXPERIMENTS.update({
    "X22_blend4_segdecile": dict(desc="X16 parts with per-segment weights by OOF risk decile (nested CV per segment)", proposer="AI (round 6)",
                                 model="blend", step=0.1, parts=["P20_135_bag_cred", "F18_bag", "A1_catboost", "B20_smoothboost_noexpo"],
                                 segments={"by": "oof_decile", "bins": 10, "min_rows": 50000}),
    "X23_blend4_segquint": dict(desc="X16 parts with per-segment weights by OOF risk quintile", proposer="AI (round 6)",
                                model="blend", step=0.1, parts=["P20_135_bag_cred", "F18_bag", "A1_catboost", "B20_smoothboost_noexpo"],
                                segments={"by": "oof_decile", "bins": 5, "min_rows": 50000}),
    "P27_modelload": dict(desc="P20_135 with a model-based large-loss loading (excess GBM, shrink 0.5, clipped) instead of the segment factor",
                          proposer="AI (round 6)", model="lgbm", params=_P135,
                          target={**FULL_T, "cap_q": 0.99, "loading": "model", "loading_params": {"shrink": 0.5, "clip": 2.0, "min_data_in_leaf": 2000}},
                          features=_F135),
    "F21_smoothfreq": dict(desc="F18 with a smooth Poisson GLM base under the frequency GBM (counters dropped, engineered ratios)",
                           proposer="AI (round 6)", model="freqsev",
                           params={"freq": {**TUNED, "min_data_in_leaf": 2000}, "freq_base": _SMOOTH_GLM,
                                   "sev": {"num_leaves": 15, "min_data_in_leaf": 500, "learning_rate": 0.03, "lambda_l2": 50.0, "n_iter": 600}},
                           target={**FULL_T, "cap_q": 0.99}, features={**HIST_F, "engineered": True, "drop": NOISE + ["make_model"] + EXPO_COUNTERS}),
})

# Round 6: cap 0.985 helped motor (+0.0006 quick) and hurt buildings -> per-line cap via target_by_line
EXPERIMENTS.update({
    "P28_capline": dict(desc="P20_135 with per-line cap: motor 98.5th pct, buildings/contents 99th", proposer="AI (round 6)", model="lgbm",
                        params=_P135, target=_T135, target_by_line={"motor": {"cap_q": 0.985}}, features=_F135),
})

# Round 6 follow-up blends on the candidate-elect X21 (CatBoost bag): + smooth-frequency freq×sev, + model-loaded LightGBM
EXPERIMENTS.update({
    "X24_blend5_f21": dict(desc="X21 plus F21 smooth-frequency freq×sev", proposer="AI (round 6)", model="blend", step=0.1,
                           parts=["P20_135_bag_cred", "F18_bag", "A1_bag", "B20_smoothboost_noexpo", "F21_smoothfreq"]),
    "X25_blend5_p27": dict(desc="X21 plus P27 model-loaded LightGBM", proposer="AI (round 6)", model="blend", step=0.1,
                           parts=["P20_135_bag_cred", "F18_bag", "A1_bag", "B20_smoothboost_noexpo", "P27_modelload"]),
})

EXPERIMENTS.update({
    "X26_blend5_p28": dict(desc="X21 plus P28 per-line-cap LightGBM", proposer="AI (round 6)", model="blend", step=0.1,
                           parts=["P20_135_bag_cred", "F18_bag", "A1_bag", "B20_smoothboost_noexpo", "P28_capline"]),
})

# User decision 18 Oct 23:10 UTC: written_premium allowed as a BLEND PART only (one model carries it; nested weight)
EXPERIMENTS.update({
    "P29_wp": dict(desc="P20_135 plus written_premium (blend-part-only use of the incumbent price)", proposer="AI (round 6, user-approved scope)",
                   model="lgbm", params=_P135, target=_T135, features={**_F135, "premium": True}),
    "X27_blend6_wp": dict(desc="X21 parts plus the written-premium model as a blend part", proposer="AI (round 6)", model="blend", step=0.1,
                          parts=["P20_135_bag_cred", "F18_bag", "A1_bag", "B20_smoothboost_noexpo", "P29_wp"]),
})

# Brainstorm B3/B4 (user-selected 19 Oct): low-learning-rate part and an unconstrained diversity part on the X21 parts
EXPERIMENTS.update({
    "P30_lowlr": dict(desc="P20_135 at learning rate 0.01 (early stopping sets the rounds)", proposer="AI (B3)", model="lgbm",
                      params={**_P135, "learning_rate": 0.01}, target=_T135, features=_F135),
    "P31_free": dict(desc="P20_135 without monotone constraints (diversity part only; constrained parts stay in the core)", proposer="AI (B4)",
                     model="lgbm", params={k: v for k, v in _P135.items() if k != "monotone"}, target=_T135, features=_F135),
    "X28_blend5_lowlr": dict(desc="X21 parts plus P30 low-lr LightGBM", proposer="AI (B3)", model="blend", step=0.1,
                             parts=["P20_135_bag_cred", "F18_bag", "A1_bag", "B20_smoothboost_noexpo", "P30_lowlr"]),
    "X29_blend5_free": dict(desc="X21 parts plus P31 unconstrained LightGBM", proposer="AI (B4)", model="blend", step=0.1,
                            parts=["P20_135_bag_cred", "F18_bag", "A1_bag", "B20_smoothboost_noexpo", "P31_free"]),
})

# Audit 19 Oct: annual_mileage == 0 is a sentinel (1.2 % of rows) -> NaN + indicator; free quick screen
EXPERIMENTS.update({
    "P32_mileage": dict(desc="P20_135 with annual_mileage 0 -> NaN + missing indicator", proposer="AI (data audit 19 Oct)", model="lgbm",
                        params=_P135, target=_T135, features={**_F135, "mileage_zero_nan": True}),
})
