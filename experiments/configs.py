"""Experiment definitions. Each builds on the previous accepted one (see docs/experiment_log.md)."""

RAW = {"develop": False, "cap_q": None, "trend": "none", "weight_power": 0.0}
DEV = {**RAW, "develop": True}
DEV_CAP = {**DEV, "cap_q": 0.995}
FULL_T = {**DEV_CAP, "trend": "premium", "weight_power": 1.0}

BASE_F = {"history": False}
HIST_F = {"history": True}

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
}
