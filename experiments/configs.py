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
}
