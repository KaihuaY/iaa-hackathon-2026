"""Round-2 decisions (17 Oct, after Day-1 submissions and Day-2 local experiments). Rendered on the results dashboard."""

DECISIONS = [
    {
        "id": "next_model", "impact": "High", "multi": True,
        "title": "Next modelling investment (choose all you want tried, locally first)",
        "why": "Nothing tried today beats X1 by more than the noise. The next gain needs a new idea, not tuning.",
        "options": [
            {"key": "A", "label": "Per-peril model for buildings (storm, flood, subsidence, escape of water, fire)",
             "pros": "Weather and location map directly onto perils; buildings is where trees found the most extra signal.",
             "cons": "Small perils (subsidence 1.3k claims, fire 1.1k) are noisy; ~half a day of work.", "rec": True},
            {"key": "B", "label": "Frequency x severity split (Poisson GBM x Gamma GLM) per line",
             "pros": "Avoids the Tweedie GBM's weakness with zeros; severity can be modelled on claims only.",
             "cons": "Two models per line; severity data thin for home.", "rec": True},
            {"key": "C", "label": "Open-claim development factor (motor 2023-24 severity still ~6 % below 2021-22)",
             "pros": "Directly targets the level gap seen in the target analysis.",
             "cons": "Single valuation, so the factor is a judgement; mainly moves level, which CV can't see."},
            {"key": "D", "label": "Seed / bagging ensemble of the current best (5 seeds averaged)",
             "pros": "Cheap, reliable small gain from lower variance.", "cons": "Small gain (~+0.0005 CV typical)."},
            {"key": "E", "label": "Hand-built interaction features for the GLM (dry summer x clay, flood zone x rainfall...)",
             "pros": "Makes the GLM explain what the GBM finds; good story.", "cons": "The GBM already captures these."},
        ],
        "suggest": "A + B first (largest expected gain), D as a final polish.",
    },
    {
        "id": "trend", "impact": "High", "multi": False,
        "title": "Trend method (backtest result)",
        "why": "The level of every 2025 prediction depends on it. The local backtest now answers what CV could not.",
        "options": [
            {"key": "A", "label": "Keep the premium index for all lines (your original choice D)",
             "pros": "Backtest level error 1.8 % on average vs 3.0 % for the claims trend; best on every line on average.",
             "cons": "Premium changes also include rate actions, so not a pure claims index.", "rec": True},
            {"key": "B", "label": "Claims-based trend for motor only",
             "pros": "Motor frequency has been falling.", "cons": "Backtest: motor 2024 under-priced by 6.2 % with it."},
            {"key": "C", "label": "Spend one submission to test the claims trend on the leaderboard anyway",
             "pros": "Direct check.", "cons": "Uses a submission on something the backtest already answers."},
        ],
        "suggest": "A. No submission needed.",
    },
    {
        "id": "submit_now", "impact": "High", "multi": False,
        "title": "Submit X6 (3-way de-noised, tuned, seed-bagged blend)? Predicted LB 0.1275 vs best 0.1249 (+2.1 %)",
        "why": "X6 = GLM+boost + tuned Tweedie GBM + tuned frequency x severity GBM (both averaged over 3 seeds), noise features removed. "
               "Nested CV 0.1584 vs X1 0.1552 (same as X5, more robust); "
               "first candidate whose predicted gain (+0.0026) clears the noise band (+-0.0022). "
               "Below the 5 % auto-submit bar (needs predicted 0.1311). You held X3 (+1.5 %) earlier to keep stacking.",
        "options": [
            {"key": "A", "label": "Yes, submit X6 (1 submission)",
             "pros": "Gain is consistent across lines and on the 2023 holdout, with a clear mechanism (noise features removed); "
                     "removing noise should also help on the private leaderboard.",
             "cons": "Below the 5 % auto-submit bar; gain is modest in absolute terms.", "rec": True},
            {"key": "B", "label": "Wait and stack more improvements first", "pros": "One bigger, clearer jump later.",
             "cons": "No LB confirmation that the de-noising transfers."},
        ],
        "suggest": "A.",
    },
    {
        "id": "final_rule", "impact": "Medium", "multi": False,
        "title": "How to choose the final submission (23 Oct)",
        "why": "Only one is judged, on the private leaderboard (the rows we never see).",
        "options": [
            {"key": "A", "label": "Highest nested CV among candidates whose public LB matched its prediction",
             "pros": "Robust to public-LB luck; uses all training data.", "cons": "May not be the top public score.", "rec": True},
            {"key": "B", "label": "Highest public LB", "pros": "Simple.", "cons": "Rewards luck on the public sample."},
            {"key": "C", "label": "Average of the top 2-3 candidates", "pros": "Most stable.", "cons": "Usually close to A."},
        ],
        "suggest": "A, falling back to C if the top two are within noise.",
    },
    {
        "id": "docs", "impact": "Judging (40 %)", "multi": False,
        "title": "Start the write-up and video material now?",
        "why": "Communication (15 %) and AI use / guardrails (45 %) are judged on documents due after Part 2.",
        "options": [
            {"key": "A", "label": "Yes: draft the 5-page write-up skeleton from the dashboards and experiment log now",
             "pros": "Captures decisions while fresh; Part 2 will be busy.", "cons": "Some rework after Part 2.", "rec": True},
            {"key": "B", "label": "Later, after Part 2", "pros": "Write once.", "cons": "Rushed at the end."},
        ],
        "suggest": "A. A skeleton only, filled in as we go.",
    },
    {
        "id": "mcp", "impact": "Low", "multi": False,
        "title": "Kaggle MCP submission test",
        "why": "The MCP server is configured but its tools only load after a session restart.",
        "options": [
            {"key": "A", "label": "I'll restart the session; test MCP on the next approved submission",
             "pros": "Answers your earlier question.", "cons": "None.", "rec": True},
            {"key": "B", "label": "Keep using the CLI", "pros": "Already works.", "cons": "MCP stays untested."},
        ],
        "suggest": "A.",
    },
]
