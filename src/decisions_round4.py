"""Round-4 decision (18 Oct, non-linear hunt and novel techniques). Rendered on the results dashboard, section 9.

Facts are from local nested 5-fold CV (Tweedie D2). No Kaggle submissions since 17 Oct.
"""

DECISIONS = [
    {
        "id": "final_pick_22oct", "impact": "High", "multi": False,
        "title": "Final pick on 22 Oct: submit the X16 candidate in a single batch so it can be ticked as final?",
        "why": "Predicted +0.0009 vs the current best, inside the +/-0.002 noise; the private-LB bootstrap says X16 is the better file on "
               "96-100 % of 30 % row subsets (paired sd 0.0002). X16 is predicted at about 0.1290 +/- 0.002 with level factors and is "
               "written to submissions/20261018_1300_FINAL_part1.csv. It has NOT been uploaded: no Kaggle submissions since 17 Oct. "
               "Leaderboard now: #1 0.12908, #2 0.12893, #3 0.12855, us #4 0.12815 (FINAL_avg3_levelcal, submitted 17 Oct).",
        "options": [
            {"key": "A", "label": "Submit the X16 batch on 22 Oct and tick it as final",
             "pros": "Higher predicted score (about 0.1290 vs 0.12815) and better on 96-100 % of bootstrap row subsets; "
                     "smooth-GLM base passed two guardrail reviews.",
             "cons": "Predicted gain is inside the +/-0.002 noise; uses submissions at the end."},
            {"key": "B", "label": "Keep FINAL_avg3_levelcal (0.12815)",
             "pros": "Already scored publicly; no new risk.",
             "cons": "Gives up the predicted +0.0009 and the more robust smooth-GLM base."},
        ],
        "suggest": "Your call on 22 Oct; the gain is within noise, so the bootstrap robustness is the main argument for A.",
    },
]
