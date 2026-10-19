"""Round-4 decision (18 Oct, non-linear hunt and novel techniques). Rendered on the results dashboard, section 9.

Facts are from local nested 5-fold CV (Tweedie D2). Submissions: none 18 Oct; 10 probes on 19 Oct (user-approved); candidates due 20 Oct.
"""

DECISIONS = [
    {
        "id": "final_pick_22oct", "impact": "High", "multi": False,
        "title": "Final pick on 22 Oct: submit the X16 candidate in a single batch so it can be ticked as final?",
        "why": "Predicted +0.0009 vs the current best, inside the +/-0.002 noise; the private-LB bootstrap says X16 is the better file on "
               "96-100 % of 30 % row subsets (paired sd 0.0002). X16 is predicted at about 0.1290 +/- 0.002 with level factors and is "
               "written to submissions/20261018_1300_FINAL_part1.csv. It has NOT been uploaded: no Kaggle submissions since 17 Oct. "
               "Leaderboard 19 Oct 03:00 UTC: #1 0.12996, #2 0.12910, us #3 0.12815 (FINAL_avg3_levelcal, submitted 17 Oct); the X21 candidate (0.12744 raw) is superseded by the SI-band candidates of section 10.",
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

DECISIONS.append({
    "id": "extra_probes_20oct", "impact": "Medium", "multi": False,
    "title": "20 Oct: after the two candidate files score, spend up to 4 more submissions mapping contents top-quartile and buildings top-decile?",
    "why": "The 19 Oct probes put the 2025 buildings shift in the top sum-insured quartile (x1.16 scores 0.12808 vs 0.12744; optimum "
           "about x1.22, roughly +0.0007). Contents top quartile and buildings top decile are the two untested segments, expected "
           "+0.0002 to +0.0004 public. Caveat: this is public-LB-only evidence. The training years do not contain the shift, and the "
           "public-vs-private level gap (sd about 0.005) is larger than any of these differences, so a public gain may not carry to the "
           "private score.",
    "options": [
        {"key": "A", "label": "Yes: spend up to 4 submissions on the two segment maps",
         "pros": "Expected +0.0002 to +0.0004 public; noise-free differences, so each probe is read exactly.",
         "cons": "Public-only risk: gain may not transfer to private; uses submissions before the 22 Oct final pick."},
        {"key": "B", "label": "No: keep the two candidate files and stop probing",
         "pros": "No public-only overfitting risk; submissions kept in reserve.",
         "cons": "Leaves the expected +0.0002 to +0.0004 public gain unmapped."},
    ],
    "suggest": "Default B; choose A only if you accept public-only risk.",
})
