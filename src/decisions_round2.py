"""Round-3 decisions (17 Oct, after the level probe; X10_levelcal public LB 0.12797). Rendered on the results dashboard."""

DECISIONS = [
    {
        "id": "final_rule", "impact": "High", "multi": False,
        "title": "How to choose the final submission (23 Oct)",
        "why": "Only one file is judged, on the private leaderboard (rows we never see). Current public LB: X10_levelcal 0.12797, "
               "X10 unscaled 0.12749.",
        "options": [
            {"key": "A", "label": "Average of the top-3 candidates, with the LB-calibrated level factors applied",
             "pros": "Most stable; averaging lowers variance and the level factors keep the public-LB gain.",
             "cons": "Not the single top public score; level factors were fitted on the public sample.", "rec": True},
            {"key": "B", "label": "Single best: X10_levelcal", "pros": "Highest public LB (0.12797); simple.",
             "cons": "Level factors tuned on the public rows; rewards public-LB fit."},
            {"key": "C", "label": "Hold X10 unscaled (0.12749)", "pros": "No dependence on public-LB level tuning.",
             "cons": "Gives up the +0.0005 from the calibrated level."},
        ],
        "suggest": "A.",
    },
    {
        "id": "git_rewrite", "impact": "Medium", "multi": False,
        "title": "Rewrite git history to drop 1.7 GB of previously tracked result pickles",
        "why": "The repository must be shared; old commits still hold the result pickles even though they are no longer tracked.",
        "options": [
            {"key": "A", "label": "Yes, rewrite history before Part 2 starts",
             "pros": "Small, shareable repo; done before more work is built on top.",
             "cons": "Destructive; collaborators need a fresh clone.", "rec": True},
            {"key": "B", "label": "No, keep history as is", "pros": "Nothing breaks.",
             "cons": "Repo stays 1.7 GB heavier to clone and share."},
        ],
        "suggest": "A, before Part 2 starts.",
    },
    {
        "id": "part2_plan", "impact": "Medium", "multi": False,
        "title": "Part 2 first-day plan",
        "why": "docs/part2_playbook.md lays out the steps for the first day of Part 2.",
        "options": [
            {"key": "A", "label": "Follow docs/part2_playbook.md", "pros": "Prepared and rehearsed; no time lost deciding.",
             "cons": "Less room for new ideas on day one.", "rec": True},
            {"key": "B", "label": "Improvise", "pros": "Can react to what the Part 2 data shows.",
             "cons": "Risk of skipping checks the playbook covers."},
        ],
        "suggest": "A.",
    },
    {
        "id": "writeup_len", "impact": "Judging (40 %)", "multi": False,
        "title": "Write-up length: currently ~2.7k words against the 5-page limit",
        "why": "Part 2 will add material, so the final length is not yet known.",
        "options": [
            {"key": "A", "label": "Trim after Part 2", "pros": "Trim once, with the full content known.",
             "cons": "Some end-of-project time pressure.", "rec": True},
            {"key": "B", "label": "Trim now", "pros": "Fits the limit today.", "cons": "Likely to be redone after Part 2."},
        ],
        "suggest": "A.",
    },
]
