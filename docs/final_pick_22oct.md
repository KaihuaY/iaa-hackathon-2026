# Final-pick memo — Part 1 (decide 22 Oct; deadline 23 Oct 2026 16:00 UTC)

Only one submission counts: the one ticked **"Use for Final Score"** on Kaggle. It is scored on the **private**
rows (≈ 75–80 % of the test set that the public board never shows).

## The candidates (all already submitted, so all selectable)

| File (in `submissions/`) | Public LB | What it is | sha256 (first 16) |
|---|---|---|---|
| **`20261019_2013_X30_blend4_si07_shrunk_c102.csv`** | 0.12820 | X30 blend (sum insured^0.7 in the buildings exposure) × **credibility-shrunk** factors: buildings by SI quartile 1.01 / 1.03 / 1.05 / 1.07, motor ×1.012, contents ×1.02 | 554da91e80348667 |
| `20261022_1708_X33_blend4_si07_allbags_shrunk_c102.csv` | 0.12826 | **equivalent alternative (22 Oct)**: identical shrunk factors on X33 = the X30 family with every part a 3-seed average; nested CV equal (0.1598); public +0.00006 (noise-sized) | d83ce8cba461830d |
| `20261018_2152_X30_blend4_si07_sifix.csv` | 0.12838 | same X30 × **probe-fitted** factors: buildings 1.01 / 1.055 / 1.085 / 1.16, motor ×1.029 | ebd4dd966f43f06b |
| `20261018_0146_FINAL_part1.csv` (see note below) | 0.12815 | 17 Oct final: average of three blends × flat line factors (motor ×1.029, buildings ×1.093) | c1780f47a3d55ce8 |

Name note: the file submitted 17 Oct (public 0.12815) is listed as `20261017_2143_FINAL_part1.csv` in
`docs/lb_response_fit.md` and as `20261018_0146_FINAL_part1.csv` in this memo. The sha above refers to the 0146 name.

## Recommendation: tick the shrunk file (`…shrunk_c102.csv`) — or its seed-averaged twin X33 (same factors, same CV, slightly lower variance); either is fine, tick exactly one

1. **The extra 0.00018 of the probe-fitted file is mostly fitted to the public rows' particular claims.**
   A factor read off the public slice carries sampling noise of ≈ ±5 % for a whole line and ±10–13 % for a
   sum-insured quartile (`docs/diagnostics_17oct.md` §10d). The analysis predicted ≈ 0.0001 of public gain per
   fitted factor from noise alone; the observed gap (0.00018) is that size.
2. **The prior from the forward level backtest favours shrinking.** The split-half probe (halves ×1.09 vs ≈ ×1.40,
   log gap 0.25 vs expected spread ≈ 0.24) and the wind-region probe (×1.00 vs ≈ ×1.17, z ≈ 1.25) measure the *size*
   of public-row noise; they do not by themselves prove the high-SI signal is noise. The case rests on the backtest:
   our trend method's 2023/2024 level errors are +0.025/−0.012 (motor), −0.003/−0.012 (buildings), −0.032/+0.024
   (contents); net of one-year sampling noise (≈ 0.022–0.028) the true error τ ≈ 0–0.03, against a public sampling SD of
   0.045–0.050 (motor), 0.061–0.066 (buildings), 0.050–0.054 (contents). A 10–25 % prior chance of a real uniform
   buildings shock (+9 %; severity inflation +5.4/+6.7/+9.8 % in 2022–24 vs a +5.4 % premium index) puts buildings at
   ×1.03–1.05. A buildings level error e costs ≈ 0.049·e² privately, so these choices differ by ≤ 0.00002.

   | Line | Public optimum | Z | Chosen factor |
   |---|---|---|---|
   | Motor | ×1.029 | 0.15–0.30 | ×1.012 |
   | Buildings | ×1.093 | 0.10–0.20 | ×1.04 (SI bands 1.01–1.07) |
   | Contents | ×1.019 | 0.12–0.27 | ×1.02 |
3. **The shrunk file keeps the parts that are supported**: the line levels (credibility Z ≈ 0.45 → buildings ×1.04,
   motor ×1.012, contents ×1.02 — the same values two independent methods produced, `docs/lb_response_fit.md`) and the
   model's own SI elasticity (X30), which is fitted on 2.4 M training rows rather than on the public slice.
4. **Cost if we are wrong:** if the public shift were real for all rows, the shrunk file gives up ≈ 0.0002 on the
   private board; if it is noise (our reading), the probe-fitted file loses about the same or more. Expected value
   favours the shrunk file; it is also the defensible choice for the validation/guardrail criterion.

When to change the recommendation: only if the independent clean-room rebuild (other session, `notebooks/03_cleanroom.py`)
finds a systematic difference in the target or exposure that changes the line levels — then rebuild and decide again.
**Checked 19 Oct (`docs/ai_review_log.md` Review 7): rows, claim counts and reported cost identical in every line × year,
exposure −0.01 %; differences only in immature-year development (documented method choices). Recommendation stands.**

## Kaggle steps (manual, before 23 Oct 16:00 UTC)
1. kaggle.com → competition *IAA AI-Assisted Hackathon 2026 – Part 1* → **Submissions** tab.
2. Find the submission whose description starts "credibility-shrunk final candidate: X30 x buildings SI bands
   [1.01,1.03,1.05,1.07], motor x1.012, contents x1.02" (public 0.12820, submitted 20 Oct ≈ 01:00 UTC).
3. Tick **Use for Final Score** on it; make sure every other submission is unticked (Kaggle allows a limited number of
   selected files — select only this one so the outcome is not decided by an automatic best-public pick).
   Two "shrunk" files exist — tick the one with description '…contents x1.02' (sha 554da91e…), not the 0825 file
   (contents ×1.00, sha 41efea62…).
   Public-only files (prefix PUBLIC_ONLY_) and probes may be uploaded on 22–23 Oct for the visible rank; they must never be ticked.
4. Screenshot the Submissions page for the write-up appendix.

## Submissions left
21 and 22 Oct: none planned (user allowance: up to 4/day). Use a slot only if the clean-room rebuild produces a
corrected file, and only with the user's go-ahead.
