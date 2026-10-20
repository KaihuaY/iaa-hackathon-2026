# Final-pick memo — Part 1 (decide 22 Oct; deadline 23 Oct 2026 16:00 UTC)

Only one submission counts: the one ticked **"Use for Final Score"** on Kaggle. It is scored on the **private**
rows (≈ 75–80 % of the test set that the public board never shows).

## The candidates (all already submitted, so all selectable)

| File (in `submissions/`) | Public LB | What it is | sha256 (first 16) |
|---|---|---|---|
| **`20261019_2013_X30_blend4_si07_shrunk_c102.csv`** | 0.12820 | X30 blend (sum insured^0.7 in the buildings exposure) × **credibility-shrunk** factors: buildings by SI quartile 1.01 / 1.03 / 1.05 / 1.07, motor ×1.012, contents ×1.02 | 554da91e80348667 |
| `20261018_2152_X30_blend4_si07_sifix.csv` | 0.12838 | same X30 × **probe-fitted** factors: buildings 1.01 / 1.055 / 1.085 / 1.16, motor ×1.029 | ebd4dd966f43f06b |
| `20261018_0146_FINAL_part1.csv` | 0.12815 | 17 Oct final: average of three blends × flat line factors (motor ×1.029, buildings ×1.093) | c1780f47a3d55ce8 |

## Recommendation: tick the shrunk file (`…shrunk_c102.csv`)

1. **The extra 0.00018 of the probe-fitted file is mostly fitted to the public rows' particular claims.**
   A factor read off the public slice carries sampling noise of ≈ ±5 % for a whole line and ±10–13 % for a
   sum-insured quartile (`docs/diagnostics_17oct.md` §10d). The analysis predicted ≈ 0.0001 of public gain per
   fitted factor from noise alone; the observed gap (0.00018) is that size.
2. **The two decisive probes point to noise, not a 2025 effect.** The top-SI-quartile uplift sits in one half of the
   rows only (split-half test: the even-id half wants ×1.09, implying ≈ ×1.4 in the other half — a few large
   claims). The regions where 2025 wind jumped want no uplift at all, so a storm story does not hold. Nothing in the
   data (uniform SI indexation, no SI-by-year trend, flat peril mix, the 2025 tariff) supports a real high-SI shift.
3. **The shrunk file keeps the parts that are supported**: the line levels (credibility Z ≈ 0.45 → buildings ×1.04,
   motor ×1.012, contents ×1.02 — the same values two independent methods produced, `docs/lb_response_fit.md`) and the
   model's own SI elasticity (X30), which is fitted on 2.4 M training rows rather than on the public slice.
4. **Cost if we are wrong:** if the public shift were real for all rows, the shrunk file gives up ≈ 0.0002 on the
   private board; if it is noise (our reading), the probe-fitted file loses about the same or more. Expected value
   favours the shrunk file; it is also the defensible choice for the validation/guardrail criterion.

When to change the recommendation: only if the independent clean-room rebuild (other session, `notebooks/03_cleanroom.py`)
finds a systematic difference in the target or exposure that changes the line levels — then rebuild and decide again.

## Kaggle steps (manual, before 23 Oct 16:00 UTC)
1. kaggle.com → competition *IAA AI-Assisted Hackathon 2026 – Part 1* → **Submissions** tab.
2. Find the submission whose description starts "credibility-shrunk final candidate: X30 x buildings SI bands
   [1.01,1.03,1.05,1.07], motor x1.012, contents x1.02" (public 0.12820, submitted 20 Oct ≈ 01:00 UTC).
3. Tick **Use for Final Score** on it; make sure every other submission is unticked (Kaggle allows a limited number of
   selected files — select only this one so the outcome is not decided by an automatic best-public pick).
4. Screenshot the Submissions page for the write-up appendix.

## Submissions left
21 and 22 Oct: none planned (user allowance: up to 4/day). Use a slot only if the clean-room rebuild produces a
corrected file, and only with the user's go-ahead.
