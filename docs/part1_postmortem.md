# Part 1 post-mortem — private leaderboard (final #5, private 0.13777)

Source: `docs/part1_submissions_final.csv` (Kaggle CLI, all 38 scored submissions with public and private scores).
Ticked final: `20261022_1708_X33_blend4_si07_allbags_shrunk_c102.csv` — public 0.12826, **private 0.13777**.

## What the private scores say about each decision

| Decision | Files compared | Private | Verdict |
|---|---|---|---|
| Model choice by nested CV | X10 raw 0.13744 → X21 raw 0.13762 | **+0.00018** | CV gains *did* transfer (they were invisible on the public slice: −0.00005) |
| Shrunk vs probe-fitted factors | X30 shrunk 0.13765 vs X30 probe-fitted (best public, 0.12838) 0.13749 | **+0.00016** | shrinking was right |
| Seed bagging of every part | X33 shrunk 0.13777 vs X30 shrunk 0.13765 | +0.00012 | the bag helped privately although nested CV was equal |
| Flat buildings level uplift (public-probed ×1.093) | X21 raw 0.13762 vs ×1.09 0.13740, ×1.15 0.13691; 17 Oct FINAL (flat levels) 0.13718 vs X10 raw 0.13744 | **−0.0002 to −0.0007** | the line-level probe did **not** transfer — public-specific |
| Top sum-insured quartile uplift | X21 raw 0.13762 vs Q4 ×1.08 0.13785, ×1.16 0.13791; even-id half ×1.20 0.13815 (public −0.00001) | **+0.0002 to +0.0005** | the SI *tilt* was real for all rows (both halves) — the split-half probe's public reading was the noisy one |
| High-wind-region uplift | ×1.08 → 0.13720 | −0.0004 | correctly rejected |
| Outlier ("tail") models | T3 0.13532, motor power 1.7 0.13617, uplift 0.13736 | −0.0003 to −0.0023 | correctly rejected, as CV said |
| Public/private correlation across our 38 files | — | corr 0.995 | the level differs, the ranking of files mostly holds |

## Lessons for Part 2
1. **Choose models by nested CV and the forward holdout.** It transferred to the private board even when the public
   board showed nothing.
2. **Never apply a public-probed flat line level at full strength.** The flat buildings uplift was the costliest
   leaderboard-informed move (−0.0002 to −0.0007 private). Shrink line levels much harder (Z ≤ 0.2) or skip them.
3. **Structure (a tilt by an exposure measure such as sum insured) transferred, a level did not.** Model it locally
   first (the SI-elasticity offset was the right direction; it applied about a third of the tilt), and if a probe
   confirms a tilt on two independent halves, keep more of it than the credibility rule suggested.
4. **Seed-average every part of the final blend** — nested CV under-states its private value.
5. The best file we ever submitted privately was a probe (0.13815): the private optimum lay between our shrunk
   factors and a stronger SI tilt with no flat level uplift.
