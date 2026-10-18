# Public/private LB subset-noise bootstrap

Rows n=2,409,311; 300 random subsets of 30% (n=722,793) drawn without replacement; complement = 70%. D2 = Tweedie p=1.5 vs subset-mean baseline on OOF predictions. Top-k = equal average of the k best candidates by full-data D2: X11_blend4, X10_blend4, X7_blend4, X8_blend_cred, X6_blend3bag.

## (a) Per-candidate D2 across subsets

| model | full-data D2 | mean (30%) | sd (30%) | mean (70%) | sd (70%) |
|---|---|---|---|---|---|
| X10_blend4 | 0.1593 | 0.1593 | 0.0035 | 0.1592 | 0.0015 |
| X11_blend4 | 0.1593 | 0.1594 | 0.0035 | 0.1593 | 0.0015 |
| X7_blend4 | 0.1591 | 0.1591 | 0.0035 | 0.1591 | 0.0015 |
| X8_blend_cred | 0.1589 | 0.1589 | 0.0035 | 0.1588 | 0.0015 |
| X6_blend3bag | 0.1584 | 0.1585 | 0.0035 | 0.1584 | 0.0015 |
| top2 | 0.1593 | 0.1594 | 0.0035 | 0.1593 | 0.0015 |
| top3 | 0.1593 | 0.1593 | 0.0035 | 0.1593 | 0.0015 |
| top5 | 0.1592 | 0.1592 | 0.0035 | 0.1592 | 0.0015 |

## (b) Pairwise (30% subset): P(A beats B), sd of paired difference, mean difference

| A | B | full-data A-B | mean diff | sd diff (paired) | P(A>B) | sd diff if unpaired | P(A>B) on 70% |
|---|---|---|---|---|---|---|---|
| X10_blend4 | X11_blend4 | -0.0001 | -0.0001 | 0.0001 | 0.20 | 0.0049 | 0.03 |
| X10_blend4 | X7_blend4 | +0.0002 | +0.0002 | 0.0001 | 0.88 | 0.0049 | 0.99 |
| X10_blend4 | X8_blend_cred | +0.0004 | +0.0004 | 0.0002 | 0.98 | 0.0050 | 1.00 |
| X10_blend4 | X6_blend3bag | +0.0008 | +0.0008 | 0.0003 | 1.00 | 0.0050 | 1.00 |
| X11_blend4 | X7_blend4 | +0.0002 | +0.0002 | 0.0002 | 0.93 | 0.0050 | 1.00 |
| X11_blend4 | X8_blend_cred | +0.0005 | +0.0005 | 0.0002 | 0.98 | 0.0050 | 1.00 |
| X11_blend4 | X6_blend3bag | +0.0009 | +0.0009 | 0.0003 | 1.00 | 0.0050 | 1.00 |
| X7_blend4 | X8_blend_cred | +0.0002 | +0.0002 | 0.0003 | 0.83 | 0.0050 | 0.99 |
| X7_blend4 | X6_blend3bag | +0.0007 | +0.0006 | 0.0002 | 1.00 | 0.0050 | 1.00 |
| X8_blend_cred | X6_blend3bag | +0.0004 | +0.0004 | 0.0002 | 0.98 | 0.0050 | 1.00 |

## (c) Top-k averages (30% subset)

| k | members | full-data D2 | E[D2] | sd | E - sd |
|---|---|---|---|---|---|
| 1 | X11_blend4 | 0.1593 | 0.1594 | 0.0035 | 0.1559 |
| 2 | X11_blend4, X10_blend4 | 0.1593 | 0.1594 | 0.0035 | 0.1559 |
| 3 | X11_blend4, X10_blend4, X7_blend4 | 0.1593 | 0.1593 | 0.0035 | 0.1559 |
| 5 | X11_blend4, X10_blend4, X7_blend4, X8_blend_cred, X6_blend3bag | 0.1592 | 0.1592 | 0.0035 | 0.1557 |

Paired difference of top-k vs top-1 (30% subsets):

| k | mean diff vs k=1 | sd diff | P(topk > top1) |
|---|---|---|---|
| 2 | -0.0000 | 0.0001 | 0.47 |
| 3 | -0.0000 | 0.0001 | 0.30 |
| 5 | -0.0001 | 0.0001 | 0.12 |

## (d) Public vs private split noise for the same candidate

| model | sd of (30% D2 - 70% D2) | mean abs gap | P(gap > 0.0006) |
|---|---|---|---|
| X10_blend4 | 0.0050 | 0.0041 | 0.94 |
| X11_blend4 | 0.0050 | 0.0041 | 0.95 |
| X7_blend4 | 0.0050 | 0.0041 | 0.95 |
| X8_blend_cred | 0.0050 | 0.0041 | 0.93 |
| X6_blend3bag | 0.0050 | 0.0041 | 0.93 |
| top2 | 0.0050 | 0.0041 | 0.94 |
| top3 | 0.0050 | 0.0041 | 0.94 |
| top5 | 0.0050 | 0.0041 | 0.95 |

Recommendation by max(E - sd): k = 2 (E-sd = 0.1559).

## Interpretation

- Row-resampling noise only: it ignores the test-set distribution shift, so it understates real private-LB risk.
- Level sd across subsets is much larger than between-candidate gaps; levels of one candidate are not comparable across subsets.
- Paired differences are far tighter than levels (see sd diff vs unpaired column); ranking is judged on paired differences.
- Pairs with P(A>B) near 0.5 are indistinguishable by subset; pairs near 0/1 have a real gap on these rows.
- A 0.0006 public margin is compared with the paired sd above; where P(A>B) < ~0.9 that margin is mostly subset noise.
- The public-vs-private gap of one candidate (section d) shows how much the same model moves across the split.
- Differences between top-k averages are small relative to their sd; the criterion E - sd is nearly flat in k, so k is a low-stakes choice.
