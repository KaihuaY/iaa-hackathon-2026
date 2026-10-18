# Public/private LB subset-noise bootstrap

Rows n=2,409,311; 300 random subsets of 30% (n=722,793) drawn without replacement; complement = 70%. D2 = Tweedie p=1.5 vs subset-mean baseline on OOF predictions. Top-k = equal average of the k best candidates by full-data D2: X16_blend4_noexpo, X12_blend4_f18, X14_blend4_bags, X11_blend4, X10_blend4.

## (a) Per-candidate D2 across subsets

| model | full-data D2 | mean (30%) | sd (30%) | mean (70%) | sd (70%) |
|---|---|---|---|---|---|
| X16_blend4_noexpo | 0.1599 | 0.1599 | 0.0035 | 0.1599 | 0.0015 |
| X14_blend4_bags | 0.1595 | 0.1595 | 0.0035 | 0.1594 | 0.0015 |
| X12_blend4_f18 | 0.1595 | 0.1595 | 0.0035 | 0.1595 | 0.0015 |
| X10_blend4 | 0.1593 | 0.1593 | 0.0035 | 0.1592 | 0.0015 |
| X11_blend4 | 0.1593 | 0.1594 | 0.0035 | 0.1593 | 0.0015 |
| top2 | 0.1598 | 0.1599 | 0.0035 | 0.1598 | 0.0015 |
| top3 | 0.1597 | 0.1598 | 0.0035 | 0.1597 | 0.0015 |
| top5 | 0.1596 | 0.1597 | 0.0035 | 0.1596 | 0.0015 |

## (b) Pairwise (30% subset): P(A beats B), sd of paired difference, mean difference

| A | B | full-data A-B | mean diff | sd diff (paired) | P(A>B) | sd diff if unpaired | P(A>B) on 70% |
|---|---|---|---|---|---|---|---|
| X16_blend4_noexpo | X14_blend4_bags | +0.0004 | +0.0004 | 0.0002 | 0.98 | 0.0050 | 1.00 |
| X16_blend4_noexpo | X12_blend4_f18 | +0.0004 | +0.0004 | 0.0002 | 0.96 | 0.0050 | 1.00 |
| X16_blend4_noexpo | X10_blend4 | +0.0006 | +0.0006 | 0.0002 | 1.00 | 0.0050 | 1.00 |
| X16_blend4_noexpo | X11_blend4 | +0.0005 | +0.0005 | 0.0002 | 0.99 | 0.0050 | 1.00 |
| X14_blend4_bags | X12_blend4_f18 | -0.0000 | -0.0000 | 0.0001 | 0.40 | 0.0050 | 0.33 |
| X14_blend4_bags | X10_blend4 | +0.0002 | +0.0002 | 0.0001 | 0.93 | 0.0049 | 1.00 |
| X14_blend4_bags | X11_blend4 | +0.0001 | +0.0001 | 0.0001 | 0.90 | 0.0050 | 1.00 |
| X12_blend4_f18 | X10_blend4 | +0.0002 | +0.0002 | 0.0001 | 0.96 | 0.0050 | 1.00 |
| X12_blend4_f18 | X11_blend4 | +0.0002 | +0.0001 | 0.0001 | 0.95 | 0.0050 | 1.00 |
| X10_blend4 | X11_blend4 | -0.0001 | -0.0001 | 0.0001 | 0.20 | 0.0049 | 0.03 |

## (c) Top-k averages (30% subset)

| k | members | full-data D2 | E[D2] | sd | E - sd |
|---|---|---|---|---|---|
| 1 | X16_blend4_noexpo | 0.1599 | 0.1599 | 0.0035 | 0.1564 |
| 2 | X16_blend4_noexpo, X12_blend4_f18 | 0.1598 | 0.1599 | 0.0035 | 0.1564 |
| 3 | X16_blend4_noexpo, X12_blend4_f18, X14_blend4_bags | 0.1597 | 0.1598 | 0.0035 | 0.1563 |
| 5 | X16_blend4_noexpo, X12_blend4_f18, X14_blend4_bags, X11_blend4, X10_blend4 | 0.1596 | 0.1597 | 0.0035 | 0.1562 |

Paired difference of top-k vs top-1 (30% subsets):

| k | mean diff vs k=1 | sd diff | P(topk > top1) |
|---|---|---|---|
| 2 | -0.0001 | 0.0001 | 0.31 |
| 3 | -0.0001 | 0.0001 | 0.16 |
| 5 | -0.0003 | 0.0002 | 0.06 |

## (d) Public vs private split noise for the same candidate

| model | sd of (30% D2 - 70% D2) | mean abs gap | P(gap > 0.0006) |
|---|---|---|---|
| X16_blend4_noexpo | 0.0050 | 0.0041 | 0.93 |
| X14_blend4_bags | 0.0050 | 0.0041 | 0.94 |
| X12_blend4_f18 | 0.0050 | 0.0041 | 0.95 |
| X10_blend4 | 0.0050 | 0.0041 | 0.94 |
| X11_blend4 | 0.0050 | 0.0041 | 0.95 |
| top2 | 0.0050 | 0.0041 | 0.94 |
| top3 | 0.0050 | 0.0041 | 0.94 |
| top5 | 0.0050 | 0.0041 | 0.94 |

Recommendation by max(E - sd): k = 1 (E-sd = 0.1564).

## Interpretation

- Row-resampling noise only: it ignores the test-set distribution shift, so it understates real private-LB risk.
- Level sd across subsets is much larger than between-candidate gaps; levels of one candidate are not comparable across subsets.
- Paired differences are far tighter than levels (see sd diff vs unpaired column); ranking is judged on paired differences.
- Pairs with P(A>B) near 0.5 are indistinguishable by subset; pairs near 0/1 have a real gap on these rows.
- A 0.0006 public margin is compared with the paired sd above; where P(A>B) < ~0.9 that margin is mostly subset noise.
- The public-vs-private gap of one candidate (section d) shows how much the same model moves across the split.
- Differences between top-k averages are small relative to their sd; the criterion E - sd is nearly flat in k, so k is a low-stakes choice.
