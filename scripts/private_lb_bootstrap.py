"""Public/private LB subset-noise bootstrap on cached OOF predictions.

  python -P -E scripts/private_lb_bootstrap.py
Writes docs/private_lb_robustness.md. Rows are the unit; subsets drawn without replacement.
"""
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]
from pipeline import load  # noqa: E402
from metrics import tweedie_deviance, d2_tweedie  # noqa: E402

CANDS = ["X21_blend4_a1bag", "X16_blend4_noexpo", "X14_blend4_bags", "X12_blend4_f18", "X10_blend4"]   # 18 Oct: X21 (CatBoost bag) added
KS = [1, 2, 3, 5]
B, FRAC, SEED = 300, 0.30, 20260
OUT = ROOT / "docs" / "private_lb_robustness.md"


def sums_dev(y, mu, idx):
    return tweedie_deviance(y[idx], mu[idx]).sum()


def d2_sub(y, dev, idx):
    """D2 on rows idx from per-row model deviance `dev`, baseline = subset mean (closed-form sums)."""
    ys = y[idx]
    m = ys.mean()
    base = 2 * (ys ** 0.5 / (-0.25) + 2 * ys * m ** -0.5 + 2 * m ** 0.5).sum()
    return 1 - dev[idx].sum() / base


def main():
    ys, oofs = None, {}
    for c in CANDS:
        r = load(c)
        y = np.concatenate([np.asarray(l["y_eval"], float) for l in r["lines"].values()])
        o = np.concatenate([np.asarray(l["oof"], float) for l in r["lines"].values()])
        if ys is None:
            ys = y
        assert len(y) == len(ys) and np.allclose(y, ys), c
        oofs[c] = o
    n = len(ys)
    full = {c: d2_tweedie(ys, oofs[c]) for c in CANDS}
    order = sorted(CANDS, key=lambda c: -full[c])
    preds = dict(oofs)
    for k in KS:
        if k > 1:
            preds[f"top{k}"] = np.mean([oofs[c] for c in order[:k]], axis=0)
    names = CANDS + [f"top{k}" for k in KS if k > 1]
    dev = {nm: tweedie_deviance(ys, preds[nm]) for nm in names}
    fullall = {nm: d2_tweedie(ys, preds[nm]) for nm in names}

    rng = np.random.default_rng(SEED)
    ns = int(round(FRAC * n))
    S = {nm: np.empty(B) for nm in names}
    C = {nm: np.empty(B) for nm in names}
    for b in range(B):
        perm = rng.permutation(n)
        sub, comp = perm[:ns], perm[ns:]
        for nm in names:
            S[nm][b] = d2_sub(ys, dev[nm], sub)
            C[nm][b] = d2_sub(ys, dev[nm], comp)

    L = []
    L.append("# Public/private LB subset-noise bootstrap\n")
    L.append(f"Rows n={n:,}; {B} random subsets of {FRAC:.0%} (n={ns:,}) drawn without replacement; complement = {1-FRAC:.0%}. "
             "D2 = Tweedie p=1.5 vs subset-mean baseline on OOF predictions. "
             f"Top-k = equal average of the k best candidates by full-data D2: {', '.join(order)}.\n")
    L.append("## (a) Per-candidate D2 across subsets\n")
    L.append("| model | full-data D2 | mean (30%) | sd (30%) | mean (70%) | sd (70%) |\n|---|---|---|---|---|---|")
    for nm in names:
        L.append(f"| {nm} | {fullall[nm]:.4f} | {S[nm].mean():.4f} | {S[nm].std(ddof=1):.4f} | {C[nm].mean():.4f} | {C[nm].std(ddof=1):.4f} |")
    L.append("\n## (b) Pairwise (30% subset): P(A beats B), sd of paired difference, mean difference\n")
    L.append("| A | B | full-data A-B | mean diff | sd diff (paired) | P(A>B) | sd diff if unpaired | P(A>B) on 70% |\n|---|---|---|---|---|---|---|---|")
    for a, b2 in itertools.combinations(CANDS, 2):
        d = S[a] - S[b2]
        dc = C[a] - C[b2]
        unp = np.sqrt(S[a].var(ddof=1) + S[b2].var(ddof=1))
        L.append(f"| {a} | {b2} | {fullall[a]-fullall[b2]:+.4f} | {d.mean():+.4f} | {d.std(ddof=1):.4f} | {(d>0).mean():.2f} | {unp:.4f} | {(dc>0).mean():.2f} |")
    L.append("\n## (c) Top-k averages (30% subset)\n")
    L.append("| k | members | full-data D2 | E[D2] | sd | E - sd |\n|---|---|---|---|---|---|")
    crit = {}
    for k in KS:
        nm = order[0] if k == 1 else f"top{k}"
        crit[k] = S[nm].mean() - S[nm].std(ddof=1)
        L.append(f"| {k} | {', '.join(order[:k])} | {fullall[nm]:.4f} | {S[nm].mean():.4f} | {S[nm].std(ddof=1):.4f} | {crit[k]:.4f} |")
    L.append("\nPaired difference of top-k vs top-1 (30% subsets):\n")
    L.append("| k | mean diff vs k=1 | sd diff | P(topk > top1) |\n|---|---|---|---|")
    for k in KS[1:]:
        d = S[f"top{k}"] - S[order[0]]
        L.append(f"| {k} | {d.mean():+.4f} | {d.std(ddof=1):.4f} | {(d>0).mean():.2f} |")
    best = max(crit, key=crit.get)
    L.append("\n## (d) Public vs private split noise for the same candidate\n")
    L.append("| model | sd of (30% D2 - 70% D2) | mean abs gap | P(gap > 0.0006) |\n|---|---|---|---|")
    for nm in names:
        g = S[nm] - C[nm]
        L.append(f"| {nm} | {g.std(ddof=1):.4f} | {np.abs(g).mean():.4f} | {(np.abs(g)>0.0006).mean():.2f} |")
    L.append(f"\nRecommendation by max(E - sd): k = {best} (E-sd = {crit[best]:.4f}).\n")
    L.append("## Interpretation\n")
    L.append("- Row-resampling noise only: it ignores the test-set distribution shift, so it understates real private-LB risk.")
    L.append("- Level sd across subsets is much larger than between-candidate gaps; levels of one candidate are not comparable across subsets.")
    L.append("- Paired differences are far tighter than levels (see sd diff vs unpaired column); ranking is judged on paired differences.")
    L.append("- Pairs with P(A>B) near 0.5 are indistinguishable by subset; pairs near 0/1 have a real gap on these rows.")
    L.append("- A 0.0006 public margin is compared with the paired sd above; where P(A>B) < ~0.9 that margin is mostly subset noise.")
    L.append("- The public-vs-private gap of one candidate (section d) shows how much the same model moves across the split.")
    L.append("- Differences between top-k averages are small relative to their sd; the criterion E - sd is nearly flat in k, so k is a low-stakes choice.")
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
