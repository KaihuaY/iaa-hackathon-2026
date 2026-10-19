"""Grand blend: do any combinations of ALL cached experiments beat the best 4-part blend (X21)?
Three combiners evaluated by nested CV (outer GroupKFold by policy_id): greedy Caruana selection, penalised
non-negative Tweedie weights, and a geometric (log) blend with scalar level rebalance. Writes (never uploads) a file."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]

BASE_NAME = "X21_blend4_a1bag"
EXCLUDE = ("probe", "levelcal", "FINAL", "bug")
LAMBDAS = (0.0, 1e-4, 1e-3)
TOPK, GREEDY_POOL, GREEDY_STEPS = 10, 40, 25


# ---------------------------------------------------------------- metric helpers (p = 1.5, matches metrics.d2_tweedie)
def _dev_sum(y, mu):
    s = np.sqrt(np.clip(mu, 1e-9, None))
    return float(np.sum(-8 * np.sqrt(y) + 4 * (y / s + s)))


def d2(y, mu):
    return 1 - _dev_sum(y, mu) / _dev_sum(y, np.full(len(y), y.mean()))


def _softmax(t):
    e = np.exp(t - t.max())
    return e / e.sum()


# ---------------------------------------------------------------- combiners: fit(y, P, groups) -> model dict; apply(model, P)
def single_scores(y, P):
    return np.array([d2(y, P[:, j]) for j in range(P.shape[1])])


def fit_greedy(y, P, groups=None, steps=GREEDY_STEPS, pool=GREEDY_POOL):
    """Caruana forward selection with replacement on the arithmetic mean; best-scoring prefix is kept."""
    sc = single_scores(y, P)
    cand = np.argsort(-sc)[:pool]
    cnt = np.zeros(P.shape[1])
    j0 = int(cand[0])
    cnt[j0] = 1
    S = P[:, j0].astype(float).copy()
    n = 1
    best = (sc[j0], cnt.copy())
    for _ in range(steps - 1):
        scores = [d2(y, (S + P[:, j]) / (n + 1)) for j in cand]
        k = int(np.argmax(scores))
        j = int(cand[k])
        cnt[j] += 1
        S += P[:, j]
        n += 1
        if scores[k] > best[0]:
            best = (scores[k], cnt.copy())
    cnt = best[1]
    return {"w": cnt / cnt.sum()}


def _fit_softmax(y, M, idx, lam, log_space, null_dev):
    """Weights on columns idx of M minimising normalised deviance + lam*||w - 1/k||^2."""
    k = len(idx)
    X = M[:, idx].astype(float)
    w0 = np.full(k, 1.0 / k)

    def f(t):
        w = _softmax(t)
        eta = X @ w
        mu = np.clip(np.exp(eta) if log_space else eta, 1e-9, None)
        s = np.sqrt(mu)
        loss = np.sum(-8 * np.sqrt(y) + 4 * (y / s + s)) / null_dev
        dmu = 4 * (-0.5 * y * mu ** -1.5 + 0.5 * mu ** -0.5) / null_dev
        if log_space:
            dmu = dmu * mu
        g_w = X.T @ dmu + 2 * lam * (w - w0)
        loss += lam * np.sum((w - w0) ** 2)
        return loss, w * (g_w - w @ g_w)          # softmax Jacobian

    r = minimize(f, np.zeros(k), jac=True, method="L-BFGS-B", options={"maxiter": 200})
    return _softmax(r.x)


def _fit_pen(y, P, groups, log_space, lams=LAMBDAS):
    M = np.log(np.clip(P, 1e-6, None)) if log_space else P
    top = np.argsort(-single_scores(y, P))[:TOPK]
    null = _dev_sum(y, np.full(len(y), y.mean()))
    best_lam, best_s = lams[0], -np.inf
    if len(lams) > 1:                    # inner split chooses lambda
        tr, va = next(iter(GroupKFold(3).split(y, groups=groups)))
        for lam in lams:
            w = _fit_softmax(y[tr], M[tr], top, lam, log_space, _dev_sum(y[tr], np.full(len(tr), y[tr].mean())))
            x = M[va][:, top] @ w
            s = d2(y[va], np.exp(x) if log_space else x)
            if s > best_s:
                best_lam, best_s = lam, s
    w = _fit_softmax(y, M, top, best_lam, log_space, null)
    full = np.zeros(P.shape[1])
    full[top] = w
    return full, best_lam


def fit_arith(y, P, groups):
    w, lam = _fit_pen(y, P, groups, False)
    return {"w": w, "lam": lam}


def fit_geo(y, P, groups):
    w, lam = _fit_pen(y, P, groups, True)
    wa, _ = _fit_pen(y, P, groups, False)
    geo = np.exp(np.log(np.clip(P, 1e-6, None)) @ w)
    return {"w": w, "lam": lam, "scale": float((P @ wa).sum() / geo.sum())}


def apply_arith(m, P):
    return P @ m["w"]


def apply_geo(m, P):
    return m["scale"] * np.exp(np.log(np.clip(P, 1e-6, None)) @ m["w"])


COMBINERS = {"greedy": (fit_greedy, apply_arith), "pen_arith": (fit_arith, apply_arith), "geo_rebal": (fit_geo, apply_geo)}


def nested(y, P, groups, fit, apply, n_folds=5):
    """Outer GroupKFold: fit on train folds ONLY, predict held-out rows."""
    out = np.zeros(len(y))
    for tr, va in GroupKFold(n_folds).split(y, groups=groups):
        m = fit(y[tr], P[tr], groups[tr])
        out[va] = apply(m, P[va])
    return out


# ---------------------------------------------------------------- data
def load_pool():
    import pipeline
    from market_spec import get_spec
    spec = get_spec()
    ref = pipeline.load(BASE_NAME)
    names = sorted(p.stem for p in spec.results_path.glob("*.pkl"))
    pool, oofs, tests, seen = [], {l: [] for l in spec.lines}, {l: [] for l in spec.lines}, set()
    for nm in names:
        if any(x in nm for x in EXCLUDE):
            continue
        try:
            res = pipeline.load(nm)
        except Exception:
            continue
        if res.get("unsafe") or set(spec.lines) - set(res.get("lines", {})):
            continue
        cfg = res.get("cfg") or {}
        if cfg.get("model") == "blend" and not cfg.get("average"):   # weighted blends: OOF weights fitted in-sample -> leakage
            continue
        ok = True
        for l in spec.lines:
            pl, rl = res["lines"][l], ref["lines"][l]
            if pl.get("unsafe") or len(pl["row_id"]) != len(rl["row_id"]) or not (pl["row_id"] == rl["row_id"]).all() \
                    or len(pl["test_row_id"]) != len(rl["test_row_id"]) or not (pl["test_row_id"] == rl["test_row_id"]).all() \
                    or not np.isfinite(pl["oof"]).all() or not np.isfinite(pl["test_pred"]).all():
                ok = False
        if not ok:
            continue
        key = tuple(round(float(np.sum(res["lines"][l]["oof"])), 3) for l in spec.lines)
        if key in seen:         # blend whose OOF is identical to an existing part
            continue
        seen.add(key)
        pool.append(nm)
        for l in spec.lines:
            oofs[l].append(res["lines"][l]["oof"].astype(np.float32))
            tests[l].append(res["lines"][l]["test_pred"].astype(np.float32))
    return spec, ref, pool, {l: np.column_stack(v) for l, v in oofs.items()}, {l: np.column_stack(v) for l, v in tests.items()}


def groups_for(spec, line, row_id):
    from features import build_base
    tb = build_base()[f"train_{line}"]
    g = pd.Series(tb[spec.group_col].values, index=tb.row_id.values).reindex(row_id)
    assert g.notna().all()
    return g.values


def x21_nested(spec, ref, y, line, groups):
    """Reproduce X21's nested predictions (pipeline.run_blend logic) from its parts."""
    import pipeline
    cfg = ref["cfg"]
    P = np.column_stack([pipeline.load(p)["lines"][line]["oof"] for p in cfg["parts"]])
    grid = [np.full(P.shape[1], 1 / P.shape[1])] if cfg.get("average") else \
        pipeline._simplex(P.shape[1], cfg.get("step", 0.05 if P.shape[1] <= 3 else 0.1))
    out = np.zeros(len(y))
    for tr, va in GroupKFold(spec.n_folds).split(y, groups=groups):
        out[va] = P[va] @ pipeline._best_weights(y[tr], P[tr], grid)[0]
    return out


def bootstrap(y, a, b, n=300, frac=0.25, seed=0):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = rng.random(len(y)) < frac
        d.append(d2(y[i], a[i]) - d2(y[i], b[i]))
    d = np.array(d)
    return d.mean(), d.std(), float((d > 0).mean())


def main():
    import pipeline
    import submit
    spec, ref, pool, OOF, TEST = load_pool()
    print(f"POOL SIZE {len(pool)}: {', '.join(pool)}", flush=True)
    res = {c: {} for c in COMBINERS}
    base_n, ys, fullw = {}, {}, {}
    for line in spec.lines:
        y = ref["lines"][line]["y_eval"].astype(float)
        g = groups_for(spec, line, ref["lines"][line]["row_id"])
        ys[line] = y
        base_n[line] = x21_nested(spec, ref, y, line, g)
        print(f"{line}: X21 reproduced nested {d2(y, base_n[line]):.5f} (stored {ref['lines'][line]['cv_nested']:.5f})", flush=True)
        P = OOF[line].astype(np.float64)
        for c, (fit, app) in COMBINERS.items():
            res[c][line] = nested(y, P, g, fit, app)
            fullw[(c, line)] = fit(y, P, g)
            print(f"  {line} {c}: nested {d2(y, res[c][line]):.5f}", flush=True)
        del P
    ycat = np.concatenate([ys[l] for l in spec.lines])
    xcat = np.concatenate([base_n[l] for l in spec.lines])
    print(f"\n{'combiner':12s} " + " ".join(f"{l:>10s}" for l in spec.lines) + "    overall   boot mean     sd     P>0")
    print(f"{'X21':12s} " + " ".join(f"{d2(ys[l], base_n[l]):10.5f}" for l in spec.lines) + f" {d2(ycat, xcat):10.5f}")
    overall = {}
    for c in COMBINERS:
        cat = np.concatenate([res[c][l] for l in spec.lines])
        overall[c] = d2(ycat, cat)
        bm, bs, bp = bootstrap(ycat, cat, xcat)
        print(f"{c:12s} " + " ".join(f"{d2(ys[l], res[c][l]):10.5f}" for l in spec.lines) +
              f" {overall[c]:10.5f}  {bm:+.5f}  {bs:.5f}  {bp:.2f}", flush=True)
    for c in COMBINERS:
        for line in spec.lines:
            w = fullw[(c, line)]["w"]
            extra = {k: v for k, v in fullw[(c, line)].items() if k != "w"}
            print(f"[{c}/{line}] {extra} " + ", ".join(f"{pool[i]}={w[i]:.3f}" for i in np.argsort(-w)[:8] if w[i] > 0))
    best = max(overall, key=overall.get)
    print(f"\nBEST combiner: {best} overall {overall[best]:.5f} vs X21 {d2(ycat, xcat):.5f}")
    app = COMBINERS[best][1]
    df = pd.concat([pd.DataFrame({"row_id": ref["lines"][l]["test_row_id"],
                                  "prediction": app(fullw[(best, l)], TEST[l].astype(np.float64))}) for l in spec.lines],
                   ignore_index=True)
    x21 = pipeline.test_predictions(ref)
    path = submit.write(df, "GB_grand_blend")
    m = df.merge(x21, on="row_id", suffixes=("", "_x21"))
    print(f"WROTE {path}  mean ratio vs X21 {m.prediction.mean() / m.prediction_x21.mean():.4f}  "
          f"corr {np.corrcoef(m.prediction, m.prediction_x21)[0, 1]:.4f}")


if __name__ == "__main__":
    main()
