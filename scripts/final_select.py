"""Prepare the Part 1 final submission: average of the top candidates (robust) rather than the single best public score.
Kaggle's final-selection is manual on the website; this only writes the file and prints instructions."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]

import explog  # noqa: E402
import pipeline  # noqa: E402
import submit  # noqa: E402
from metrics import d2_tweedie  # noqa: E402

FLAGS = ("bug", "diagnostic", "excluded", "rejected", "void")


def pick(top: int) -> list[str]:
    df = pd.read_csv(explog.LOG)
    ok = df.exp.map(lambda e: (pipeline.RESULTS / f"{e}.pkl").exists()) & ~df.decision.fillna("").str.lower().map(
        lambda d: any(f in d for f in FLAGS))
    df = df[ok].copy()
    df["score"] = pd.to_numeric(df.cv_nested, errors="coerce").fillna(pd.to_numeric(df.cv, errors="coerce"))
    return list(df.dropna(subset=["score"]).sort_values("score", ascending=False).exp[:top])


def score_of(res: dict) -> float:
    return float(res.get("cv_nested", res["cv"]))


def weights(results: list[dict], mode: str) -> np.ndarray:
    if mode == "equal":
        return np.full(len(results), 1 / len(results))
    w = np.exp(np.array([score_of(r) for r in results]) * 100)
    return w / w.sum()


def blend(results: list[dict], w: np.ndarray, factors: dict[str, float]):
    """Returns (submission df, {line: (blend oof D2, [candidate D2s])}, overall tuple)."""
    parts, d2, ally, allo = [], {}, [], []
    for line in results[0]["lines"]:
        L = [r["lines"][line] for r in results]
        ids = np.asarray(L[0]["test_row_id"])
        for l in L:  # align each candidate on row_id
            assert set(l["test_row_id"]) == set(ids), f"test rows differ in {line}"
        tp = sum(wi * pd.Series(l["test_pred"], index=l["test_row_id"]).reindex(ids).to_numpy() for wi, l in zip(w, L))
        parts.append(pd.DataFrame({"row_id": ids, "prediction": tp * factors.get(line, 1.0)}))
        y = np.asarray(L[0]["y_eval"])
        oof = sum(wi * np.asarray(l["oof"]) for wi, l in zip(w, L))
        d2[line] = (d2_tweedie(y, oof), [d2_tweedie(y, l["oof"]) for l in L])
        ally.append(y), allo.append((oof, [l["oof"] for l in L]))
    y = np.concatenate(ally)
    d2["ALL"] = (d2_tweedie(y, np.concatenate([a for a, _ in allo])),
                 [d2_tweedie(y, np.concatenate([c[i] for _, c in allo])) for i in range(len(results))])
    return pd.concat(parts, ignore_index=True), d2


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", nargs="+")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--weights", choices=["equal", "cv"], default="equal")
    ap.add_argument("--level-factors", nargs="*", default=[], metavar="LINE=F")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    names = a.candidates or pick(a.top)
    factors = {k: float(v) for k, v in (s.split("=") for s in a.level_factors)}
    results = [pipeline.load(n) for n in names]
    log = pd.read_csv(explog.LOG).set_index("exp")
    print(f"{'exp':<22}{'cv_nested':>10}{'lb_pred':>10}{'lb_public':>10}")
    for n, r in zip(names, results):
        g = lambda c: "" if n not in log.index or pd.isna(log.at[n, c]) else log.at[n, c]  # noqa: E731
        print(f"{n:<22}{score_of(r):>10.5f}{g('lb_pred'):>10}{g('lb_public'):>10}")
    w = weights(results, a.weights)
    print("weights:", dict(zip(names, np.round(w, 4))), "level factors:", factors or "none")
    df, d2 = blend(results, w, factors)
    for line, (b, ind) in d2.items():
        print(f"OOF D2 {line:<10} blend={b:.5f}  candidates={[round(x, 5) for x in ind]}  "
              f"{'OK' if b >= min(ind) - 1e-12 else 'WARN: below min candidate'}")
    print("rows:", len(df))
    lines = pd.concat([pd.Series(r["lines"][l]["test_row_id"]).to_frame("row_id").assign(line=l) for r in results[:1] for l in r["lines"]])
    print("mean prediction per line:", df.merge(lines, on="row_id").groupby("line").prediction.mean().round(4).to_dict())
    if not a.dry_run:
        print("written:", submit.write(df, "FINAL_part1"))
    print("MANUAL: On kaggle.com -> competition -> Submissions, submit this file, then tick 'Use for Final Score' on it "
          "before 23 Oct 2026 16:00 UTC; untick others.")


if __name__ == "__main__":
    main()
