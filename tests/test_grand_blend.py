import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import grand_blend as gb  # noqa: E402


def _data(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    mu = rng.gamma(2, 0.5, n)
    y = rng.gamma(2.0, mu / 2) * (rng.random(n) < 0.5)
    truth = mu * 0.5
    parts = np.column_stack([truth * np.exp(rng.normal(0, 0.05, n))] +
                            [truth * np.exp(rng.normal(0, 0.6, n)) for _ in range(4)])
    return y, parts, np.arange(n) // 3


def test_nested_never_fits_on_heldout():
    y, P, g = _data()
    fit_rows, held = [], []

    def fit(yy, PP, gg):
        fit_rows.append((yy.copy(), set(gg)))
        return {"w": np.ones(PP.shape[1]) / PP.shape[1]}

    def app(m, PP):
        held.append(PP)
        return PP @ m["w"]

    out = gb.nested(y, P, g, fit, app, n_folds=5)
    assert len(out) == len(y) and len(fit_rows) == 5
    for (tr, va), (yy, gs), PPv in zip(GroupKFold(5).split(y, groups=g), fit_rows, held):
        assert not (gs & set(g[va]))             # no held-out policy seen by fit
        assert len(yy) == len(tr) and np.array_equal(yy, y[tr])
        assert np.array_equal(PPv, P[va])


def test_combiners_select_truth_part():
    y, P, g = _data()
    for fit in (gb.fit_greedy, gb.fit_arith, gb.fit_geo):
        m = fit(y, P, g)
        assert int(np.argmax(m["w"])) == 0, fit.__name__
        assert m["w"][0] > 0.4
