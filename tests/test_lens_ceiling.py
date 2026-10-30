import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

import lens_ceiling as lc  # noqa: E402


def _truth(n=200_000, seed=1):
    rng = np.random.default_rng(seed)
    mu = rng.lognormal(0, 0.8, n)
    mu *= 400 / mu.mean()
    sev = rng.lognormal(7, 1.0, 50_000)
    sbar = sev.mean()
    y = lc.simulate_compound(mu, sev, sbar, rng)
    return mu, sev, sbar, y


def _stats(y, mu, sev, sbar, seed):
    rng = np.random.default_rng(seed)
    return lc.summarize(lc.ceiling_stats(y, mu, (lc.simulate_compound(mu, sev, sbar, rng) for _ in range(20))))


def test_true_mu_is_consistent():
    mu, sev, sbar, y = _truth()
    s = _stats(y, mu, sev, sbar, 5)
    assert abs(s["z"]) < 3 and abs(s["excess_share"]) < 0.01


def test_flat_mu_has_headroom():
    mu, sev, sbar, y = _truth()
    s = _stats(y, np.full_like(mu, mu.mean()), sev, sbar, 6)
    assert s["excess_share"] > 0 and s["z"] > 5


def test_simulate_preserves_mean():
    mu, sev, sbar, _ = _truth()
    rng = np.random.default_rng(3)
    assert abs(lc.simulate_compound(mu, sev, sbar, rng).mean() / mu.mean() - 1) < 0.02
    g = lc.simulate_compound(mu, None, sbar, rng, gamma_cv=sev.std() / sbar)
    assert abs(g.mean() / mu.mean() - 1) < 0.03
