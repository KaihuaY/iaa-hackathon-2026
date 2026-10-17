"""Synthetic test of the level-frequency stability screen."""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import market_spec as ms  # noqa: E402
from stability import stability_table, suggest  # noqa: E402


def test_planted_stable_and_noise_factors():
    rng = np.random.default_rng(0)
    n = 120_000
    spec = replace(ms.PART1, name="stab_t", train_years=(2019, 2020, 2021, 2022), mature_years=(2019, 2020, 2021, 2022))
    year = rng.choice(spec.train_years, n)
    stable = rng.integers(0, 6, n)
    noise = rng.integers(0, 6, n)
    shuffle = {y: rng.permutation(6) for y in spec.train_years}               # noise levels are re-ranked every year
    rel = np.exp(0.25 * stable) * np.exp(0.05 * np.array([shuffle[y][v] for y, v in zip(year, noise)]))
    expo = rng.uniform(0.3, 1.0, n)
    cc = rng.poisson(0.08 * expo * rel)
    train = pd.DataFrame({"year": year, "claim_count": cc, "exposure": expo})
    X = pd.DataFrame({"stable_f": pd.Categorical(stable), "noise_f": pd.Categorical(noise),
                      "num_f": rng.normal(size=n)})
    tab = stability_table(train, X, ["stable_f", "noise_f"], spec, min_rows=300).set_index("feature")
    assert tab.loc["stable_f", "stability"] > 0.9
    assert tab.loc["noise_f", "stability"] < 0.3
    assert tab.index[0] == "stable_f"
    noise_list, _ = suggest(tab.reset_index())
    assert "stable_f" not in noise_list
