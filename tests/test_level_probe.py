import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from level_probe import fit_optimum  # noqa: E402


def test_known_parabola():
    s = [0.95, 1.0, 1.05]
    y = [0.07 - 2.0 * (x - 1.03) ** 2 for x in s]
    opt, interior, _ = fit_optimum(s, y)
    assert interior and abs(opt - 1.03) < 1e-6


def test_upward_parabola_clips():
    s = [0.95, 1.0, 1.05]
    y = [0.07 + 2.0 * (x - 1.03) ** 2 for x in s]
    opt, interior, _ = fit_optimum(s, y)
    assert not interior and opt in (0.85, 1.15)
