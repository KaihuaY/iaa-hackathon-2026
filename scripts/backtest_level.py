"""Run the forward level backtest for each trend method; saves experiments/results/backtest_level.csv."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

from backtest import level_backtest  # noqa: E402
from configs import FULL_T  # noqa: E402

if __name__ == "__main__":
    cfg = {**FULL_T, "cap_q": 0.99}
    df = level_backtest(cfg)
    out = ROOT / "experiments" / "results" / "backtest_level.csv"
    df.to_csv(out, index=False)
    print(df.pivot_table(index=["line", "target_year"], columns="trend", values="level_ratio").round(3))
    print(df.pivot_table(index=["line", "target_year"], columns="trend", values="d2").round(4))
