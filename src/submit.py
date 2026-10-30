"""Validate, write and submit a prediction file; fetch its leaderboard score."""
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import data as _data
from market_spec import get_spec

ROOT = Path(__file__).resolve().parents[1]
SUB_DIR = ROOT / "submissions"
# market -> (sub-folder, file-name prefix); markets that score on the same Kaggle competition share one folder
SUB_FOLDER = {"part2pool": ("part2", "P2_POOL_")}


def __getattr__(name):
    if name == "COMP":                          # competition slug of the active MarketSpec (PEP 562)
        return get_spec().competition
    raise AttributeError(f"module 'submit' has no attribute {name!r}")


def validate(sub: pd.DataFrame) -> pd.DataFrame:
    """Order like sample_submission and enforce the competition's format rules."""
    spec = get_spec()
    sample = _data.load("sample_submission")
    assert sub.row_id.is_unique, "duplicate row_id"
    assert set(sub.row_id) == set(sample.row_id), "row_id set differs from sample_submission"
    out = sample[["row_id"]].merge(sub, on="row_id", how="left")
    p = out.prediction.to_numpy(float)
    assert np.isfinite(p).all(), "non-finite predictions"
    assert (p >= 0).all(), "negative predictions"
    expected = len(sample) if spec.expected_rows is None else spec.expected_rows
    assert len(out) == expected, len(out)
    return out


def write(sub: pd.DataFrame, name: str) -> Path:
    """Part 1: submissions/<local time>_<name>.csv (unchanged). Other markets: submissions/<market>/<UTC time>_<name>.csv."""
    spec = get_spec()
    if spec.name == "part1":
        SUB_DIR.mkdir(exist_ok=True)
        path = SUB_DIR / f"{datetime.now():%Y%m%d_%H%M}_{name}.csv"
    else:
        folder, prefix = SUB_FOLDER.get(spec.name, (spec.name, ""))
        (SUB_DIR / folder).mkdir(parents=True, exist_ok=True)
        path = SUB_DIR / folder / f"{datetime.now(timezone.utc):%Y%m%d_%H%MZ}_{prefix}{name}.csv"
    validate(sub).to_csv(path, index=False, float_format="%.4f")
    return path


MARKER = ROOT / "experiments" / "logs" / "tests_passed_at"


def _code_state() -> str | None:
    """HEAD commit if src/tests/configs have no uncommitted changes, else None."""
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    if git("status", "--porcelain", "--", "src", "tests", "experiments/configs.py", "experiments/configs_part2.py"):
        return None
    return git("rev-parse", "HEAD")


def run_tests() -> None:
    """pytest gate. Skipped only when the suite already passed on this exact, clean commit.
    The subprocess inherits IAA_MARKET, so under another market the Part 1 data tests (marker part1_data) are
    skipped by tests/conftest.py and the market-generic tests (incl. submission validation) still run."""
    state = _code_state()
    name = get_spec().name                       # per-market marker: a part2 pass (Part 1 tests skipped) must not
    marker = MARKER if name == "part1" else MARKER.with_name(f"{MARKER.name}_{name}")   # satisfy the Part 1 gate
    if state and marker.exists() and marker.read_text().strip() == state:
        print(f"  tests already passed on {state[:7]} (code unchanged), gate satisfied")
        return
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", str(ROOT / "tests")], cwd=ROOT,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise SystemExit("tests failed, not submitting:\n" + r.stdout[-3000:])
    if state:
        marker.write_text(state)


def _kaggle(*args) -> str:
    r = subprocess.run([sys.executable, "-m", "kaggle", "competitions", *args], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-2000:] or r.stdout[-2000:])
    return r.stdout


def remaining_today() -> int:
    out = _kaggle("submission-limits", get_spec().competition)
    return int(next(l for l in out.splitlines() if l.startswith("Remaining today")).split(":")[1])


def submit(path: Path, message: str, wait: int = 600) -> float | None:
    """Upload via the Kaggle CLI and poll until the public score is available."""
    _kaggle("submit", get_spec().competition, "-f", str(path), "-m", message)
    t0 = time.time()
    while time.time() - t0 < wait:
        time.sleep(15)
        subs = json.loads(_kaggle("submissions", get_spec().competition, "--format", "json"))
        mine = [s for s in subs if s.get("description") == message or s.get("fileName") == path.name]
        if mine:
            s = mine[0]
            status = str(s.get("status", "")).lower()
            score = s.get("publicScore")
            if score not in (None, "", "None"):
                return float(score)
            if "error" in status:
                raise RuntimeError(f"submission error: {s}")
    return None
