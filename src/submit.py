"""Validate, write and submit a prediction file; fetch its leaderboard score."""
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from data import load

ROOT = Path(__file__).resolve().parents[1]
SUB_DIR = ROOT / "submissions"
COMP = "iaa-ai-assisted-hackathon-2026-part-1"


def validate(sub: pd.DataFrame) -> pd.DataFrame:
    """Order like sample_submission and enforce the competition's format rules."""
    sample = load("sample_submission")
    assert sub.row_id.is_unique, "duplicate row_id"
    assert set(sub.row_id) == set(sample.row_id), "row_id set differs from sample_submission"
    out = sample[["row_id"]].merge(sub, on="row_id", how="left")
    p = out.prediction.to_numpy(float)
    assert np.isfinite(p).all(), "non-finite predictions"
    assert (p >= 0).all(), "negative predictions"
    assert len(out) == 626_639, len(out)
    return out


def write(sub: pd.DataFrame, name: str) -> Path:
    SUB_DIR.mkdir(exist_ok=True)
    path = SUB_DIR / f"{datetime.now():%Y%m%d_%H%M}_{name}.csv"
    validate(sub).to_csv(path, index=False, float_format="%.4f")
    return path


MARKER = ROOT / "experiments" / "logs" / "tests_passed_at"


def _code_state() -> str | None:
    """HEAD commit if src/tests/configs have no uncommitted changes, else None."""
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    if git("status", "--porcelain", "--", "src", "tests", "experiments/configs.py"):
        return None
    return git("rev-parse", "HEAD")


def run_tests() -> None:
    """pytest gate. Skipped only when the suite already passed on this exact, clean commit."""
    state = _code_state()
    if state and MARKER.exists() and MARKER.read_text().strip() == state:
        print(f"  tests already passed on {state[:7]} (code unchanged), gate satisfied")
        return
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", str(ROOT / "tests")], cwd=ROOT,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise SystemExit("tests failed, not submitting:\n" + r.stdout[-3000:])
    if state:
        MARKER.write_text(state)


def _kaggle(*args) -> str:
    r = subprocess.run([sys.executable, "-m", "kaggle", "competitions", *args], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-2000:] or r.stdout[-2000:])
    return r.stdout


def remaining_today() -> int:
    out = _kaggle("submission-limits", COMP)
    return int(next(l for l in out.splitlines() if l.startswith("Remaining today")).split(":")[1])


def submit(path: Path, message: str, wait: int = 600) -> float | None:
    """Upload via the Kaggle CLI and poll until the public score is available."""
    _kaggle("submit", COMP, "-f", str(path), "-m", message)
    t0 = time.time()
    while time.time() - t0 < wait:
        time.sleep(15)
        subs = json.loads(_kaggle("submissions", COMP, "--format", "json"))
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
