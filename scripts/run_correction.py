"""Usage (from repo root): python -P -E scripts/run_correction.py X6_blend3bag [--no-buhlmann] [--no-segments]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from credibility import correct  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--no-buhlmann", action="store_true")
    ap.add_argument("--no-segments", action="store_true")
    a = ap.parse_args()
    res = correct(a.name, use_buhlmann=not a.no_buhlmann, use_segments=not a.no_segments)
    print(f"\n{res['name']}: overall nested D2 {res['cv']:.5f}")
    for line, l in res["lines"].items():
        print(f"  {line}: {l['cv']:.5f}  k={res['k'].get(line)}  alpha={res['alpha'].get(line)}")
