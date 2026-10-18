"""Read-only leaderboard watcher: one CSV line per call in docs/leaderboard_watch.csv + a one-line summary.

Uses the Kaggle CLI (`competitions leaderboard <slug> --show`); never submits anything.
"""
import csv
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]

OUR_TEAM = "kaiY"
OUT = ROOT / "docs" / "leaderboard_watch.csv"
FIELDS = ["timestamp_utc", "our_rank", "our_score", "top1_team", "top1_score", "top2_score", "margin_to_next"]
MARGIN_ALERT = 0.0003
_ROW = re.compile(r"^\s*(\d+)\s{2,}(.+?)\s{2,}(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d(?:\.\d+)?)\s+(-?[\d.]+)\s*$")


def parse(text: str) -> list[dict]:
    """Table rows in leaderboard order (rank = position): [{rank, team, score}]."""
    rows = []
    for line in text.splitlines():
        m = _ROW.match(line.rstrip("\r"))
        if m:
            rows.append({"rank": len(rows) + 1, "team": m.group(2).strip(), "score": float(m.group(4))})
    return rows


def summarise(rows: list[dict], us: str = OUR_TEAM) -> dict:
    assert rows, "no leaderboard rows parsed"
    mine = next((r for r in rows if r["team"] == us), None)
    top1 = rows[0]
    top2 = rows[1]["score"] if len(rows) > 1 else float("nan")
    if mine is None:
        return {"our_rank": "", "our_score": "", "top1_team": top1["team"], "top1_score": top1["score"],
                "top2_score": top2, "margin_to_next": ""}
    if mine["rank"] == 1:
        margin = mine["score"] - top2
    else:
        margin = top1["score"] - mine["score"]
    return {"our_rank": mine["rank"], "our_score": mine["score"], "top1_team": top1["team"],
            "top1_score": top1["score"], "top2_score": top2, "margin_to_next": round(margin, 6)}


def alert(s: dict) -> bool:
    return s["our_rank"] == "" or s["our_rank"] > 1 or s["margin_to_next"] < MARGIN_ALERT


def main() -> dict:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")       # team names may be non-ASCII on a cp1252 console
    from market_spec import get_spec
    from submit import _kaggle
    rows = parse(_kaggle("leaderboard", get_spec().competition, "--show"))
    s = summarise(rows)
    rec = {"timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), **s}
    OUT.parent.mkdir(exist_ok=True)
    new = not OUT.exists()
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, FIELDS)
        if new:
            w.writeheader()
        w.writerow(rec)
    top5 = ", ".join(f"#{r['rank']} {r['team']} {r['score']:.5f}" for r in rows[:5])
    print(f"{rec['timestamp_utc']} us #{s['our_rank']} {s['our_score']} | margin_to_next {s['margin_to_next']} | top5: {top5}"
          + (" | ALERT" if alert(s) else ""))
    return s


if __name__ == "__main__":
    main()
