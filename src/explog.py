"""Experiment log (CSV + rendered markdown). Doubles as the record of AI-assisted work and its verification."""
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "experiments" / "log.csv"
MD = ROOT / "docs" / "experiment_log.md"
COLS = ["exp", "time", "description", "proposer", "cv", "cv_nested", "cv_motor", "cv_buildings", "cv_contents",
        "holdout_2023", "lb_pred", "lb_public", "decision", "verification", "config", "file"]
KEEP = ["lb_public", "decision"]  # re-running an experiment must not wipe its leaderboard score or decision


def record(res: dict, desc: str, proposer: str, lb=None, decision="", verification="", file="", lb_pred="") -> None:
    L = res["lines"]
    row = {"exp": res["name"], "time": f"{datetime.now():%Y-%m-%d %H:%M}", "description": desc, "proposer": proposer,
           "cv": round(res["cv"], 5), **{f"cv_{l}": round(L[l]["cv"], 5) for l in L},
           "holdout_2023": round(res["holdout"], 5) if res["holdout"] == res["holdout"] else "",
           "cv_nested": round(res["cv_nested"], 5) if "cv_nested" in res else "", "lb_pred": lb_pred,
           "lb_public": lb if lb is not None else "", "decision": decision, "verification": verification,
           "config": json.dumps(res["cfg"], default=str), "file": file}
    df = pd.read_csv(LOG) if LOG.exists() else pd.DataFrame(columns=COLS)
    for c in COLS:
        if c not in df:
            df[c] = ""
    old = df[df.exp == res["name"]]
    if len(old):
        for k in KEEP:
            if row[k] in ("", None) and pd.notna(old.iloc[0][k]):
                row[k] = old.iloc[0][k]
    df = pd.concat([df[df.exp != res["name"]], pd.DataFrame([row])], ignore_index=True)[COLS]
    LOG.parent.mkdir(exist_ok=True)
    df.to_csv(LOG, index=False)
    render(df)


def update(exp: str, **fields) -> None:
    df = pd.read_csv(LOG)
    for k, v in fields.items():
        df[k] = df[k].astype(object)
        df.loc[df.exp == exp, k] = v
    df.to_csv(LOG, index=False)
    render(df)


def render(df: pd.DataFrame | None = None) -> None:
    df = pd.read_csv(LOG) if df is None else df
    show = ["exp", "time", "description", "proposer", "cv", "cv_nested", "cv_motor", "cv_buildings", "cv_contents",
            "holdout_2023", "lb_pred", "lb_public", "decision", "verification"]
    show = [c for c in show if c in df]
    lines = ["# Experiment log", "",
             "Every model run, how it was checked, and what was decided. `cv` = 5-fold Tweedie D² grouped by policy "
             "(all years, developed labels); `holdout_2023` = train 2021–22 → score 2023; `lb_public` = Kaggle public "
             "leaderboard; `lb_pred` = LB predicted from CV (src/lbmodel.py); `cv_nested` = blend CV with weights chosen out-of-fold. "
             "Proposer = who suggested the change (user / AI). Keep rule: Kaggle and CV must agree; submissions need user approval.", "",
             "| " + " | ".join(show) + " |", "|" + "---|" * len(show)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(r[c]) else str(r[c]) for c in show) + " |")
    MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
