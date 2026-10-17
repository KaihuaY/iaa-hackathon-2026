"""Run several experiments one after another, each in a fresh process (frees memory between runs).

  python scripts/run_chain.py G3_dev G4_cap G5_hist:submit ...     (":submit" submits that one)

Writes its PID to experiments/logs/chain.pid. Stop everything with:  taskkill /T /F /PID <pid>
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
(ROOT / "experiments" / "logs").mkdir(parents=True, exist_ok=True)
(ROOT / "experiments" / "logs" / "chain.pid").write_text(str(os.getpid()))

for item in sys.argv[1:]:
    exp, _, flag = item.partition(":")
    cmd = [sys.executable, "-P", "-E", str(ROOT / "scripts" / "run_experiment.py"), exp] + (["--submit"] if flag == "submit" else [])
    print(f"### start {exp}", flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    print(f"### end {exp} rc={r.returncode}", flush=True)
print("CHAIN DONE", flush=True)
