"""Rebuild the submission from raw data. Usage (repo root):
python -P -E scripts/reproduce.py [--final X10_blend4] [--plan] [--check] [--skip-download] [--skip-tests] [--force]"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from configs import EXPERIMENTS  # noqa: E402

RAW = ROOT / "data" / "raw" / "public"
CACHE = ROOT / "experiments" / "cache"
RESULTS = ROOT / "experiments" / "results"
LOG = ROOT / "experiments" / "logs" / "reproduce.log"
COMP = "iaa-ai-assisted-hackathon-2026-part-1"
RAW_FILES = ["addresses", "claims", "customers", "policies_buildings", "policies_contents", "policies_motor",
             "sample_submission", "test_buildings", "test_contents", "test_motor", "weather"]
CACHE_FILES = [f"{k}_{l}" for k in ("train", "test") for l in ("motor", "buildings", "contents")] + ["claims"]


def order(final):
    """Dependency-ordered [(name, kind)], kind in train/blend/correction."""
    seen, out = set(), []

    def visit(name):
        if name in seen:
            return
        seen.add(name)
        if name in EXPERIMENTS:
            cfg = EXPERIMENTS[name]
            for p in cfg.get("parts", []):
                visit(p)
            out.append((name, "blend" if cfg["model"] == "blend" else "catboost" if cfg["model"] == "catboost" else "train"))
        elif name.endswith("_cred"):
            visit(name[:-5])
            out.append((name, "correction"))
        else:
            sys.exit(f"unknown experiment {name!r}")
    visit(final)
    return out


MINUTES = {"train": 45, "catboost": 60, "correction": 10, "blend": 5}


def steps(final, skip_download, skip_tests):
    """[(label, minutes, artefact_exists, command or callable)]"""
    py = [sys.executable, "-P", "-E"]
    s = []
    if not skip_download:
        s.append(("download raw data", 2, all((RAW / f"{f}.parquet").exists() for f in RAW_FILES), "download"))
    s.append(("build feature cache", 25, all((CACHE / f"{f}.parquet").exists() for f in CACHE_FILES), "cache"))
    if not skip_tests:
        s.append(("run tests", 2, False, py + ["-m", "pytest", "-q", "tests"]))
    for name, kind in order(final):
        base = name[:-5] if kind == "correction" else name
        script = "run_correction.py" if kind == "correction" else "run_experiment.py"
        s.append((f"{kind:10s} {name}", MINUTES[kind], (RESULTS / f"{name}.pkl").exists(),
                  py + [str(ROOT / "scripts" / script), base]))
    s.append((f"write submission {final}_reproduced", 1, False, "submit"))
    return s


def run_cmd(label, cmd, final):
    """Run one step; raise SystemExit with a clear message on failure."""
    LOG.parent.mkdir(parents=True, exist_ok=True)
    if cmd == "download":
        cmd = ["kaggle", "competitions", "download", "-c", COMP, "-p", str(ROOT / "data" / "raw")]
    if cmd == "cache":
        import features
        features.build_base(refresh=True)
        return
    if cmd == "submit":
        import pipeline
        import submit
        res = pipeline.load(final)
        sub = pipeline.test_predictions(res)
        path = submit.write(sub, f"{final}_reproduced")
        print(f"wrote {path}: {len(sub)} rows, mean prediction {sub.prediction.mean():.4f}")
        for line, l in res["lines"].items():
            print(f"  {line}: {len(l['test_pred'])} rows, mean {l['test_pred'].mean():.4f}")
        return
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"\n===== {label} =====\n")
        f.flush()
        rc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc:
        sys.exit(f"FAILED: {label} (exit {rc}). See {LOG}. Fix and re-run; finished steps are skipped.")
    if label.startswith("download"):
        import zipfile
        for z in (ROOT / "data" / "raw").glob("*.zip"):
            zipfile.ZipFile(z).extractall(ROOT / "data" / "raw")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", default="X10_blend4")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-run steps whose artefact exists")
    a = ap.parse_args()
    plan = steps(a.final, a.skip_download, a.skip_tests)
    if a.plan or a.check:
        total = 0
        for i, (label, mins, done, _) in enumerate(plan, 1):
            if a.check:
                print(f"{i:2d}. [{'present' if done else 'WOULD RUN'}] {label}")
            else:
                print(f"{i:2d}. {label:55s} ~{mins} min")
            total += 0 if (a.check and done) else mins
        print(f"Estimated runtime{' (remaining)' if a.check else ''}: ~{total} min (~{total / 60:.1f} h)")
        sys.exit(0)
    for i, (label, mins, done, cmd) in enumerate(plan, 1):
        if done and not a.force:
            print(f"[{i}/{len(plan)}] skip (artefact exists): {label}")
            continue
        print(f"[{i}/{len(plan)}] {label} (~{mins} min)", flush=True)
        run_cmd(label, cmd, a.final)
