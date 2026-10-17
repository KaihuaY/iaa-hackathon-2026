"""Level-frequency stability screen for the active market (reads the feature cache only).

  python scripts/stability_screen.py [--out docs/<market>_stability.md] [--min-rows 300]
  IAA_MARKET=rehearsal python scripts/stability_screen.py --out docs/rehearsal_stability.md

See src/stability.py for the definition and the weather-variable caveat.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]

from features import build_base, model_matrix  # noqa: E402
from market_spec import get_spec  # noqa: E402
from stability import suggest, stability_table  # noqa: E402


def main():
    spec = get_spec()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"docs/{spec.name}_stability.md")
    ap.add_argument("--min-rows", type=int, default=300)
    a = ap.parse_args()
    base = build_base()
    early = sorted(spec.train_years)[:-2]
    md = [f"# Level-frequency stability screen - market `{spec.name}`", "",
          f"stability = exposure-weighted correlation of level claim frequencies, years {early[0]}-{early[-1]} vs "
          f"{spec.last_mature_year}; spread = exposure-weighted sd of level frequencies / overall frequency. "
          f"Levels need >= {a.min_rows} rows in both periods. noise: stability < 0.2 and spread < 0.1; watch: 0.2-0.5. "
          "Weather variables vary by year by design: do not drop them on this screen alone.", ""]
    for line in spec.lines:
        train = base[f"train_{line}"]
        X, _, cats = model_matrix(train, base[f"test_{line}"], {"history": True})
        tab = stability_table(train, X, cats, spec, a.min_rows)
        noise, watch = suggest(tab)
        print(f"[{line}] noise: {noise}\n[{line}] watch: {watch}", flush=True)
        md += [f"## {line}", "", f"noise: {', '.join(noise) or '-'}", "", f"watch: {', '.join(watch) or '-'}", "",
               "| feature | kind | levels | stability | spread |", "|---|---|---|---|---|"]
        md += [f"| {r.feature} | {r.kind} | {r.n_levels} | {'' if r.stability != r.stability else f'{r.stability:.2f}'} | "
               f"{'' if r.spread != r.spread else f'{r.spread:.3f}'} |" for r in tab.itertuples()]
        md.append("")
    out = Path(a.out)
    out = out if out.is_absolute() else ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
