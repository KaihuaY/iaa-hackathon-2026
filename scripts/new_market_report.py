"""New-market readiness report: everything that differs from Part 1, before any modelling.

  python -P -E scripts/new_market_report.py <new_data_dir> [--ref data/raw/public] [--out docs/new_market_report.md]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import market  # noqa: E402
from market_spec import get_spec  # noqa: E402


def engineered_proxy_table():
    """Calendar-proxy table on the engineered training frame of the ACTIVE spec (IAA_MARKET), all lines pooled."""
    import pandas as pd
    import features
    base = features.build_base()
    tr = pd.concat([base[k] for k in base if k.startswith("train_")], ignore_index=True)
    skip = {"row_id", "claim_count", "claim_amount", "exposure", "year"}
    feats = [c for c in tr.columns if c not in skip and pd.api.types.is_numeric_dtype(tr[c])]
    return market.calendar_proxy_table(tr, "year", feats)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("new_dir")
    ap.add_argument("--ref", default=str(get_spec().data_path), help="reference (current market) data dir; default from the active MarketSpec")
    ap.add_argument("--out", default=str(ROOT / "docs" / "new_market_report.md"))
    ap.add_argument("--no-proxy", action="store_true", help="skip the calendar-proxy section (needs features.build_base, heavy)")
    a = ap.parse_args()
    if not Path(a.new_dir).is_dir():
        sys.exit(f"not a directory: {a.new_dir}")
    parts = market.readiness_parts(a.new_dir, a.ref)
    if not a.no_proxy:
        try:
            parts["proxy"] = engineered_proxy_table()
            parts["proxy"].to_csv(Path(a.out).with_name("calendar_proxy_features.csv"), index=False)
        except Exception as e:  # engineered frame needs the active spec's data; never block the report
            print(f"calendar-proxy section skipped: {e}")
    md = market.readiness_report(a.new_dir, a.ref, parts=parts)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print("\n".join(market.summary_lines(parts)))
    print(f"report written to {out}")


if __name__ == "__main__":
    main()
