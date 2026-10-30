"""Office premium build-up for the 20 Part 2 reference policies.

Joins data/part2/emerging/reference_policies.parquet to a submission CSV (row_id, prediction = risk premium for the
row's period) and loads it with the `expenses` assumptions (table-driven, either market column):

    RP   = prediction (row basis, i.e. for the row's exposure)
    CAT  = cat_rate * RP for property lines (buildings, contents), 0 for motor
    F    = fixed expense per policy (CU, not pro-rated by exposure)
    D    = 1 - variable - commission[channel] - premium_tax - profit          (all "share of premium" items)
    P    = (RP + CAT + F) / D                                                 (primary: shares of the office premium)
    Palt = (RP + CAT + F) / (D + premium_tax) * (1 + premium_tax)             (alternative: tax on top of net premium)

Writes a Markdown section (docs/part2_office_premium.md) and a CSV (reports/part2_office_premium.csv).
Usage: python -P -E scripts/office_premium.py --predictions <csv> [--market emerging] [--placeholder]
"""
import argparse
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "part2" / "emerging"
PROPERTY_LINES = ("buildings", "contents")
LINES = ("motor", "buildings", "contents")
UNIT_CU = "CU"
UNIT_SHARE = "share of premium"
UNIT_PROPERTY_RP = "share of property risk premium"
COMMISSION_PREFIX = "commission_"
TAX_ITEM = "premium_tax"


# --------------------------------------------------------------------------------------------- assumptions

def load_expenses(path=None, market="emerging", table=None):
    """Parse the expenses table into loadings. Pass `table` (DataFrame with item, <market>, unit) to bypass the file.

    Returns dict: fixed (CU sum), cat (property-RP share sum), shares {item: rate} (non-commission share-of-premium
    items, incl. premium_tax), commission {channel: rate or NaN}, tax (premium_tax rate), raw (item -> (value, unit)).
    """
    t = table if table is not None else pd.read_parquet(path or DATA / "expenses.parquet")
    if market not in t.columns:
        raise ValueError(f"market column {market!r} not in expenses table ({list(t.columns)})")
    fixed, cat, shares, commission, raw = 0.0, 0.0, {}, {}, {}
    for _, r in t.iterrows():
        item, unit, v = str(r["item"]), str(r["unit"]).strip(), float(r[market]) if pd.notna(r[market]) else math.nan
        raw[item] = (v, unit)
        if item.startswith(COMMISSION_PREFIX):
            if unit != UNIT_SHARE:
                raise ValueError(f"commission item {item} has unexpected unit {unit!r}")
            commission[item[len(COMMISSION_PREFIX):]] = v
        elif unit == UNIT_CU:
            fixed += 0.0 if math.isnan(v) else v
        elif unit == UNIT_PROPERTY_RP:
            cat += 0.0 if math.isnan(v) else v
        elif unit == UNIT_SHARE:
            shares[item] = 0.0 if math.isnan(v) else v
        else:
            raise ValueError(f"unknown unit {unit!r} for item {item}")
    return dict(fixed=fixed, cat=cat, shares=shares, commission=commission, tax=shares.get(TAX_ITEM, 0.0),
                raw=raw, market=market)


def commission_rate(exp, channel):
    """Commission share for a channel; NaN / missing -> 0 (flagged as an assumption by the caller)."""
    v = exp["commission"].get(str(channel), math.nan)
    return (0.0, True) if math.isnan(v) else (v, False)


# --------------------------------------------------------------------------------------------- build-up

def build(policies, rp, exp):
    """policies: DataFrame with row_id, line, channel, exposure, written_premium (+ optional sum_insured, cover_type).
    rp: array-like risk premium per row (row basis). Returns one row per policy with every component."""
    p = policies.reset_index(drop=True).copy()
    p["rp"] = np.asarray(rp, float)
    if p["rp"].isna().any():
        raise ValueError("missing risk premium for: " + ", ".join(p.loc[p["rp"].isna(), "row_id"].astype(str)))
    e = p["exposure"].astype(float)
    is_prop = p["line"].isin(PROPERTY_LINES)
    p["rp_pa"] = p["rp"] / e
    p["cat_rate"] = np.where(is_prop, exp["cat"], 0.0)
    p["cat"] = p["cat_rate"] * p["rp"]
    p["fixed"] = exp["fixed"]
    cr = [commission_rate(exp, ch) for ch in p["channel"]]
    p["commission_rate"] = [c for c, _ in cr]
    p["commission_assumed_zero"] = [f for _, f in cr]
    other = sum(exp["shares"].values())                      # variable + tax + profit (+ any other share items)
    p["denominator"] = 1.0 - other - p["commission_rate"]
    if (p["denominator"] <= 0).any():
        raise ValueError("share-of-premium loadings sum to >= 100%")
    p["cost_base"] = p["rp"] + p["cat"] + p["fixed"]
    p["office"] = p["cost_base"] / p["denominator"]
    # component split of the primary office premium (sums back to `office`)
    for item, rate in exp["shares"].items():
        p[f"part_{item}"] = rate * p["office"]
    p["part_commission"] = p["commission_rate"] * p["office"]
    # alternative: premium tax charged on top of the net office premium
    tax = exp["tax"]
    p["denominator_net"] = p["denominator"] + tax
    p["office_net"] = p["cost_base"] / p["denominator_net"]
    p["office_tax_on_top"] = p["office_net"] * (1.0 + tax)
    # alternative: fixed expense pro-rated by exposure
    p["office_fixed_prorated"] = (p["rp"] + p["cat"] + p["fixed"] * e) / p["denominator"]
    # annual-equivalent office premium (full year of cover, one fixed expense) for comparison with written_premium
    p["office_pa"] = (p["rp_pa"] * (1.0 + p["cat_rate"]) + p["fixed"]) / p["denominator"]
    p["ratio_to_written"] = p["office_pa"] / p["written_premium"].astype(float)
    return p


def summary_by_line(b):
    g = b.groupby("line", sort=False)
    s = pd.DataFrame({
        "n": g.size(), "sum_rp": g["rp"].sum(), "sum_cat": g["cat"].sum(), "sum_fixed": g["fixed"].sum(),
        "sum_office": g["office"].sum(), "sum_office_tax_on_top": g["office_tax_on_top"].sum(),
        "sum_written": g["written_premium"].sum(), "mean_ratio": g["ratio_to_written"].mean(),
    })
    s = s.reindex([l for l in LINES if l in s.index])
    tot = s.sum(numeric_only=True)
    tot["mean_ratio"] = b["ratio_to_written"].mean()
    s.loc["total"] = tot
    s["n"] = s["n"].astype(int)
    return s


# --------------------------------------------------------------------------------------------- rendering

def _f(x, d=2):
    if isinstance(x, (float, np.floating)):
        if math.isnan(x):
            return "n/a"
        return f"{x:,.{d}f}"
    return str(x)


def md_table(header, rows, align=None):
    align = align or ["---"] * len(header)
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(align) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _pct(x):
    return f"{100 * x:.1f}%" if not math.isnan(x) else "n/a"


def assumptions_table(exp):
    rows = []
    for item, (v, unit) in exp["raw"].items():
        if item.startswith(COMMISSION_PREFIX):
            use = f"denominator, if channel = {item[len(COMMISSION_PREFIX):]}"
            if math.isnan(v):
                use += " (blank in table: taken as 0, assumption)"
        elif unit == UNIT_CU:
            use = "added per policy (CU), not pro-rated"
        elif unit == UNIT_PROPERTY_RP:
            use = "x risk premium, buildings and contents only"
        else:
            use = "denominator (share of office premium)"
        rows.append([f"`{item}`", "blank" if math.isnan(v) else (_f(v) if unit == UNIT_CU else _pct(v)), unit, use])
    return md_table(["Item", exp["market"], "Unit", "How it enters"], rows, ["---", "---:", "---", "---"])


def worked_example(r, exp):
    e, rp = r["exposure"], r["rp"]
    shares = exp["shares"]
    v = shares.get("variable_expense", 0.0)
    t = exp["tax"]
    m = shares.get("target_profit_margin", 0.0)
    other_items = {k: x for k, x in shares.items() if k not in ("variable_expense", TAX_ITEM, "target_profit_margin")}
    c = r["commission_rate"]
    prop = r["line"] in PROPERTY_LINES
    desc = f"sum insured {_f(r['sum_insured'], 0)} CU" if prop else f"cover `{r['cover_type']}`"
    L = [f"**{r['row_id']}** ({r['line']}, {r['channel']}, exposure {e:.4f} years, {desc}, "
         f"current written premium {_f(r['written_premium'])} CU a year).", ""]
    L.append(f"1. Risk premium for the row: RP = {_f(rp)} CU (annualised: {_f(rp)} / {e:.4f} = {_f(r['rp_pa'])} CU a year).")
    if prop:
        L.append(f"2. Catastrophe reinsurance: CAT = {_pct(exp['cat'])} x {_f(rp)} = {_f(r['cat'])} CU.")
    else:
        L.append("2. Catastrophe reinsurance: motor is not a property line, so CAT = 0.")
    L.append(f"3. Fixed expense per policy: F = {_f(r['fixed'])} CU (not pro-rated).")
    L.append(f"4. Costs to recover before percentage loadings: RP + CAT + F = {_f(rp)} + {_f(r['cat'])} + "
             f"{_f(r['fixed'])} = {_f(r['cost_base'])} CU.")
    parts = [f"{_pct(v)} variable", f"{_pct(c)} commission ({r['channel']})", f"{_pct(t)} premium tax",
             f"{_pct(m)} profit"] + [f"{_pct(x)} {k}" for k, x in other_items.items()]
    L.append(f"5. Share-of-premium loadings: {' + '.join(parts)} = {_pct(1 - r['denominator'])}; "
             f"denominator D = 1 - {1 - r['denominator']:.4f} = {r['denominator']:.4f}.")
    L.append(f"6. Office premium P = {_f(r['cost_base'])} / {r['denominator']:.4f} = **{_f(r['office'])} CU** for the row.")
    chk = [f"variable {_f(r['part_variable_expense'])}" if "variable_expense" in shares else None,
           f"commission {_f(r['part_commission'])}",
           f"premium tax {_f(r['part_' + TAX_ITEM])}" if TAX_ITEM in shares else None,
           f"profit {_f(r['part_target_profit_margin'])}" if "target_profit_margin" in shares else None]
    chk += [f"{k} {_f(r['part_' + k])}" for k in other_items]
    chk = [x for x in chk if x]
    total = r["cost_base"] + sum(r["part_" + k] for k in shares) + r["part_commission"]
    L.append(f"7. Check: {_f(rp)} risk + {_f(r['cat'])} cat + {_f(r['fixed'])} fixed + " + " + ".join(chk)
             + f" = {_f(total)} CU = P.")
    L.append(f"8. Tax-on-top alternative: net premium = {_f(r['cost_base'])} / {r['denominator_net']:.4f} = "
             f"{_f(r['office_net'])}; x (1 + {t:.2f}) = {_f(r['office_tax_on_top'])} CU.")
    L.append(f"9. Comparison with the current price: annual-equivalent office premium = "
             f"({_f(r['rp_pa'])} x {1 + r['cat_rate']:.2f} + {_f(r['fixed'])}) / {r['denominator']:.4f} = "
             f"{_f(r['office_pa'])} CU a year, {r['ratio_to_written']:.2f} x the written premium of "
             f"{_f(r['written_premium'])}.")
    return "\n".join(L)


def pick_examples(b, rows=None):
    if rows:
        return [b.index[b["row_id"] == r][0] for r in rows]
    out = []
    for line, pref in (("motor", "direct"), ("buildings", "broker")):
        sub = b[(b["line"] == line) & (b["exposure"] > 0.9)]
        sub_p = sub[sub["channel"] == pref]
        sub = sub_p if len(sub_p) else (sub if len(sub) else b[b["line"] == line])
        if len(sub):
            out.append(sub.index[0])
    return out


def render(b, exp, pred_path, placeholder, examples):
    s = summary_by_line(b)
    t = exp["tax"]
    L = []
    if placeholder:
        L += ["> **PLACEHOLDER - rerun with the final submission file.** The risk premiums below are NOT model "
              "predictions: they are a stand-in (0.6 x written premium x exposure) used only to render the layout. "
              "Every number in this document changes when the script is rerun with the final Part 2 submission.", ""]
    L += ["# Part 2: office premium build-up for the 20 reference policies", "",
          f"Generated {datetime.now():%Y-%m-%d %H:%M} by `scripts/office_premium.py` from `{pred_path}` "
          f"(risk premiums) and `expenses.parquet` (column `{exp['market']}`). All amounts in CU.", "",
          "## Convention", "",
          "The risk premium RP is our predicted expected ultimate claim cost for the row's period (net of excess and "
          "recoveries), so it already reflects the row's exposure. The office premium is built on that same row basis:",
          "",
          "```",
          "CAT = cat_reinsurance x RP            (buildings and contents only; 0 for motor)",
          "F   = fixed_expense_per_policy        (CU per policy, not pro-rated)",
          "D   = 1 - variable_expense - commission[channel] - premium_tax - target_profit_margin",
          "P   = (RP + CAT + F) / D              (primary)",
          "",
          "Alternative (premium tax on top of the net premium):",
          "P_net = (RP + CAT + F) / (1 - variable_expense - commission[channel] - target_profit_margin)",
          "P_alt = P_net x (1 + premium_tax)",
          "```", "",
          "Every item whose unit is \"share of premium\" is read as a share of the office premium actually charged, so "
          "it sits in the denominator: after paying variable expenses, commission, premium tax and the profit margin "
          "out of P, exactly RP + CAT + F is left to pay claims, reinsurance and fixed costs.", "",
          "### Assumptions used", "", assumptions_table(exp), "",
          "1. RP is the submission prediction for the row (row basis, i.e. already for its exposure). RP p.a. = RP / "
          "exposure is shown for readability only.",
          "2. Catastrophe reinsurance is a cost loading on property risk premium (buildings, contents); motor carries "
          "none. It is added to the risk premium, not netted off it: we price the gross risk and buy the cover.",
          "3. The fixed expense is a per-policy cost and is charged in full on each reference row, whatever its "
          "exposure (see the note on pro-rating below).",
          "4. Variable expense, commission, premium tax and profit margin are shares of the office premium "
          "(gross of tax) and enter the denominator.",
          "5. Commission follows the policy's `channel`. A blank commission rate (comparison site in the emerging "
          "market) is taken as 0; no reference policy is sold through that channel, so this does not affect any "
          "number here.",
          "6. `written_premium` is an annual rate (in the training data its mean is flat across exposure bands, it "
          "does not scale with exposure), so the comparison column uses an annual-equivalent office premium: "
          "P p.a. = (RP p.a. x (1 + cat rate) + F) / D, i.e. one full year of cover and one fixed expense. "
          "Ratio = P p.a. / written premium.",
          "7. No investment income, no allowance for claims-handling costs beyond what the risk premium already "
          "includes, and no rounding or minimum premium.", ""]
    # worked examples
    L += ["## Worked examples", ""]
    for i in examples:
        L += [worked_example(b.loc[i], exp), ""]
    # per-policy table
    hdr = ["row_id", "line", "channel", "exposure", "sum insured / cover", "RP", "RP p.a.", "CAT", "F", "D",
           "P (primary)", "P (tax on top)", "P p.a.", "written p.a.", "P p.a. / written"]
    rows = []
    for _, r in b.iterrows():
        si = _f(r["sum_insured"], 0) if r["line"] in PROPERTY_LINES else str(r.get("cover_type", ""))
        rows.append([r["row_id"], r["line"], r["channel"], f"{r['exposure']:.4f}", si, _f(r["rp"]), _f(r["rp_pa"]),
                     _f(r["cat"]), _f(r["fixed"]), f"{r['denominator']:.4f}", _f(r["office"]),
                     _f(r["office_tax_on_top"]), _f(r["office_pa"]), _f(r["written_premium"]),
                     f"{r['ratio_to_written']:.2f}"])
    L += ["## Office premium by reference policy", "",
          "RP, CAT, P (primary) and P (tax on top) are for the row's period; P p.a. is the annual equivalent used for "
          "the comparison with the current written premium.", "",
          md_table(hdr, rows, ["---", "---", "---", "---:", "---"] + ["---:"] * (len(hdr) - 5)), ""]
    # summary
    srows = [[k, int(r["n"]), _f(r["sum_rp"]), _f(r["sum_cat"]), _f(r["sum_fixed"]), _f(r["sum_office"]),
              _f(r["sum_office_tax_on_top"]), f"{r['mean_ratio']:.2f}"] for k, r in s.iterrows()]
    L += ["## Summary by line", "",
          md_table(["line", "n", "sum RP", "sum CAT", "sum F", "sum P (primary)", "sum P (tax on top)",
                    "mean P p.a. / written"], srows, ["---"] + ["---:"] * 7), ""]
    # fixed-expense floor: the office premium cannot fall below F / D even at zero risk premium
    floor = (b["fixed"] / b["denominator"])
    is_below = b["written_premium"] < floor
    below = int(is_below.sum())
    by_line = ", ".join(f"{n} {l}" for l, n in b.loc[is_below, "line"].value_counts().items())
    fl = {ch: (exp["fixed"] / d) for ch, d in b.groupby("channel")["denominator"].first().items()}
    L += ["**Fixed-expense floor.** With zero risk premium the office premium is still F / D: "
          + ", ".join(f"{_f(v)} CU ({ch})" for ch, v in fl.items())
          + f". {below} of the {len(b)} current written premiums are below this floor"
          + (f" ({by_line})" if below else "") + ", so for low-premium property policies the fixed expense, not the "
          "risk premium, drives the price and the ratio to the current premium.", ""]
    # primary vs tax-on-top
    ratio = b["office_tax_on_top"] / b["office"]
    L += ["## Why tax inside the denominator is the primary convention", "",
          f"The `expenses` table gives premium tax as a share of premium, like the other percentage items, so the "
          f"primary build makes the tax {_pct(t)} of the premium the customer pays, and commission and profit are "
          f"also measured on that gross premium. The alternative treats the tax as a levy added on top of a net "
          f"premium (P_alt = P_net x (1 + t)): the tax is then {_pct(t)} of the net premium, i.e. "
          f"{_pct(t / (1 + t))} of the gross, and commission, variable expense and profit are earned on the net "
          f"premium only. The alternative is therefore always lower; the ratio P_alt / P = (1 + t)(D) / (D + t) "
          f"ranges from {ratio.min():.4f} to {ratio.max():.4f} here (about {_pct(1 - ratio.mean())} lower). We "
          f"recommend the primary convention because it applies every \"share of premium\" item to the same base, "
          f"the premium charged, which is how the table states them; the tax-on-top figure is the right one only if "
          f"the local tax is legally levied on the premium net of tax.", ""]
    # pro-rating note
    short = b[b["exposure"] < 0.5]
    L += ["## Note: pro-rating the fixed expense", "",
          "We charge the full fixed expense per policy on each row because it is stated per policy (issuing, "
          "administration and claims set-up costs are incurred once, however short the period). The alternative "
          "pro-rates it by exposure: P = (RP + CAT + F x exposure) / D. It matters only for short rows. Caveat: a row "
          "is a policy-year segment, so a policy that starts late in 2025 continues in a 2026 row; charging F on both "
          "would double count it, and in that case pro-rating (or charging F once per policy term) is the better "
          "choice. Rows that end early because the policy was cancelled or lapsed have incurred the full set-up "
          "cost, which supports charging F in full."]
    if len(short):
        L += ["", md_table(["row_id", "line", "exposure", "P (F in full)", "P (F pro-rated)"],
                           [[r["row_id"], r["line"], f"{r['exposure']:.4f}", _f(r["office"]),
                             _f(r["office_fixed_prorated"])] for _, r in short.iterrows()],
                           ["---", "---", "---:", "---:", "---:"])]
    L.append("")
    return "\n".join(L)


# --------------------------------------------------------------------------------------------- main

def load_predictions(path, col="prediction"):
    d = pd.read_csv(path)
    if "row_id" not in d.columns or col not in d.columns:
        raise ValueError(f"{path}: need columns row_id and {col}; got {list(d.columns)}")
    if d["row_id"].duplicated().any():
        raise ValueError(f"{path}: duplicated row_id")
    return d.set_index("row_id")[col].astype(float)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--predictions", required=True, help="submission CSV: row_id, prediction (risk premium per row)")
    ap.add_argument("--risk-premium-col", default="prediction")
    ap.add_argument("--market", default="emerging", help="expenses column: emerging or developed")
    ap.add_argument("--expenses", default=str(DATA / "expenses.parquet"))
    ap.add_argument("--reference", default=str(DATA / "reference_policies.parquet"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "part2_office_premium.md"))
    ap.add_argument("--csv", default=str(ROOT / "reports" / "part2_office_premium.csv"))
    ap.add_argument("--placeholder", action="store_true",
                    help="mark output as PLACEHOLDER (also automatic if the predictions file name contains it)")
    ap.add_argument("--example-rows", nargs="*", default=None, help="row_ids for the worked examples")
    a = ap.parse_args(argv)

    exp = load_expenses(a.expenses, a.market)
    ref = pd.read_parquet(a.reference)
    pred = load_predictions(a.predictions, a.risk_premium_col)
    missing = sorted(set(ref["row_id"]) - set(pred.index))
    if missing:
        raise SystemExit(f"predictions file lacks {len(missing)} reference row_ids: {missing}")
    b = build(ref, pred.reindex(ref["row_id"]).to_numpy(), exp)
    placeholder = a.placeholder or "placeholder" in Path(a.predictions).name.lower()
    try:
        shown = Path(a.predictions).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        shown = Path(a.predictions).name + (" (scratch file, outside the repo)" if placeholder else "")
    md = render(b, exp, shown, placeholder, pick_examples(b, a.example_rows))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(md, encoding="utf-8")
    keep = ["row_id", "line", "channel", "exposure", "sum_insured", "cover_type", "written_premium", "rp", "rp_pa",
            "cat_rate", "cat", "fixed", "commission_rate", "commission_assumed_zero", "denominator", "cost_base",
            "office"] + [c for c in b.columns if c.startswith("part_")] + [
            "denominator_net", "office_net", "office_tax_on_top", "office_fixed_prorated", "office_pa",
            "ratio_to_written"]
    out = b[keep].copy()
    out.insert(0, "placeholder", placeholder)
    Path(a.csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.csv, index=False, float_format="%.6f")
    s = summary_by_line(b)
    print(("PLACEHOLDER run. " if placeholder else "") + f"wrote {a.out} and {a.csv}")
    print(s.round(2).to_string())


if __name__ == "__main__":
    main()
