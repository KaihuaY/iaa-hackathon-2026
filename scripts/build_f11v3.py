"""Build the ticked Part 2 final, F11 v3, from the core_a3 file (30 Oct split-half readings).

F11 v3 = core_a3 with the motor new-business level 1.05 (core_a3 carried 1.30) and renewals 1.03
(was 1.09), no 17-24 or 65+ tilt, buildings sum-insured top-quartile x1.03 and flood zone >= 3 x1.05
with the buildings cells renormalised to the core level; the stage-2 gender/fuel layer is kept.

Inputs (all produced by the chain in README "Final files"):
  submissions/part2/20261028_0403Z_P2_F6_stage2_core_a3.csv   core_a3 (file 6 x stage-2 relativities)
  experiments/cache/part2pool/test_{motor,buildings,contents}.parquet   pooled-market test rows
  docs/part2_response.json   fitted public-score response (beta per sqrt(mu), per-segment alphas)
Output: submissions/part2/20261030_0145Z_P2_F11v3_robust_nb105_ren103_si103_fl105.csv
        (SHA-256 prefix 378f7b835d3ace59 when built from the same core_a3)

  python -P -E scripts/build_f11v3.py [--core <csv>] [--out <csv>]

The predicted public / ex-excess score changes printed at the end are the decision evidence recorded in
docs/part2_plan.md (30 Oct 01:40); they do not affect the file.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "submissions/part2/20261028_0403Z_P2_F6_stage2_core_a3.csv"
OUT = ROOT / "submissions/part2/20261030_0145Z_P2_F11v3_robust_nb105_ren103_si103_fl105.csv"
CACHE = ROOT / "experiments/cache/part2pool"
NB, REN, YOUNG, OLD, SI, FL = 1.05, 1.03, 1.00, 1.00, 1.03, 1.05  # F11 v3 factors
CORE_NB, CORE_REN = 1.30, 1.09  # levels already inside core_a3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", default=str(CORE))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    resp = json.load(open(ROOT / "docs/part2_response.json"))
    bps = float(resp["_meta"]["beta_per_sqrt_mu"])
    segs = resp["fits"]["20261028_0403Z_P2_F6_stage2_core_a3.csv"]["segments"]
    C = pd.read_csv(a.core).set_index("row_id").prediction
    te = {l: pd.read_parquet(CACHE / f"test_{l}.parquet") for l in ("motor", "buildings", "contents")}
    for l, d in te.items():
        d["line"] = l
        d["nb"] = d.new_business.astype(bool)
    T = pd.concat(te.values(), ignore_index=True).set_index("row_id").reindex(C.index)
    m, b = T.line.eq("motor").to_numpy(), T.line.eq("buildings").to_numpy()
    nb = T.nb.fillna(False).to_numpy()
    young = m & T.driver_age_band.eq("17-24").to_numpy()
    old65 = m & T.driver_age_band.eq("65+").to_numpy()
    last = C.index.to_series().str.extract(r"(\d+)$")[0].astype(int)
    odd = (last % 2 == 1).to_numpy()
    even = ~odd
    siq4 = b & (T.sum_insured >= te["buildings"].sum_insured.quantile(.75)).fillna(False).to_numpy()
    fl3 = b & (pd.to_numeric(T.flood_zone, errors="coerce") >= 3).fillna(False).to_numpy()

    def dl(alpha, beta, s):  # exact public-score change for factor s on a segment
        return -alpha * (s ** -0.5 - 1) - beta * (s ** 0.5 - 1)

    def inv(d, beta, s):
        return (d + beta * (np.sqrt(s) - 1)) / (1 - s ** -0.5)

    def B(mask, mu):
        return bps * np.sqrt(mu[mask]).sum()

    # reading 3 (30 Oct): motor NB excluding 17-24, even half x1.10 scored 0.16370 vs core 0.16452
    rest = m & nb & ~young
    a_even = inv(0.16370 - 0.16452, B(rest & even, C.values), 1.10)
    a_rest = 0.1369
    a_odd = a_rest - a_even
    print(f"R3: NB non-young even s* {a_even / B(rest & even, C.values):.3f}; "
          f"odd (subtraction) {a_odd / B(rest & odd, C.values):.3f}; whole {a_rest / B(rest, C.values):.3f}")

    # the file
    f = np.ones(len(C))
    f[m & nb] *= NB / CORE_NB
    f[m & ~nb] *= REN / CORE_REN
    f[young] *= YOUNG
    f[old65] *= OLD
    f[siq4] *= SI
    f[fl3] *= FL
    new = C.values * f
    cell = T.line.astype(str) + np.where(nb, "_nb", "_ren")
    for c in ("buildings_nb", "buildings_ren"):
        i = (cell == c).to_numpy()
        r = C.values[i].sum() / new[i].sum()
        new[i] *= r
        print(f"  {c} renorm x{r:.4f}")
    for l in ("motor", "buildings", "contents"):
        i = T.line.eq(l).to_numpy()
        print(f"  {l}: level vs core_a3 x{new[i].sum() / C.values[i].sum():.4f}")

    # predicted score change (public, with the extreme public claims; ex-excess = private-relevant estimate)
    EXC = 0.0445
    a_young = segs["motor_age1724_x1.10"]["alpha"]
    b_young_ren = B(young & ~nb, C.values)
    a_young_ren = 0.90 * b_young_ren
    a_young_nb = a_young - a_young_ren
    a_cell_ren = B(m & ~nb, C.values) * 1.0

    def predict(excess_out):
        tot = 0.0
        for c in pd.unique(cell):
            i = (cell == c).to_numpy()
            line = c.split("_")[0]
            isnb = c.endswith("_nb")
            parts = []
            if line == "motor":
                if isnb:
                    parts = [(young & i, a_young_nb - (EXC if excess_out else 0)), (rest, a_rest)]
                else:
                    parts = [(young & i, a_young_ren), (i & ~young, a_cell_ren - a_young_ren)]
            else:
                b_cell = B(i, C.values)
                sopt = {("buildings", True): 1.079, ("contents", True): 1.017}.get((line, isnb), 1.0)
                a_cell = b_cell * sopt
                used = i.copy()
                a_used = 0.0
                for mk, key in ((siq4, "bld_SIq4_x1.10"), (fl3 & ~siq4, "bld_flood3p_x1.12")):
                    mi = mk & i
                    if mi.any():
                        bs = B(mi, C.values)
                        a_s = bs * (1.04 if excess_out else segs[key]["s_opt"])  # even-half ex-excess ~1.04
                        parts.append((mi, a_s))
                        a_used += a_s
                        used &= ~mi
                if used.any():
                    parts.append((used, a_cell - a_used))
            d = 0.0
            for mi, a_s in parts:
                if not mi.any():
                    continue
                d += dl(a_s, B(mi, C.values), float(np.median(new[mi] / C.values[mi])))
            print(f"    cell {c:14s} {'ex-excess' if excess_out else 'public   '} {d:+.5f}")
            tot += d
        return tot

    pub = predict(False)
    priv = predict(True)
    print(f"F11v3 predicted PUBLIC change vs core_a3 {pub:+.5f} -> {0.16452 + pub:.5f}")
    print(f"F11v3 expected ex-excess change vs core_a3 {priv:+.5f} (the private-relevant estimate)")
    print("corr with core_a3", round(np.corrcoef(new, C.values)[0, 1], 5))

    out = pd.DataFrame({"row_id": C.index, "prediction": new})
    assert len(out) == 209063 and np.isfinite(new).all() and (new >= 0).all()
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(p, index=False)
    print("wrote", p.name, hashlib.sha256(p.read_bytes()).hexdigest()[:16])


if __name__ == "__main__":
    main()
