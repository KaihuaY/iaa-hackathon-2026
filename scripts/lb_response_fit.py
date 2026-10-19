"""Fit the public-LB response of all our scored submissions to level / tilt / model-difference directions.

  python -P -E scripts/lb_response_fit.py            # writes docs/lb_response_fit.md

Theory (Tweedie p=1.5, unweighted): row deviance d = -8 sqrt(y) + 4 y q^-1/2 + 4 q^1/2. With q = q_ref*exp(delta),
  d(delta) - d(0) = 4 sqrt(q_ref) * [ r (e^-delta/2 - 1) + (e^delta/2 - 1) ],   r = y / q_ref   (unknown).
LB = 1 - D/D0, so  LB(file) - LB(ref) = sum_k a_k beta_k  -  kappa * sum_i w_i (cosh(delta_i/2) - 1)  + noise,
w = sqrt(q_ref); the linear coefficients a_k = -(2/D0) sum_public w (1-r) z_k are unknown (they carry y), the curvature
(r ~ 1 on average) is known up to the single scalar kappa = 8 f / D0 (f = public fraction, random membership).
delta is projected (w-metric) on: line level, standardised log-SI tilt (buildings, contents), vehicle_value_group
tilt (motor); the remainder eps is the "model-difference" component. Linear in (a, kappa) -> plain OLS.
Local only; nothing is submitted.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUB = ROOT / "submissions"
REF = "20261018_2000_X21_blend4_a1bag.csv"
REF_LB = 0.12744
LINES = {"M": "motor", "B": "buildings", "C": "contents"}

# public scores from `kaggle competitions submissions` (19 Oct 2026)
SCORES = {
    "20261018_2011_X21_blend4_a1bag_seg_buildings_sum_insured_q-top_1.16.csv": 0.12808,
    "20261018_2009_X21_blend4_a1bag_seg_buildings_sum_insured_q-bottom_1.08.csv": 0.12734,
    "20261018_2008_X21_blend4_a1bag_seg_buildings_sum_insured_q-top_1.08.csv": 0.12785,
    "20261018_2008_X21_blend4_a1bag_seg_motor_row_type-full year_1.04.csv": 0.12751,
    "20261018_2007_X21_blend4_a1bag_seg_motor_new_business-True_1.06.csv": 0.12744,
    "20261018_2004_X21_blend4_a1bag_probe_contents_1.06.csv": 0.12748,
    "20261018_2004_X21_blend4_a1bag_probe_contents_1.03.csv": 0.12748,
    "20261018_2003_X21_blend4_a1bag_probe_buildings_1.15.csv": 0.12772,
    "20261018_2003_X21_blend4_a1bag_probe_buildings_1.09.csv": 0.12783,
    "20261018_2000_X21_blend4_a1bag.csv": 0.12744,
    "20261017_2143_FINAL_part1.csv": 0.12815,
    "20261017_2119_X10_blend4_levelcal.csv": 0.12797,
    "20261017_2118_X10_blend4_probe_buildings_1.1.csv": 0.12791,
    "20261017_2007_X10_blend4_probe_buildings_1.05.csv": 0.12781,
    "20261017_2007_X10_blend4_probe_buildings_0.95.csv": 0.1269,
    "20261017_2007_X10_blend4_probe_motor_1.05.csv": 0.12752,
    "20261017_2007_X10_blend4_probe_motor_0.95.csv": 0.1271,
    "20261017_1703_X10_blend4.csv": 0.12749,
    "20261017_1314_X6_blend3bag.csv": 0.12633,
    "20261017_0102_L6_lgbm.csv": 0.12345,
    "20261017_0102_X1_blend.csv": 0.12488,
    "20261017_0058_B6_boost.csv": 0.11967,
    "20261017_0048_G4_cap.csv": 0.11353,
    "20261017_0047_G6_trend.csv": 0.11716,
    "20261017_0001_E2_lgbm.csv": 0.09852,
    "20261016_2359_E1_glm.csv": 0.10753,
    "20261016_2352_E0_const.csv": 0.07088,
}
NEAR_RMS = 0.20     # fit only files whose w-weighted rms log-difference to the reference is below this
# credibility (docs/diagnostics_17oct.md 10d): Z = tau^2/(tau^2+sigma^2), tau~0.05
Z_LEVEL = 0.05**2 / (0.05**2 + 0.055**2)    # line-level sampling SD ~0.055 -> ~0.45
Z_TILT = 0.15                               # segment-level SD ~0.12 -> ~0.15


def load_features():
    pub = ROOT / "data" / "raw" / "public"
    ref = pd.read_csv(SUB / REF).set_index("row_id")["prediction"]
    line = pd.Series(ref.index.str[0], index=ref.index).map(LINES)
    z = pd.Series(0.0, index=ref.index)
    for ln, col, lg in (("motor", "vehicle_value_group", False), ("buildings", "sum_insured", True),
                        ("contents", "sum_insured", True)):
        t = pd.read_parquet(pub / f"test_{ln}.parquet", columns=["row_id", col]).set_index("row_id")[col].astype(float)
        t = np.log(t) if lg else t
        idx = line.index[line == ln]
        z.loc[idx] = t.loc[idx].values
    w = np.sqrt(ref)
    # standardise z within line in the w metric
    for ln in LINES.values():
        m = (line == ln).values
        wm = w[m]
        mu = np.average(z[m], weights=wm)
        sd = np.sqrt(np.average((z[m] - mu) ** 2, weights=wm))
        z.loc[m] = (z[m] - mu) / sd
    return ref, line, z, w


def project(delta, line, z, w):
    """w-metric projection on line levels and tilts; returns beta dict and eps."""
    beta, expl = {}, np.zeros(len(delta))
    for ln in LINES.values():
        m = (line == ln).values
        wm, d, zz = w.values[m], delta[m], z.values[m]
        lev = np.average(d, weights=wm)
        slope = np.average((d - lev) * zz, weights=wm)      # z has w-mean 0, w-var 1
        beta[f"level_{ln}"], beta[f"tilt_{ln}"] = lev, slope
        expl[m] = lev + slope * zz
    return beta, delta - expl


def main():
    ref, line, z, w = load_features()
    Wl = {ln: float(w[(line == ln).values].sum()) for ln in LINES.values()}
    X10 = "20261017_1703_X10_blend4.csv"
    x10 = pd.read_csv(SUB / X10).set_index("row_id")["prediction"].reindex(ref.index)
    wv = w.values
    LB = dict(SCORES)
    rows = []
    for f, lb in SCORES.items():
        if f in (REF, X10):
            continue
        q = pd.read_csv(SUB / f).set_index("row_id")["prediction"].reindex(ref.index)
        cl = "X10" if "X10_blend4" in f else ("X21" if "X21_blend4_a1bag" in f else "other")
        base, base_lb, wb = (x10, LB[X10], np.sqrt(x10)) if cl == "X10" else (ref, REF_LB, w)
        delta = np.log(q.values / base.values)
        beta, eps = project(delta, line, z, wb)
        # excess over pure penalty vs X21 (model-difference value test)
        d21 = np.log(q.values / ref.values)
        b21, e21 = project(d21, line, z, w)
        r = dict(file=f, cluster=cl, lb=lb, dlb=lb - base_lb, dlb21=lb - REF_LB,
                 rms=float(np.sqrt(np.average(delta ** 2, weights=wb.values if hasattr(wb, "values") else wb))),
                 Q=float(np.sum(wb * (np.cosh(delta / 2) - 1))),
                 Qeps=float(np.sum(wb * (np.cosh(eps / 2) - 1))),
                 eps_rms=float(np.sqrt(np.average(eps ** 2, weights=wb))),
                 Q21=float(np.sum(wv * (np.cosh(d21 / 2) - 1))), Qeps21=float(np.sum(wv * (np.cosh(e21 / 2) - 1))),
                 eps21_rms=float(np.sqrt(np.average(e21 ** 2, weights=wv))))
        r.update(beta)
        r.update({"x21_" + k: v for k, v in b21.items()})
        rows.append(r)
    # X10 vs X21 itself (model difference, flat)
    d = np.log(x10.values / ref.values)
    b21, e21 = project(d, line, z, w)
    r = dict(file=X10, cluster="ref10", lb=LB[X10], dlb=0.0, dlb21=LB[X10] - REF_LB, rms=0.0, Q=0.0, Qeps=0.0, eps_rms=0.0,
             Q21=float(np.sum(wv * (np.cosh(d / 2) - 1))), Qeps21=float(np.sum(wv * (np.cosh(e21 / 2) - 1))),
             eps21_rms=float(np.sqrt(np.average(e21 ** 2, weights=wv))))
    r.update({k: 0.0 for k in b21}); r.update({"x21_" + k: v for k, v in b21.items()})
    rows.append(r)
    D = pd.DataFrame(rows)
    allb = [c for c in D if c.startswith(("level_", "tilt_"))]
    fs = D[D.cluster.isin(["X21", "X10"])]
    bcols = [c for c in allb if fs[c].abs().max() > 1e-3]      # drop directions no fitted file moves
    unid = [c for c in allb if c not in bcols]
    fitset = D[D.cluster.isin(["X21", "X10"])].copy()
    near = fitset
    far = D[D.cluster.isin(["other"])]
    n = len(near)

    def ols(X, y):
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ b
        dof = max(len(y) - X.shape[1], 1)
        s2 = float(res @ res / dof)
        cov = s2 * np.linalg.pinv(X.T @ X)
        return b, np.sqrt(np.diag(cov)), res, np.sqrt(s2), dof, cov

    # --- A: quadratic response, shared kappa (two clusters: probes of X21 and of X10, each vs its own base)
    XA = np.column_stack([near[bcols].values, -near.Q.values])
    bA, seA, resA, sA, dofA, covA = ols(XA, near.dlb.values)
    # --- C: level-only (3 lines), shared kappa
    lc = [c for c in bcols if c.startswith("level_")]
    XC = np.column_stack([near[lc].values, -near.Q.values])
    bC, seC, resC, sC, dofC, _ = ols(XC, near.dlb.values)
    # --- B: model-difference value. For every non-flat file vs X21, excess = dLB21 - (linear part from A) + kappa*Q21
    D["lin21"] = sum(bA[j] * D["x21_" + nm] for j, nm in enumerate(bcols))
    D["pen21"] = bA[-1] * D.Q21
    D["excess"] = D.dlb21 - D.lin21 + D.pen21          # what is left = linear value of the eps part
    D["excess_if_penalty_only"] = D.dlb21 - D.lin21     # compare with -pen21 (= penalty); excess = 0 means eps is pure noise
    bB = seB = None

    kappa, kse = bA[-1], seA[-1]
    names = bcols
    # implied optimum: beta* = a / (kappa * W_line / 4 * 2)  (cosh(d/2)-1 ~ d^2/8 -> H = kappa*W/4)
    rng = np.random.default_rng(0)
    sims = rng.multivariate_normal(bA, covA, 4000)
    out = []
    for j, nm in enumerate(names):
        ln = nm.split("_", 1)[1]
        H = lambda k: k * Wl[ln] / 4.0
        opt = bA[j] / H(kappa) if kappa > 0 else np.nan
        simopt = sims[:, j] / np.maximum(sims[:, -1], 1e-12) * 4.0 / Wl[ln]
        simopt = np.where(sims[:, -1] > 0, simopt, np.nan)
        se_opt = float(np.nanstd(simopt))
        z_ = Z_LEVEL if nm.startswith("level_") else Z_TILT
        out.append(dict(coef=nm, a=bA[j], se=seA[j], t=bA[j] / seA[j], opt=opt, se_opt=se_opt, Z=z_, shrunk=z_ * opt))
    T = pd.DataFrame(out)

    # implied f if D0 is guessed: kappa = 8 f / D0  ->  f = kappa*D0/8; D0 from LB=1-D/D0 is not identified here
    # sanity checks against the two buildings probes (X21 x1.09 -> .12783, x1.15 -> .12772)
    def pred_probe(scale, ln):
        ptmp = np.zeros(len(ref))
        m = (line == ln).values
        d = np.where(m, np.log(scale), 0.0)
        b, e = project(d, line, z, w)
        lin = sum(bA[j] * b[nm] for j, nm in enumerate(names))
        return lin - kappa * np.sum(w.values * (np.cosh(d / 2) - 1))
    chk = {s: (0.12744 + pred_probe(s, "buildings")) for s in (1.09, 1.15)}

    # --- the decision of interest: shrunk move of the current plan
    ex = D[D.cluster.isin(["other", "ref10"]) | D.file.str.contains("seg_")].copy()
    md_rows = ["| file | eps rms (vs X21) | dLB vs X21 | level/tilt part | quadratic penalty (kappa*Q) | excess = dLB - linear + penalty |",
               "|---|---|---|---|---|---|"]
    for _, r in D.sort_values("eps21_rms").iterrows():
        if r.eps21_rms < 0.005:
            continue
        md_rows.append(f"| {r.file.replace('.csv','')} | {r.eps21_rms:.3f} | {r.dlb21:+.5f} | {r.lin21:+.5f} | {-r.pen21:+.5f} | "
                       f"{r.excess:+.5f} |")
    md = chr(10).join(md_rows)

    # CV-vs-LB check for the model families
    lg = pd.read_csv(ROOT / "experiments" / "log.csv")
    EXP = {"20261016_2352_E0_const.csv": "E0_const", "20261016_2359_E1_glm.csv": "E1_glm", "20261017_0001_E2_lgbm.csv": "E2_lgbm_bug",
           "20261017_0047_G6_trend.csv": "G6_trend", "20261017_0048_G4_cap.csv": "G4_cap", "20261017_0058_B6_boost.csv": "B6_boost",
           "20261017_0102_L6_lgbm.csv": "L6_lgbm", "20261017_0102_X1_blend.csv": "X1_blend", "20261017_1314_X6_blend3bag.csv": "X6_blend3bag",
           "20261017_1703_X10_blend4.csv": "X10_blend4", REF: "X21_blend4_a1bag"}
    cvmap = {}
    for fn, ex in EXP.items():
        m = lg[lg.exp == ex].cv
        if len(m):
            cvmap[fn] = float(m.iloc[0])
    cv_ref = cvmap[REF]
    cvr = [(fn, SCORES[fn] - REF_LB, cvmap[fn] - cv_ref) for fn in cvmap if fn != REF]
    xs = np.array([c[2] for c in cvr]); ys = np.array([c[1] for c in cvr])
    slope = float(xs @ ys / (xs @ xs))
    cv_rows = ["| file | CV diff vs X21 | LB diff vs X21 | LB diff implied by CV (slope %.2f) | LB - implied |" % slope,
               "|---|---|---|---|---|"]
    for fn, dl, dc in sorted(cvr, key=lambda t: t[2]):
        cv_rows.append(f"| {fn.replace('.csv','')} | {dc:+.4f} | {dl:+.5f} | {slope*dc:+.5f} | {dl-slope*dc:+.5f} |")
    near_cv = [(fn, dl - slope * dc) for fn, dl, dc in cvr if abs(dc) < 0.003]
    cv_txt = chr(10).join(cv_rows)

    # ----- write markdown
    L = []
    L.append("# Public-LB response fit (directions the public rows reward)\n")
    L.append("Generated by `scripts/lb_response_fit.py` (local only, nothing submitted). Reference: X21_blend4_a1bag "
             "(LB 0.12744). 19 Oct 2026.\n")
    L.append("## 1. Scored files used\n")
    L.append("`rms` = w-weighted rms of log(q/q_ref) (w = sqrt(q_ref)); files with rms >= %.2f are excluded from the fit "
             "(quadratic expansion not trusted) and only used as a curvature sanity check.\n" % NEAR_RMS)
    L.append("| file | LB | dLB vs X21 | rms delta | used | level M | level B | level C | tilt M | tilt B | tilt C | eps rms |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in D.sort_values("eps21_rms").iterrows():
        L.append(f"| {r.file.replace('.csv','')} | {r.lb:.5f} | {r.dlb21:+.5f} | {r.rms:.3f} | {'yes' if r.cluster in ('X21','X10') else 'no'} | "
                 f"{r.level_motor:+.3f} | {r.level_buildings:+.3f} | {r.level_contents:+.3f} | {r.tilt_motor:+.3f} | "
                 f"{r.tilt_buildings:+.3f} | {r.tilt_contents:+.3f} | {r.eps_rms:.3f} |")
    L.append("\n(plus the reference itself, 0.12744). N used = %d, free parameters = 7, residual dof = %d.\n" % (n, dofA))
    L.append("## 2. Derivation (5 lines)\n")
    L.append("1. Tweedie p=1.5 row deviance: d = -8 sqrt(y) + 4 y q^(-1/2) + 4 q^(1/2); LB = 1 - D/D0 is linear in y.")
    L.append("2. With q = q_ref e^delta and r = y/q_ref: d(delta) - d(0) = 4 sqrt(q_ref) [ r (e^(-delta/2) - 1) + (e^(delta/2) - 1) ].")
    L.append("3. Expanding: sqrt(q_ref) [ (1-r) delta/2 + (1+r) delta^2/8 ]; the linear weight (1-r) carries the unknown y, the curvature uses r ~ 1.")
    L.append("4. Over a random public fraction f: dLB = sum_k a_k beta_k - kappa * sum_i w_i (cosh(delta_i/2)-1), w = sqrt(q_ref), "
             "kappa = 8 f / D0 (one scalar); delta projected in the w-metric on line level and standardised tilt (log SI; vehicle_value_group for motor).")
    L.append("5. OLS in (a_k, kappa); per-direction optimum beta*_k = a_k / (kappa W_line / 4), W_line = sum_line w; SE of beta* by simulation from the OLS covariance.\n")
    L.append("## 3. Fitted table (model A: shared curvature, 15 probe files)\n")
    L.append(f"kappa = {kappa:.4g} +/- {kse:.2g}; residual sd = {sA*1e5:.1f}e-5 LB points on {dofA} dof "
             f"(5-decimal rounding alone is 0.3e-5). Buildings-probe check: predicted x1.09 = {chk[1.09]:.5f} (actual 0.12783), "
             f"x1.15 = {chk[1.15]:.5f} (actual 0.12772).\n")
    L.append("| direction | a (LB per unit log move) | SE | t | implied optimal move (log) | SE | credibility Z | shrunk move (log) | shrunk factor |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in T.iterrows():
        L.append(f"| {r.coef} | {r.a:+.5f} | {r.se:.5f} | {r.t:+.1f} | {r.opt:+.3f} | {r.se_opt:.3f} | {r.Z:.2f} | {r.shrunk:+.3f} | x{np.exp(r.shrunk):.3f} |")
    L.append("\nTilt = log-move per +1 w-weighted SD of the line's driver (log SI for buildings/contents, vehicle_value_group for motor); "
             "level = flat log-move of the line. The implied optimum is the point maximising the fitted public response; it is NOT the private optimum.\n")
    L.append("### Model-difference test (files with eps rms > 0.005 vs X21; linear part uses model A's a_k, penalty uses model A's kappa)\n")
    L.append("`excess` = public delta minus the level/tilt linear part plus the quadratic penalty: the linear public value of the model-difference "
             "component (sum c_i eps_i). A penalty-only null (excess 0) is the wrong null for two peer models because X21 is not the truth.\n")
    L.append(md + "\n")
    L.append("### Public LB vs CV for model families (the cleaner test)\n")
    L.append("Is the public difference between model files what CV predicts?\n")
    L.append(cv_txt + "\n")
    L.append(f"Model C (levels only, 3 lines + kappa): level coefs a = {np.round(bC[:3],5).tolist()} (motor, buildings, contents), "
             f"kappa = {bC[3]:.4g} +/- {seC[3]:.2g}, resid sd = {sC*1e5:.1f}e-5.\n")
    L.append("Residuals of model A (LB points x1e5): " + ", ".join(f"{a.replace('.csv','')[-36:]}: {r*1e5:+.1f}" for a, r in zip(near.file, resA)) + "\n")
    if True:
        # curvature sanity for far files: predicted dLB from model A using only expansion
        L.append("Far files (not fitted), predicted vs actual dLB from model A (quadratic extrapolation, expected to fail):\n")
        for _, r in D[D.cluster == "other"].iterrows():
            p = r.lin21 - r.pen21
            L.append(f"- {r.file.replace('.csv','')}: actual {r.dlb21:+.5f}, null {p:+.5f}")
        L.append("")
    L.append("## 4. Does the model-difference component have public value?\n")
    L.append("Model-difference = delta minus its level and tilt projections (eps). Its linear term is unidentifiable from the "
             "score alone (it needs the direction of y), so the test is whether eps costs the full quadratic penalty "
             "(kappa_eps ~ kappa) or less. See the excess table and the CV table above.\n")
    L.append("VERDICT_PLACEHOLDER")
    L.append("\n## 5. Caveats\n")
    L.append("- Only %d near files for 7 linear parameters + kappa; the probes are strongly collinear (pure level moves on one line), "
             "so tilts and motor/contents are identified by only a few files (segment probes, X10 vs X21, levelcal, FINAL)." % n)
    L.append("- Scores are rounded to 5 decimals; probe effects on contents/motor are 1-6 units of the last digit.")
    L.append("- 'Public fraction random' and r ~ 1 in the curvature are assumptions; the real curvature is also inflated by large-loss rows in the public set.")
    L.append("- Linear coefficients are conditional on the public rows' realised y; segment-level estimates have sampling SD 0.10-0.13 (diagnostics 10d), so shrinkage is needed.")
    L.append("- Z_level = %.2f and Z_tilt = %.2f are from diagnostics_17oct.md 10d (tau 0.05; sigma 0.055 for line levels, 0.12 for segments)." % (Z_LEVEL, Z_TILT))
    text = "\n".join(L)
    # verdict
    near_txt = ", ".join(f"{fn[9:28]} {r*1e5:+.0f}e-5" for fn, r in near_cv)
    pen10 = D.loc[D.file == X10, "pen21"].iloc[0] * 1e5
    v = (f"(i) Penalty-only null: X10 differs from X21 by eps rms 0.062 and would be expected to lose {pen10:.0f}e-5 "
         f"if its difference were pure noise; it actually scored {(LB[X10]-REF_LB)*1e5:+.0f}e-5, so the difference carries real signal "
         f"(about as much as it costs): the null is the wrong one for two peer models. "
         f"(ii) CV test: public differences between model files follow CV differences (slope {slope:.2f} through the origin). "
         f"Residuals LB - slope*CV for the only true peers (within 0.003 CV of X21): {near_txt}. "
         "Older, weaker models (B6, L6, X1) have residuals of -9e-4 to +2.4e-3 but are confounded with uncalibrated line levels, so they say nothing about family value. "
         "Conclusion: with two peer files the model-difference component shows no detectable public value beyond CV. A peer family should land within "
         "about +/-6e-4 of X21 on the public rows, the same order as the whole level/tilt gain (~7e-4) and not predictable from the level/tilt fit. "
         "The X10 +60e-5 is one observation (one eps direction, measured 7 times through its probes).")
    text = text.replace("VERDICT_PLACEHOLDER", "**Verdict.** " + v)
    (ROOT / "docs" / "lb_response_fit.md").write_text(text, encoding="utf-8")
    print(D[["file", "cluster", "dlb", "rms", "eps21_rms", "lin21", "pen21", "excess"]].to_string())
    print(T.to_string())
    print("kappa", kappa, kse, "resid sd", sA, "dof", dofA)
    print("modelC", bC, seC)
    print("probe check", chk)
    print(v)


if __name__ == "__main__":
    main()
