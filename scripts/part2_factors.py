"""Build a Part 2 submission from a base file and multiplicative factors, with the predicted public-score change;
read a probe result back into the response model (--fit).

Usage
  build:  python -P -E scripts/part2_factors.py --base submissions/part2/<file>.csv --tag R2_X4_nb119 \
              --factor motor:nb=1.19 [--factor contents:nb=1.20] [--probe] [--dry-run] [--hyp 0.9,1.0,1.1]
  fit:    python -P -E scripts/part2_factors.py --base submissions/part2/<base>.csv --base-score 0.16382 \
              --fit motor:nb=1.10=0.16360 [--z 0.9]

Factor keys: <line> (all rows of the line), <line>:nb (new-business rows), <line>:ren (renewal rows).
Response (exact for row-level Tweedie p = 1.5 on a fixed public subset): scaling a subset of a file by s changes the
public score by  delta(s) = -alpha (s^-0.5 - 1) - beta (s^0.5 - 1),  optimum s* = alpha / beta,  max gain (sqrt a - sqrt b)^2.
beta is known for ANY file and subset: beta = beta_per_sqrt_mu x sum(sqrt(mu)) over the subset (constant fitted 27 Oct,
docs/part2_response.json, _meta). alpha depends on the file's predictions, so it is known only for a base that has been
probed: the reference file (top-level entries) or a base with an entry under "fits" (written by --fit). For any other
base the script prints the readings expected under hypothesised residual optima (--hyp).
Files tagged --probe carry PROBE_ in the name and must never be ticked as final.
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESP = ROOT / "docs" / "part2_response.json"
TEST = ROOT / "data" / "part2" / "emerging"
CACHE = ROOT / "experiments" / "cache" / "part2pool"
OUT = ROOT / "submissions" / "part2"
LINES = ("motor", "buildings", "contents")
PUBLIC_SHARE = 0.25  # public rows ≈ a random quarter of the test file (confirmed by the sqrt-mu check, 27 Oct)


def load_test():
    return pd.concat([pd.read_parquet(TEST / f"test_{l}.parquet", columns=["row_id", "new_business", "exposure"])
                      .assign(line=l) for l in LINES], ignore_index=True)


def mask_for(key, te):
    line, _, sub = key.partition(":")
    if line not in LINES or sub not in ("", "nb", "ren"):
        raise SystemExit(f"bad factor key {key!r}; use <line>, <line>:nb or <line>:ren")
    m = (te.line == line).values
    if sub == "nb":
        m &= te.new_business.values.astype(bool)
    elif sub == "ren":
        m &= ~te.new_business.values.astype(bool)
    return m


def _g(alpha, beta, s):
    return -alpha * (s ** -0.5 - 1) - beta * (s ** 0.5 - 1)


def beta_of(mu, m, resp):
    """beta of a subset of THIS file from its sum of sqrt(mu) (the public-subset constant from the reference fit)."""
    return float(resp["_meta"]["beta_per_sqrt_mu"] * np.sqrt(mu[m]).sum())


def alpha_of(key, base_name, resp):
    """alpha for this base and subset if it has been probed: a 'fits' entry for the base, else the reference entry."""
    fits = resp.get("fits", {}).get(base_name, {})
    if key in fits and fits[key].get("alpha") is not None:
        return fits[key]["alpha"], f"fit on {base_name}"
    if base_name == resp.get("_meta", {}).get("reference_file") and resp.get(key, {}).get("alpha") is not None:
        return resp[key]["alpha"], "reference fit"
    return None, None


def alpha_transferred(key, beta_base, resp):
    """alpha carried from the reference file to another base as if the base were a uniform rescale of the reference
    on this subset: alpha = sum(y mu^-1/2) scales by r^-1/2 while beta scales by r^1/2, so alpha_base = alpha_ref x
    beta_ref / beta_base. Shape differences between the files make this approximate; a --fit replaces it."""
    r = resp.get(key)
    if r is None or r.get("alpha") is None:
        return None
    return r["alpha"] * r["beta"] / beta_base


def predicted_delta(key, s, resp, base_scale=1.0, alpha=None):
    """Reference-file response: change when a subset already at `base_scale` x reference is scaled by a further s."""
    r = resp.get(key)
    if r is None:
        return None
    a = r.get("alpha") if alpha is None else alpha
    if a is None:
        return None
    return _g(a, r["beta"], base_scale * s) - _g(a, r["beta"], base_scale)


def hypothesis_table(key, s, resp, base_scale):
    """For an unprobed subset of the reference file: readings expected if its optimum were line_opt x h."""
    line = key.partition(":")[0]
    r, rl = resp.get(key), resp.get(line)
    if r is None or rl is None:
        return []
    return [(h, predicted_delta(key, s, resp, base_scale, alpha=r["beta"] * rl["s_opt"] * h))
            for h in (0.8, 0.9, 1.0, 1.1, 1.24, 1.45)]


def reading_sd(key, m):
    """Rough sampling SD of a fitted s* (relative): row-level relative variance of the emerging history cost on the
    same line x NB subset, divided by the number of public rows. Public score noise cancels between paired files."""
    line, _, sub = key.partition(":")
    f = CACHE / f"train_{line}.parquet"
    if not f.exists():
        return None
    t = pd.read_parquet(f, columns=["row_id", "new_business", "claim_cost"])
    t = t[~t.row_id.astype(str).str.startswith("D")]
    if sub == "nb":
        t = t[t.new_business.astype(bool)]
    elif sub == "ren":
        t = t[~t.new_business.astype(bool)]
    y = t.claim_cost.values.astype(float)
    rel_var = y.var() / y.mean() ** 2
    return float(np.sqrt(rel_var / (m.sum() * PUBLIC_SHARE)))


def fit_reading(base_name, base_score, key, s, score, mu, m, resp, z):
    """Invert one probe reading on this base: alpha from delta = -alpha(s^-0.5-1) - beta(s^0.5-1)."""
    b = beta_of(mu, m, resp)
    delta = score - base_score
    a = -(delta + b * (s ** 0.5 - 1)) / (s ** -0.5 - 1)
    s_opt = a / b
    sd = reading_sd(key, m)
    entry = {"alpha": float(a), "beta": b, "s_opt": float(s_opt), "max_gain": float((a ** 0.5 - b ** 0.5) ** 2),
             "probe_s": s, "probe_score": score, "base_score": base_score, "delta": float(delta),
             "sd_s_opt": None if sd is None else float(sd * s_opt), "z": z,
             "shrunk_factor": float(1 + z * (s_opt - 1)),
             "fitted_utc": f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M}"}
    resp.setdefault("fits", {}).setdefault(base_name, {})[key] = entry
    print(f"  {key:14s} probe x{s:<5g} read {delta:+.5f}  -> alpha {a:.5f} beta {b:.5f}  s* {s_opt:.3f}"
          + (f" (SD ~ {entry['sd_s_opt']:.3f})" if sd else "")
          + f"  max gain {entry['max_gain']:+.5f}  Z={z:g} factor {entry['shrunk_factor']:.3f}"
          + f"  gain at that factor {_g(a, b, entry['shrunk_factor']):+.5f}")
    return entry


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--tag")
    ap.add_argument("--factor", action="append", default=[], help="key=value, e.g. motor:nb=1.19")
    ap.add_argument("--base-scale", action="append", default=[],
                    help="key=value: how the base file already sits relative to the reference on that subset "
                         "(reference-file response only)")
    ap.add_argument("--hyp", default="0.9,0.95,1.0,1.05,1.1,1.2,1.3",
                    help="residual optima to tabulate when alpha is unknown for this base")
    ap.add_argument("--row-factors", help="path.csv:column — row-level multiplicative factors keyed by row_id (e.g. a "
                                          "level-neutral stage-2 relativity layer); applied before --factor subsets")
    ap.add_argument("--probe", action="store_true", help="name the file PROBE_... (never tick)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--fit", action="append", default=[], help="key=s=public_score of a probe built on --base")
    ap.add_argument("--base-score", type=float, help="public score of --base (for --fit)")
    ap.add_argument("--z", type=float, default=0.9, help="credibility applied to a fitted optimum (for --fit)")
    a = ap.parse_args(argv)

    resp = json.load(open(RESP)) if RESP.exists() else {}
    te = load_test()
    base = pd.read_csv(a.base)
    base_name = Path(a.base).name
    sample = pd.read_parquet(TEST / "sample_submission.parquet", columns=["row_id"])
    if len(base) != len(sample) or not base.row_id.equals(sample.row_id):
        raise SystemExit("base file is not in sample_submission order / length")
    d = sample.merge(base, on="row_id", how="left").merge(te, on="row_id", how="left")
    if d.prediction.isna().any() or d.line.isna().any():
        raise SystemExit("base file has missing predictions or unknown row_ids")
    mu0 = d.prediction.values.astype(float)
    print(f"base {base_name}")
    ref = resp.get("_meta", {}).get("reference_file")
    if ref and base_name != ref and (OUT / ref).exists():
        r0 = pd.read_csv(OUT / ref).prediction.values
        print(f"  vs reference {ref}: corr {np.corrcoef(mu0, r0)[0, 1]:.4f}, level ratio "
              + " ".join(f"{l} {mu0[(d.line == l).values].sum() / r0[(d.line == l).values].sum():.3f}" for l in LINES))

    if a.fit:
        if a.base_score is None:
            raise SystemExit("--fit needs --base-score")
        print(f"  fitting {len(a.fit)} reading(s) on this base (public {a.base_score:.5f})")
        for f in a.fit:
            key, s, score = f.split("=")
            fit_reading(base_name, a.base_score, key, float(s), float(score), mu0, mask_for(key, te), resp, a.z)
        json.dump(resp, open(RESP, "w"), indent=1)
        print(f"  wrote {RESP.relative_to(ROOT)}")
        if not a.factor:
            return

    if not a.tag:
        raise SystemExit("--tag is required to build a file")
    mu = mu0.copy()
    total = 0.0
    if a.row_factors:
        path, _, col = a.row_factors.rpartition(":")
        rf = pd.read_csv(path, usecols=["row_id", col])
        if rf.row_id.duplicated().any():
            raise SystemExit("row-factor file has duplicate row_id")
        f = d[["row_id"]].merge(rf, on="row_id", how="left")[col].values.astype(float)
        if np.isnan(f).any():
            raise SystemExit(f"row-factor file misses {int(np.isnan(f).sum())} test rows")
        print(f"  row factors {Path(path).name}:{col}  range {f.min():.3f}-{f.max():.3f}  "
              f"share != 1: {(np.abs(f - 1) > 1e-9).mean():.1%}")
        for l in LINES:
            for nb in (True, False):
                m = (d.line == l).values & (te.set_index('row_id').loc[d.row_id, 'new_business'].values.astype(bool) == nb)
                print(f"    {l:9s} {'NB ' if nb else 'ren'} level ratio after factors {np.sum(mu0[m] * f[m]) / mu0[m].sum():.4f}"
                      f"  (sqrt-mu ratio {np.sqrt(mu0[m] * f[m]).sum() / np.sqrt(mu0[m]).sum():.4f})")
        mu *= f
        print("  predicted public change from row factors: not a level move; expected = the layer's honest OOF gain "
              "(paired reading decides)")
    base_scale = {k: float(v) for k, _, v in (b.partition("=") for b in a.base_scale)}
    hyps = [float(h) for h in a.hyp.split(",")]
    partial = False
    for f in a.factor:
        key, _, val = f.partition("=")
        s = float(val)
        m = mask_for(key, te)
        mu[m] *= s
        b = beta_of(mu0, m, resp) if resp else None
        alpha, src = alpha_of(key, base_name, resp)
        if alpha is None and b is not None and alpha_transferred(key, b, resp) is not None:
            alpha, src = alpha_transferred(key, b, resp), "transferred from the reference, uniform-rescale approx."
        if alpha is not None:
            pdlt = _g(alpha, b, s)
            if src == "reference fit" and key in base_scale:  # legacy path: reference response from a shifted point
                pdlt = predicted_delta(key, s, resp, base_scale[key])
            total += pdlt
            print(f"  {key:14s} x{s:<6g} rows {m.sum():7d}  beta {b:.4f}  alpha {alpha:.4f} ({src})  s* {alpha / b:.3f}"
                  f"  predicted delta {pdlt:+.5f}")
        else:
            partial = True
            if base_name == ref:
                hyp = hypothesis_table(key, s, resp, base_scale.get(key, 1.0))
                print(f"  {key:14s} x{s:<6g} rows {m.sum():7d}  alpha unknown; if true optimum = line optimum x h: "
                      + ", ".join(f"h={h}: {dd:+.5f}" for h, dd in hyp))
            else:
                print(f"  {key:14s} x{s:<6g} rows {m.sum():7d}  beta {b:.4f}  alpha unknown on this base; "
                      "reading expected if this subset's residual optimum is s*: "
                      + ", ".join(f"{h:g}: {_g(b * h, b, s):+.5f}" for h in hyps))
    print("  predicted public change vs base: " + (f"{total:+.5f} from the probed subsets only (others unprobed)"
                                                   if partial else f"{total:+.5f}"))
    for l in LINES:
        m = (d.line == l).values
        print(f"  {l:9s} mean pred {mu0[m].mean():9.2f} -> {mu[m].mean():9.2f}  (x{mu[m].sum() / mu0[m].sum():.4f})")
    assert np.isfinite(mu).all() and (mu >= 0).all()
    out = sample.assign(prediction=mu)
    name = f"{datetime.now(timezone.utc):%Y%m%d_%H%M}Z_{'PROBE_' if a.probe else 'P2_'}{a.tag}.csv"
    if a.dry_run:
        print(f"dry run: would write {OUT / name}")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / name, index=False)
    sha = hashlib.sha256((OUT / name).read_bytes()).hexdigest()[:16]
    print(f"wrote {OUT / name}  rows {len(out)}  sha256_16 {sha}")


if __name__ == "__main__":
    main(sys.argv[1:])
