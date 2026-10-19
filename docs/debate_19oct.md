# Three-agent strategy debate, 19 Oct 2026 11:30–12:30 UTC (Fable synthesis)

Brief: the evidence file given to all three (data, target, models, CV vs public board, 19 Oct probes, constraints).
Standpoints: a P&C pricing actuary (generator mechanisms), a Kaggle tactician (probe design, public/private), a
validation skeptic (target definition, noise). Each ran light read-only checks on the cached data.

## Facts established during the debate (verified by Fable)
| Check | Result |
|---|---|
| Sum-insured indexation 2024→2025, same policies | +3.0 % buildings, +2.0 % contents; identical across SI quartiles and identical to every earlier year step → no 2025 exposure-base break |
| Written premium (2025 tariff) vs our prediction by SI quartile | same shape as 2024 (buildings 1.02/0.95/1.04/0.99); only the level moved (+5 % bld, +4 % cts, +10 % motor) → the tariff does not encode a high-SI shift |
| Sampling SD of a segment's Tweedie-optimal factor on public-sized (20 %) samples, X21 OOF | SI quartile 0.10–0.13; top decile 0.17–0.29; whole line 0.05–0.06; full-year segment optima swing 15–20 % between adjacent years |
| Credibility Z = year-shock var / (year-shock + public-noise var), 25 % public | buildings level 0.03, buildings Q4 0.03, Q1 0.53; motor level 0.26; contents level 0.29, Q4 0.12 (in-sample years bias these low; a forward view gives ≈ 0.2–0.4) |
| Paired X21−X10 gain on public-sized subsets (skeptic) | mean +0.00066, SD 0.00065, P(≤ 0) = 0.17 → our observed −0.00005 is an ordinary draw |
| Target definition vs organisers' wording (skeptic, tactician) | matches (net of excess and recoveries, repudiated excluded, in-period loss date); open claims at case estimate; open share flat across SI → cannot create a tilt |
| Severity elasticity to SI by peril (actuary) | 0.55–0.72 for every major peril (accidental damage 0.16); no SI-by-year trend in training; peril mix flat across SI; storm share rises mildly with SI |

## Where the three agree
1. Local CV gains did not fail to transfer — the public board cannot resolve +0.0007 (one draw in six shows ≤ 0).
2. The public high-SI signal (Q3 optimum ≈ ×1.22) is ≈ 1.3–1.7 SD of sampling noise; no mechanism in the data
   supports a 2025 tilt; the transferable part is the model's under-elastic SI fit (≈ +4 % at the top), which the
   SI^0.7 offset (X30) already halves.
3. A 0.1300 team with ~60 submissions is most likely ≈ +0.0015 of public-fitted noise ahead (tactician: each fitted
   factor buys ≈ +0.0001 public from noise alone); the private 75–80 % will not reward that. The ticked file should
   be credibility-shrunk; the probe-fitted file is for information.
4. Only large segments are readable: cells under ~10 % of a line have σ ≈ 0.15–0.25.

## Where they differ, and Fable's ruling
- Probe choice: actuary → split-half Q3 test; tactician → whole-line tilts, high-wind excluding Q3; skeptic → high-wind
  regions ×1.08. Ruling: keep the user's two large probes (contents Q4 ×1.10; motor value groups 15–20 ×1.06) and
  replace the two small cells with buildings high-wind-jump regions ×1.08 (39.6 % of rows) and buildings Q3
  split-half ×1.20 (12.7 %). User approved 12:30 UTC.
- Final file: all three say shrink; they differ on Z (0.15–0.4). Ruling: build the shrunk file now (X30 base,
  bands [1.01, 1.03, 1.05, 1.07], motor ×1.012, contents ×1.0) and decide on 22 Oct after the large-segment probes
  (user's choice); raise Z only if an independent segment (contents tilt, wind regions) reproduces the pattern.
- Written premium as a blend part (X27): the tariff carries no 2025 structure; expected +0.0003 CV; low priority,
  overnight filler at most.
- Hygiene for Part 2 (skeptic): nominal SI plus a premium-index trend double-counts SI growth (≈ 5–7 % level);
  a deflated "real SI" feature fixes it; CV will not see it, so it is a playbook item, not a Part 1 change.

## Submission plan (user-approved budget: 4 probes/day on 20–22 Oct + the 2 candidates)
20 Oct 00:03 UTC: X21_sifix, X30_sifix, then the four probes above (cron e1f75ba7). Read each probe with the exact
Tweedie response (ΔLB = −α(s^−0.5 − 1) − b(s^0.5 − 1), b ∝ Σμ^0.5 on the segment's test rows; s* = α/b).
21 Oct: second-scale follow-ups on whatever large segment responded; a region-half or driver-age tilt for motor.
22 Oct: submit the shrunk file (so it is selectable) and the probe-fitted file; the user ticks one.
