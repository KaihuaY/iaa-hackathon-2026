"""PART2POOL: developed (2021-24) + emerging (2023-24) history pooled in one market; test = emerging 2025 only.

Data: data/part2/pooled, written by scripts/make_pooled_market.py (developed ids prefixed "D", ages banded, construction
labels canonical, developed written_premium rescaled per line to the emerging price level, `market` column on every
policy / test row, which model_matrix turns into a categorical feature because it is a string column).

Imported by market_spec.py (guarded hook at its end) or explicitly; registering is idempotent.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

from market_spec import PART1, MarketSpec, register

_GONE = {"driver_age", "policyholder_age", "annual_mileage", "telematics_opt_in", "telematics_score",
         "date_of_birth", "credit_band", "customer_age"}


@dataclass(frozen=True)
class PooledSpec(MarketSpec):
    """MarketSpec plus per-market claims-development overrides, read by target._market_specs:
    {market: {"curve_years": ..., "mature_years": ..., "ref_sev_year": ..., "completeness_last_year": ...}}.
    completeness_last_year here is documentation / for market-aware consumers (credibility.py is not market-aware and
    is not used by the pooled configs)."""
    market_dev: dict = field(default_factory=dict)


_P1_POOLED = PooledSpec(**{f.name: getattr(PART1, f.name) for f in dataclasses.fields(MarketSpec)})   # not registered
PART2POOL = register(dataclasses.replace(
    _P1_POOLED, name="part2pool", competition="iaa-ai-assisted-hackathon-2026-part-2", data_dir="data/part2/pooled",
    train_years=(2021, 2022, 2023, 2024), test_year=2025, holdout_year=2024, backtest_years=(2023, 2024),
    mature_years=(2021, 2022, 2023), curve_years=(2021, 2022), ref_sev_year=2023, immature_years=(2024,),
    # Part 1 values (developed lag curve). Residual risk: emerging 2024 completeness may differ (lag p50 8-9 days in both
    # markets, but emerging open share 2024 is 20 % motor / 25 % buildings).
    completeness_last_year={"motor": 0.920, "buildings": 0.939, "contents": 0.960},
    valuation_date="2024-12-31",
    customer_id_width=None,              # ids are already padded by make_pooled_market ("D" + 8 digits = 9 chars)
    expected_rows=None,                  # len(sample_submission) = 209,063
    suspect_claims=("DCL30973963",),     # the Part 1 11.9M motor BI case estimate, now prefixed; none in emerging
    construction_map={**PART1.construction_map, "brk": "brick", "stn": "stone", "wod": "timber", "mix": "non_standard"},
    normalise_labels=("occupation",),    # as PART2: emerging 'Retired' / 'retired' / 'Retired ' variants
    # no driver_age: motor large-loss segments by cover x years licensed (bands ~ Part 1 ages 25/35/60 minus 17)
    segments={**PART1.segments,
              "motor": {"kind": "cross", "cat": "cover_type", "num": "years_licensed", "edges": (8, 18, 43),
                        "labels": ("<8", "8-17", "18-42", "43+")}},
    # Review 11 (stability on the pooled frame, scratch stab_pool.py): previous_insurer level-frequency correlation
    # emerging 2023~2024 -0.05/-0.06/-0.08, developed 2021-22~2023-24 -0.10/-0.14/-0.13, developed~emerging
    # -0.09/-0.05/+0.02 (motor/buildings/contents): no persistent signal; email_domain / campaign_code / branch_id also
    # unstable in the developed half. Same list as Part 1. (Review 10 configs used () and are kept reproducible.)
    noise_features=("previous_insurer", "branch_id", "email_domain", "campaign_code"),
    irrelevant={k: [c for c in v if c not in _GONE] for k, v in PART1.irrelevant.items()},
    monotone={k: v for k, v in PART1.monotone.items() if k not in _GONE},
    tuned={**PART1.tuned, "min_data_in_leaf": 2000},
    # Review 11: claims development per market (target._market_specs). Developed rows keep the fields above (2021-22
    # lag curve); emerging rows use the emerging-only PART2 choices: 2023 the only curve / mature / severity year.
    # completeness_last_year = PART2's (mean 2024-row completeness from the 2023 lag curve; tests/test_target_market.py).
    market_dev={"emerging": {"curve_years": (2023,), "mature_years": (2023,), "ref_sev_year": 2023,
                             "completeness_last_year": {"motor": 0.923, "buildings": 0.944, "contents": 0.961}}},
))
