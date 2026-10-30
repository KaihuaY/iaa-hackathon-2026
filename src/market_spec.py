"""MarketSpec: every market-specific assumption of the pipeline in one place.

The defaults of `PART1` reproduce the Part 1 competition exactly (guarded by tests/test_golden_part1.py).
A new market is a copy with overrides, registered and selected without code changes:

    PART2 = dataclasses.replace(PART1, name="part2", data_dir="data/raw/part2", ...)
    register(PART2)            # then  IAA_MARKET=part2 python -P -E scripts/...   or   use_spec("part2")

Modules read the active spec at call time (`get_spec()`), never at import, so `use_spec()` takes effect
immediately without re-importing anything.
"""
from __future__ import annotations

import contextlib
import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class MarketSpec:
    name: str = "part1"
    competition: str = "iaa-ai-assisted-hackathon-2026-part-1"
    # ---- paths (relative to the repo root unless absolute)
    data_dir: str = "data/raw/public"
    cache_dir: str | None = None      # None -> "experiments/cache" for part1, "experiments/cache/<name>" otherwise
    results_dir: str | None = None    # None -> "experiments/results" for part1, "experiments/results/<name>" otherwise
    # ---- structure
    lines: tuple = ("motor", "buildings", "contents")
    train_years: tuple = (2021, 2022, 2023, 2024)
    test_year: int = 2025
    curve_years: tuple = (2021, 2022)            # loss years used for the reporting-lag curve
    mature_years: tuple = (2021, 2022, 2023)     # loss years treated as (mostly) developed
    ref_sev_year: int = 2023                     # "recent, mostly mature" severity year for the BF IBNR
    immature_years: tuple = (2024,)              # years whose claim counts are scaled by completeness
    completeness_last_year: dict = field(default_factory=lambda: {"motor": 0.920, "buildings": 0.939, "contents": 0.960})
    holdout_year: int = 2023                     # time holdout: train years < holdout_year, score holdout_year
    backtest_years: tuple = (2023, 2024)
    max_lag: int = 730
    valuation_date: str = "2024-12-31"
    # ---- column aliases: {name in this market's files: canonical Part 1 name}, applied by data.load() right after
    # reading a table. The CANONICAL SCHEMA is Part 1's: all downstream code (cleaning, features, target, credibility,
    # configs) keeps using the Part 1 column names. Columns a market drops are tolerated (skipped, logged once).
    column_aliases: dict = field(default_factory=dict)
    # ---- cleaning
    customer_id_width: int | None = 8            # zero-pad customer_id to this width; None = skip
    exposure_days_check: bool = True             # repair exposure recorded in days rather than years
    exposure_days_tol: float = 0.02
    min_licence_age: int = 17
    si_unit_error: dict = field(default_factory=lambda: {"buildings": 5e6, "contents": 1e6})
    construction_map: dict = field(default_factory=lambda: {"brick": "brick", "stone": "stone", "non_standard": "non_standard",
                                                            "t": "timber", "w": "timber", "timber": "timber"})
    leaky_cols: tuple = ("renewal_decision", "customer_service_contacts")
    excluded_claim_statuses: tuple = ("repudiated",)
    # text columns whose labels differ only by case / surrounding whitespace: each variant is mapped to the most
    # frequent spelling of its stripped, lower-cased key (data.build_all). () = no change (Part 1).
    normalise_labels: tuple = ()
    # ---- target
    suspect_claims: tuple = ("CL30973963",)
    peril_min_claims: int = 1000
    segment_credibility_k: float = 200
    # per-line segment definitions for the segment large-loss loading
    segments: dict = field(default_factory=lambda: {
        "motor": {"kind": "cross", "cat": "cover_type", "num": "driver_age", "edges": (25, 35, 60),
                  "labels": ("<25", "25-34", "35-59", "60+")},
        "buildings": {"kind": "quantile", "col": "sum_insured", "q": (0.25, 0.5, 0.75), "prefix": "SIQ"},
        "contents": {"kind": "quantile", "col": "sum_insured", "q": (0.25, 0.5, 0.75), "prefix": "SIQ"},
    })
    # ---- modelling protocol
    n_folds: int = 5
    group_col: str = "policy_id"
    seed: int = 11
    quick_test_size: float = 0.2
    # ---- submission
    expected_rows: int | None = 626_639          # None -> len(sample_submission)
    # ---- experiment-config constants (experiments/configs.py)
    noise_features: tuple = ("previous_insurer", "branch_id", "email_domain", "campaign_code")
    irrelevant: dict = field(default_factory=lambda: {
        "motor": ["roof_type", "soil_type", "energy_rating", "basement", "trees_within_5m", "listed_building", "solar_panels",
                  "bedrooms", "vehicle_colour", "flood_zone", "distance_to_river_km", "elevation_m", "property_age"],
        "contents": ["roof_type", "soil_type"],
        "buildings": ["document_preference"],
    })
    monotone: dict = field(default_factory=lambda: {
        "voluntary_excess": -1, "ncb_years": -1, "prior_claims_5y": 1, "licence_points": 1, "pol_claims": 1,
        "cust_claims": 1, "missed_payments_12m": 1, "vehicle_value_group": 1, "annual_mileage": 1,
        "sum_insured": 1, "flood_zone": 1, "high_value_items": 1})
    tuned: dict = field(default_factory=lambda: {"min_data_in_leaf": 8000, "feature_fraction": 0.5})

    # ---- derived
    @property
    def all_years(self) -> list:
        return list(self.train_years) + [self.test_year]

    @property
    def first_year(self) -> int:
        return min(self.train_years)

    @property
    def last_mature_year(self) -> int:
        return max(self.mature_years)

    @property
    def day0(self) -> pd.Timestamp:
        return pd.Timestamp(f"{self.first_year}-01-01")

    @property
    def valuation_ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.valuation_date)

    def _path(self, p: str) -> Path:
        p = Path(p)
        return p if p.is_absolute() else ROOT / p

    @property
    def data_path(self) -> Path:
        return self._path(self.data_dir)

    @property
    def cache_path(self) -> Path:
        if self.cache_dir:
            return self._path(self.cache_dir)
        return ROOT / "experiments" / "cache" / ("" if self.name == "part1" else self.name)

    @property
    def results_path(self) -> Path:
        if self.results_dir:
            return self._path(self.results_dir)
        return ROOT / "experiments" / "results" / ("" if self.name == "part1" else self.name)


PART1 = MarketSpec()
REGISTRY: dict[str, MarketSpec] = {"part1": PART1}


def register(spec: MarketSpec) -> MarketSpec:
    REGISTRY[spec.name] = spec
    return spec


def _resolve(spec_or_name) -> MarketSpec:
    if isinstance(spec_or_name, MarketSpec):
        return spec_or_name
    try:
        return REGISTRY[spec_or_name]
    except KeyError:
        raise KeyError(f"unknown market {spec_or_name!r}; registered: {sorted(REGISTRY)}") from None


_ACTIVE: MarketSpec | None = None   # resolved lazily on first get_spec(), so specs registered later in this file
                                    # (or by the caller) can be selected through the IAA_MARKET env var


def get_spec() -> MarketSpec:
    global _ACTIVE
    if _ACTIVE is None:
        _ACTIVE = _resolve(os.environ.get("IAA_MARKET", "part1"))
    return _ACTIVE


def __getattr__(name):
    if name == "SPEC":                  # back-compat: market_spec.SPEC
        return get_spec()
    raise AttributeError(f"module 'market_spec' has no attribute {name!r}")


def use_spec(spec_or_name) -> MarketSpec:
    """Make a spec (instance or registered name) the active one. Takes effect immediately in every module."""
    global _ACTIVE
    _ACTIVE = _resolve(spec_or_name)
    return _ACTIVE


@contextlib.contextmanager
def using_spec(spec_or_name):
    """Temporarily activate a spec."""
    prev = get_spec()
    try:
        yield use_spec(spec_or_name)
    finally:
        use_spec(prev)


replace = dataclasses.replace

_WARNED: set = set()


def warn_once(msg: str) -> None:
    """Print a message the first time it is seen (used for columns a market does not provide)."""
    if msg not in _WARNED:
        _WARNED.add(msg)
        print(f"[market_spec] {msg}")


# ---- dress-rehearsal market (synthetic, see scripts/make_rehearsal_market.py)
REHEARSAL = register(dataclasses.replace(
    PART1, name="rehearsal", competition="none", data_dir="data/rehearsal",
    lines=("motor", "buildings"), train_years=(2019, 2020, 2021, 2022), test_year=2023,
    curve_years=(2019, 2020), mature_years=(2019, 2020, 2021, 2022), ref_sev_year=2022,
    immature_years=(), completeness_last_year={"motor": 1.0, "buildings": 1.0},
    holdout_year=2022, backtest_years=(2021, 2022), valuation_date="2022-12-31",
    customer_id_width=None, si_unit_error={"buildings": 5e6 * 50}, expected_rows=None,
    noise_features=tuple(f for f in PART1.noise_features if f != "previous_insurer"),
    column_aliases={"age_driver": "driver_age"},
    segments={**{k: v for k, v in PART1.segments.items() if k != "contents"},
              "motor": {**PART1.segments["motor"], "num": "driver_age"}},
    irrelevant={k: v for k, v in PART1.irrelevant.items() if k != "contents"},
))


# ---- Part 2: the emerging market (data/part2/emerging; evidence and reasons per field in docs/part2_decisions.md)
PART2 = register(dataclasses.replace(
    PART1, name="part2", competition="iaa-ai-assisted-hackathon-2026-part-2", data_dir="data/part2/emerging",
    train_years=(2023, 2024), test_year=2025,
    curve_years=(2023,), mature_years=(2023,), ref_sev_year=2023,     # 2023 open share 4 %, 2024 20-25 %
    immature_years=(2024,),
    # unweighted mean over 2024 rows of the row completeness from the 2023 lag curve (the Part 1 derivation, which
    # reproduces Part 1's 0.920 / 0.939 / 0.960 exactly); cross-check: 2024/2023 reported frequency 0.921 / 0.973 / 0.974
    completeness_last_year={"motor": 0.923, "buildings": 0.944, "contents": 0.961},
    holdout_year=2024, backtest_years=(2024,), max_lag=730, valuation_date="2024-12-31",   # last report 2024-12-30
    expected_rows=None,                                              # 209,063 rows from sample_submission
    construction_map={**PART1.construction_map, "brk": "brick", "stn": "stone", "wod": "timber", "mix": "non_standard"},
    normalise_labels=("occupation",),                                # 'Retired' / 'retired' / 'Retired ' etc.
    suspect_claims=(),
    peril_min_claims=60,                                             # 9.1k claims: pools buildings subsidence (52) only
    segments={**PART1.segments,                                      # no numeric driver age: value-group quartiles
              "motor": {"kind": "quantile", "col": "vehicle_value_group", "q": (0.25, 0.5, 0.75), "prefix": "VVG"}},
    noise_features=(),                                               # re-derive with the stability screen
    monotone={k: v for k, v in PART1.monotone.items() if k != "annual_mileage"},
    tuned={"min_data_in_leaf": 300, "feature_fraction": 0.5},        # ~47k / 27k / 35k rows per line
))

# Other modules may register more specs (e.g. a pooled developed + emerging market); optional.
import importlib  # noqa: E402

for _m in ("market_spec_pool",):
    try:
        importlib.import_module(_m)
    except ImportError:
        pass
