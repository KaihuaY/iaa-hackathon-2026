"""Skip tests marked `part1_data` unless the active market spec is part1 (IAA_MARKET unset or 'part1')."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def pytest_collection_modifyitems(config, items):
    from market_spec import get_spec
    name = get_spec().name
    if name == "part1":
        return
    skip = pytest.mark.skip(reason=f"Part 1 data test; active market is {name!r}")
    for it in items:
        if "part1_data" in it.keywords:
            it.add_marker(skip)
