"""Headless integration: gym registration through dishware load/evaluate."""

from __future__ import annotations

import pytest

from table_bussing import DishwareCounts, TableBussingConfig, make_env

pytestmark = pytest.mark.integration


def test_make_env_reset_loads_requested_dishware():
    counts = DishwareCounts(plates=1, bowls=0, cups=0, mugs=0)
    env = make_env(
        counts,
        TableBussingConfig(use_coacd=False, acronym_seed=0),
        render_mode=None,
    )
    try:
        env.reset(seed=0)
        raw = env.unwrapped
        assert len(raw.dishware) == 1
        assert raw.dishware_categories == ["Plate"]
        info = raw.evaluate()
        for key in (
            "success",
            "all_on_tray",
            "all_static",
            "frac_on_tray",
            "dishware_meta",
            "dishware_grasp_counts",
        ):
            assert key in info
        assert len(info["dishware_meta"]) == 1
        assert info["dishware_meta"][0]["category"] == "Plate"
        assert len(info["dishware_grasp_counts"]) == 1
    finally:
        env.close()


def test_make_env_rejects_conflicting_counts_and_num_kwargs():
    with pytest.raises(ValueError, match="conflicting"):
        make_env(DishwareCounts(plates=1, bowls=0), num_plates=3)
