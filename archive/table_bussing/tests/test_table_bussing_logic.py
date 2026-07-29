"""Lightweight tests for TableBussing helpers that do not need a GPU sim."""

from __future__ import annotations

import pytest

from table_bussing import DishwareCounts, TableBussingConfig, make_env


def test_dishware_counts_validate_and_merge_cups():
    counts = DishwareCounts(plates=1, bowls=0, cups=2, mugs=0)
    counts.validate()
    assert counts.as_category_counts()["Cup"] == 2

    with pytest.raises(ValueError, match="At least one"):
        DishwareCounts(plates=0, bowls=0, cups=0, mugs=0).validate()


def test_make_env_rejects_bad_split():
    with pytest.raises(ValueError, match="acronym_split"):
        make_env(
            DishwareCounts(plates=1),
            TableBussingConfig(acronym_split="nope"),  # type: ignore[arg-type]
        )
