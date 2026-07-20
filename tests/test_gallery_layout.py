"""Tests for pure gallery layout helpers and CLI/factory wiring."""

from __future__ import annotations

import pytest

from render_dishware_gallery import (
    _layout_grid,
    _parse_gallery_args,
    _resolve_categories,
    make_gallery_env,
)


def test_layout_grid_places_rows_and_columns():
    # One category row with two items of known radius/center.
    rows = [
        [
            (object(), 0.05, 0.1, (0.0, 0.0)),
            (object(), 0.05, 0.1, (0.0, 0.0)),
        ]
    ]
    poses = _layout_grid(rows, gap=0.06, origin_xy=(0.0, 0.0))
    assert len(poses) == 1
    assert len(poses[0]) == 2
    # Columns advance along Y; same row shares X.
    assert poses[0][0][0] == poses[0][1][0]
    assert poses[0][1][1] > poses[0][0][1]
    # Spawn z preserved from input.
    assert poses[0][0][2] == 0.05


def test_layout_grid_empty():
    assert _layout_grid([], gap=0.06) == []


def test_parse_gallery_args_rebuild_and_categories():
    args = _parse_gallery_args(["--rebuild", "--include-mugs", "--no-save"])
    assert args.rebuild is True
    assert args.include_mugs is True
    assert args.no_save is True
    assert _resolve_categories(args) == ["Plate", "Bowl", "Cup", "Mug"]


@pytest.mark.integration
def test_make_gallery_env_headless_reset():
    env = make_gallery_env(
        categories=["Plate"],
        rebuild=False,
        render_mode=None,
    )
    try:
        env.reset(seed=0)
        raw = env.unwrapped
        assert len(raw.gallery_objects) >= 1
        assert all(o.category == "Plate" for o in raw.gallery_objects)
        assert len(raw.actors) == len(raw.gallery_objects)
    finally:
        env.close()
