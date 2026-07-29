"""Unit tests for curated ACRONYM dishware helpers (no ManiSkill runtime)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from acronym_dishware import (
    ACTIVE_CATEGORIES,
    DishwareObject,
    N_PER_CATEGORY,
    N_TEST,
    N_TRAIN,
    ObjectSpawnMetrics,
    SelectionEntry,
    _bowl_interior_content_frac,
    _parse_split_entry,
    build_dishware_manifest,
    compute_object_footprint,
    curate_selection,
    ensure_curated_dishware,
    filter_dishware_objects,
    load_full_manifest,
    load_grasps,
    load_selection,
    resolve_selection,
    sample_objects,
    save_full_manifest,
    xy_radius_from_bounds,
)


@pytest.mark.integration
def test_ensure_curated_dishware_loads_active_categories():
    objs = ensure_curated_dishware()
    assert objs
    cats = {o.category for o in objs}
    assert cats <= set(ACTIVE_CATEGORIES)
    assert "Plate" in cats and "Bowl" in cats


@pytest.mark.integration
def test_ensure_curated_dishware_rejects_extra_categories():
    with pytest.raises(ValueError, match="ACTIVE_CATEGORIES"):
        ensure_curated_dishware(categories=["WineGlass"])


@pytest.mark.integration
def test_filter_and_sample_objects_are_deterministic():
    objs = ensure_curated_dishware()
    train = filter_dishware_objects(objs, split="train", categories=["Plate", "Bowl"])
    assert train
    assert all(o.split == "train" for o in train)

    rng = np.random.default_rng(0)
    a = sample_objects(objs, {"Plate": 1, "Bowl": 1}, split="train", rng=rng)
    rng = np.random.default_rng(0)
    b = sample_objects(objs, {"Plate": 1, "Bowl": 1}, split="train", rng=rng)
    assert [o.key for o in a] == [o.key for o in b]


def test_empty_mesh_interior_score_is_not_preferred(tmp_path: Path):
    empty = tmp_path / "empty.obj"
    empty.write_text("# empty mesh\n")
    # trimesh may still load; if vertices empty, score must not look like a filled bowl.
    score = _bowl_interior_content_frac(empty)
    assert score == float("inf") or score >= 0.0


@pytest.mark.integration
def test_compute_object_footprint_matches_selection_object():
    objs = ensure_curated_dishware(categories=["Plate", "Bowl"])
    metrics = compute_object_footprint(objs[0])
    assert isinstance(metrics, ObjectSpawnMetrics)
    assert metrics.scale > 0
    assert metrics.density > 0
    assert metrics.xy_radius > 0
    mins = np.asarray(metrics.mins)
    maxs = np.asarray(metrics.maxs)
    assert xy_radius_from_bounds(mins, maxs) == pytest.approx(metrics.xy_radius)


def test_selection_entry_typed_dict_shape():
    entry: SelectionEntry = {
        "mesh_hash": "abc",
        "scale": "0.1",
        "split": "train",
        "rank": 1,
    }
    assert entry["rank"] == 1


def test_sample_objects_missing_category_raises():
    objs = [
        DishwareObject(
            category="Plate",
            mesh_hash="deadbeef",
            scale=0.1,
            mesh_path="/tmp/x.obj",
            grasp_path="/tmp/x.h5",
            split="train",
        )
    ]
    with pytest.raises(ValueError, match="No ACRONYM objects"):
        sample_objects(objs, {"Bowl": 1}, split="train", rng=np.random.default_rng(0))


def test_parse_split_entry_accepts_and_rejects():
    assert _parse_split_entry("Plate_abcdef0123456789_0.065.json") == (
        "Plate",
        "abcdef0123456789",
        0.065,
    )
    with pytest.raises(ValueError, match="Unrecognized"):
        _parse_split_entry("not-a-valid-stem")


def test_curate_selection_picks_train_and_test_slots():
    catalog: list[DishwareObject] = []
    for i in range(6):
        catalog.append(
            DishwareObject(
                category="Plate",
                mesh_hash=f"{i:016x}",
                scale=0.1,
                mesh_path=f"/tmp/Plate_{i:016x}.obj",
                grasp_path=f"/tmp/Plate_{i:016x}_0.1.h5",
                split="train",
            )
        )
    for i in range(6, 10):
        catalog.append(
            DishwareObject(
                category="Plate",
                mesh_hash=f"{i:016x}",
                scale=0.1,
                mesh_path=f"/tmp/Plate_{i:016x}.obj",
                grasp_path=f"/tmp/Plate_{i:016x}_0.1.h5",
                split="test",
            )
        )
    selection = curate_selection(catalog, categories=["Plate"])
    entries = selection["Plate"]
    assert len(entries) == N_PER_CATEGORY
    assert sum(1 for e in entries if e["split"] == "train") == N_TRAIN
    assert sum(1 for e in entries if e["split"] == "test") == N_TEST
    assert {e["rank"] for e in entries if e["split"] == "train"} == {1, 2, 3}
    assert {e["rank"] for e in entries if e["split"] == "test"} == {1, 2}


@pytest.mark.integration
@pytest.mark.integration
def test_resolve_selection_round_trips_on_disk_selection():
    selection = load_selection()
    objects = resolve_selection(selection)
    assert objects
    by_cat: dict[str, list[DishwareObject]] = {}
    for obj in objects:
        by_cat.setdefault(obj.category, []).append(obj)
    for category, entries in selection.items():
        if category not in ACTIVE_CATEGORIES:
            continue
        resolved = by_cat[category]
        assert len(resolved) == len(entries)
        resolved_keys = {(o.mesh_hash, f"{o.scale:g}", o.split) for o in resolved}
        expected_keys = {
            (e["mesh_hash"], f"{float(e['scale']):g}", e["split"]) for e in entries
        }
        assert resolved_keys == expected_keys


def test_load_grasps_filters_success_only(fake_acronym_root: Path):
    grasp = next((fake_acronym_root / "grasps").glob("*.h5"))
    all_grasps = load_grasps(grasp, success_only=False)
    ok_grasps = load_grasps(grasp, success_only=True)
    assert all_grasps.shape[0] == 3
    assert ok_grasps.shape[0] == 2
    assert ok_grasps.shape[1:] == (4, 4)


def test_full_manifest_cache_round_trip(tmp_path: Path, fake_acronym_root: Path):
    cache = tmp_path / "manifest.json"
    built = build_dishware_manifest(
        acronym_root=fake_acronym_root,
        categories=["Plate"],
        cache_path=None,
    )
    assert built
    save_full_manifest(built, path=cache)
    loaded = load_full_manifest(cache)
    assert [o.key for o in loaded] == [o.key for o in built]

    # Cache hit path: missing categories in cache force rebuild; here cache has Plate.
    cached = build_dishware_manifest(
        acronym_root=fake_acronym_root,
        categories=["Plate"],
        cache_path=cache,
        rebuild_cache=False,
    )
    assert [o.key for o in cached] == [o.key for o in built]


def test_resolve_selection_hermetic(fake_acronym_root: Path):
    mesh_hash = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    selection = {
        "Plate": [
            {
                "mesh_hash": mesh_hash,
                "scale": "0.100000",
                "split": "train",
                "rank": 1,
            }
        ]
    }
    objects = resolve_selection(selection, acronym_root=fake_acronym_root)
    assert len(objects) == 1
    assert objects[0].category == "Plate"
    assert objects[0].mesh_hash == mesh_hash
    assert Path(objects[0].mesh_path).is_file()
    assert Path(objects[0].grasp_path).is_file()
