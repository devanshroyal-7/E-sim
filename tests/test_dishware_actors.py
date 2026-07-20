"""Tests for dishware actor helper contracts (hermetic fake scene + ACRONYM tree)."""

from __future__ import annotations

from pathlib import Path

from acronym_dishware import DishwareObject, compute_object_footprint, resolve_selection
from dishware_actors import ActorBuildOptions, build_dishware_actor


class _FakeBuilder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def add_convex_collision_from_file(self, **kwargs) -> None:
        self.calls.append("convex")

    def add_multiple_convex_collisions_from_file(self, **kwargs) -> None:
        self.calls.append("coacd")

    def add_visual_from_file(self, **kwargs) -> None:
        self.calls.append("visual")

    def set_initial_pose(self, pose) -> None:
        self.calls.append("pose")

    def build_kinematic(self, name: str):
        self.calls.append(f"kinematic:{name}")
        return {"kinematic": True, "name": name}

    def build(self, name: str, **kwargs):
        self.calls.append(f"build:{name}")
        return {"kinematic": False, "name": name}


class _FakeScene:
    def __init__(self) -> None:
        self.builder = _FakeBuilder()

    def create_actor_builder(self) -> _FakeBuilder:
        return self.builder


def _plate_from_fake_root(fake_acronym_root: Path) -> DishwareObject:
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
    return resolve_selection(selection, acronym_root=fake_acronym_root)[0]


def test_build_dishware_actor_convex_kinematic(fake_acronym_root: Path):
    obj = _plate_from_fake_root(fake_acronym_root)
    metrics = compute_object_footprint(obj)
    scene = _FakeScene()
    actor = build_dishware_actor(
        scene,
        obj,
        name="plate0",
        pose_xyz=(0.1, 0.2, metrics.spawn_z),
        options=ActorBuildOptions(
            metrics=metrics, use_coacd=False, kinematic=True
        ),
    )
    assert actor["kinematic"] is True
    assert "convex" in scene.builder.calls
    assert "visual" in scene.builder.calls
    assert "kinematic:plate0" in scene.builder.calls


def test_build_dishware_actor_coacd_dynamic(fake_acronym_root: Path):
    obj = _plate_from_fake_root(fake_acronym_root)
    scene = _FakeScene()
    actor = build_dishware_actor(
        scene,
        obj,
        name="bowl0",
        pose_xyz=(0.0, 0.0, 0.1),
        options=ActorBuildOptions(use_coacd=True, kinematic=False),
    )
    assert actor["kinematic"] is False
    assert "coacd" in scene.builder.calls
    assert "build:bowl0" in scene.builder.calls
