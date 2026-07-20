"""Shared Sapien actor construction for ACRONYM dishware meshes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import sapien

from acronym_dishware import DishwareObject, ObjectSpawnMetrics, compute_object_footprint


@runtime_checkable
class _ActorBuilder(Protocol):
    def add_convex_collision_from_file(self, **kwargs: Any) -> Any: ...
    def add_multiple_convex_collisions_from_file(self, **kwargs: Any) -> Any: ...
    def add_visual_from_file(self, **kwargs: Any) -> Any: ...
    def set_initial_pose(self, pose: sapien.Pose) -> Any: ...
    def build(self, name: str) -> Any: ...
    def build_kinematic(self, name: str) -> Any: ...


@runtime_checkable
class SceneWithActorBuilder(Protocol):
    def create_actor_builder(self) -> _ActorBuilder: ...


@dataclass(frozen=True)
class ActorBuildOptions:
    """Optional collision/physics knobs for :func:`build_dishware_actor`."""

    metrics: ObjectSpawnMetrics | None = None
    scale: float | None = None
    density: float | None = None
    use_coacd: bool = False
    kinematic: bool = False


def build_dishware_actor(
    scene: SceneWithActorBuilder,
    obj: DishwareObject,
    *,
    name: str,
    pose_xyz: tuple[float, float, float],
    options: ActorBuildOptions | None = None,
) -> Any:
    """Build a Sapien actor from a ``DishwareObject``.

    Collision policy is controlled by ``options.use_coacd`` (convex decomposition
    vs single convex hull). Gallery actors pass ``kinematic=True`` so physics
    cannot knock them out of place.
    """
    opts = options or ActorBuildOptions()
    metrics = opts.metrics
    scale = opts.scale
    density = opts.density
    if metrics is None and (scale is None or density is None):
        metrics = compute_object_footprint(obj)
    if scale is None:
        if metrics is None:
            raise ValueError("scale is required when metrics is not provided")
        scale = metrics.scale
    if density is None:
        if metrics is None:
            raise ValueError("density is required when metrics is not provided")
        density = metrics.density

    scale_xyz = [scale, scale, scale]
    builder = scene.create_actor_builder()
    if opts.use_coacd:
        builder.add_multiple_convex_collisions_from_file(
            filename=obj.mesh_path,
            scale=scale_xyz,
            density=density,
            decomposition="coacd",
        )
    else:
        builder.add_convex_collision_from_file(
            filename=obj.mesh_path,
            scale=scale_xyz,
            density=density,
        )
    builder.add_visual_from_file(filename=obj.mesh_path, scale=scale_xyz)
    builder.set_initial_pose(
        sapien.Pose(p=[pose_xyz[0], pose_xyz[1], pose_xyz[2]])
    )
    if opts.kinematic:
        return builder.build_kinematic(name=name)
    return builder.build(name=name)
