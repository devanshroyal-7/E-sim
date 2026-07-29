"""Render all curated dishware objects on a table for visual review.

Layout (default: Plate + Bowl + Cup; pass --include-mugs for Mug too):
  - One row per category along table width (Plate, Bowl, Cup, …)
  - Columns along table length: train ranks 1-3, then test ranks 1-2
  - Spacing is size-aware; mesh AABB centers are aligned to the grid
  - Actors are kinematic so physics cannot scramble the grid
  - Console prints a legend so you can map positions to mesh hashes

Edit ``data/dishware_selection.json`` to swap an object, then re-run this script.

Examples:
  python render_dishware_gallery.py
  python render_dishware_gallery.py --include-mugs --save gallery.png
  python render_dishware_gallery.py --rebuild
"""

from __future__ import annotations

import argparse
from pathlib import Path
import gymnasium as gym
import numpy as np
import sapien
import torch
from PIL import Image

import mani_skill.envs  # noqa: F401
from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig

from typing import TypedDict

from acronym_dishware import (
    ACTIVE_CATEGORIES,
    DishwareObject,
    SELECTION_PATH,
    compute_object_footprint,
    ensure_curated_dishware,
    print_selection_summary,
)
from dishware_actors import ActorBuildOptions, build_dishware_actor


class GalleryEvalInfo(TypedDict):
    """Empty success payload for the kinematic gallery (no task metrics)."""


class GalleryObsExtra(TypedDict):
    """Empty extra obs for the gallery env."""


def _layout_grid(
    rows: list[list[tuple[DishwareObject, float, float, tuple[float, float]]]],
    gap: float,
    origin_xy: tuple[float, float] = (-0.12, 0.0),
) -> list[list[tuple[float, float, float]]]:
    """Place each category row with size-aware spacing; return actor poses.

    ManiSkill's table visual is long along **Y** (~2.4m) and short along **X**
    (~1.2m), so columns (train1..test2) run along Y and category rows along X.
    ``rows[r][c]`` is ``(obj, spawn_z, xy_radius, center_xy)``. Returned poses
    put each mesh AABB center on the grid (pose = grid - center_xy).
    """
    if not rows:
        return []

    n_cols = max(len(row) for row in rows)
    all_radii = [radius for row in rows for _, _, radius, _ in row]
    pitch = 2.0 * max(all_radii, default=0.1) + gap

    # Uniform pitch so the grid reads as evenly ordered.
    ys = [c * pitch for c in range(n_cols)]
    xs = [r * pitch for r in range(len(rows))]

    x_min = xs[0] - 0.5 * pitch + 0.5 * gap
    x_max = xs[-1] + 0.5 * pitch - 0.5 * gap
    y_min = ys[0] - 0.5 * pitch + 0.5 * gap
    y_max = ys[-1] + 0.5 * pitch - 0.5 * gap
    x_shift = origin_xy[0] - 0.5 * (x_min + x_max)
    y_shift = origin_xy[1] - 0.5 * (y_min + y_max)

    poses: list[list[tuple[float, float, float]]] = []
    for r, row in enumerate(rows):
        row_poses = []
        for c, (_, spawn_z, _, center_xy) in enumerate(row):
            px = xs[r] + x_shift - center_xy[0]
            py = ys[c] + y_shift - center_xy[1]
            row_poses.append((px, py, spawn_z))
        poses.append(row_poses)
    return poses


def _plan_gallery(
    categories: list[str],
    acronym_root: str | None,
    rebuild: bool,
    gap: float,
    gallery_objects: list[DishwareObject] | None = None,
):
    """Compute ordered objects, size-aware poses, labels, and camera framing."""
    if gallery_objects is None:
        gallery_objects = ensure_curated_dishware(
            acronym_root=acronym_root,
            rebuild=rebuild,
            categories=categories,
        )
    else:
        gallery_objects = list(gallery_objects)
    gallery_objects.sort(key=lambda o: (o.category, o.split != "train", o.rank))

    by_cat: dict[str, list[DishwareObject]] = {}
    for obj in gallery_objects:
        by_cat.setdefault(obj.category, []).append(obj)

    # One row per category: footprint metrics feed layout directly (no parallel copies).
    rows: list[list[tuple[DishwareObject, float, float, float, tuple[float, float]]]] = []
    for category in categories:
        if category not in by_cat:
            continue
        row = []
        for obj in by_cat[category]:
            # ACRONYM meshes are often not origin-centered; center_xy lets the
            # layout place the visible AABB on the grid rather than the mesh origin.
            m = compute_object_footprint(obj)
            row.append((obj, m.scale, m.spawn_z, m.xy_radius, m.center_xy))
        rows.append(row)

    pose_rows = _layout_grid(
        [[(obj, spawn_z, radius, center_xy) for obj, _, spawn_z, radius, center_xy in row]
         for row in rows],
        gap=gap,
    )

    planned: list[dict] = []
    labels: list[str] = []
    grid_poses: list[tuple[float, float, float]] = []
    for row_i, (row, poses) in enumerate(zip(rows, pose_rows)):
        for col_i, ((obj, scale, spawn_z, xy_radius, center_xy), pose) in enumerate(
            zip(row, poses)
        ):
            x, y, z = pose
            vx, vy = x + center_xy[0], y + center_xy[1]
            planned.append(
                {
                    "obj": obj,
                    "scale": scale,
                    "spawn_z": spawn_z,
                    "xy_radius": xy_radius,
                    "pose": pose,
                    "visual_center": (vx, vy),
                    "name": f"{obj.category}_{obj.split}{obj.rank}_{obj.mesh_hash[:8]}",
                }
            )
            grid_poses.append(pose)
            labels.append(
                f"row={row_i} col={col_i} | {obj.category} {obj.split}#{obj.rank} "
                f"hash={obj.mesh_hash} scale={obj.scale:.6g} xy_r={xy_radius:.3f} "
                f"center=({vx:.3f},{vy:.3f})"
            )

    if planned:
        xs = [item["visual_center"][0] for item in planned]
        ys = [item["visual_center"][1] for item in planned]
        cx = 0.5 * (min(xs) + max(xs))
        cy = 0.5 * (min(ys) + max(ys))
        span_y = max(max(ys) - min(ys), 0.4)
        # Near top-down so the 2×N grid is obvious; slight +X bias for depth.
        camera_eye = [cx + 0.35, cy, 1.55 + 0.15 * span_y]
        camera_target = [cx, cy, 0.02]
    else:
        camera_eye = [0.2, 0.0, 1.6]
        camera_target = [-0.12, 0.0, 0.02]

    return gallery_objects, planned, labels, grid_poses, camera_eye, camera_target


@register_env("DishwareGallery-v1", max_episode_steps=10_000)
class DishwareGalleryEnv(BaseEnv):
    """Tabletop gallery of the curated ACRONYM dishware selection."""

    SUPPORTED_ROBOTS = ["panda"]
    SUPPORTED_REWARD_MODES = ["none"]
    agent: Panda

    def __init__(
        self,
        *args,
        categories: list[str] | None = None,
        acronym_root: str | None = None,
        rebuild: bool = False,
        gap: float = 0.06,
        gallery_objects: list[DishwareObject] | None = None,
        **kwargs,
    ):
        self._gallery_categories = categories or ["Plate", "Bowl", "Cup"]
        self._acronym_root = acronym_root
        self._rebuild = rebuild
        self._gap = gap
        (
            self.gallery_objects,
            self._planned,
            self.gallery_labels,
            self._grid_poses,
            self._camera_eye,
            self._camera_target,
        ) = _plan_gallery(
            self._gallery_categories,
            acronym_root=acronym_root,
            rebuild=rebuild,
            gap=gap,
            gallery_objects=gallery_objects,
        )
        self.actors = []
        super().__init__(*args, robot_uids="panda", **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=2**20,
                max_rigid_patch_count=2**18,
            )
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=self._camera_eye, target=self._camera_target)
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        # High viewpoint framed on the size-aware grid.
        pose = sapien_utils.look_at(eye=self._camera_eye, target=self._camera_target)
        return CameraConfig(
            "render_camera", pose=pose, width=1280, height=720, fov=0.85, near=0.01, far=100
        )

    def _load_agent(self, options: dict):
        # Initial load pose; episode init re-parks after TableSceneBuilder reset.
        super()._load_agent(options, sapien.Pose(p=[-1.6, -1.2, 0.0]))

    def _park_agent(self):
        """Move the panda fully off the table after TableSceneBuilder.initialize."""
        self.agent.robot.set_pose(sapien.Pose(p=[-1.6, -1.2, 0.0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(env=self, robot_init_qpos_noise=0.0)
        self.table_scene.build()

        self.actors = []
        for item in self._planned:
            actor = build_dishware_actor(
                self.scene,
                item["obj"],
                name=item["name"],
                pose_xyz=item["pose"],
                options=ActorBuildOptions(
                    scale=item["scale"], use_coacd=False, kinematic=True
                ),
            )
            self.actors.append(actor)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            self._park_agent()
            for actor, (x, y, z) in zip(self.actors, self._grid_poses):
                xyz = torch.zeros((b, 3), device=self.device)
                xyz[:, 0] = x
                xyz[:, 1] = y
                xyz[:, 2] = z
                actor.set_pose(Pose.create_from_pq(p=xyz, q=[1, 0, 0, 0]))

    def evaluate(self) -> GalleryEvalInfo:
        return {}

    def _get_obs_extra(self, info: dict[str, object]) -> GalleryObsExtra:
        return {}


def make_gallery_env(
    categories: list[str] | None = None,
    *,
    acronym_root: str | None = None,
    rebuild: bool = False,
    gap: float = 0.06,
    obs_mode: str = "state",
    num_envs: int = 1,
    **kwargs,
) -> gym.Env:
    """Canonical factory for ``DishwareGallery-v1`` (mirrors ``table_bussing.make_env``)."""
    cats = categories or ["Plate", "Bowl", "Cup"]
    objs = ensure_curated_dishware(
        acronym_root=acronym_root, rebuild=rebuild, categories=cats
    )
    return gym.make(
        "DishwareGallery-v1",
        obs_mode=obs_mode,
        reward_mode=kwargs.pop("reward_mode", "none"),
        num_envs=num_envs,
        categories=cats,
        acronym_root=acronym_root,
        rebuild=False,
        gap=gap,
        gallery_objects=objs,
        **kwargs,
    )


def _parse_gallery_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-mugs",
        action="store_true",
        help="Also show curated Mug objects (Cups are on by default)",
    )
    parser.add_argument(
        "--include-cups-mugs",
        action="store_true",
        help=argparse.SUPPRESS,  # backwards-compatible alias of --include-mugs
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Override categories (default: Plate Bowl Cup)",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Regenerate data/dishware_selection.json before rendering",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=Path("data/dishware_gallery.png"),
        help="Path to save an RGB screenshot",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip writing a screenshot",
    )
    parser.add_argument(
        "--no-human",
        action="store_true",
        help="Skip the interactive human viewer (still saves unless --no-save)",
    )
    parser.add_argument("--steps", type=int, default=300, help="Viewer loop steps")
    return parser.parse_args(argv)


def _resolve_categories(args: argparse.Namespace) -> list[str]:
    if args.categories:
        return list(args.categories)
    if args.include_mugs or args.include_cups_mugs:
        return list(ACTIVE_CATEGORIES)
    return ["Plate", "Bowl", "Cup"]


def _save_gallery_frame(raw_env: BaseEnv, save_path: Path) -> None:
    # Human mode's env.render() returns a Viewer; always capture rgb_array.
    frame = raw_env.render_rgb_array()
    if isinstance(frame, torch.Tensor):
        frame_np = frame[0].detach().cpu().numpy()
    else:
        frame_np = np.asarray(frame)
        if frame_np.ndim == 4:
            frame_np = frame_np[0]
    save_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame_np.astype(np.uint8)).save(save_path)
    print(f"Saved screenshot: {save_path.resolve()}")


def main(argv: list[str] | None = None) -> None:
    args = _parse_gallery_args(argv)
    categories = _resolve_categories(args)

    render_mode = "rgb_array" if args.no_human else "human"
    env = make_gallery_env(
        categories,
        rebuild=args.rebuild,
        render_mode=render_mode,
    )
    obs, info = env.reset(seed=0)
    raw = env.unwrapped

    print(f"Selection file: {SELECTION_PATH}")
    print_selection_summary(raw.gallery_objects)
    print()
    print(
        "Gallery layout legend "
        "(rows along +X = category; cols along +Y = train1-3 then test1-2):"
    )
    for line in raw.gallery_labels:
        print(f"  {line}")

    # Brief no-op steps so the scene is fully initialized before capture.
    for _ in range(10):
        env.step(env.action_space.sample() * 0.0)

    if not args.no_save:
        _save_gallery_frame(raw, args.save)

    if not args.no_human:
        print("Close the viewer window or Ctrl+C to exit.")
        try:
            for _ in range(args.steps):
                env.step(env.action_space.sample() * 0.0)
                env.render()
        except KeyboardInterrupt:
            print("Interrupted — closing gallery.")

    env.close()


if __name__ == "__main__":
    main()
