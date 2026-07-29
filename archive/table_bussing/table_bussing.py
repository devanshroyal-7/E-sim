from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NotRequired, TypedDict, Union

import gymnasium as gym
import numpy as np
import sapien
import torch

import mani_skill.envs  # noqa: F401
from mani_skill.agents.robots import Fetch, Panda, UR10e
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig

from acronym_dishware import (
    DishwareObject,
    SplitName,
    compute_object_footprint,
    ensure_curated_dishware,
    load_grasps,
    sample_objects,
)
from dishware_actors import ActorBuildOptions, build_dishware_actor


class DishwareMeta(TypedDict):
    category: str
    mesh_hash: str
    scale: float
    mesh_path: str
    grasp_path: str
    split: str


class EvalInfo(TypedDict):
    """Fixed keys returned by ``TableBussingEnv.evaluate``."""

    success: torch.Tensor
    all_on_tray: torch.Tensor
    all_static: torch.Tensor
    frac_on_tray: torch.Tensor
    dishware_meta: list[DishwareMeta]
    dishware_grasp_counts: list[int]


class ObsExtra(TypedDict):
    """Extra observation keys from ``TableBussingEnv._get_obs_extra``."""

    tcp_pose: torch.Tensor
    tray_pose: torch.Tensor
    dishware_poses: NotRequired[torch.Tensor]
    dishware_to_tray: NotRequired[torch.Tensor]


@dataclass(frozen=True)
class LoadedDishware:
    """One spawned dishware actor plus the spawn/meta data that travels with it."""

    actor: Actor
    category: str
    xy_radius: float
    spawn_z: float
    meta: DishwareMeta
    grasps: np.ndarray


@dataclass(frozen=True)
class DishwareCounts:
    """Grouped dishware instance counts for curated ``ACTIVE_CATEGORIES``."""

    plates: int = 2
    bowls: int = 1
    cups: int = 0  # ACRONYM Cup tumblers
    mugs: int = 0

    def as_category_counts(self) -> dict[str, int]:
        return {
            "Plate": self.plates,
            "Bowl": self.bowls,
            "Cup": self.cups,
            "Mug": self.mugs,
        }

    def validate(self) -> None:
        values = list(self.as_category_counts().values())
        if any(n < 0 for n in values):
            raise ValueError("Dishware counts must be non-negative")
        if sum(values) == 0:
            raise ValueError("At least one dishware item is required")


@dataclass(frozen=True)
class TableBussingConfig:
    """Non-count options for ``make_env`` / ``TableBussing-v1``."""

    acronym_root: str | None = None
    acronym_split: SplitName = "train"
    acronym_seed: int | None = None
    use_coacd: bool = True
    obs_mode: str = "state"
    num_envs: int = 1


def make_env(
    counts: DishwareCounts | None = None,
    config: TableBussingConfig | None = None,
    **kwargs: Any,
) -> gym.Env:
    """Create a registered ``TableBussing-v1`` environment.

    This is the canonical factory. It validates dishware counts and
    ``acronym_split``, then applies shared defaults from ``TableBussingConfig``.
    Pass ManiSkill options such as ``render_mode`` via ``kwargs``.
    """
    counts = counts or DishwareCounts()
    cfg = config or TableBussingConfig()
    counts.validate()
    if cfg.acronym_split not in ("train", "test", "all"):
        raise ValueError(
            f"acronym_split must be 'train', 'test', or 'all'; got {cfg.acronym_split!r}"
        )
    return gym.make(
        "TableBussing-v1",
        obs_mode=cfg.obs_mode,
        num_envs=cfg.num_envs,
        counts=counts,
        acronym_root=cfg.acronym_root,
        acronym_split=cfg.acronym_split,
        acronym_seed=cfg.acronym_seed,
        use_coacd=cfg.use_coacd,
        **kwargs,
    )


def item_centers_on_tray(
    item_xyz: torch.Tensor,
    tray_xyz: torch.Tensor,
    tray_half_sizes: Union[torch.Tensor, tuple[float, float, float]],
) -> torch.Tensor:
    """Batch mask: item xy inside the tray footprint and center above the tray top.

    ``item_xyz`` / ``tray_xyz`` are ``(B, 3)``. Half-sizes are ``(hx, hy, hz)``.
    Stacking is allowed: any ``z`` strictly above the tray top surface counts.
    """
    half = torch.as_tensor(
        tray_half_sizes, device=item_xyz.device, dtype=item_xyz.dtype
    )
    in_xy = (torch.abs(item_xyz[:, 0] - tray_xyz[:, 0]) < half[0]) & (
        torch.abs(item_xyz[:, 1] - tray_xyz[:, 1]) < half[1]
    )
    tray_top_z = tray_xyz[:, 2] + half[2]
    on_top = item_xyz[:, 2] > tray_top_z
    return in_xy & on_top


@register_env("TableBussing-v1", max_episode_steps=100)
class TableBussingEnv(BaseEnv):
    """
    **Task Description:**
    Clear a table by moving ACRONYM dishware onto a tray. Stack dishware on the tray to
    achieve a stable configuration (table bussing).

    **Randomizations:**
    - dishware instances are sampled from the ACRONYM dishware catalog (train split by default)
    - dishware xy positions are randomized on top of a table in the region
      [-0.25, -0.35] x [0.25, 0.05], sampled so objects do not overlap
    - dishware yaw about the table normal is randomized
    - the tray (goal region) xy position is lightly randomized around [0.0, 0.28]

    **Success Conditions:**
    - every dishware item's xy position is within the tray bounds
    - every dishware item rests on top of the tray (z above the tray surface)
    - all dishware is static

    **Configurable counts (constructor kwargs / ``DishwareCounts``):**
    - ``num_plates``, ``num_bowls``, ``num_cups`` (ACRONYM ``Cup``), ``num_mugs``
    """

    SUPPORTED_ROBOTS = ["panda", "fetch", "ur_10e"]
    agent: Union[Panda, Fetch, UR10e]

    spawn_region = [[-0.25, -0.35], [0.25, 0.05]]
    tray_half_sizes = (0.18, 0.22, 0.01)
    tray_center_xy = (0.0, 0.28)

    def __init__(
        self,
        *args,
        robot_uids="panda",
        robot_init_qpos_noise=0.02,
        counts: DishwareCounts | None = None,
        # Flat num_* kwargs remain for gym.make compatibility; prefer ``counts=``.
        # Defaults are None so ``counts=`` alone is never treated as conflicting.
        num_plates: int | None = None,
        num_bowls: int | None = None,
        num_cups: int | None = None,
        num_mugs: int | None = None,
        acronym_root: str | None = None,
        acronym_split: SplitName = "train",
        acronym_seed: int | None = None,
        use_coacd: bool = True,
        num_envs: int = 1,
        reconfiguration_freq: int | None = None,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        flat_kwargs = {
            "plates": num_plates,
            "bowls": num_bowls,
            "cups": num_cups,
            "mugs": num_mugs,
        }
        explicit_flat = {k: v for k, v in flat_kwargs.items() if v is not None}
        if counts is not None:
            if explicit_flat:
                merged = DishwareCounts(
                    plates=explicit_flat.get("plates", counts.plates),
                    bowls=explicit_flat.get("bowls", counts.bowls),
                    cups=explicit_flat.get("cups", counts.cups),
                    mugs=explicit_flat.get("mugs", counts.mugs),
                )
                if merged != counts:
                    raise ValueError(
                        "pass counts= or num_* kwargs, not both with conflicting values"
                    )
            dish_counts = counts
        else:
            dish_counts = DishwareCounts(
                plates=explicit_flat.get("plates", 2),
                bowls=explicit_flat.get("bowls", 1),
                cups=explicit_flat.get("cups", 0),
                mugs=explicit_flat.get("mugs", 0),
            )
        dish_counts.validate()
        self.num_plates = dish_counts.plates
        self.num_bowls = dish_counts.bowls
        self.num_cups = dish_counts.cups
        self.num_mugs = dish_counts.mugs
        self.acronym_root = acronym_root
        if acronym_split not in ("train", "test", "all"):
            raise ValueError(
                f"acronym_split must be 'train', 'test', or 'all'; got {acronym_split!r}"
            )
        self.acronym_split = acronym_split
        self.acronym_seed = acronym_seed
        self.use_coacd = bool(use_coacd)

        self._dishware_counts = dish_counts.as_category_counts()
        # Defer catalog I/O until scene load so importing/constructing the class
        # does not touch the filesystem before BaseEnv initializes.
        self._curated_dishware: list[DishwareObject] | None = None
        self._sample_rng = np.random.default_rng(acronym_seed)
        self._selected_objects: list[DishwareObject] = []

        if reconfiguration_freq is None:
            reconfiguration_freq = 1 if num_envs == 1 else 0

        super().__init__(
            *args,
            robot_uids=robot_uids,
            num_envs=num_envs,
            reconfiguration_freq=reconfiguration_freq,
            **kwargs,
        )

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=2**21,
                max_rigid_patch_count=2**19,
                found_lost_pairs_capacity=2**25,
            )
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [
            CameraConfig(
                "base_camera",
                pose=pose,
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _sample_dishware_objects(self) -> list[DishwareObject]:
        if self._curated_dishware is None:
            self._curated_dishware = ensure_curated_dishware(
                acronym_root=self.acronym_root
            )
        return sample_objects(
            self._curated_dishware,
            self._dishware_counts,
            split=self.acronym_split,
            rng=self._sample_rng,
        )

    def _build_acronym_dish(
        self, obj: DishwareObject, index: int, initial_xy: tuple[float, float]
    ) -> tuple[Actor, float, float]:
        metrics = compute_object_footprint(obj)
        actor = build_dishware_actor(
            self.scene,
            obj,
            name=f"{obj.category}_{obj.mesh_hash[:8]}_{index}",
            pose_xyz=(initial_xy[0], initial_xy[1], metrics.spawn_z),
            options=ActorBuildOptions(
                metrics=metrics, use_coacd=self.use_coacd, kinematic=False
            ),
        )
        return actor, metrics.xy_radius, metrics.spawn_z

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self._selected_objects = self._sample_dishware_objects()

        self.actors_by_category: dict[str, list[Actor]] = {
            "Plate": [],
            "Bowl": [],
            "Cup": [],
            "Mug": [],
        }
        self.plates = self.actors_by_category["Plate"]
        self.bowls = self.actors_by_category["Bowl"]
        self.cups = self.actors_by_category["Cup"]
        self.mugs = self.actors_by_category["Mug"]
        self.loaded_dishware: list[LoadedDishware] = []

        # Park actors off-table before episode init randomizes poses.
        for i, obj in enumerate(self._selected_objects):
            actor, xy_radius, spawn_z = self._build_acronym_dish(
                obj, i, initial_xy=(0.5 + i * 0.25, 0.5)
            )
            item = LoadedDishware(
                actor=actor,
                category=obj.category,
                xy_radius=xy_radius,
                spawn_z=spawn_z,
                meta={
                    "category": obj.category,
                    "mesh_hash": obj.mesh_hash,
                    "scale": obj.scale,
                    "mesh_path": obj.mesh_path,
                    "grasp_path": obj.grasp_path,
                    "split": obj.split,
                },
                grasps=load_grasps(obj.grasp_path, success_only=True),
            )
            self.loaded_dishware.append(item)
            self.actors_by_category[obj.category].append(actor)

        self.all_dishware = Actor.merge(self.dishware, name="all_dishware")

        hx, hy, hz = self.tray_half_sizes
        self.tray = actors.build_box(
            self.scene,
            half_sizes=[hx, hy, hz],
            color=[0.25, 0.25, 0.28, 1.0],
            name="tray",
            body_type="kinematic",
            initial_pose=sapien.Pose(
                p=[self.tray_center_xy[0], self.tray_center_xy[1], hz]
            ),
        )
        self.tray_half_sizes_t = common.to_tensor(self.tray_half_sizes, device=self.device)
        self.dishware_radii_t = common.to_tensor(
            [d.xy_radius for d in self.loaded_dishware], device=self.device
        )
        self.dishware_spawn_z_t = common.to_tensor(
            [d.spawn_z for d in self.loaded_dishware], device=self.device
        )

    @property
    def dishware(self) -> list[Actor]:
        return [d.actor for d in self.loaded_dishware]

    @property
    def dishware_categories(self) -> list[str]:
        return [d.category for d in self.loaded_dishware]

    @property
    def dishware_radii(self) -> list[float]:
        return [d.xy_radius for d in self.loaded_dishware]

    @property
    def dishware_spawn_z(self) -> list[float]:
        return [d.spawn_z for d in self.loaded_dishware]

    @property
    def dishware_meta(self) -> list[DishwareMeta]:
        return [d.meta for d in self.loaded_dishware]

    @property
    def dishware_grasps(self) -> list[np.ndarray]:
        return [d.grasps for d in self.loaded_dishware]

    def get_world_grasps(self, index: int, env_idx: int = 0) -> np.ndarray:
        """Return successful grasps for dishware ``index`` in the world frame.

        ACRONYM grasp transforms are stored in the object frame. This left-multiplies
        by the current object pose: ``T_world_grasp = T_world_obj @ T_obj_grasp``.
        """
        if index < 0 or index >= len(self.loaded_dishware):
            raise IndexError(f"dishware index out of range: {index}")
        item = self.loaded_dishware[index]
        if item.grasps.size == 0:
            return np.zeros((0, 4, 4), dtype=np.float64)

        pose = item.actor.pose
        p = pose.p[env_idx].detach().cpu().numpy()
        q = pose.q[env_idx].detach().cpu().numpy()  # wxyz
        T_world_obj = sapien.Pose(p=p, q=q).to_transformation_matrix()
        return np.asarray(T_world_obj, dtype=np.float64) @ item.grasps

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            tray_xyz = torch.zeros((b, 3))
            tray_xyz[:, 0] = self.tray_center_xy[0] + (torch.rand(b) * 0.06 - 0.03)
            tray_xyz[:, 1] = self.tray_center_xy[1] + (torch.rand(b) * 0.06 - 0.03)
            tray_xyz[:, 2] = self.tray_half_sizes[2]
            self.tray.set_pose(Pose.create_from_pq(p=tray_xyz, q=[1, 0, 0, 0]))

            sampler = randomization.UniformPlacementSampler(
                bounds=self.spawn_region, batch_size=b, device=self.device
            )
            for item in self.loaded_dishware:
                xy = sampler.sample(item.xy_radius, max_trials=100, verbose=False)
                xyz = torch.zeros((b, 3))
                xyz[:, :2] = xy
                xyz[:, 2] = item.spawn_z
                qs = randomization.random_quaternions(
                    b, lock_x=True, lock_y=True, lock_z=False
                )
                item.actor.set_pose(Pose.create_from_pq(p=xyz, q=qs))

    def _item_on_tray(self, item: Actor) -> torch.Tensor:
        """True if item xy is inside the tray and its center is above the tray surface."""
        return item_centers_on_tray(
            item.pose.p, self.tray.pose.p, self.tray_half_sizes_t
        )

    def _stack_per_dish(
        self, fn, *, dim: int = -1
    ) -> torch.Tensor:
        """Stack a per-item tensor computation across ``self.dishware``."""
        return torch.stack([fn(item) for item in self.dishware], dim=dim)

    def evaluate(self) -> EvalInfo:
        on_tray = self._stack_per_dish(self._item_on_tray, dim=-1)  # (B, N)
        all_on_tray = torch.all(on_tray, dim=-1)
        static_flags = self._stack_per_dish(
            lambda item: item.is_static(lin_thresh=1e-2, ang_thresh=0.5), dim=-1
        )
        all_static = torch.all(static_flags, dim=-1)
        success = all_on_tray & all_static
        return {
            "success": success,
            "all_on_tray": all_on_tray,
            "all_static": all_static,
            "frac_on_tray": on_tray.float().mean(dim=-1),
            "dishware_meta": self.dishware_meta,
            "dishware_grasp_counts": [int(g.shape[0]) for g in self.dishware_grasps],
        }

    def _get_obs_extra(self, info: dict) -> ObsExtra:
        obs: ObsExtra = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            tray_pose=self.tray.pose.raw_pose,
        )
        if self.obs_mode_struct.use_state:
            # Flatten (B, N, ...) -> (B, N * ...) so state-dict obs can be hstacked.
            poses = self._stack_per_dish(
                lambda item: item.pose.raw_pose, dim=1
            ).flatten(start_dim=1)
            to_tray = self._stack_per_dish(
                lambda item: self.tray.pose.p - item.pose.p, dim=1
            ).flatten(start_dim=1)
            obs.update(dishware_poses=poses, dishware_to_tray=to_tray)
        return obs

    def compute_dense_reward(
        self, obs: Any, action: Array, info: EvalInfo
    ) -> torch.Tensor:
        # Encourage reaching the nearest off-tray item, then placing items onto the tray.
        tray_p = self.tray.pose.p
        tcp_p = self.agent.tcp.pose.p

        dists_tcp = self._stack_per_dish(
            lambda item: torch.linalg.norm(item.pose.p - tcp_p, axis=1), dim=-1
        )
        on_tray = self._stack_per_dish(self._item_on_tray, dim=-1)
        masked = dists_tcp.clone()
        masked[on_tray] = 1e6
        nearest = masked.min(dim=-1).values
        all_placed = on_tray.all(dim=-1)
        reaching = 1 - torch.tanh(5 * nearest)
        reaching = torch.where(all_placed, torch.ones_like(reaching), reaching)

        place_dists = self._stack_per_dish(
            lambda item: torch.linalg.norm(item.pose.p[:, :2] - tray_p[:, :2], axis=1),
            dim=-1,
        )
        place_reward = (1 - torch.tanh(5 * place_dists)).mean(dim=-1)

        reward = reaching + 2 * place_reward + 2 * info["frac_on_tray"]
        reward[info["success"]] = 6
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: Array, info: dict[str, Any]
    ) -> torch.Tensor:
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 6.0
