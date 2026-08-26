"""MuJoCo PushT + the same SimPlanner used on ManiSkill.

A planar T and a cylinder pusher, dimensioned to match ManiSkill PushT-v1, so
`SimPlanner` can run unchanged: 8 xy primitives, landmark heuristic, 64x64
pseudo-render intersection, get/set of the full physics state.

    uv run eim_mujoco.py
    uv run eim_mujoco.py --smoke
    uv run eim_mujoco.py --max-expansions 2000 --seed 0
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from geometry import landmarks_world_xy, wrap_pi
from plan_io import save_plan_result
from planner import SimPlanner
from search_checkpoint import Checkpointer, fingerprint, load_checkpoint
from search_log import format_report
from search_recorder import SearchRecorder
from viz_search import plot_expansion_heatmap

# --------------------------------------------------------------------------- #
# Matching ManiSkill PushT-v1
# --------------------------------------------------------------------------- #

T_MASS = 0.8
T_HALF_THICKNESS = 0.02
COM_Y = 0.0375
GOAL_OFFSET = np.array([-0.156, -0.1], dtype=np.float64)
GOAL_YAW = (5.0 / 3.0) * np.pi
EE_START_XY = np.array([-0.321, 0.284], dtype=np.float64)
TEE_SPAWNBOX_XLENGTH = 0.2
TEE_SPAWNBOX_YLENGTH = 0.3
TEE_SPAWNBOX_XOFFSET = -0.1
TEE_SPAWNBOX_YOFFSET = -0.1
INTERSECTION_THRESH = 0.90

# Raster matching ManiSkill's pseudo_render_intersection.
RASTER_RES = 64
UV_HALF_WIDTH = 0.15

# Control: clip the commanded EE delta so a step_size=0.2 primitive over K=10
# travels ~80 mm, in the same ballpark as panda_stick + pd_ee_delta_pose.
MAX_EE_SPEED = 0.20  # m/s
TIMESTEP = 0.002
FRAME_SKIP = 20  # 40 ms per env step

K_SUBSTEPS = 10
STEP_SIZE = 0.2
SEED = 0
PLAN_PATH = Path(__file__).resolve().parent / "last_mujoco_plan.txt"
RESULT_NPZ_PATH = Path(__file__).resolve().parent / "last_mujoco_plan_result.npz"
CHECKPOINT_PATH = (
    Path(__file__).resolve().parent / "results" / "checkpoints" / "last_mujoco_search.npz"
)

# T bar (horizontal) and stem (vertical), COM-centered, same as ManiSkill.
_MJCF = f"""
<mujoco model="pusht">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{TIMESTEP}" gravity="0 0 0" cone="elliptic"
          impratio="10" iterations="50" ls_iterations="20" noslip_iterations="3"/>
  <statistic center="-0.15 0.05 0" extent="0.55"/>

  <visual>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.4 0.4 0.4" specular="0.1 0.1 0.1"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="120" elevation="-50"/>
  </visual>

  <default>
    <geom condim="3" solref="0.015 1" solimp="0.9 0.95 0.001"
          friction="1.4 0.05 0.001"/>
    <joint limited="false"/>
  </default>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.92 0.92 0.90"
             rgb2="0.84 0.84 0.82" width="256" height="256"/>
    <material name="grid" texture="grid" texrepeat="8 8" texuniform="true"
              reflectance="0.02"/>
  </asset>

  <worldbody>
    <light directional="true" diffuse="0.55 0.55 0.55" specular="0.2 0.2 0.2"
           pos="0 0 4" dir="0 0 -1"/>
    <light directional="true" diffuse="0.25 0.25 0.28" pos="1.5 1.0 2.5"
           dir="-0.4 -0.3 -1"/>
    <camera name="top" pos="0 0 0.95" xyaxes="1 0 0  0 1 0"/>
    <camera name="oblique" pos="0.42 -0.38 0.55"
            xyaxes="0.75 0.66 0  -0.28 0.32 0.90"/>

    <geom name="table" type="box" pos="0 0 -0.04" size="0.6 0.6 0.04"
          material="grid" contype="0" conaffinity="0"/>

    <body name="goal_tee" pos="{GOAL_OFFSET[0]} {GOAL_OFFSET[1]} 0.0008"
          euler="0 0 {GOAL_YAW}">
      <geom name="goal_bar" type="box" pos="0 {-COM_Y} 0"
            size="0.1 0.025 0.0004" rgba="0.45 0.45 0.48 0.45"
            contype="0" conaffinity="0"/>
      <geom name="goal_stem" type="box" pos="0 0.0625 0"
            size="0.025 0.075 0.0004" rgba="0.45 0.45 0.48 0.45"
            contype="0" conaffinity="0"/>
    </body>

    <body name="ee_goal" pos="{EE_START_XY[0]} {EE_START_XY[1]} 0.001"
          euler="0 {np.pi/2} 0">
      <geom type="cylinder" size="0.02 0.0004" rgba="0.45 0.45 0.48 0.5"
            contype="0" conaffinity="0"/>
    </body>

    <body name="tee" pos="0 0 {T_HALF_THICKNESS}">
      <inertial pos="0 0 0" mass="{T_MASS}" diaginertia="0.0045 0.0045 0.008"/>
      <joint name="tee_x" type="slide" axis="1 0 0" damping="0.35" frictionloss="0.08"/>
      <joint name="tee_y" type="slide" axis="0 1 0" damping="0.35" frictionloss="0.08"/>
      <joint name="tee_yaw" type="hinge" axis="0 0 1" damping="0.02" frictionloss="0.004"/>
      <geom name="tee_bar" type="box" pos="0 {-COM_Y} 0" size="0.1 0.025 {T_HALF_THICKNESS}"
            rgba="0.761 0.075 0.086 1" friction="1.6 0.08 0.002"/>
      <geom name="tee_stem" type="box" pos="0 0.0625 0" size="0.025 0.075 {T_HALF_THICKNESS}"
            rgba="0.761 0.075 0.086 1" friction="1.6 0.08 0.002"/>
    </body>

    <body name="pusher" mocap="true" pos="-0.321 0.284 0.03">
      <geom name="pusher_cyl" type="cylinder" size="0.02 0.028"
            rgba="0.15 0.35 0.85 1" friction="1.8 0.1 0.002"/>
    </body>
  </worldbody>
</mujoco>
"""


def _quat_wxyz_from_yaw(yaw: float) -> np.ndarray:
    h = 0.5 * float(yaw)
    return np.array([np.cos(h), 0.0, 0.0, np.sin(h)], dtype=np.float64)


def _pose7(xy, z, yaw) -> np.ndarray:
    p = np.empty(7, dtype=np.float64)
    p[0] = xy[0]
    p[1] = xy[1]
    p[2] = z
    p[3:] = _quat_wxyz_from_yaw(yaw)
    return p


def _rigid_xy(xy, yaw) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    m = np.eye(3, dtype=np.float64)
    m[0, 0], m[0, 1], m[0, 2] = c, -s, xy[0]
    m[1, 0], m[1, 1], m[1, 2] = s, c, xy[1]
    return m


def _build_tee_raster():
    """Egocentric T occupancy, same construction as ManiSkill PushTEnv."""
    res = RASTER_RES
    box1 = np.array(
        [[-0.1, 0.025], [0.1, 0.025], [-0.1, -0.025], [0.1, -0.025]],
        dtype=np.float64,
    )
    box2 = np.array(
        [[-0.025, 0.175], [0.025, 0.175], [-0.025, 0.025], [0.025, 0.025]],
        dtype=np.float64,
    )
    box1[:, 1] -= COM_Y
    box2[:, 1] -= COM_Y
    scale = (res / 2) / UV_HALF_WIDTH
    for box in (box1, box2):
        box *= scale
        box += res / 2
    box1 = box1.astype(np.int64)
    box2 = box2.astype(np.int64)
    mask = np.zeros((res, res), dtype=np.float64)
    mask.T[box1[0, 0] : box1[1, 0], box1[2, 1] : box1[0, 1]] = 1.0
    mask.T[box2[0, 0] : box2[1, 0], box2[2, 1] : box2[0, 1]] = 1.0
    mask = np.flip(mask, axis=0)

    oned = np.arange(res, dtype=np.float64).reshape(1, res).repeat(res, axis=0) - (
        res / 2
    )
    uv = np.stack([oned, -oned.T], axis=0)
    uv = (uv + 0.5) / ((res / 2) / UV_HALF_WIDTH)
    homo = np.concatenate([uv, np.ones((1, res, res), dtype=np.float64)], axis=0)
    return mask, homo


_TEE_MASK, _HOMO_UV = _build_tee_raster()
_GOAL_AREA = float(_TEE_MASK.astype(bool).sum())


def raster_intersection(obj_xy, obj_yaw, goal_xy, goal_yaw) -> float:
    """Fraction of the goal T covered by the object T, 64x64 raster."""
    world_to_goal = np.linalg.inv(_rigid_xy(goal_xy, goal_yaw))
    tee_to_goal = world_to_goal @ _rigid_xy(obj_xy, obj_yaw)
    res = RASTER_RES
    mapped = tee_to_goal @ _HOMO_UV.reshape(3, -1)
    mapped = mapped[:2] / mapped[2]
    mapped = mapped.reshape(2, res, res)
    coords = mapped[:, _TEE_MASK.astype(bool)]
    idx = (coords * ((res / 2) / UV_HALF_WIDTH) + (res / 2)).astype(np.int64)
    valid = (
        (idx[0] >= 0) & (idx[0] < res) & (idx[1] >= 0) & (idx[1] < res)
    )
    canvas = np.zeros((res, res), dtype=np.float64)
    canvas[idx[0, valid], idx[1, valid]] = 1.0
    # ManiSkill: (x, y) image -> transpose then flip y so it lines up with tee_render.
    canvas = np.flip(canvas.T, axis=0)
    return float((canvas.astype(bool) & _TEE_MASK.astype(bool)).sum()) / _GOAL_AREA


class PushTMujocoEnv(gym.Env):
    """Duck-types the ManiSkill PushT bits SimPlanner actually calls."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None, max_ee_speed=MAX_EE_SPEED):
        super().__init__()
        self.render_mode = render_mode
        self.max_ee_speed = float(max_ee_speed)
        self.frame_skip = FRAME_SKIP
        self.dt_ctrl = TIMESTEP * FRAME_SKIP
        self.intersection_thresh = INTERSECTION_THRESH

        self.model = mujoco.MjModel.from_xml_string(_MJCF)
        self.data = mujoco.MjData(self.model)
        self._state_sig = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        self._state_n = mujoco.mj_stateSize(self.model, self._state_sig)

        self._adr = {
            "tee_x": int(self.model.jnt_qposadr[self.model.joint("tee_x").id]),
            "tee_y": int(self.model.jnt_qposadr[self.model.joint("tee_y").id]),
            "tee_yaw": int(self.model.jnt_qposadr[self.model.joint("tee_yaw").id]),
        }

        goal_pose = _pose7(GOAL_OFFSET, 1e-3, GOAL_YAW)
        self.goal_tee = SimpleNamespace(
            pose=SimpleNamespace(raw_pose=goal_pose, p=goal_pose[:3], q=goal_pose[3:])
        )

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(
            {
                "extra": spaces.Dict(
                    {
                        "obj_pose": spaces.Box(-np.inf, np.inf, (7,), np.float64),
                        "tcp_pose": spaces.Box(-np.inf, np.inf, (7,), np.float64),
                    }
                )
            }
        )
        self._viewer = None
        self._renderer = None
        self._np_random = np.random.default_rng()

    # -- state --------------------------------------------------------------

    def get_state(self) -> np.ndarray:
        state = np.empty(self._state_n, dtype=np.float64)
        mujoco.mj_getState(self.model, self.data, state, self._state_sig)
        return state

    def set_state(self, state):
        state = np.asarray(state, dtype=np.float64).reshape(-1)
        mujoco.mj_setState(self.model, self.data, state, self._state_sig)
        mujoco.mj_forward(self.model, self.data)

    def _tee_xy(self) -> np.ndarray:
        return np.array(
            [self.data.qpos[self._adr["tee_x"]], self.data.qpos[self._adr["tee_y"]]],
            dtype=np.float64,
        )

    def _tee_yaw(self) -> float:
        return wrap_pi(float(self.data.qpos[self._adr["tee_yaw"]]))

    def _pusher_xy(self) -> np.ndarray:
        return np.array(self.data.mocap_pos[0, :2], dtype=np.float64)

    def _obj_pose(self) -> np.ndarray:
        return _pose7(self._tee_xy(), T_HALF_THICKNESS, self._tee_yaw())

    def _tcp_pose(self) -> np.ndarray:
        xy = self._pusher_xy()
        return _pose7(xy, 0.03, 0.0)

    def get_obs(self):
        return {
            "extra": {
                "obj_pose": self._obj_pose(),
                "tcp_pose": self._tcp_pose(),
            }
        }

    def pseudo_render_intersection(self):
        xy = self._tee_xy()
        yaw = self._tee_yaw()
        cov = raster_intersection(xy, yaw, GOAL_OFFSET, GOAL_YAW)
        return np.array([cov], dtype=np.float64)

    # -- gym ----------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)

        rng = self._np_random
        tee_xy = GOAL_OFFSET.copy()
        tee_xy[0] += rng.random() * TEE_SPAWNBOX_XLENGTH + TEE_SPAWNBOX_XOFFSET
        tee_xy[1] += rng.random() * TEE_SPAWNBOX_YLENGTH + TEE_SPAWNBOX_YOFFSET
        tee_yaw = rng.random() * 2.0 * np.pi

        self.data.qpos[self._adr["tee_x"]] = tee_xy[0]
        self.data.qpos[self._adr["tee_y"]] = tee_xy[1]
        self.data.qpos[self._adr["tee_yaw"]] = tee_yaw
        self.data.qvel[:] = 0.0
        self.data.mocap_pos[0, :2] = EE_START_XY
        self.data.mocap_pos[0, 2] = 0.03
        self.data.mocap_quat[0] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(self.model, self.data)
        return self.get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        delta = action[:2].copy()
        n = float(np.linalg.norm(delta))
        max_delta = self.max_ee_speed * self.dt_ctrl
        if n > max_delta > 0:
            delta *= max_delta / n
        # Spread the mocap move across physics substeps so contacts stay stable.
        step_delta = delta / self.frame_skip
        for _ in range(self.frame_skip):
            self.data.mocap_pos[0, :2] += step_delta
            mujoco.mj_step(self.model, self.data)

        obs = self.get_obs()
        inter = float(self.pseudo_render_intersection()[0])
        terminated = inter >= self.intersection_thresh
        return obs, inter, terminated, False, {"intersection": inter}

    def render(self):
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=512, width=512)
            self._renderer.update_scene(self.data, camera="oblique")
            return self._renderer.render()
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(
                    self.model, self.data, show_left_ui=False, show_right_ui=False
                )
                self._viewer.cam.lookat[:] = [-0.16, 0.05, 0.0]
                self._viewer.cam.distance = 0.95
                self._viewer.cam.azimuth = 90
                self._viewer.cam.elevation = -70
            if self._viewer.is_running():
                self._viewer.sync()
        return None

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        self._renderer = None


gym.register(
    id="PushTMujoco-v0",
    entry_point="eim_mujoco:PushTMujocoEnv",
    max_episode_steps=200,
)


def smoke(env: PushTMujocoEnv, n_check=12):
    """Determinism + a single primitive, so we know pushing actually works."""
    obs, _ = env.reset(seed=SEED)
    root = env.get_state().copy()
    inter0 = float(env.pseudo_render_intersection()[0])
    tcp0 = obs["extra"]["tcp_pose"][:2].copy()
    obj0 = obs["extra"]["obj_pose"][:2].copy()
    print(
        f"reset seed={SEED}: tee={obj0}, tcp={tcp0}, "
        f"inter={inter0:.4f}, nq={env.model.nq} nv={env.model.nv} "
        f"nmocap={env.model.nmocap}"
    )
    print(f"  landmarks[0]={landmarks_world_xy(obs['extra']['obj_pose'])[0]}")

    env.data.qpos[env._adr["tee_x"]] = GOAL_OFFSET[0]
    env.data.qpos[env._adr["tee_y"]] = GOAL_OFFSET[1]
    env.data.qpos[env._adr["tee_yaw"]] = GOAL_YAW
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)
    on_goal = float(env.pseudo_render_intersection()[0])
    print(f"  intersection on goal pose: {on_goal:.4f} (want ~1.0)")
    env.set_state(root)

    action = np.array([0.2, 0.0], dtype=np.float32)
    env.set_state(root)
    for _ in range(n_check):
        env.step(action)
    a = env.get_state().copy()
    pose_a = env._obj_pose().copy()
    tcp_a = env._tcp_pose()[:2].copy()

    env.set_state(root)
    for _ in range(n_check):
        env.step(action)
    b = env.get_state().copy()
    max_diff = float(np.max(np.abs(a - b)))
    if max_diff > 1e-8:
        raise RuntimeError(f"set_state is not deterministic (max abs {max_diff:.3e})")
    print(f"set_state determinism ok (max abs {max_diff:.3e})")
    print(
        f"  after {n_check} steps of +x 0.2: "
        f"Δtcp={np.linalg.norm(tcp_a - tcp0)*1e3:.1f} mm  "
        f"Δtee={np.linalg.norm(pose_a[:2] - obj0)*1e3:.1f} mm  "
        f"inter={float(env.pseudo_render_intersection()[0]):.4f}"
    )

    env.set_state(root)
    planner = SimPlanner(env, K_substeps=K_SUBSTEPS, step_size=STEP_SIZE)
    for act_idx, prim in enumerate(planner.action_primitives):
        env.set_state(root)
        for _ in range(planner.K):
            env.step(prim)
        zeros = np.zeros_like(prim)
        for _ in range(2):
            env.step(zeros)
        d_tcp = np.linalg.norm(env._tcp_pose()[:2] - tcp0)
        d_tee = np.linalg.norm(env._tee_xy() - obj0)
        print(
            f"  prim {act_idx} {prim[:2]}  "
            f"|d_tcp|={d_tcp*1e3:.1f} mm  |d_tee|={d_tee*1e3:.1f} mm"
        )
    env.set_state(root)

    # Park the pusher against the T and shove it in -x so contact is guaranteed.
    env.data.mocap_pos[0, 0] = obj0[0] + 0.16
    env.data.mocap_pos[0, 1] = obj0[1]
    env.data.mocap_pos[0, 2] = 0.03
    mujoco.mj_forward(env.model, env.data)
    yaw0 = env._tee_yaw()
    shove = np.array([-0.2, 0.0], dtype=np.float32)
    for _ in range(20):
        env.step(shove)
    d_tee = np.linalg.norm(env._tee_xy() - obj0)
    d_yaw = wrap_pi(env._tee_yaw() - yaw0)
    print(
        f"  contact shove: |d_tee|={d_tee*1e3:.1f} mm  "
        f"Δyaw={np.rad2deg(d_yaw):+.1f} deg  ncon={env.data.ncon}  "
        f"inter={float(env.pseudo_render_intersection()[0]):.4f}"
    )
    env.set_state(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-expansions", type=int, default=15000)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument(
        "--resume",
        nargs="?",
        const=str(CHECKPOINT_PATH),
        default=None,
        metavar="PATH",
    )
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--step-size", type=float, default=STEP_SIZE)
    parser.add_argument("--k-substeps", type=int, default=K_SUBSTEPS)
    parser.add_argument("--smoke", action="store_true", help="Physics / set_state check")
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Skip the MuJoCo viewer replay after planning",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.02,
        help="Seconds to sleep after each replay substep",
    )
    args = parser.parse_args()

    plan_env = PushTMujocoEnv(render_mode=None)
    plan_env.reset(seed=args.seed)
    initial_state = plan_env.get_state().copy()

    if args.smoke:
        smoke(plan_env)
        plan_env.close()
        return

    planner = SimPlanner(plan_env, K_substeps=args.k_substeps, step_size=args.step_size)
    search_id = fingerprint(planner, args.threshold)

    resume = None
    if args.resume:
        resume = load_checkpoint(args.resume, search_id, force=args.force)
        print(
            f"Resuming {args.resume} at {resume.expansions} expansions "
            f"({len(resume.open_nodes['g'])} open)"
        )

    recorder = SearchRecorder()
    checkpointer = Checkpointer(
        args.checkpoint, search_id, recorder, every=args.checkpoint_every
    )
    print(
        f"MuJoCo PushT  threshold={args.threshold}  "
        f"max_expansions={args.max_expansions}  seed={args.seed}  "
        f"step_size={args.step_size}  K={args.k_substeps}"
    )
    t0 = time.perf_counter()
    result = planner.plan(
        max_expansions=args.max_expansions,
        threshold_value=args.threshold,
        recorder=recorder,
        resume=resume,
        checkpointer=checkpointer,
    )
    elapsed = time.perf_counter() - t0
    plan = result.actions
    print(f"Plan length: {len(plan)}  ({elapsed:.1f}s)")
    print(f"Plan action indices: {plan}")
    print(
        f"Expansions: {len(result.expansion_xy)}, "
        f"unique tee xy≈{len(np.unique(np.round(result.expansion_xy, 3), axis=0))}, "
        f"traj Δ={np.linalg.norm(result.trajectory_xy[-1] - result.trajectory_xy[0]):.4f} m"
    )
    print(format_report(result.log, actions=plan))

    PLAN_PATH.write_text(
        f"# step_size={planner.step_size}\n"
        f"# k_substeps={planner.K}\n"
        f"# seed={args.seed}\n"
        + ",".join(str(a) for a in plan)
        + "\n"
    )
    print(f"Saved plan to {PLAN_PATH}")
    save_plan_result(result, RESULT_NPZ_PATH)
    print(f"Saved plan arrays to {RESULT_NPZ_PATH}")
    heatmap_path = plot_expansion_heatmap(result)
    print(f"Saved expansion heatmap to {heatmap_path}")
    print(
        f"Search further with: uv run eim_mujoco.py --resume {args.checkpoint} "
        f"--max-expansions {args.max_expansions * 2} --seed {args.seed}"
    )
    plan_env.close()

    if args.no_render:
        return

    render_env = PushTMujocoEnv(render_mode="human")
    render_env.reset(seed=args.seed)
    render_planner = SimPlanner(
        render_env, K_substeps=args.k_substeps, step_size=args.step_size
    )
    print("Replaying in MuJoCo viewer (close the window or Ctrl-C to stop)")
    try:
        render_planner.execute_plan(
            initial_state, plan, step_delay=args.delay
        )
        # Hold the last frame so the T pose is visible.
        hold_until = time.time() + 3.0
        while (
            render_env._viewer is not None
            and render_env._viewer.is_running()
            and time.time() < hold_until
        ):
            render_env.render()
            time.sleep(0.03)
    except KeyboardInterrupt:
        pass
    finally:
        render_env.close()


if __name__ == "__main__":
    main()
