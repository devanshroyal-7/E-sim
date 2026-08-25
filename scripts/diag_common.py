"""Shared helpers for the throwaway drift diagnostics on this branch.

Every check here is about one question: the search builds each edge as
`set_state(parent) + K env.steps`, but `execute_plan` / heatmap replay run the
same actions open-loop. Those two transition models are not the same under
contact, so a plan that hits intersection>=0.9 in the search can miss the goal
when you watch it in the viewer.
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from geometry import TEE_LANDMARKS_XY, landmarks_world_xy, to_numpy, yaw_from_quat
from planner import SimPlanner

ENV_KWARGS = dict(obs_mode="state_dict", control_mode="pd_ee_delta_pose")
DEFAULT_K = 10
DEFAULT_STEP = 0.2
DEFAULT_SEED = 0
DEFAULT_PLAN_PATH = ROOT / "last_plan.txt"

_META_RE = re.compile(
    r"^\s*#\s*(step_size|k_substeps|seed)\s*=\s*"
    r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*$"
)


def _parse_plan_text(text: str) -> list[int]:
    tokens = re.findall(r"-?\d+", text)
    if not tokens:
        raise SystemExit(f"Plan must contain at least one action index, got {text!r}")
    return [int(t) for t in tokens]


def _parse_plan_file(path: Path) -> tuple[list[int], dict]:
    meta: dict = {}
    action_lines: list[str] = []
    for line in path.read_text().splitlines():
        match = _META_RE.match(line)
        if match is not None:
            key, raw = match.group(1), match.group(2)
            meta[key] = float(raw) if key == "step_size" else int(float(raw))
            continue
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        action_lines.append(stripped)
    if not action_lines:
        raise SystemExit(f"No action indices found in {path}")
    return _parse_plan_text("\n".join(action_lines)), meta


@dataclass
class Snapshot:
    obj: np.ndarray
    tcp: np.ndarray
    intersection: float
    tee_speed: float
    tee_omega: float
    qvel: float
    sim_state: object | None = None


@dataclass
class Rollout:
    snaps: list[Snapshot] = field(default_factory=list)

    @property
    def obj(self):
        return np.stack([s.obj for s in self.snaps])

    @property
    def tcp(self):
        return np.stack([s.tcp for s in self.snaps])

    @property
    def intersection(self):
        return np.array([s.intersection for s in self.snaps])


def parse_plan_source(values: list[str] | None, plan_file: Path | None):
    """Load (actions, meta) from a file, CLI tokens, or last_plan.txt."""
    if plan_file is not None:
        return _parse_plan_file(plan_file)
    if values:
        candidate = Path(values[0])
        if len(values) == 1 and candidate.is_file():
            return _parse_plan_file(candidate)
        return _parse_plan_text(" ".join(values)), {}
    if DEFAULT_PLAN_PATH.is_file():
        return _parse_plan_file(DEFAULT_PLAN_PATH)
    raise SystemExit(
        f"No plan given and {DEFAULT_PLAN_PATH} not found. "
        "Pass a plan file or action indices."
    )


def add_plan_args(parser: argparse.ArgumentParser, *, need_search=False):
    parser.add_argument(
        "plan",
        nargs="*",
        help="Action indices or a plan file. Omit to load last_plan.txt.",
    )
    parser.add_argument("--plan-file", type=Path, default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Env reset seed (default: value saved with the plan, else 0)",
    )
    parser.add_argument("--step-size", type=float, default=None)
    parser.add_argument("--k-substeps", type=int, default=None)
    if need_search:
        parser.add_argument("--max-expansions", type=int, default=5000)
        parser.add_argument("--threshold", type=float, default=0.90)
    return parser


def resolve_run(args):
    plan, meta = parse_plan_source(args.plan, args.plan_file)
    step_size = (
        args.step_size
        if args.step_size is not None
        else float(meta.get("step_size", DEFAULT_STEP))
    )
    k_substeps = (
        args.k_substeps
        if args.k_substeps is not None
        else int(meta.get("k_substeps", DEFAULT_K))
    )
    seed = args.seed if args.seed is not None else int(meta.get("seed", DEFAULT_SEED))
    return plan, seed, step_size, k_substeps


def setup(args):
    """Env + planner + cloned root state for a diagnostic run."""
    plan, seed, step_size, k_substeps = resolve_run(args)
    env = gym.make("PushT-v1", **ENV_KWARGS, render_mode=None)
    env.reset(seed=seed)
    planner = SimPlanner(env, K_substeps=k_substeps, step_size=step_size)
    root = env.unwrapped.get_state().clone()
    print(
        f"plan len={len(plan)} seed={seed} step_size={step_size} K={k_substeps}\n"
        f"  actions={plan}"
    )
    return env, planner, root, plan, seed, step_size, k_substeps


def obj_pose(env):
    return to_numpy(env.unwrapped.get_obs()["extra"]["obj_pose"]).reshape(-1).copy()


def tcp_pose(env):
    return to_numpy(env.unwrapped.get_obs()["extra"]["tcp_pose"]).reshape(-1).copy()


def intersection(env):
    return float(to_numpy(env.unwrapped.pseudo_render_intersection()).reshape(-1)[0])


def snapshot(env, *, keep_state=False) -> Snapshot:
    v = to_numpy(env.unwrapped.tee.get_linear_velocity()).reshape(-1)
    w = to_numpy(env.unwrapped.tee.get_angular_velocity()).reshape(-1)
    qv = to_numpy(env.unwrapped.agent.robot.get_qvel()).reshape(-1)
    return Snapshot(
        obj=obj_pose(env),
        tcp=tcp_pose(env),
        intersection=intersection(env),
        tee_speed=float(np.linalg.norm(v)),
        tee_omega=float(np.linalg.norm(w)),
        qvel=float(np.linalg.norm(qv)),
        sim_state=env.unwrapped.get_state().clone() if keep_state else None,
    )


def landmark_err(obj, goal_pose):
    return float(
        np.max(np.linalg.norm(landmarks_world_xy(obj) - landmarks_world_xy(goal_pose), axis=-1))
    )


def yaw_deg(pose):
    return float(np.rad2deg(yaw_from_quat(pose[3:7])))


def rollout(
    env,
    planner,
    root,
    plan,
    *,
    restore_every_prim=False,
    settle=0,
    perturb=0.0,
    keep_state=False,
    substep=False,
) -> Rollout:
    """restore_every_prim=True is the search's transition; False is execute_plan."""
    env.unwrapped.set_state(root)
    if perturb:
        s = root.clone()
        # Tee actor is the second 13-wide block; quat starts at index 13+3=16.
        # Nudging a table-workspace quat slot (index 4) is also enough to tickle
        # float32 chaos, but the Tee quat is the physically relevant one.
        s[0, 16] = s[0, 16] + perturb
        env.unwrapped.set_state(s)
    out = Rollout()
    out.snaps.append(snapshot(env, keep_state=keep_state))
    zero = np.zeros_like(planner.action_primitives[0])
    state = root
    for act_idx in plan:
        if restore_every_prim:
            env.unwrapped.set_state(state)
        action = planner.action_primitives[act_idx]
        for _ in range(planner.K):
            env.step(action)
            if substep:
                out.snaps.append(snapshot(env, keep_state=keep_state))
        for _ in range(settle):
            env.step(zero)
            if substep:
                out.snaps.append(snapshot(env, keep_state=keep_state))
        state = env.unwrapped.get_state().clone()
        if not substep:
            out.snaps.append(snapshot(env, keep_state=keep_state))
    return out


def point_in_tee_body(pts):
    box1 = (np.abs(pts[:, 0]) <= 0.1) & (pts[:, 1] >= -0.0625) & (pts[:, 1] <= -0.0125)
    box2 = (np.abs(pts[:, 0]) <= 0.025) & (pts[:, 1] >= -0.0125) & (pts[:, 1] <= 0.1375)
    return box1 | box2


def exact_intersection(obj, goal, n=200):
    """Goal-T area covered by the current T, by dense sampling of the goal body."""
    lo, hi = TEE_LANDMARKS_XY.min(0), TEE_LANDMARKS_XY.max(0)
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], n), np.linspace(lo[1], hi[1], n))
    body = np.stack([gx.ravel(), gy.ravel()], axis=1)
    goal_pts = body[point_in_tee_body(body)]

    def rot(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s], [s, c]])

    world = goal_pts @ rot(float(yaw_from_quat(goal[3:7]))).T + goal[:2]
    local = (world - obj[:2]) @ rot(float(yaw_from_quat(obj[3:7])))
    return float(point_in_tee_body(local).mean())
