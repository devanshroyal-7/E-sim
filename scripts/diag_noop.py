"""Measure what env.step(zeros) vs env.step(None) does to the EE after a primitive."""
import sys
from pathlib import Path

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geometry import to_numpy
from planner import SimPlanner

ENV_KWARGS = dict(obs_mode="state_dict", control_mode="pd_ee_delta_pose")
K = 10
N_HOLD = 20
SEED = 0


def tcp_xy(env):
    p = to_numpy(env.unwrapped.agent.tcp.pose.p).reshape(-1)
    return np.array([float(p[0]), float(p[1]), float(p[2])])


def qpos(env):
    return to_numpy(env.unwrapped.agent.robot.get_qpos()).reshape(-1).copy()


def drive_targets(env):
    ctrl = env.unwrapped.agent.controller
    # CombinedController wraps arm
    inner = ctrl.controllers["arm"] if hasattr(ctrl, "controllers") else ctrl
    tgt = getattr(inner, "_target_qpos", None)
    pose = getattr(inner, "_target_pose", None)
    return inner, tgt, pose


def main():
    env = gym.make("PushT-v1", **ENV_KWARGS, render_mode=None)
    env.reset(seed=SEED)
    planner = SimPlanner(env, K_substeps=K, step_size=0.2)
    action = planner.action_primitives[0]  # +x
    zero = np.zeros_like(action)

    ctrl, _, _ = drive_targets(env)
    space = env.action_space
    print("control_mode:", env.unwrapped.control_mode)
    print("controller:", type(ctrl).__name__)
    print("  use_delta:", ctrl.config.use_delta, " use_target:", ctrl.config.use_target)
    print("  normalize_action:", ctrl.config.normalize_action)
    print("  pos_lower/upper:", ctrl.config.pos_lower, ctrl.config.pos_upper)
    print("  gym action_space:", space)
    print("  primitive[0] (planner):", action[:3], "  ||a||", float(np.linalg.norm(action[:2])))
    print("  zeros:", zero[:3])

    raw = torch.as_tensor(zero, device=ctrl.device).reshape(1, -1)
    processed = to_numpy(ctrl._preprocess_action(raw)).reshape(-1)
    print("  zeros after _preprocess_action (physical units):", processed[:6])

    raw_a = torch.as_tensor(action, device=ctrl.device).reshape(1, -1)
    processed_a = to_numpy(ctrl._preprocess_action(raw_a)).reshape(-1)
    print("  primitive[0] after _preprocess_action:", processed_a[:6])
    print(
        "  (clipped/scaled pos. gym Box is likely [-1,1]; 0.2 in -> physical 0.1 if clipped)"
    )

    def rollout(hold_kind):
        env.reset(seed=SEED)
        pts = [tcp_xy(env)]
        q0 = qpos(env)
        kinds = []
        for _ in range(K):
            env.step(action)
            pts.append(tcp_xy(env))
            kinds.append("push")
        q_after_push = qpos(env)
        _, tgt_after_push, pose_after_push = drive_targets(env)
        tgt_after_push = to_numpy(tgt_after_push).reshape(-1).copy() if tgt_after_push is not None else None
        pose_after_push = (
            to_numpy(pose_after_push.raw_pose).reshape(-1).copy() if pose_after_push is not None else None
        )
        tcp_at_hold_start = pts[-1].copy()
        for i in range(N_HOLD):
            if hold_kind == "zeros":
                env.step(zero)
            elif hold_kind == "none":
                env.step(None)
            else:
                raise ValueError(hold_kind)
            pts.append(tcp_xy(env))
            kinds.append(hold_kind)
            if i == 0:
                q_first_hold = qpos(env)
                _, tgt_hold, pose_hold = drive_targets(env)
        q_end = qpos(env)
        pts = np.asarray(pts)
        d_push = pts[K] - pts[0]
        d_hold = pts[-1] - pts[K]
        step_hold = np.diff(pts[K:], axis=0)
        step_push = np.diff(pts[: K + 1], axis=0)
        push_dir = d_push[:2] / (np.linalg.norm(d_push[:2]) + 1e-12)
        hold_dir = d_hold[:2] / (np.linalg.norm(d_hold[:2]) + 1e-12)
        cos = float(np.dot(push_dir, hold_dir))
        first_hold = pts[K + 1] - pts[K]
        print(f"\n=== hold={hold_kind!r} ===")
        print(f"  TCP start          {pts[0]}")
        print(f"  TCP after {K} push  {pts[K]}   delta {d_push}  |dxy|={np.linalg.norm(d_push[:2])*1000:.2f} mm")
        print(f"  TCP after {N_HOLD} hold {pts[-1]}   delta {d_hold}  |dxy|={np.linalg.norm(d_hold[:2])*1000:.2f} mm")
        print(f"  first hold step    {first_hold}  |dxy|={np.linalg.norm(first_hold[:2])*1000:.2f} mm  |dz|={abs(first_hold[2])*1000:.2f} mm")
        print(f"  mean |dxy|/push step {np.mean(np.linalg.norm(step_push[:, :2], axis=1))*1000:.2f} mm")
        print(f"  mean |dxy|/hold step {np.mean(np.linalg.norm(step_hold[:, :2], axis=1))*1000:.2f} mm")
        print(f"  max  |dxy|/hold step {np.max(np.linalg.norm(step_hold[:, :2], axis=1))*1000:.2f} mm")
        print(f"  cosine(push_dir, hold_dir) = {cos:+.4f}   (+1 = kept going the same way)")
        print(f"  |qpos| change first hold step {np.linalg.norm(q_first_hold - q_after_push):.4f} rad")
        print(f"  |qpos| change over all holds  {np.linalg.norm(q_end - q_after_push):.4f} rad")
        if tgt_after_push is not None:
            tgt_h = to_numpy(tgt_hold).reshape(-1)
            print(f"  |drive target qpos| jump on first hold {np.linalg.norm(tgt_h - tgt_after_push):.4f} rad")
        if pose_after_push is not None and pose_hold is not None:
            ph = to_numpy(pose_hold.raw_pose).reshape(-1)
            print(f"  EE target pose after push {pose_after_push[:3]}")
            print(f"  EE target pose first hold {ph[:3]}")
            print(f"  |target p| jump {np.linalg.norm(ph[:3] - pose_after_push[:3])*1000:.2f} mm")
        return pts, cos

    rollout("zeros")
    rollout("none")

    # zeros from rest, no prior push
    env.reset(seed=SEED)
    p0 = tcp_xy(env)
    for _ in range(N_HOLD):
        env.step(zero)
    p1 = tcp_xy(env)
    print(f"\n=== zeros from rest (no prior push) ===")
    print(f"  |dxy| over {N_HOLD} steps: {np.linalg.norm((p1-p0)[:2])*1000:.2f} mm  dz={(p1-p0)[2]*1000:.2f} mm")

    env.reset(seed=SEED)
    p0 = tcp_xy(env)
    for _ in range(N_HOLD):
        env.step(None)
    p1 = tcp_xy(env)
    print(f"=== None from rest ===")
    print(f"  |dxy| over {N_HOLD} steps: {np.linalg.norm((p1-p0)[:2])*1000:.2f} mm  dz={(p1-p0)[2]*1000:.2f} mm")

    env.close()


if __name__ == "__main__":
    main()
