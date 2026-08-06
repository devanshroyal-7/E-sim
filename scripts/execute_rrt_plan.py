"""Replay a saved RRT PushT plan with RRTPlanner.execute_plan.

Examples:
  uv run scripts/execute_rrt_plan.py                  # loads last_rrt_plan.txt
  uv run scripts/execute_rrt_plan.py last_rrt_plan.txt
  uv run scripts/execute_rrt_plan.py --delay 0.05
"""
import argparse
import sys
import time
from pathlib import Path

import gymnasium as gym
import mani_skill.envs

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rrt_planner import RRTPlanner, load_plan

ENV_KWARGS = dict(
    obs_mode="state_dict",
    control_mode="pd_ee_delta_pose",
)
K_SUBSTEPS = 10
STEP_SIZE = 0.5
SEED = 0
DEFAULT_PLAN_PATH = Path(__file__).resolve().parents[1] / "last_rrt_plan.txt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "plan_file",
        nargs="?",
        type=Path,
        default=None,
        help=f"Path to a saved RRT plan file (default: {DEFAULT_PLAN_PATH.name})",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.05,
        help="Seconds to sleep after each env step (default: 0.05)",
    )
    parser.add_argument(
        "--hold",
        type=float,
        default=2.0,
        help="Seconds to leave the viewer open after the plan finishes",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Env reset seed (default: {SEED})",
    )
    parser.add_argument(
        "--k-substeps",
        type=int,
        default=K_SUBSTEPS,
        help=f"Env steps per primitive (default: {K_SUBSTEPS})",
    )
    args = parser.parse_args()

    plan_path = args.plan_file or DEFAULT_PLAN_PATH
    if not plan_path.is_file():
        raise SystemExit(f"Plan file not found: {plan_path}")

    env = gym.make("PushT-v1", **ENV_KWARGS, render_mode="human")
    env.reset(seed=args.seed)
    initial_state = env.unwrapped.get_state().clone()

    plan = load_plan(plan_path, env.action_space.shape[-1])
    planner = RRTPlanner(
        env, K_substeps=args.k_substeps, step_size=STEP_SIZE, allow_render=True
    )
    print(
        f"Executing RRT plan of length {len(plan)} from {plan_path}\n"
        f"  seed={args.seed} K={args.k_substeps} delay={args.delay}s"
    )
    planner.execute_plan(initial_state, plan, step_delay=args.delay, K=args.k_substeps)

    if args.hold > 0:
        time.sleep(args.hold)
    env.close()


if __name__ == "__main__":
    main()
