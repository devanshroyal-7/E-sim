"""Replay a saved PushT plan slowly so the motion is easy to watch."""
import argparse
import sys
import time
from pathlib import Path

import gymnasium as gym
import mani_skill.envs

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from planner import SimPlanner

# Plan from: uv run main.py (2000 expansions, seed=0)
PLAN = [2, 3, 3, 4, 6, 7]
K_SUBSTEPS = 10
STEP_SIZE = 0.5
SEED = 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
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
    args = parser.parse_args()

    env = gym.make(
        "PushT-v1",
        obs_mode="state_dict",
        control_mode="pd_ee_delta_pose",
        render_mode="human",
    )
    env.reset(seed=SEED)
    initial_state = env.unwrapped.get_state().clone()

    planner = SimPlanner(env, K_substeps=K_SUBSTEPS, step_size=STEP_SIZE)
    print(f"Replaying plan {PLAN} with step_delay={args.delay}s")
    planner.execute_plan(initial_state, PLAN, step_delay=args.delay)

    if args.hold > 0:
        time.sleep(args.hold)
    env.close()


if __name__ == "__main__":
    main()
