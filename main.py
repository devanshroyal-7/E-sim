"""Plan and render a PushT-v1 episode with SimPlanner."""
import argparse
from pathlib import Path

import gymnasium as gym
import mani_skill.envs

from planner import SimPlanner

ENV_KWARGS = dict(
    obs_mode="state_dict",
    control_mode="pd_ee_delta_pose",
)
K_SUBSTEPS = 10
STEP_SIZE = 0.1
PLAN_PATH = Path(__file__).resolve().parent / "last_plan.txt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--open-policy",
        choices=("base", "best-g"),
        default="base",
        help="Open-list duplicate policy (default: base)",
    )
    parser.add_argument(
        "--max-expansions",
        type=int,
        default=15000,
        help="Maximum A* expansions (default: 15000)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.90,
        help="Intersection success threshold (default: 0.90)",
    )
    args = parser.parse_args()

    plan_env = gym.make("PushT-v1", **ENV_KWARGS, render_mode=None)
    plan_env.reset(seed=0)
    initial_state = plan_env.unwrapped.get_state().clone()

    planner = SimPlanner(plan_env, K_substeps=K_SUBSTEPS, step_size=STEP_SIZE)
    print(
        f"Planning with open_policy={args.open_policy}, "
        f"threshold={args.threshold}, max_expansions={args.max_expansions}"
    )
    plan = planner.plan(
        max_expansions=args.max_expansions,
        threshold_value=args.threshold,
        open_policy=args.open_policy,
    )
    print(f"Plan length: {len(plan)}")
    print(f"Plan action indices: {plan}")

    PLAN_PATH.write_text(",".join(str(a) for a in plan) + "\n")
    print(f"Saved plan to {PLAN_PATH}")
    print(f"Replay with: uv run scripts/execute_plan.py")
    plan_env.close()

    render_env = gym.make("PushT-v1", **ENV_KWARGS, render_mode="human")
    render_env.reset(seed=0)
    render_planner = SimPlanner(render_env, K_substeps=K_SUBSTEPS, step_size=STEP_SIZE)
    render_planner.execute_plan(initial_state, plan)
    render_env.close()


if __name__ == "__main__":
    main()
