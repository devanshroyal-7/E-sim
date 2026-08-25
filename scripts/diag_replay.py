"""Compare search transitions against a straight replay of the same actions.

Same measurement as diag_roundtrip, plus: a second restore-each-prim pass
must match the first (search is deterministic), and a fresh env with the
same seed must start at the same T pose.

    uv run scripts/diag_replay.py
"""
import argparse
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_common import ENV_KWARGS, add_plan_args, landmark_err, obj_pose, rollout, setup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_plan_args(parser)
    args = parser.parse_args()
    env, planner, root, plan, seed, *_ = setup(args)
    try:
        search = rollout(env, planner, root, plan, restore_every_prim=True)
        replay = rollout(env, planner, root, plan, restore_every_prim=False)
        again = rollout(env, planner, root, plan, restore_every_prim=True)

        print("\nstep | search inter | replay inter | repeat inter | |dxy| search-replay")
        for i, s in enumerate(search.snaps):
            r, a = replay.snaps[i], again.snaps[i]
            d = 1000 * np.linalg.norm(s.obj[:2] - r.obj[:2])
            print(
                f"{i:4d} | {s.intersection:12.4f} | {r.intersection:12.4f} | "
                f"{a.intersection:12.4f} | {d:8.2f} mm"
            )
        print(
            "search vs second restore-pass identical: "
            f"{np.allclose(search.obj, again.obj, atol=1e-9)}"
        )
        goal = planner.goal_pose
        print(
            f"search landmark err {landmark_err(search.snaps[-1].obj, goal):.4f} m, "
            f"replay {landmark_err(replay.snaps[-1].obj, goal):.4f} m"
        )
    finally:
        env.close()

    env2 = gym.make("PushT-v1", **ENV_KWARGS, render_mode=None)
    env2.reset(seed=seed)
    p2 = obj_pose(env2)
    print(f"\nfresh env same seed obj pose {p2}")
    print(f"  delta from first env: {np.abs(p2 - search.snaps[0].obj).max():.2e}")
    env2.close()


if __name__ == "__main__":
    main()
