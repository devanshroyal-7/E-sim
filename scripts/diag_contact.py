"""Contact chaos vs hidden-state, plus pseudo_render vs exact overlap.

Walks the plan restore-style, reports tee speed / robot |qvel| at every
primitive boundary, then at the first sliding boundary compares:
  continue open-loop
  restore then step
  nudge the stored state by 1e-8 then step

If restore≈nudge >> 0, the restore is not dropping a contact cache — it is
just another 1e-8 kick that contact then amplifies.

    uv run scripts/diag_contact.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_common import add_plan_args, exact_intersection, obj_pose, rollout, setup
from diag_drift import check_metric


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_plan_args(parser)
    args = parser.parse_args()
    env, planner, root, plan, *_ = setup(args)
    try:
        search = rollout(env, planner, root, plan, restore_every_prim=True, keep_state=True)
        print("\nprim | tee speed (m/s) | tee omega (rad/s) | robot |qvel|")
        first_slide = None
        for i, snap in enumerate(search.snaps):
            print(
                f"{i:4d} | {snap.tee_speed:15.5f} | {snap.tee_omega:17.5f} | {snap.qvel:.5f}"
            )
            if first_slide is None and i > 0 and snap.tee_speed > 1e-3:
                first_slide = i

        if first_slide is None:
            print("T never moved. nothing to compare at a contact boundary.")
        else:
            # first_slide is the node *after* the primitive that started motion.
            # Compare continuations of the *parent* of that primitive.
            parent_i = first_slide - 1
            nxt = plan[parent_i]
            parent = search.snaps[parent_i].sim_state

            def go(state):
                env.unwrapped.set_state(state)
                for _ in range(planner.K):
                    env.step(planner.action_primitives[nxt])
                return obj_pose(env)

            restored = go(parent)
            nudged_state = parent.clone()
            nudged_state[0, 16] = nudged_state[0, 16] + 1e-8
            nudged = go(nudged_state)

            env.unwrapped.set_state(root)
            for a in plan[: first_slide]:
                for _ in range(planner.K):
                    env.step(planner.action_primitives[a])
            straight = obj_pose(env)

            print(
                f"\nafter primitive {first_slide} (first contact), tee |dxy| vs open-loop continue:"
            )
            print(f"  restore then step : {1000 * np.linalg.norm(restored[:2] - straight[:2]):.3f} mm")
            print(f"  nudge 1e-8 + step : {1000 * np.linalg.norm(nudged[:2] - straight[:2]):.3f} mm")
            print(
                "  restore≈nudge => restore is a float-level kick, not a dropped contact cache."
            )

        check_metric(env, planner, root, plan, search)
    finally:
        env.close()


if __name__ == "__main__":
    main()
