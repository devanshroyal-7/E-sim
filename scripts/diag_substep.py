"""Env-step (not primitive) resolution: when do search and execute first split?

Replays both transition models, dumping tee xy / intersection every `env.step`.
The first row with |dxy| > 0.1 mm is the moment contact diverges — usually a
few substeps into the primitive where the stick first catches the T.

    uv run scripts/diag_substep.py
    uv run scripts/diag_substep.py --after 6     # only print from primitive 6
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_common import add_plan_args, rollout, setup, yaw_deg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_plan_args(parser)
    parser.add_argument(
        "--after",
        type=int,
        default=0,
        help="Skip printing until this primitive index (0 = start)",
    )
    parser.add_argument(
        "--tol-mm",
        type=float,
        default=0.1,
        help="|dxy| in mm that counts as the first split (default 0.1)",
    )
    args = parser.parse_args()
    env, planner, root, plan, *_ = setup(args)
    try:
        search = rollout(env, planner, root, plan, restore_every_prim=True, substep=True)
        execute = rollout(env, planner, root, plan, restore_every_prim=False, substep=True)
        steps_per_prim = planner.K
        first = None
        start_i = args.after * steps_per_prim
        print(
            "t | prim | sub | search inter | exec inter | |dxy| mm | |d tcp| mm | |d yaw| deg"
        )
        n = min(len(search.snaps), len(execute.snaps))
        for t in range(n):
            if t < start_i:
                continue
            s, e = search.snaps[t], execute.snaps[t]
            prim = 0 if t == 0 else (t - 1) // steps_per_prim
            sub = 0 if t == 0 else (t - 1) % steps_per_prim
            dxy = 1000 * np.linalg.norm(s.obj[:2] - e.obj[:2])
            dtcp = 1000 * np.linalg.norm(s.tcp[:2] - e.tcp[:2])
            dyaw = abs(((yaw_deg(s.obj) - yaw_deg(e.obj) + 180) % 360) - 180)
            mark = ""
            if first is None and dxy > args.tol_mm:
                first = (t, prim, sub, dxy)
                mark = "  <- first split"
            print(
                f"{t:4d} | {prim:4d} | {sub:3d} | {s.intersection:12.4f} | "
                f"{e.intersection:10.4f} | {dxy:7.2f} | {dtcp:8.2f} | "
                f"{dyaw:8.2f}{mark}"
            )
        if first is None:
            print("never split past the tolerance.")
        else:
            t, prim, sub, dxy = first
            print(
                f"\nfirst |dxy| > {args.tol_mm} mm at env-step {t}, "
                f"primitive {prim} substep {sub} ({dxy:.2f} mm). "
                f"that primitive's action index is {plan[prim]}."
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
