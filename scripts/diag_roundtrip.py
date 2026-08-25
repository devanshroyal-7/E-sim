"""Search-style restore-each-prim vs execute_plan open-loop vs heatmap replay.

    uv run scripts/diag_roundtrip.py
    uv run scripts/diag_roundtrip.py --seed 2 last_plan.txt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_common import add_plan_args, setup
from diag_drift import check_nudge, check_roundtrip


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_plan_args(parser)
    args = parser.parse_args()
    env, planner, root, plan, *_ = setup(args)
    try:
        search, execute, _ = check_roundtrip(env, planner, root, plan)
        check_nudge(env, planner, root, plan, execute, search)
    finally:
        env.close()


if __name__ == "__main__":
    main()
