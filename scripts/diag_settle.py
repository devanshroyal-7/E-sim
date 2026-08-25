"""Does a settle/hold after each primitive make search and execute agree?

    uv run scripts/diag_settle.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_common import add_plan_args, setup
from diag_drift import check_settle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_plan_args(parser)
    args = parser.parse_args()
    env, planner, root, plan, *_ = setup(args)
    try:
        check_settle(env, planner, root, plan)
    finally:
        env.close()


if __name__ == "__main__":
    main()
