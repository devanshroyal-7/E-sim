"""set_state identity + (parent, action) independent of env history.

    uv run scripts/diag_setstate.py
    uv run scripts/diag_setstate.py --seed 3
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_common import add_plan_args, setup
from diag_drift import check_restore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_plan_args(parser)
    args = parser.parse_args()
    env, planner, root, *_ = setup(args)
    try:
        check_restore(env, planner, root)
    finally:
        env.close()


if __name__ == "__main__":
    main()
