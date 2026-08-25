"""Re-run A* and check the returned action list rebuilds the goal node.

Slow: actually plans. Use this when you want to prove the search itself is
self-consistent, not just that a saved plan is chaotic under open-loop replay.

    uv run scripts/diag_chain.py --max-expansions 5000 --seed 2
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_common import add_plan_args, setup
from diag_drift import check_search


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_plan_args(parser, need_search=True)
    args = parser.parse_args()
    env, planner, root, *_ = setup(args)
    try:
        env.unwrapped.set_state(root)
        check_search(env, planner, args)
    finally:
        env.close()


if __name__ == "__main__":
    main()
