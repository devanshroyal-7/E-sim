"""Replay any PushT plan with SimPlanner.execute_plan.

Examples:
  uv run scripts/execute_plan.py                  # loads last_plan.txt
  uv run scripts/execute_plan.py last_plan.txt
  uv run scripts/execute_plan.py 6 6 6 6 6 6 6 6
  uv run scripts/execute_plan.py 6,6,6,6 --delay 0.05
  uv run scripts/execute_plan.py "[6, 6, 6, 6]"
"""
import argparse
import re
import sys
import time
from pathlib import Path

import gymnasium as gym
import mani_skill.envs

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from planner import SimPlanner

ENV_KWARGS = dict(
    obs_mode="state_dict",
    control_mode="pd_ee_delta_pose",
)
K_SUBSTEPS = 10
STEP_SIZE = 0.04
SEED = 0
DEFAULT_PLAN_PATH = Path(__file__).resolve().parents[1] / "last_plan.txt"

_META_RE = re.compile(
    r"^\s*#\s*(step_size|k_substeps)\s*=\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*$"
)


def _parse_plan_text(text: str) -> list[int]:
    """Parse integers from CSV, space-separated, or Python-list-like text."""
    tokens = re.findall(r"-?\d+", text)
    if not tokens:
        raise argparse.ArgumentTypeError(
            f"Plan must contain at least one action index, got {text!r}"
        )
    return [int(t) for t in tokens]


def _parse_plan_file(path: Path) -> tuple[list[int], dict]:
    """Load action indices plus optional `# step_size=` / `# k_substeps=` headers."""
    meta: dict = {}
    action_lines: list[str] = []
    for line in path.read_text().splitlines():
        match = _META_RE.match(line)
        if match is not None:
            key, raw = match.group(1), match.group(2)
            meta[key] = float(raw) if key == "step_size" else int(float(raw))
            continue
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        action_lines.append(stripped)
    if not action_lines:
        raise argparse.ArgumentTypeError(f"No action indices found in {path}")
    return _parse_plan_text("\n".join(action_lines)), meta


def _parse_plan(
    values: list[str] | None, plan_file: Path | None
) -> tuple[list[int], dict]:
    if plan_file is not None:
        return _parse_plan_file(plan_file)

    if not values:
        if DEFAULT_PLAN_PATH.is_file():
            return _parse_plan_file(DEFAULT_PLAN_PATH)
        raise argparse.ArgumentTypeError(
            f"No plan given and {DEFAULT_PLAN_PATH.name} not found. "
            "Pass indices or a plan file path."
        )

    if len(values) == 1:
        candidate = Path(values[0])
        if candidate.is_file():
            return _parse_plan_file(candidate)
        return _parse_plan_text(values[0]), {}

    return _parse_plan_text(" ".join(values)), {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "plan",
        nargs="*",
        help=(
            "Action indices, a plan file, or omit to load last_plan.txt. "
            "Accepts '6 6 6', '6,6,6', or '[6, 6, 6]'."
        ),
    )
    parser.add_argument(
        "--plan-file",
        type=Path,
        default=None,
        help="Path to a saved plan file (CSV / list text)",
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
        "--step-size",
        type=float,
        default=None,
        help=(
            "Primitive step size (default: value saved with the plan, "
            f"else {STEP_SIZE})"
        ),
    )
    parser.add_argument(
        "--k-substeps",
        type=int,
        default=None,
        help=(
            "Env steps per primitive (default: value saved with the plan, "
            f"else {K_SUBSTEPS})"
        ),
    )
    args = parser.parse_args()
    plan, meta = _parse_plan(args.plan, args.plan_file)

    step_size = (
        args.step_size
        if args.step_size is not None
        else float(meta.get("step_size", STEP_SIZE))
    )
    k_substeps = (
        args.k_substeps
        if args.k_substeps is not None
        else int(meta.get("k_substeps", K_SUBSTEPS))
    )

    env = gym.make("PushT-v1", **ENV_KWARGS, render_mode="human")
    env.reset(seed=args.seed)
    initial_state = env.unwrapped.get_state().clone()

    planner = SimPlanner(env, K_substeps=k_substeps, step_size=step_size)
    print(
        f"Executing plan of length {len(plan)}: {plan}\n"
        f"  seed={args.seed} step_size={step_size} "
        f"K={k_substeps} delay={args.delay}s"
    )
    planner.execute_plan(
        initial_state,
        plan,
        step_delay=args.delay,
        K=k_substeps,
        step_size=step_size,
    )

    if args.hold > 0:
        time.sleep(args.hold)
    env.close()


if __name__ == "__main__":
    main()
