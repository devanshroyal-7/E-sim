"""Plan and render a PushT-v1 episode with SimPlanner."""
import argparse
from pathlib import Path

import gymnasium as gym
import mani_skill.envs
import numpy as np

from plan_io import save_plan_result
from planner import SimPlanner
from search_checkpoint import Checkpointer, fingerprint, load_checkpoint
from search_log import format_report
from search_recorder import SearchRecorder
from viz_search import plot_expansion_heatmap

ENV_KWARGS = dict(
    obs_mode="state_dict",
    control_mode="pd_ee_delta_pose",
)
K_SUBSTEPS = 10
STEP_SIZE = 0.2
SEED = 0
PLAN_PATH = Path(__file__).resolve().parent / "last_plan.txt"
RESULT_NPZ_PATH = Path(__file__).resolve().parent / "last_plan_result.npz"
CHECKPOINT_PATH = (
    Path(__file__).resolve().parent / "results" / "checkpoints" / "last_search.npz"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument(
        "--resume",
        nargs="?",
        const=str(CHECKPOINT_PATH),
        default=None,
        metavar="PATH",
        help=(
            "Continue a checkpointed search instead of starting from the root. "
            "--max-expansions is the total budget, so resuming a 15000-expansion "
            f"checkpoint with 30000 runs 15000 more (default: {CHECKPOINT_PATH})"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT_PATH,
        help=f"Where to write the search checkpoint (default: {CHECKPOINT_PATH})",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=2000,
        metavar="N",
        help="Autosave every N expansions; 0 saves only at the end (default: 2000)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Resume even if the search parameters or heuristic have changed",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Env reset seed (default: {SEED})",
    )
    args = parser.parse_args()

    plan_env = gym.make("PushT-v1", **ENV_KWARGS, render_mode=None)
    plan_env.reset(seed=args.seed)
    initial_state = plan_env.unwrapped.get_state().clone()

    planner = SimPlanner(plan_env, K_substeps=K_SUBSTEPS, step_size=STEP_SIZE)
    # Hashes the env's state as the search root, so this has to happen while the
    # env is still where reset left it.
    search_id = fingerprint(planner, args.threshold)

    resume = None
    if args.resume:
        resume = load_checkpoint(args.resume, search_id, force=args.force)
        print(
            f"Resuming {args.resume} at {resume.expansions} expansions "
            f"({len(resume.open_nodes['g'])} open)"
        )

    recorder = SearchRecorder()
    checkpointer = Checkpointer(
        args.checkpoint, search_id, recorder, every=args.checkpoint_every
    )
    print(
        f"Planning with threshold={args.threshold}, "
        f"max_expansions={args.max_expansions}, seed={args.seed}"
    )
    result = planner.plan(
        max_expansions=args.max_expansions,
        threshold_value=args.threshold,
        recorder=recorder,
        resume=resume,
        checkpointer=checkpointer,
    )
    plan = result.actions
    print(f"Plan length: {len(plan)}")
    print(f"Plan action indices: {plan}")
    print(
        f"Expansions: {len(result.expansion_xy)}, "
        f"unique tee xy≈{len(np.unique(np.round(result.expansion_xy, 3), axis=0))}, "
        f"traj Δ={np.linalg.norm(result.trajectory_xy[-1] - result.trajectory_xy[0]):.4f} m"
    )
    print(format_report(result.log, actions=plan))

    PLAN_PATH.write_text(
        f"# step_size={planner.step_size}\n"
        f"# k_substeps={planner.K}\n"
        f"# seed={args.seed}\n"
        + ",".join(str(a) for a in plan)
        + "\n"
    )
    print(
        f"Saved plan to {PLAN_PATH} "
        f"(step_size={planner.step_size}, k_substeps={planner.K})"
    )

    save_plan_result(result, RESULT_NPZ_PATH)
    print(f"Saved plan arrays to {RESULT_NPZ_PATH}")

    heatmap_path = plot_expansion_heatmap(result)
    print(f"Saved expansion heatmap to {heatmap_path}")
    print(f"Replay with: uv run scripts/execute_plan.py")
    print(
        f"Search further with: uv run main.py --resume {args.checkpoint} "
        f"--max-expansions {args.max_expansions * 2} --seed {args.seed}"
    )
    plan_env.close()

    render_env = gym.make("PushT-v1", **ENV_KWARGS, render_mode="human")
    render_env.reset(seed=args.seed)
    render_planner = SimPlanner(render_env, K_substeps=K_SUBSTEPS, step_size=STEP_SIZE)
    render_planner.execute_plan(initial_state, plan)
    render_env.close()


if __name__ == "__main__":
    main()
