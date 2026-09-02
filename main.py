"""Plan and render a PushT-v1 episode with SimPlanner (A*) or RRTPlanner (RRT)."""
import argparse
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import mani_skill.envs
import numpy as np

from plan_io import save_plan_result
from planner import SimPlanner
from rrt_checkpoint import (
    Checkpointer as RRTCheckpointer,
    fingerprint as rrt_fingerprint,
    load_checkpoint as rrt_load_checkpoint,
)
from rrt_planner import RRTPlanner, save_plan, verify_set_state_determinism
from rrt_recorder import RRTRecorder
from search_checkpoint import (
    Checkpointer as AStarCheckpointer,
    fingerprint as astar_fingerprint,
    load_checkpoint as astar_load_checkpoint,
)
from search_log import format_report
from search_recorder import SearchRecorder
from viz_search import plot_expansion_heatmap, results_path

ENV_KWARGS = dict(
    obs_mode="state_dict",
    control_mode="pd_ee_delta_pose",
)
K_SUBSTEPS = 10
STEP_SIZE = 0.2
RRT_STEP_SIZE = 0.3
SEED = 0
PLAN_PATH = Path(__file__).resolve().parent / "last_plan.txt"
RESULT_NPZ_PATH = Path(__file__).resolve().parent / "last_plan_result.npz"
CHECKPOINT_PATH = (
    Path(__file__).resolve().parent / "results" / "checkpoints" / "last_search.npz"
)
RRT_PLAN_PATH = Path(__file__).resolve().parent / "last_rrt_plan.txt"
RRT_RESULT_NPZ_PATH = Path(__file__).resolve().parent / "last_rrt_plan_result.npz"
RRT_CHECKPOINT_PATH = (
    Path(__file__).resolve().parent / "results" / "checkpoints" / "last_rrt_search.npz"
)


def run_astar(args):
    plan_env = gym.make("PushT-v1", **ENV_KWARGS, render_mode=None)
    plan_env.reset(seed=args.seed)
    initial_state = plan_env.unwrapped.get_state().clone()

    planner = SimPlanner(plan_env, K_substeps=K_SUBSTEPS, step_size=STEP_SIZE)
    # Hashes the env's state as the search root, so this has to happen while the
    # env is still where reset left it.
    search_id = astar_fingerprint(planner, args.threshold)

    resume = None
    if args.resume:
        resume = astar_load_checkpoint(args.resume, search_id, force=args.force)
        print(
            f"Resuming {args.resume} at {resume.expansions} expansions "
            f"({len(resume.open_nodes['g'])} open)"
        )

    recorder = SearchRecorder()
    checkpointer = AStarCheckpointer(
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


def run_rrt(args):
    rng_seed = args.rng_seed if args.rng_seed is not None else int(np.random.randint(0, 2**31 - 1))
    np.random.seed(rng_seed)
    print(f"env_seed={args.seed} rng_seed={rng_seed}")

    plan_env = gym.make("PushT-v1", **ENV_KWARGS, render_mode=None)
    plan_env.reset(seed=args.seed)

    verify_set_state_determinism(plan_env)
    plan_env.reset(seed=args.seed)  # discard the warmup/check rollout, start planning clean

    initial_state = plan_env.unwrapped.get_state().clone()

    planner = RRTPlanner(
        plan_env,
        K_substeps=K_SUBSTEPS,
        step_size=RRT_STEP_SIZE,
        goal_bias=args.goal_bias,
        n_candidates=args.n_candidates,
        reposition_prob=args.reposition_prob,
        max_extra_contacts=args.max_extra_contacts,
    )

    search_id = rrt_fingerprint(planner, args.threshold)

    resume = None
    if args.resume:
        resume = rrt_load_checkpoint(args.resume, search_id, force=args.force)
        print(f"Resuming {args.resume} at {resume.count} nodes")

    recorder = RRTRecorder()
    checkpointer = RRTCheckpointer(
        args.checkpoint, search_id, recorder, every=args.checkpoint_every
    )

    result = planner.plan(
        max_iters=args.max_expansions,
        threshold_value=args.threshold,
        extra_params={"env_seed": args.seed, "rng_seed": rng_seed},
        recorder=recorder,
        resume=resume,
        checkpointer=checkpointer,
    )
    plan = result.actions
    print(f"Plan length: {len(plan)}")

    when = datetime.now()

    save_plan(RRT_PLAN_PATH, plan)
    print(f"Saved plan to {RRT_PLAN_PATH}")
    archived_plan_path = results_path("last_rrt_plan", ext=".txt", when=when)
    save_plan(archived_plan_path, plan)

    save_plan_result(result, RRT_RESULT_NPZ_PATH)
    print(f"Saved plan arrays to {RRT_RESULT_NPZ_PATH}")
    archived_npz_path = results_path("last_rrt_plan_result", ext=".npz", when=when)
    save_plan_result(result, archived_npz_path)
    print(f"Archived this run's plan + arrays as {archived_plan_path.stem}.*")

    heatmap_path = plot_expansion_heatmap(result, out_path=results_path("expansion_heatmap", when=when))
    print(f"Saved expansion heatmap to {heatmap_path}")
    print(
        f"Search further with: uv run main.py --planner rrt --resume {args.checkpoint} "
        f"--max-expansions {args.max_expansions * 2} --seed {args.seed} --rng-seed {rng_seed}"
    )

    plan_env.close()
    render_env = gym.make("PushT-v1", **ENV_KWARGS, render_mode="human")
    render_env.reset(seed=args.seed)
    render_planner = RRTPlanner(
        render_env, K_substeps=K_SUBSTEPS, step_size=STEP_SIZE, allow_render=True
    )
    render_planner.execute_plan(initial_state, plan, step_delay=0.05)

    print("Replay finished. Press Enter in this terminal to close the window...")
    while True:
        render_env.render()
        try:
            import msvcrt
            if msvcrt.kbhit() and msvcrt.getch() in (b"\r", b"\n"):
                break
        except ImportError:
            input()
            break
    render_env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--planner",
        choices=["astar", "rrt"],
        default="astar",
        help="Which planner to run (default: astar)",
    )
    parser.add_argument(
        "--max-expansions",
        type=int,
        default=15000,
        help="Maximum A* expansions / RRT iterations (default: 15000)",
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
        help="Autosave every N expansions/nodes; 0 saves only at the end (default: 2000)",
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
    parser.add_argument(
        "--goal-bias",
        type=float,
        default=0.1,
        help="[rrt only] Probability of sampling the goal as the RRT target (default: 0.1)",
    )
    parser.add_argument(
        "--n-candidates",
        type=int,
        default=8,
        help="[rrt only] Candidate directions sampled per RRT expansion (default: 8)",
    )
    parser.add_argument(
        "--reposition-prob",
        type=float,
        default=0.5,
        help=(
            "[rrt only] Probability of attempting a reposition move (lift the "
            "end-effector to a different contact point on the T) instead of an "
            "extend each iteration (default: 0.5)"
        ),
    )
    parser.add_argument(
        "--max-extra-contacts",
        type=int,
        default=4,
        help="[rrt only] Max additional contact points remembered per tree node (default: 4)",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=None,
        help="[rrt only] numpy RNG seed for RRT's target/candidate sampling (default: "
        "random, but it's always printed so a good run can be reproduced later with "
        "--rng-seed <value>)",
    )
    args = parser.parse_args()

    if args.planner == "astar":
        run_astar(args)
    else:
        run_rrt(args)


if __name__ == "__main__":
    main()
