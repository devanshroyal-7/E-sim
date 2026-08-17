import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from geometry import TEE_LANDMARKS_XY, to_numpy, yaw_from_quat
from plan_io import PlanResult, save_plan_result
from rrt_recorder import RRTRecorder
from viz_search import plot_expansion_heatmap

PLAN_PATH = Path(__file__).resolve().parent / "last_rrt_plan.txt"
COVERAGE_PLOT_PATH = Path(__file__).resolve().parent / "coverage_vs_time.png"
RESULT_NPZ_PATH = Path(__file__).resolve().parent / "last_rrt_plan_result.npz"


def verify_set_state_determinism(env, n_warmup_steps=5, n_check_steps=10):
    action_dim = env.action_space.shape[-1]
    warmup_action = np.zeros(action_dim, dtype=np.float32)
    warmup_action[:2] = [0.5, 0.3]
    for _ in range(n_warmup_steps):
        env.step(warmup_action)

    state = env.unwrapped.get_state().clone()

    check_action = np.zeros(action_dim, dtype=np.float32)
    check_action[:2] = [-0.4, 0.2]

    env.unwrapped.set_state(state)
    for _ in range(n_check_steps):
        env.step(check_action)
    result_a = env.unwrapped.get_state().clone().detach().cpu().numpy()

    env.unwrapped.set_state(state)
    for _ in range(n_check_steps):
        env.step(check_action)
    result_b = env.unwrapped.get_state().clone().detach().cpu().numpy()

    if not np.allclose(result_a, result_b, atol=1e-6):
        max_diff = np.abs(result_a - result_b).max()
        raise RuntimeError(
            "set_state() is not a full deterministic state restore "
            f"(max abs diff after replay = {max_diff:.3e}). RRT candidate "
            "comparisons in _expand() are not trustworthy until this is fixed."
        )
    print("set_state determinism check passed.")


def save_plan(path, plan_actions):
    """Save an RRT plan (list of action vectors) as one 'dx,dy' line per action."""
    lines = [f"{float(action[0]):.6f},{float(action[1]):.6f}" for action in plan_actions]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def load_plan(path, action_dim):
    """Load a plan saved by save_plan back into zero-padded action vectors."""
    plan_actions = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        dx_str, dy_str = line.split(",")
        action = np.zeros(action_dim, dtype=np.float32)
        action[0] = float(dx_str)
        action[1] = float(dy_str)
        plan_actions.append(action)
    return plan_actions


class RRTNode:
    def __init__(
        self,
        sim_state,
        parent,
        action_from_parent,
        key,
        g_value,
        h_value,
        intersection,
        tcp_xy,
        rel_xy,
    ):
        self.sim_state = sim_state
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.key = key  # (obj_x, obj_y, obj_yaw)
        self.g_value = g_value
        self.h_value = h_value
        self.intersection = intersection
        self.tcp_xy = tcp_xy
        self.rel_xy = rel_xy  # tcp position in the T's body frame


class RRTPlanner:
    def __init__(
        self,
        env,
        K_substeps,
        step_size,
        goal_bias=0.1,
        n_candidates=8,
        angle_jitter_scale=0.5,
        mag_range=(0.3, 1.0),
        workspace_xy_bounds=None,
        workspace_margin=0.3,
        allow_render=False,
    ):
        if not allow_render and getattr(env.unwrapped, "render_mode", None) is not None:
            raise ValueError(
                "RRTPlanner requires a planning env built with render_mode=None; "
                "use a separate render_mode='human' env for execute_plan()."
            )

        self.env = env
        self.K = K_substeps
        self.step_size = step_size
        self.goal_bias = goal_bias
        self.n_candidates = n_candidates
        self.angle_jitter_scale = angle_jitter_scale
        self.mag_range = mag_range
        self.action_dim = env.action_space.shape[-1]

        if workspace_xy_bounds is None:
            goal_offset = to_numpy(env.unwrapped.goal_offset).reshape(-1)
            m = workspace_margin
            workspace_xy_bounds = (
                (float(goal_offset[0]) - m, float(goal_offset[0]) + m),
                (float(goal_offset[1]) - m, float(goal_offset[1]) + m),
            )
        self.workspace_xy_bounds = workspace_xy_bounds

    def _clone_state(self, state):
        if hasattr(state, "clone"):
            return state.clone()
        return np.array(state, copy=True)

    def _rot2d(self, theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s], [s, c]], dtype=np.float64)

    def _rot2d_batch(self, thetas):
        cos_t = np.cos(thetas)
        sin_t = np.sin(thetas)
        rot = np.empty((thetas.shape[0], 2, 2), dtype=np.float64)
        rot[:, 0, 0] = cos_t
        rot[:, 0, 1] = -sin_t
        rot[:, 1, 0] = sin_t
        rot[:, 1, 1] = cos_t
        return rot

    def _landmarks_world_xy(self, x, y, theta):
        rot = self._rot2d(theta)
        return TEE_LANDMARKS_XY @ rot.T + np.array([x, y], dtype=np.float64)

    def _landmark_dist(self, a_xytheta, b_xytheta):
        la = self._landmarks_world_xy(*a_xytheta)
        lb = self._landmarks_world_xy(*b_xytheta)
        # Max over the 8 landmarks (worst-corner error), not mean: a rigid
        # body match needs every corner close, not just the average corner.
        return float(np.linalg.norm(la - lb, axis=-1).max())

    def _obj_xytheta(self, obs_extra):
        obj = to_numpy(obs_extra["obj_pose"])[0]
        return (float(obj[0]), float(obj[1]), float(yaw_from_quat(obj[3:7])))

    def _goal_xytheta(self):
        p = to_numpy(self.env.unwrapped.goal_tee.pose.p).reshape(-1)
        q = to_numpy(self.env.unwrapped.goal_tee.pose.q).reshape(-1)
        return (float(p[0]), float(p[1]), float(yaw_from_quat(q)))

    def _intersection(self):
        inter = self.env.unwrapped.pseudo_render_intersection()
        return float(to_numpy(inter).reshape(-1)[0])

    def _heuristic(self, obs_extra):
        """Landmark distance to goal + tcp-to-object distance. Fallback scoring only."""
        obj_xytheta = self._obj_xytheta(obs_extra)
        goal_xytheta = self._goal_xytheta()
        tcp = to_numpy(obs_extra["tcp_pose"])[0]
        tcp_to_obj = float(np.linalg.norm(tcp[:2] - np.array(obj_xytheta[:2])))
        pose_err = self._landmark_dist(obj_xytheta, goal_xytheta)
        return pose_err + 0.25 * tcp_to_obj

    def _pose_features(self, obs_extra, obj_xytheta):
        """TCP xy, and TCP position in the T's body frame, for logging/plots."""
        obj_x, obj_y, yaw = obj_xytheta
        tcp = to_numpy(obs_extra["tcp_pose"])[0]
        dx, dy = float(tcp[0]) - obj_x, float(tcp[1]) - obj_y
        c, s = np.cos(yaw), np.sin(yaw)
        tcp_xy = np.array([tcp[0], tcp[1]], dtype=np.float64)
        rel_xy = np.array([c * dx + s * dy, -s * dx + c * dy], dtype=np.float64)
        return tcp_xy, rel_xy

    def _obj_raw_pose(self, obs_extra):
        return to_numpy(obs_extra["obj_pose"])[0].astype(np.float64)

    def _goal_raw_pose(self):
        return to_numpy(self.env.unwrapped.goal_tee.pose.raw_pose).reshape(-1).astype(np.float64)

    def _is_better_inter(self, candidate, incumbent):
        if candidate.intersection > incumbent.intersection:
            return True
        if candidate.intersection < incumbent.intersection:
            return False
        return candidate.g_value < incumbent.g_value

    def _is_better_h(self, candidate, incumbent):
        if candidate.h_value < incumbent.h_value:
            return True
        if candidate.h_value > incumbent.h_value:
            return False
        return candidate.g_value < incumbent.g_value

    def _pick_fallback(self, root_node, best_inter_node, best_h_node):
        if best_inter_node.intersection > root_node.intersection:
            return best_inter_node
        return best_h_node

    def _sample_target(self):
        if np.random.rand() < self.goal_bias:
            return self._goal_xytheta()
        (x_lo, x_hi), (y_lo, y_hi) = self.workspace_xy_bounds
        x = np.random.uniform(x_lo, x_hi)
        y = np.random.uniform(y_lo, y_hi)
        theta = np.random.uniform(0.0, 2.0 * np.pi)
        return (float(x), float(y), float(theta))

    def _nearest_index(self, keys_xy, keys_theta, count, target_landmarks):
        # Vectorized keypoint-max distance to every tree node at once:
        # batch-rotate the 8 landmarks by each node's theta, not a raw
        # (x, y, theta) norm (theta isn't in the same units as x, y).
        rot = self._rot2d_batch(keys_theta[:count])  # (count, 2, 2)
        world = np.einsum("nij,pj->npi", rot, TEE_LANDMARKS_XY) + keys_xy[:count, None, :]
        dist = np.linalg.norm(world - target_landmarks[None, :, :], axis=-1).max(axis=-1)
        return int(np.argmin(dist))

    def _expand(self, nearest_node, target_xytheta, target_landmarks):
        nearest_xy = np.array(nearest_node.key[:2], dtype=np.float64)
        direction = np.array(target_xytheta[:2], dtype=np.float64) - nearest_xy
        base_angle = float(np.arctan2(direction[1], direction[0]))

        angles = base_angle + np.random.normal(0.0, self.angle_jitter_scale, size=self.n_candidates)
        magnitudes = (
            np.random.uniform(self.mag_range[0], self.mag_range[1], size=self.n_candidates)
            * self.step_size
        )

        best_dist = np.inf
        best_state, best_obj_xytheta, best_obs_extra, best_action = None, None, None, None

        for angle, magnitude in zip(angles, magnitudes):
            # Every candidate propagates from the same parent state -- no
            # candidate may build on another candidate's result.
            self.env.unwrapped.set_state(nearest_node.sim_state)

            action = np.zeros(self.action_dim, dtype=np.float32)
            action[0] = magnitude * np.cos(angle)
            action[1] = magnitude * np.sin(angle)

            for _ in range(self.K):
                self.env.step(action)

            obs_extra = self.env.unwrapped.get_obs()["extra"]
            obj_xytheta = self._obj_xytheta(obs_extra)
            candidate_landmarks = self._landmarks_world_xy(*obj_xytheta)
            dist = float(np.linalg.norm(candidate_landmarks - target_landmarks, axis=-1).max())

            if dist < best_dist:
                best_dist = dist
                best_state = self._clone_state(self.env.unwrapped.get_state())
                best_obj_xytheta = obj_xytheta
                best_obs_extra = obs_extra
                best_action = action

        # Only the winning candidate is kept; the sim is currently sitting at
        # whichever candidate ran last, so restore it before reading intersection.
        self.env.unwrapped.set_state(best_state)
        inter = self._intersection()
        h_value = self._heuristic(best_obs_extra)
        # Edge cost = actual worst-corner displacement of the T between parent
        # and child (same metric as _landmark_dist), not a flat per-step cost.
        step_cost = self._landmark_dist(nearest_node.key, best_obj_xytheta)
        g_value = nearest_node.g_value + step_cost
        tcp_xy, rel_xy = self._pose_features(best_obs_extra, best_obj_xytheta)

        return RRTNode(
            best_state,
            nearest_node,
            best_action,
            best_obj_xytheta,
            g_value,
            h_value,
            inter,
            tcp_xy,
            rel_xy,
        )

    def _extract_path(self, node):
        actions = []
        while node.parent is not None:
            actions.append(node.action_from_parent)
            node = node.parent
        actions.reverse()
        return actions

    def _extract_trajectory_xy(self, node):
        points = [node.key[:2]]
        while node.parent is not None:
            node = node.parent
            points.append(node.key[:2])
        points.reverse()
        return np.asarray(points, dtype=np.float64)

    def _save_coverage_plot(self, history, path):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            tqdm.write("matplotlib not available; skipping coverage-vs-time plot.")
            return

        times, coverage = zip(*history)
        coverage_pct = [c * 100.0 for c in coverage]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(times, coverage_pct, drawstyle="steps-post")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Best intersection coverage (%)")
        ax.set_title("RRT planning progress")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        tqdm.write(f"Saved coverage-vs-time plot to {path}")

    def plan(
        self,
        max_iters=2000,
        threshold_value=None,
        plot_path=COVERAGE_PLOT_PATH,
        recorder=None,
        extra_params=None,
    ):
        if threshold_value is None:
            threshold_value = float(self.env.unwrapped.intersection_thresh)
        if recorder is None:
            recorder = RRTRecorder()

        root_state = self._clone_state(self.env.unwrapped.get_state())
        obs_extra = self.env.unwrapped.get_obs()["extra"]
        root_key = self._obj_xytheta(obs_extra)
        root_inter = self._intersection()
        root_h = self._heuristic(obs_extra)
        root_tcp_xy, root_rel_xy = self._pose_features(obs_extra, root_key)
        root_node = RRTNode(
            root_state, None, None, root_key, 0.0, root_h, root_inter, root_tcp_xy, root_rel_xy
        )
        start_pose = self._obj_raw_pose(obs_extra)
        goal_pose = self._goal_raw_pose()

        if root_inter >= threshold_value:
            tqdm.write(f"Already at goal (intersection={root_inter:.4f})")
            return PlanResult(
                actions=[],
                expansion_xy=np.zeros((0, 2), dtype=np.float64),
                start_pose=start_pose,
                goal_pose=goal_pose,
                trajectory_xy=self._extract_trajectory_xy(root_node),
                log=None,
            )

        keys_xy = np.zeros((max_iters + 1, 2), dtype=np.float64)
        keys_theta = np.zeros((max_iters + 1,), dtype=np.float64)
        nodes = [root_node]
        keys_xy[0] = root_key[:2]
        keys_theta[0] = root_key[2]
        count = 1

        best_inter_node = root_node
        best_h_node = root_node

        start_time = time.time()
        coverage_history = [(0.0, root_node.intersection)]

        params = {
            "threshold": threshold_value,
            "step_size": self.step_size,
            "k_substeps": self.K,
            "n_candidates": self.n_candidates,
            "goal_bias": self.goal_bias,
        }
        if extra_params:
            params.update(extra_params)

        result_node = None
        goal_reached = False
        with recorder.run(max_iters, params, root_node) as rec:
            rec.log_root(root_node)

            while count <= max_iters:
                target = self._sample_target()
                target_landmarks = self._landmarks_world_xy(*target)

                nearest_idx = self._nearest_index(keys_xy, keys_theta, count, target_landmarks)
                nearest_node = nodes[nearest_idx]

                child = self._expand(nearest_node, target, target_landmarks)
                nodes.append(child)
                keys_xy[count] = child.key[:2]
                keys_theta[count] = child.key[2]
                count += 1

                if self._is_better_inter(child, best_inter_node):
                    best_inter_node = child
                if self._is_better_h(child, best_h_node):
                    best_h_node = child

                coverage_history.append((time.time() - start_time, best_inter_node.intersection))
                rec.inserted(child, nearest_idx, best_h_node, best_inter_node, count)

                if child.intersection >= threshold_value:
                    rec.goal_reached(child, count)
                    result_node = child
                    goal_reached = True
                    break

            if result_node is None:
                result_node = self._pick_fallback(root_node, best_inter_node, best_h_node)
                rec.exhausted(best_inter_node, best_h_node, len(self._extract_path(result_node)))

            log = rec.finish(result_node, goal_reached, count)

        self.env.unwrapped.set_state(root_state)
        self._save_coverage_plot(coverage_history, plot_path)

        return PlanResult(
            actions=self._extract_path(result_node),
            expansion_xy=recorder.expansion_xy(),
            start_pose=start_pose,
            goal_pose=goal_pose,
            trajectory_xy=self._extract_trajectory_xy(result_node),
            log=log,
        )

    def execute_plan(self, initial_state, plan_actions, step_delay=0.0, K=None):
        K = self.K if K is None else K
        self.env.unwrapped.set_state(initial_state)

        for action in tqdm(plan_actions, desc="Executing", unit="action"):
            for _ in range(K):
                self.env.step(action)
                self.env.render()
                if step_delay > 0:
                    time.sleep(step_delay)


if __name__ == "__main__":
    import argparse

    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    ENV_KWARGS = dict(obs_mode="state_dict", control_mode="pd_ee_delta_pose")
    K_SUBSTEPS = 10
    STEP_SIZE = 0.5

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-iters", type=int, default=5000, help="Maximum RRT iterations (default: 5000)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Intersection success threshold (default: env's intersection_thresh)",
    )
    parser.add_argument(
        "--goal-bias",
        type=float,
        default=0.1,
        help="Probability of sampling the goal as the RRT target (default: 0.1)",
    )
    parser.add_argument(
        "--n-candidates",
        type=int,
        default=8,
        help="Candidate directions sampled (and simulated serially on CPU) "
        "per RRT expansion (default: 8)",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Record the final replay to an mp4 instead of opening an interactive "
        "window. Replays inside the SAME sim instance that found the plan (not a "
        "freshly-constructed env) -- a fresh instance is not guaranteed to "
        "reproduce the same physical outcome, see scripts/check_replay_intersection.py.",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default="videos",
        help="Directory to save the replay video to when --save-video is set (default: videos)",
    )
    parser.add_argument(
        "--env-seed",
        type=int,
        default=0,
        help="Env reset seed -- controls the T/goal/robot starting layout (default: 0)",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=None,
        help="numpy RNG seed for RRT's target/candidate sampling (default: random, "
        "but it's always printed and saved so a good run can be reproduced later "
        "with --rng-seed <value>)",
    )
    args = parser.parse_args()

    # Locked in below via np.random.seed() -- drawn from the ambient (unseeded)
    # state first if the user didn't pin one, so every run still gets genuine
    # randomness by default while still being reproducible after the fact.
    rng_seed = args.rng_seed if args.rng_seed is not None else int(np.random.randint(0, 2**31 - 1))
    np.random.seed(rng_seed)
    print(f"env_seed={args.env_seed} rng_seed={rng_seed}")

    # rgb_array costs nothing extra during planning (render() is never called
    # mid-search) but lets us record the final replay in this exact instance.
    plan_render_mode = "rgb_array" if args.save_video else None
    plan_env = gym.make("PushT-v1", **ENV_KWARGS, render_mode=plan_render_mode)
    plan_env.reset(seed=args.env_seed)

    verify_set_state_determinism(plan_env)
    plan_env.reset(seed=args.env_seed)  # discard the warmup/check rollout, start planning clean

    initial_state = plan_env.unwrapped.get_state().clone()

    planner = RRTPlanner(
        plan_env,
        K_substeps=K_SUBSTEPS,
        step_size=STEP_SIZE,
        goal_bias=args.goal_bias,
        n_candidates=args.n_candidates,
        allow_render=args.save_video,
    )
    result = planner.plan(
        max_iters=args.max_iters,
        threshold_value=args.threshold,
        extra_params={"env_seed": args.env_seed, "rng_seed": rng_seed},
    )
    plan = result.actions
    print(f"Plan length: {len(plan)}")

    save_plan(PLAN_PATH, plan)
    print(f"Saved plan to {PLAN_PATH}")

    save_plan_result(result, RESULT_NPZ_PATH)
    print(f"Saved plan arrays to {RESULT_NPZ_PATH}")

    heatmap_path = plot_expansion_heatmap(result)
    print(f"Saved expansion heatmap to {heatmap_path}")

    if args.save_video:
        from mani_skill.utils.wrappers.record import RecordEpisode

        # Wrap (not recreate) plan_env: same underlying sim instance/state history
        # that the search actually produced `plan` from.
        plan_env = RecordEpisode(
            plan_env, output_dir=args.video_dir, save_trajectory=False, save_video=True
        )
        plan_env.reset(seed=args.env_seed)
        plan_env.unwrapped.set_state(initial_state)

        replay_planner = RRTPlanner(
            plan_env, K_substeps=K_SUBSTEPS, step_size=STEP_SIZE, allow_render=True
        )
        replay_planner.execute_plan(initial_state, plan, step_delay=0.0)
        plan_env.close()
        print(f"Saved replay video under {args.video_dir}/")
    else:
        plan_env.close()
        render_env = gym.make("PushT-v1", **ENV_KWARGS, render_mode="human")
        render_env.reset(seed=args.env_seed)
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
