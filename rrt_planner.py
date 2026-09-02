import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from geometry import (
    TEE_CIRCUMRADIUS,
    TEE_LANDMARKS_XY,
    ContactRecord,
    sample_contact_point_body_xy,
    to_numpy,
    yaw_from_quat,
)
from plan_io import PlanResult
from rrt_checkpoint import NullCheckpointer
from rrt_recorder import RRTRecorder

_DEFAULT_COVERAGE_PLOT_PATH = Path(__file__).resolve().parent / "coverage_vs_time.png"

# Within this band, two candidates count as making the same object progress
# (typically: neither touched the T), so the winner is picked by whichever
# leaves the TCP closer to the object instead.
CANDIDATE_TIE_EPS = 1e-3


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
    """Save an RRT plan (list of action vectors) as one 'dx,dy,dz' line per
    action. dz is needed for reposition's lift/lower segments -- push actions
    just save as dz=0."""
    lines = [
        f"{float(action[0]):.6f},{float(action[1]):.6f},{float(action[2]):.6f}"
        for action in plan_actions
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def load_plan(path, action_dim):
    """Load a plan saved by save_plan back into zero-padded action vectors.

    Accepts older 2-column (dx,dy) files too, treating a missing dz as 0 --
    those predate reposition and never had a z-component to save."""
    plan_actions = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        action = np.zeros(action_dim, dtype=np.float32)
        action[0] = float(parts[0])
        action[1] = float(parts[1])
        if len(parts) > 2:
            action[2] = float(parts[2])
        plan_actions.append(action)
    return plan_actions


class RRTNode:
    def __init__(
        self,
        sim_state,
        parent,
        action_from_parent,
        key,
        intersection,
        tcp_xy,
        rel_xy,
        parent_contact_index=0,
        extra_contacts=None,
    ):
        self.sim_state = sim_state
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.key = key  # (obj_x, obj_y, obj_yaw)
        self.intersection = intersection
        self.tcp_xy = tcp_xy
        self.rel_xy = rel_xy  # tcp position in the T's body frame
        # Index into the *parent's* contact list (see `contacts`) that this
        # node's action_from_parent pushed off from. Meaningless for the root.
        self.parent_contact_index = parent_contact_index
        # Alternative contact points discovered at this same T pose via
        # reposition moves. This node's own (sim_state, tcp_xy, rel_xy) is
        # always contact index 0; see `contacts`.
        self.extra_contacts = [] if extra_contacts is None else extra_contacts

    def contacts(self):
        """Every way to be at this node's T pose: this node's own contact
        (index 0) followed by any reposition-discovered ones."""
        return [ContactRecord(self.sim_state, self.tcp_xy, self.rel_xy)] + self.extra_contacts


class RRTPlanner:
    def __init__(
        self,
        env,
        K_substeps,
        step_size,
        goal_bias=0.1,
        n_candidates=8,
        angle_jitter_scale=0.5,
        mag_range=(0.15, 0.5),
        tcp_engage_radius=TEE_CIRCUMRADIUS,
        workspace_xy_bounds=None,
        workspace_margin=0.3,
        reposition_prob=0.5,
        contact_standoff=0.025,
        reposition_lift_dz_total=0.15,
        obj_pose_tolerance=5e-3,
        contact_xy_tolerance=0.01,
        max_extra_contacts=4,
        max_translate_retries=20,
        max_contact_sample_attempts=5,
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
        # Beyond this distance from the T, the TCP is not "still pushing" --
        # aim candidates back at the T instead of blindly toward the target.
        self.tcp_engage_radius = tcp_engage_radius
        self.action_dim = env.action_space.shape[-1]

        self.reposition_prob = reposition_prob
        self.contact_standoff = contact_standoff
        self.reposition_lift_dz_total = reposition_lift_dz_total
        self.obj_pose_tolerance = obj_pose_tolerance
        self.contact_xy_tolerance = contact_xy_tolerance
        self.max_extra_contacts = max_extra_contacts
        self.max_translate_retries = max_translate_retries
        self.max_contact_sample_attempts = max_contact_sample_attempts

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

    def _pose_features(self, obs_extra, obj_xytheta):
        """TCP xy, and TCP position in the T's body frame, for logging/plots."""
        obj_x, obj_y, yaw = obj_xytheta
        tcp = to_numpy(obs_extra["tcp_pose"])[0]        # (7, )
        tcp_x, tcp_y = float(tcp[0]), float(tcp[1])
        dx, dy = tcp_x - obj_x, tcp_y - obj_y
        c, s = np.cos(yaw), np.sin(yaw)
        tcp_xy = np.array([tcp_x, tcp_y], dtype=np.float64)
        rel_xy = np.array([c * dx + s * dy, -s * dx + c * dy], dtype=np.float64)
        return tcp_xy, rel_xy

    def _obj_raw_pose(self, obs_extra):
        return to_numpy(obs_extra["obj_pose"])[0].astype(np.float64)

    def _goal_raw_pose(self):
        return to_numpy(self.env.unwrapped.goal_tee.pose.raw_pose).reshape(-1).astype(np.float64)

    def _is_better_inter(self, candidate, incumbent):
        return candidate.intersection > incumbent.intersection

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

    def _choose_contact(self, node):
        """Uniformly pick one of `node`'s known contact points (index 0 is
        the node's own primary contact; the rest came from reposition)."""
        contacts = node.contacts()
        index = int(np.random.randint(len(contacts)))
        return index, contacts[index]

    def _in_workspace(self, xy):
        (x_lo, x_hi), (y_lo, y_hi) = self.workspace_xy_bounds
        return x_lo <= xy[0] <= x_hi and y_lo <= xy[1] <= y_hi

    def _reposition(self, nearest_node):
        """Move the TCP to a different contact point on the same T pose (lift
        above the block, translate, lower back down), through real physics.
        Returns a new ContactRecord on success, else None.

        This runs the actual lift/translate/lower rollout at discovery time
        (not just an IK feasibility check), so whatever contact this yields
        is exactly what a later push or replay will see -- there's no
        separate "does this hold up for real" step to fail later, and the
        actions recorded on the ContactRecord are already the real ones."""
        source_idx, source_contact = self._choose_contact(nearest_node)

        target_world_xy = None
        for _ in range(self.max_contact_sample_attempts):
            body_xy = sample_contact_point_body_xy(self.contact_standoff)
            x, y, theta = nearest_node.key
            candidate = self._rot2d(theta) @ body_xy + np.array([x, y], dtype=np.float64)
            if self._in_workspace(candidate):
                target_world_xy = candidate
                break
        if target_world_xy is None:
            return None

        self.env.unwrapped.set_state(source_contact.sim_state)
        actions = self._physical_reposition_actions(target_world_xy)
        if actions is None:
            return None

        obs_extra = self.env.unwrapped.get_obs()["extra"]
        tcp_xy, rel_xy = self._pose_features(obs_extra, nearest_node.key)

        new_state = self._clone_state(self.env.unwrapped.get_state())
        return ContactRecord(
            new_state, tcp_xy, rel_xy, source_index=source_idx, actions=actions
        )

    def _expand(self, nearest_node, target_xytheta, target_landmarks):
        contact_idx, contact = self._choose_contact(nearest_node)
        nearest_xy = np.array(nearest_node.key[:2], dtype=np.float64)
        tcp_to_obj = nearest_xy - contact.tcp_xy
        if float(np.linalg.norm(tcp_to_obj)) > self.tcp_engage_radius:
            # TCP isn't near the T -- aim candidates back at it instead of
            # toward the target, which the TCP's own position never enters.
            direction = tcp_to_obj
        else:
            direction = np.array(target_xytheta[:2], dtype=np.float64) - nearest_xy
        base_angle = float(np.arctan2(direction[1], direction[0]))

        angles = base_angle + np.random.normal(0.0, self.angle_jitter_scale, size=self.n_candidates)
        magnitudes = (
            np.random.uniform(self.mag_range[0], self.mag_range[1], size=self.n_candidates)
            * self.step_size
        )

        best_dist = np.inf
        best_tcp_to_obj = np.inf
        best_state, best_obj_xytheta, best_obs_extra, best_action = None, None, None, None

        for angle, magnitude in zip(angles, magnitudes):
            # Every candidate propagates from the same parent state -- no
            # candidate may build on another candidate's result.
            self.env.unwrapped.set_state(contact.sim_state)

            action = np.zeros(self.action_dim, dtype=np.float32)
            action[0] = magnitude * np.cos(angle)
            action[1] = magnitude * np.sin(angle)

            for _ in range(self.K):
                self.env.step(action)

            obs_extra = self.env.unwrapped.get_obs()["extra"]
            obj_xytheta = self._obj_xytheta(obs_extra)
            candidate_landmarks = self._landmarks_world_xy(*obj_xytheta)
            dist = float(np.linalg.norm(candidate_landmarks - target_landmarks, axis=-1).max())
            tcp = to_numpy(obs_extra["tcp_pose"])[0]
            tcp_to_obj = float(np.linalg.norm(tcp[:2] - np.array(obj_xytheta[:2])))

            # Primary: get closer to the target. Candidates within
            # CANDIDATE_TIE_EPS of the current best made essentially the same
            # object progress (typically: none of them touched the T), so the
            # tiebreak falls back to whichever leaves the TCP nearer the
            # object -- the only real difference between them.
            better = dist < best_dist - CANDIDATE_TIE_EPS or (
                dist <= best_dist + CANDIDATE_TIE_EPS and tcp_to_obj < best_tcp_to_obj
            )
            if better:
                best_dist = dist
                best_tcp_to_obj = tcp_to_obj
                best_state = self._clone_state(self.env.unwrapped.get_state())
                best_obj_xytheta = obj_xytheta
                best_obs_extra = obs_extra
                best_action = action

        # Only the winning candidate is kept; the sim is currently sitting at
        # whichever candidate ran last, so restore it before reading intersection.
        self.env.unwrapped.set_state(best_state)
        inter = self._intersection()
        tcp_xy, rel_xy = self._pose_features(best_obs_extra, best_obj_xytheta)

        return RRTNode(
            best_state,
            nearest_node,
            best_action,
            best_obj_xytheta,
            inter,
            tcp_xy,
            rel_xy,
            parent_contact_index=contact_idx,
        )

    def _physical_reposition_actions(self, target_world_xy):
        """Real lift/translate/lower actions that move the TCP -- wherever
        the sim currently sits -- to `target_world_xy`. Returns the actions
        on success, or None if physics disagrees (the T moved more than
        `obj_pose_tolerance`, or translate didn't converge within
        `contact_xy_tolerance`) -- the caller treats that as an infeasible
        reposition, same as `_expand` treats a candidate that doesn't pan
        out."""
        source_obj_xytheta = self._obj_xytheta(self.env.unwrapped.get_obs()["extra"])
        actions = []

        # pd_ee_delta_pose actions are normalized to [-1, 1] and rescaled by
        # the controller to +-pos_upper meters before being applied -- a raw
        # action of `delta / K` alone only ever moves `delta * pos_upper`,
        # not `delta`. Divide the intended per-substep meters by pos_upper so
        # the raw action actually produces them.
        ctrl = self.env.unwrapped.agent.controller.controllers["arm"]
        pos_upper = np.broadcast_to(np.asarray(ctrl.config.pos_upper, dtype=np.float64), (3,))
        xy_scale = pos_upper[:2]
        z_scale = float(pos_upper[2])

        dz = self.reposition_lift_dz_total / self.K
        lift_action = np.zeros(self.action_dim, dtype=np.float32)
        lift_action[2] = dz / z_scale
        for _ in range(self.K):
            self.env.step(lift_action)
        actions.append(lift_action)

        for _ in range(self.max_translate_retries):
            obs_extra = self.env.unwrapped.get_obs()["extra"]
            current_tcp_xy = to_numpy(obs_extra["tcp_pose"])[0, :2].astype(np.float64)
            delta_xy = target_world_xy - current_tcp_xy
            # Break with margin below contact_xy_tolerance -- the lower phase
            # that follows can nudge xy slightly, and the final check below
            # is against the full tolerance.
            if float(np.linalg.norm(delta_xy)) <= self.contact_xy_tolerance / 2:
                break
            move_action = np.zeros(self.action_dim, dtype=np.float32)
            move_action[0], move_action[1] = (delta_xy / self.K) / xy_scale
            for _ in range(self.K):
                self.env.step(move_action)
            actions.append(move_action)

        lower_action = np.zeros(self.action_dim, dtype=np.float32)
        lower_action[2] = -dz / z_scale
        for _ in range(self.K):
            self.env.step(lower_action)
        actions.append(lower_action)

        obs_extra = self.env.unwrapped.get_obs()["extra"]
        obj_xytheta = self._obj_xytheta(obs_extra)
        obj_landmarks = self._landmarks_world_xy(*obj_xytheta)
        source_landmarks = self._landmarks_world_xy(*source_obj_xytheta)
        obj_moved = float(np.linalg.norm(obj_landmarks - source_landmarks, axis=-1).max())

        tcp_xy = to_numpy(obs_extra["tcp_pose"])[0, :2].astype(np.float64)
        tcp_err = float(np.linalg.norm(tcp_xy - target_world_xy))
        if obj_moved > self.obj_pose_tolerance or tcp_err > self.contact_xy_tolerance:
            return None
        return actions

    def _contact_chain_actions(self, node, contact_idx):
        """The reposition action segments needed to walk `node`'s TCP from
        its primary contact (index 0, where it lands right after
        action_from_parent) to `contact_idx`, in chronological order.

        Contacts can be reached from other non-primary contacts (a reposition
        may lift off a previously-discovered contact rather than the
        primary), so this recurses along each contact's `source_index`."""
        if contact_idx == 0:
            return []
        contact = node.contacts()[contact_idx]
        return self._contact_chain_actions(node, contact.source_index) + list(contact.actions)

    def _extract_path(self, node):
        actions = []
        while node.parent is not None:
            actions.append(node.action_from_parent)
            reposition_actions = self._contact_chain_actions(node.parent, node.parent_contact_index)
            actions.extend(reversed(reposition_actions))
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
        plot_path=_DEFAULT_COVERAGE_PLOT_PATH,
        recorder=None,
        extra_params=None,
        resume=None,
        checkpointer=None,
    ):
        """Expand until `max_iters` *total*, counting the nodes a resumed
        checkpoint already made towards the budget."""
        if threshold_value is None:
            threshold_value = float(self.env.unwrapped.intersection_thresh)
        if recorder is None:
            recorder = RRTRecorder()
        if checkpointer is None:
            checkpointer = NullCheckpointer()

        root_state = self._clone_state(self.env.unwrapped.get_state())
        obs_extra = self.env.unwrapped.get_obs()["extra"]
        root_key = self._obj_xytheta(obs_extra)
        root_inter = self._intersection()
        root_tcp_xy, root_rel_xy = self._pose_features(obs_extra, root_key)
        root_node = RRTNode(
            root_state, None, None, root_key, root_inter, root_tcp_xy, root_rel_xy
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

        if resume is not None:
            nodes, keys_xy, keys_theta, best_inter_node, iters = resume.restore(
                RRTNode, like=root_state
            )
            count = len(nodes)
            if count < max_iters + 1:
                pad = max_iters + 1 - count
                keys_xy = np.concatenate([keys_xy, np.zeros((pad, 2))], axis=0)
                keys_theta = np.concatenate([keys_theta, np.zeros((pad,))], axis=0)
            recorder.resume_from(resume)
        else:
            keys_xy = np.zeros((max_iters + 1, 2), dtype=np.float64)
            keys_theta = np.zeros((max_iters + 1,), dtype=np.float64)
            nodes = [root_node]
            keys_xy[0] = root_key[:2]
            keys_theta[0] = root_key[2]
            count = 1
            best_inter_node = root_node
            iters = 0

        start_time = time.time()
        coverage_history = [(0.0, root_node.intersection)]

        params = {
            "threshold": threshold_value,
            "step_size": self.step_size,
            "k_substeps": self.K,
            "n_candidates": self.n_candidates,
            "goal_bias": self.goal_bias,
            "reposition_prob": self.reposition_prob,
            "contact_standoff": self.contact_standoff,
            "reposition_lift_dz_total": self.reposition_lift_dz_total,
            "obj_pose_tolerance": self.obj_pose_tolerance,
            "contact_xy_tolerance": self.contact_xy_tolerance,
            "max_extra_contacts": self.max_extra_contacts,
        }
        if extra_params:
            params.update(extra_params)

        result_node = None
        goal_reached = False
        with recorder.run(max_iters, params, root_node, initial_iters=iters) as rec:
            if resume is None:
                rec.log_root(root_node)

            while iters < max_iters and not rec.stop_requested:
                checkpointer.tick(nodes, best_inter_node, iters)

                target = self._sample_target()
                target_landmarks = self._landmarks_world_xy(*target)

                nearest_idx = self._nearest_index(keys_xy, keys_theta, count, target_landmarks)
                nearest_node = nodes[nearest_idx]

                if np.random.rand() < self.reposition_prob:
                    contact = self._reposition(nearest_node)
                    iters += 1
                    accepted = (
                        contact is not None
                        and len(nearest_node.extra_contacts) < self.max_extra_contacts
                    )
                    if accepted:
                        nearest_node.extra_contacts.append(contact)
                    rec.repositioned(nearest_node, nearest_idx, accepted)
                    continue

                child = self._expand(nearest_node, target, target_landmarks)
                nodes.append(child)
                keys_xy[count] = child.key[:2]
                keys_theta[count] = child.key[2]
                count += 1
                iters += 1

                if self._is_better_inter(child, best_inter_node):
                    best_inter_node = child

                coverage_history.append((time.time() - start_time, best_inter_node.intersection))
                rec.inserted(child, nearest_idx, best_inter_node, count)

                if child.intersection >= threshold_value:
                    rec.goal_reached(child, count, iters)
                    result_node = child
                    goal_reached = True
                    break

            if result_node is None:
                result_node = best_inter_node

            final_actions = self._extract_path(result_node)

            if not goal_reached:
                if rec.stop_requested:
                    rec.interrupted(best_inter_node, len(final_actions))
                else:
                    rec.exhausted(best_inter_node, len(final_actions))

            log = rec.finish(result_node, goal_reached, count, iters)

        checkpointer.save(nodes, best_inter_node, iters, goal_reached=goal_reached)

        self.env.unwrapped.set_state(root_state)
        self._save_coverage_plot(coverage_history, plot_path)

        return PlanResult(
            actions=final_actions,
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
