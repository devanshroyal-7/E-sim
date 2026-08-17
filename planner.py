import heapq
import time

import numpy as np
from tqdm import tqdm

from geometry import landmarks_world_xy, to_numpy, wrap_pi, yaw_from_quat
from plan_io import PlanResult
from search_checkpoint import NullCheckpointer
from search_recorder import SearchRecorder

G_STEP_COST = 0.1
H_WEIGHT = 1.0

# State-key resolution: object xy to 1 cm, tee yaw to 5 deg bins.
STATE_KEY_XY_DECIMALS = 2
STATE_KEY_YAW_RES = np.deg2rad(5.0)
TEE_CIRCUMRADIUS = 0.13975      # largest dist from T's COM to the corners

class SearchNode:
    def __init__(
        self,
        sim_state,
        action_history,
        g_value,
        h_value,
        state_key,
        intersection=0.0,
        parent_expansion=-1,
    ):
        self.sim_state = sim_state
        self.action_history = action_history
        self.g_value = g_value
        self.h_value = h_value
        self.f_value = g_value + H_WEIGHT * h_value
        self.state_key = state_key
        self.intersection = intersection
        # Index into the expansion log of the node this one was generated from.
        self.parent_expansion = parent_expansion

    def __lt__(self, other):
        return self.f_value < other.f_value


class SimPlanner:
    def __init__(self, env, K_substeps, step_size):
        self.env = env
        self.K = K_substeps
        self.step_size = step_size
        self.action_primitives = self._build_action_primitives(step_size)
        self.goal_pose = to_numpy(self.env.unwrapped.goal_tee.pose.raw_pose.reshape(-1))
        self.goal_landmarks = landmarks_world_xy(self.goal_pose)

    def _build_action_primitives(self, step_size):
        s = step_size
        d = s / np.sqrt(2)
        xy = np.array(
            [
                [s, 0.0],
                [-s, 0.0],
                [0.0, s],
                [0.0, -s],
                [d, d],
                [-d, d],
                [d, -d],
                [-d, -d],
            ],
            dtype=np.float32,
        )
        action_dim = self.env.action_space.shape[-1]
        primitives = np.zeros((len(xy), action_dim), dtype=np.float32)
        primitives[:, :2] = xy
        return primitives

    def _get_action_cost(self, parent_obs_extra, child_obs_extra):
        parent_landmarks = landmarks_world_xy(parent_obs_extra["obj_pose"])  # 8x2
        child_landmarks = landmarks_world_xy(child_obs_extra["obj_pose"])

        return float(np.max(np.linalg.norm(parent_landmarks - child_landmarks, axis=-1)))

    def _get_heuristic(self, obs_extra):
        obj = to_numpy(obs_extra["obj_pose"])
        # tcp = to_numpy(obs_extra["tcp_pose"])

        current = landmarks_world_xy(obj)

        pose_feats = self._pose_features(obs_extra)

        obj_xy = pose_feats["obj_xy"]
        tcp_xy = pose_feats["tcp_xy"]

        pose_err = float(np.max(np.linalg.norm(current - self.goal_landmarks, axis=-1)))
        reach = max(0, float(np.linalg.norm(obj_xy - tcp_xy) - TEE_CIRCUMRADIUS))
        return pose_err + reach

    def _pose_features(self, obs_extra):
        """Object / TCP quantities shared by the state key and the search log."""
        tcp = to_numpy(obs_extra["tcp_pose"]).reshape(-1)
        obj = to_numpy(obs_extra["obj_pose"]).reshape(-1)
        yaw = wrap_pi(float(yaw_from_quat(obj[3:7])))

        dx, dy = tcp[0] - obj[0], tcp[1] - obj[1]
        c, s = np.cos(yaw), np.sin(yaw)
        return {
            "obj_xy": np.array([obj[0], obj[1]], dtype=np.float64),
            "yaw": yaw,
            "tcp_xy": np.array([tcp[0], tcp[1]], dtype=np.float64),
            "rel_xy": np.array([c * dx + s * dy, -s * dx + c * dy], dtype=np.float64),
        }

    def _state_key(
        self,
        features,
        xy_decimals=STATE_KEY_XY_DECIMALS,
        yaw_res=STATE_KEY_YAW_RES,
    ):
        # TCP in object frame; absolute object xy + wrap-safe yaw bins.
        # xy_decimals=2 -> 1 cm bins; yaw_res=5 deg by default.
        yaw = features["yaw"]
        rel_x, rel_y = features["rel_xy"]
        obj_x, obj_y = features["obj_xy"]

        n_yaw_bins = int(np.round(2.0 * np.pi / yaw_res))
        yaw_bin = int(np.floor((yaw + np.pi) / yaw_res)) % n_yaw_bins

        vals = np.round([rel_x, rel_y, obj_x, obj_y], decimals=xy_decimals)
        return (*vals.tolist(), yaw_bin)

    def _intersection(self):
        inter = self.env.unwrapped.pseudo_render_intersection()
        return float(to_numpy(inter).reshape(-1)[0])

    def _clone_state(self, state):
        if hasattr(state, "clone"):
            return state.clone()
        return np.array(state, copy=True)

    def _is_better_inter_node(
        self, candidate: SearchNode, incumbent: SearchNode
    ) -> bool:
        if candidate.intersection > incumbent.intersection:
            return True
        if candidate.intersection < incumbent.intersection:
            return False
        return candidate.g_value < incumbent.g_value

    def _is_better_h_node(self, candidate: SearchNode, incumbent: SearchNode) -> bool:
        if candidate.h_value < incumbent.h_value:
            return True
        if candidate.h_value > incumbent.h_value:
            return False
        return candidate.g_value < incumbent.g_value

    def _obj_xy_from_obs(self, obs_extra):
        obj = to_numpy(obs_extra["obj_pose"]).reshape(-1)
        return np.array([obj[0], obj[1]], dtype=np.float64)

    def _obj_pose_from_obs(self, obs_extra):
        return to_numpy(obs_extra["obj_pose"]).reshape(-1).astype(np.float64)

    def _search_params(self, threshold_value):
        """Search settings the recorder stores alongside the log, and that a
        resumed search has to match."""
        return {
            "threshold": threshold_value,
            "step_size": self.step_size,
            "k_substeps": self.K,
            "g_step_cost": G_STEP_COST,
            "h_weight": H_WEIGHT,
            "yaw_res_deg": float(np.rad2deg(STATE_KEY_YAW_RES)),
            "xy_decimals": STATE_KEY_XY_DECIMALS,
            "tee_circumradius": TEE_CIRCUMRADIUS,
        }

    def _replay_trajectory_xy(self, root_state, actions):
        """Tee COM after each primitive, including start. No rendering."""
        self.env.unwrapped.set_state(root_state)
        pts = [self._obj_xy_from_obs(self.env.unwrapped.get_obs()["extra"])]
        for act_idx in actions:
            action = self.action_primitives[act_idx]
            for _ in range(self.K):
                self.env.step(action)
            pts.append(self._obj_xy_from_obs(self.env.unwrapped.get_obs()["extra"]))
        return np.asarray(pts, dtype=np.float64)

    def _make_plan_result(self, root_state, start_pose, actions, expansion_xy, log):
        trajectory_xy = self._replay_trajectory_xy(root_state, actions)
        self.env.unwrapped.set_state(root_state)
        return PlanResult(
            actions=list(actions),
            expansion_xy=np.asarray(expansion_xy, dtype=np.float64).reshape(-1, 2),
            start_pose=np.asarray(start_pose, dtype=np.float64).reshape(-1),
            goal_pose=np.asarray(self.goal_pose, dtype=np.float64).reshape(-1),
            trajectory_xy=trajectory_xy,
            log=log,
        )

    def plan(
        self,
        max_expansions=200,
        threshold_value=None,
        recorder=None,
        resume=None,
        checkpointer=None,
    ):
        """Expand until `max_expansions` *total*, counting the expansions a
        resumed checkpoint already made towards the budget."""
        if threshold_value is None:
            threshold_value = float(self.env.unwrapped.intersection_thresh)
        if recorder is None:
            recorder = SearchRecorder()
        if checkpointer is None:
            checkpointer = NullCheckpointer()

        root_state = self._clone_state(self.env.unwrapped.get_state())

        obs = self.env.unwrapped.get_obs()
        obs_extra = obs["extra"]

        start_pose = self._obj_pose_from_obs(obs_extra)
        root_features = self._pose_features(obs_extra)
        root_key = self._state_key(root_features)
        root_node = SearchNode(
            root_state,
            [],
            0.0,
            self._get_heuristic(obs_extra),
            root_key,
            intersection=self._intersection(),
        )

        open_list = []
        closed_list = set()
        heapq.heappush(open_list, root_node)

        best_inter_node = root_node
        best_h_node = root_node
        best_g_by_key = {root_key: 0.0}
        expansions = 0
        goal_node = None

        if resume is not None:
            (open_list, closed_list, best_g_by_key, best_h_node,
             best_inter_node, expansions) = resume.restore(SearchNode, like=root_state)
            recorder.resume_from(resume)

        with recorder.run(
            max_expansions,
            self._search_params(threshold_value),
            root_node,
            self.action_primitives[:, :2],
        ) as rec:
            while expansions < max_expansions and not rec.stop_requested:
                # Top of the loop is the only point where the open list, the
                # closed set and the expansion log agree, so it is the only
                # safe place to snapshot.
                checkpointer.tick(
                    expansions, open_list, closed_list, best_g_by_key,
                    best_h_node, best_inter_node,
                )

                if not open_list:
                    # No more nodes to expand
                    rec.open_exhausted()
                    break

                current_node = heapq.heappop(open_list)
                if current_node.state_key in closed_list:
                    rec.pop_closed()
                    continue
                closed_list.add(current_node.state_key)
                expansion_idx = expansions
                expansions += 1

                self.env.unwrapped.set_state(current_node.sim_state)
                current_obs = self.env.unwrapped.get_obs()
                current_obs_extra = current_obs["extra"]
                current_features = self._pose_features(current_obs_extra)

                if self._is_better_inter_node(current_node, best_inter_node):
                    best_inter_node = current_node
                if self._is_better_h_node(current_node, best_h_node):
                    best_h_node = current_node

                rec.expanded(
                    current_node,
                    current_features,
                    best_h_node,
                    best_inter_node,
                    len(open_list),
                    len(closed_list),
                )

                if current_node.intersection >= threshold_value:
                    rec.goal_reached(current_node)
                    goal_node = current_node
                    break

                for act_idx, action in enumerate(self.action_primitives):
                    self.env.unwrapped.set_state(current_node.sim_state)

                    for _ in range(self.K):
                        self.env.step(action)

                    child_state = self._clone_state(self.env.unwrapped.get_state())
                    child_obs = self.env.unwrapped.get_obs()
                    child_obs_extra = child_obs["extra"]
                    child_features = self._pose_features(child_obs_extra)
                    action_cost = self._get_action_cost(current_obs_extra, child_obs_extra)
                    child_key = self._state_key(child_features)
                    child_inter = self._intersection()
                    child_heuristic = self._get_heuristic(child_obs_extra)
                    edge_cost = action_cost + self.step_size * G_STEP_COST
                    child_g_value = current_node.g_value + edge_cost

                    closed_hit = child_key in closed_list
                    prior_g = best_g_by_key.get(child_key)

                    rec.generated(
                        action=act_idx,
                        features=child_features,
                        edge_cost=edge_cost,
                        g=child_g_value,
                        h=child_heuristic,
                        intersection=child_inter,
                        closed=closed_hit,
                        prior_g=prior_g,
                    )

                    if closed_hit:
                        continue

                    if prior_g is None or child_g_value < prior_g:
                        best_g_by_key[child_key] = child_g_value

                    child_node = SearchNode(
                        child_state,
                        current_node.action_history + [act_idx],
                        child_g_value,
                        child_heuristic,
                        child_key,
                        intersection=child_inter,
                        parent_expansion=expansion_idx,
                    )

                    if self._is_better_inter_node(child_node, best_inter_node):
                        best_inter_node = child_node
                    if self._is_better_h_node(child_node, best_h_node):
                        best_h_node = child_node

                    heapq.heappush(open_list, child_node)
                    rec.pushed(len(open_list))

        if goal_node is not None:
            result_node = goal_node
        else:
            result_node = best_h_node
            recorder.exhausted(result_node, best_inter_node, best_h_node)

        checkpointer.save(
            expansions, open_list, closed_list, best_g_by_key,
            best_h_node, best_inter_node, goal_reached=goal_node is not None,
        )

        log = recorder.finish(
            result_node,
            goal_reached=goal_node is not None,
            distinct_keys=len(best_g_by_key),
        )

        return self._make_plan_result(
            root_state,
            start_pose,
            result_node.action_history,
            recorder.expansion_xy(),
            log,
        )

    def execute_plan(
        self,
        initial_state,
        plan_action_indicies,
        step_delay=0.0,
        K=None,
        step_size=None,
    ):
        K = self.K if K is None else K
        primitives = (
            self.action_primitives
            if step_size is None
            else self._build_action_primitives(step_size)
        )

        self.env.unwrapped.set_state(initial_state)

        for act_idx in tqdm(plan_action_indicies, desc="Executing", unit="prim"):
            action = primitives[act_idx]

            for _ in range(K):
                self.env.step(action)
                self.env.render()
                if step_delay > 0:
                    time.sleep(step_delay)
