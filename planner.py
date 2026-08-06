import heapq
import time
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

# Body-frame T corners (COM 0.0375 already subtracted), matching ManiSkill PushT.
TEE_LANDMARKS_XY = np.array(
    [
        [0.1, -0.0125],
        [0.1, -0.0625],
        [-0.1, -0.0625],
        [-0.1, -0.0125],
        [-0.025, -0.0125],
        [-0.025, 0.1375],
        [0.025, 0.1375],
        [0.025, -0.0125],
    ],
    dtype=np.float64,
)

G_STEP_COST = 0.1
H_WEIGHT = 1.0


@dataclass
class PlanResult:
    actions: list
    expansion_xy: np.ndarray  # (N, 2) tee COM of each expanded node
    start_pose: np.ndarray
    goal_pose: np.ndarray
    trajectory_xy: np.ndarray  # (M, 2) tee COM along chosen plan, incl. start


class SearchNode:
    def __init__(
        self, sim_state, action_history, g_value, h_value, state_key, intersection=0.0
    ):
        self.sim_state = sim_state
        self.action_history = action_history
        self.g_value = g_value
        self.h_value = h_value
        self.f_value = g_value + H_WEIGHT * h_value
        self.state_key = state_key
        self.intersection = intersection
    
    def __lt__(self, other):
        return self.f_value < other.f_value


class SimPlanner:
    def __init__(self, env, K_substeps, step_size):
        self.env = env
        self.K = K_substeps
        self.step_size = step_size
        self.action_primitives = self._build_action_primitives(step_size)
        self.goal_pose = self._to_numpy(self.env.unwrapped.goal_tee.pose.raw_pose.reshape(-1))
        self.goal_landmarks = self._landmarks_world_xy(self.goal_pose)
    
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
        parent_obj = self._to_numpy(parent_obs_extra["obj_pose"])
        child_obj = self._to_numpy(child_obs_extra["obj_pose"])

        parent_landmarks = self._landmarks_world_xy(parent_obj)     # 8x2
        child_landmarks = self._landmarks_world_xy(child_obj)

        return float(np.max(np.linalg.norm(parent_landmarks - child_landmarks, axis=-1)))

    def _to_numpy(self, x):
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _yaw_from_quat(self, quat):
        return 2.0 * np.arctan2(quat[..., 3], quat[..., 0])

    def _landmarks_world_xy(self, pose):
        pose = self._to_numpy(pose).reshape(-1)
        xy = pose[:2]
        yaw = float(self._yaw_from_quat(pose[3:7]))
        c, s = np.cos(yaw), np.sin(yaw)
        rot = np.array([[c, -s], [s, c]], dtype=np.float64)
        return TEE_LANDMARKS_XY @ rot.T + xy

    def _state_key(self, obs_extra, decimals=2):
        # decimals=2 -> states within 1 cm are the same state
        tcp = self._to_numpy(obs_extra["tcp_pose"])[0]
        obj = self._to_numpy(obs_extra["obj_pose"])[0]
        vals = np.array(
            [tcp[0], tcp[1], obj[0], obj[1], self._yaw_from_quat(obj[3:7])],
            dtype=np.float64,
        )
        return tuple(np.round(vals, decimals=decimals).tolist())

    def _intersection(self):
        inter = self.env.unwrapped.pseudo_render_intersection()
        return float(self._to_numpy(inter).reshape(-1)[0])

    def _get_heuristic(self, obs_extra):
        obj = self._to_numpy(obs_extra["obj_pose"])
        # tcp = self._to_numpy(obs_extra["tcp_pose"])
        
        current = self._landmarks_world_xy(obj)
        
        pose_err = float(np.max(np.linalg.norm(current - self.goal_landmarks, axis=-1)))
        return pose_err 

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

    def _fallback_plan_node(
        self, root_node: SearchNode, best_inter_node: SearchNode, best_h_node: SearchNode
    ) -> SearchNode:
        """Prefer any intersection progress; otherwise return best landmark h."""
        if best_inter_node.intersection > root_node.intersection:
            return best_inter_node
        return best_h_node

    def _obj_xy_from_obs(self, obs_extra):
        obj = self._to_numpy(obs_extra["obj_pose"]).reshape(-1)
        return np.array([obj[0], obj[1]], dtype=np.float64)

    def _obj_pose_from_obs(self, obs_extra):
        return self._to_numpy(obs_extra["obj_pose"]).reshape(-1).astype(np.float64)

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

    def _make_plan_result(self, root_state, start_pose, actions, expansion_xy):
        trajectory_xy = self._replay_trajectory_xy(root_state, actions)
        self.env.unwrapped.set_state(root_state)
        return PlanResult(
            actions=list(actions),
            expansion_xy=np.asarray(expansion_xy, dtype=np.float64).reshape(-1, 2),
            start_pose=np.asarray(start_pose, dtype=np.float64).reshape(-1),
            goal_pose=np.asarray(self.goal_pose, dtype=np.float64).reshape(-1),
            trajectory_xy=trajectory_xy,
        )

    def plan(self, max_expansions=200, threshold_value=None):
        if threshold_value is None:
            threshold_value = float(self.env.unwrapped.intersection_thresh)

        root_state = self._clone_state(self.env.unwrapped.get_state())

        obs = self.env.unwrapped.get_obs()
        obs_extra = obs["extra"]

        start_pose = self._obj_pose_from_obs(obs_extra)
        root_key = self._state_key(obs_extra)
        root_inter = self._intersection()
        root_heuristic = self._get_heuristic(obs_extra)
        root_node = SearchNode(
            root_state, [], 0.0, root_heuristic, root_key, intersection=root_inter
        )

        open_list = []
        closed_list = set()
        heapq.heappush(open_list, root_node)

        best_inter_node = root_node
        best_h_node = root_node
        expansions = 0
        expansion_xy = []

        with tqdm(total=max_expansions, desc="Planning", unit="exp") as pbar:
            while expansions < max_expansions:
                if not open_list:
                    tqdm.write("Open list is empty, returning best path found")
                    break

                current_node = heapq.heappop(open_list)
                if current_node.state_key in closed_list:
                    continue
                closed_list.add(current_node.state_key)
                expansions += 1
                pbar.update(1)

                self.env.unwrapped.set_state(current_node.sim_state)
                current_obs = self.env.unwrapped.get_obs()
                current_obs_extra = current_obs["extra"]
                expansion_xy.append(self._obj_xy_from_obs(current_obs_extra))

                if self._is_better_inter_node(current_node, best_inter_node):
                    best_inter_node = current_node
                if self._is_better_h_node(current_node, best_h_node):
                    best_h_node = current_node

                pbar.set_postfix(
                    best_inter=f"{best_inter_node.intersection:.4f}",
                    inter_depth=len(best_inter_node.action_history),
                    best_h=f"{best_h_node.h_value:.4f}",
                    h_depth=len(best_h_node.action_history),
                    open=len(open_list),
                )

                if current_node.intersection >= threshold_value:
                    tqdm.write(
                        f"Goal reached in {expansions} expansions "
                        f"(intersection={current_node.intersection:.4f})"
                    )
                    return self._make_plan_result(
                        root_state,
                        start_pose,
                        current_node.action_history,
                        expansion_xy,
                    )

                for act_idx, action in enumerate(self.action_primitives):
                    self.env.unwrapped.set_state(current_node.sim_state)

                    for _ in range(self.K):
                        self.env.step(action)

                    child_state = self._clone_state(self.env.unwrapped.get_state())
                    child_obs = self.env.unwrapped.get_obs()
                    child_obs_extra = child_obs["extra"]
                    action_cost = self._get_action_cost(current_obs_extra, child_obs_extra)
                    child_key = self._state_key(child_obs_extra)
                    child_inter = self._intersection()
                    child_heuristic = self._get_heuristic(child_obs_extra)
                    child_g_value = current_node.g_value + action_cost + self.step_size * G_STEP_COST

                    if child_key in closed_list:
                        continue

                    child_node = SearchNode(
                        child_state,
                        current_node.action_history + [act_idx],
                        child_g_value,
                        child_heuristic,
                        child_key,
                        intersection=child_inter,
                    )

                    if self._is_better_inter_node(child_node, best_inter_node):
                        best_inter_node = child_node
                    if self._is_better_h_node(child_node, best_h_node):
                        best_h_node = child_node

                    heapq.heappush(open_list, child_node)

        result_node = self._fallback_plan_node(root_node, best_inter_node, best_h_node)
        tqdm.write(
            "Max expansions reached. Returning trajectory for the closest node "
            f"(best_intersection={best_inter_node.intersection:.4f}, "
            f"best_h={best_h_node.h_value:.4f}, "
            f"returned_depth={len(result_node.action_history)})"
        )
        return self._make_plan_result(
            root_state,
            start_pose,
            result_node.action_history,
            expansion_xy,
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
