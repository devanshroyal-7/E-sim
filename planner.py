import heapq
import time

import numpy as np
from tqdm import tqdm


class SearchNode:
    def __init__(self, sim_state, action_history, g_value, h_value, state_key):
        self.sim_state = sim_state
        self.action_history = action_history
        self.g_value = g_value
        self.h_value = h_value
        self.f_value = g_value + h_value
        self.state_key = state_key

    def __lt__(self, other):
        return self.f_value < other.f_value


class SimPlanner:
    def __init__(self, env, K_substeps, step_size):
        self.env = env
        self.K = K_substeps
        s = step_size
        d = s / np.sqrt(2)

        xy = np.array([
            [s, 0.0],
            [-s, 0.0],
            [0.0, s],
            [0.0, -s],
            [d, d],
            [-d, d],
            [d, -d],
            [-d, -d],
        ], dtype=np.float32)

        action_dim = env.action_space.shape[-1]
        self.action_primitives = np.zeros((len(xy), action_dim), dtype=np.float32)
        self.action_primitives[:, :2] = xy

    def _to_numpy(self, x):
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _yaw_from_quat(self, quat):
        return 2.0 * np.arctan2(quat[..., 3], quat[..., 0])

    def _state_key(self, obs_extra, decimals=3):
        tcp = self._to_numpy(obs_extra["tcp_pose"])[0]
        obj = self._to_numpy(obs_extra["obj_pose"])[0]
        vals = np.array(
            [tcp[0], tcp[1], obj[0], obj[1], self._yaw_from_quat(obj[3:7])],
            dtype=np.float64,
        )
        return tuple(np.round(vals, decimals=decimals).tolist())

    def _get_heuristic(self, obs_extra):
        obj = self._to_numpy(obs_extra["obj_pose"])
        tcp = self._to_numpy(obs_extra["tcp_pose"])
        goal_xy = self._to_numpy(obs_extra["goal_pos"])[:, :2]
        pos_dist = float(np.linalg.norm(obj[:, :2] - goal_xy))
        tcp_to_obj = float(np.linalg.norm(tcp[:, :2] - obj[:, :2]))

        obj_yaw = self._yaw_from_quat(obj[:, 3:7])
        goal_q = self._to_numpy(self.env.unwrapped.goal_tee.pose.q)
        goal_yaw = self._yaw_from_quat(goal_q)
        yaw_err = obj_yaw - goal_yaw
        ang = float(np.abs(np.arctan2(np.sin(yaw_err), np.cos(yaw_err))).reshape(-1)[0])
        return pos_dist + 0.05 * ang + 0.25 * tcp_to_obj

    def _clone_state(self, state):
        if hasattr(state, "clone"):
            return state.clone()
        return np.array(state, copy=True)

    def plan(self, max_expansions=200, threshold_value=0.02):
        root_state = self._clone_state(self.env.unwrapped.get_state())

        obs = self.env.unwrapped.get_obs()
        obs_extra = obs["extra"]

        root_key = self._state_key(obs_extra)
        root_heuristic = self._get_heuristic(obs_extra)
        root_node = SearchNode(root_state, [], 0.0, root_heuristic, root_key)

        open_list = []
        closed_list = set()
        heapq.heappush(open_list, root_node)

        best_node = root_node
        expansions = 0

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

                if current_node.h_value < best_node.h_value:
                    best_node = current_node

                pbar.set_postfix(
                    best_h=f"{best_node.h_value:.4f}",
                    depth=len(best_node.action_history),
                    open=len(open_list),
                )

                if current_node.h_value < threshold_value:
                    tqdm.write(f"Goal reached in {expansions} expansions")
                    self.env.unwrapped.set_state(root_state)
                    return current_node.action_history

                for act_idx, action in enumerate(self.action_primitives):
                    self.env.unwrapped.set_state(current_node.sim_state)

                    for _ in range(self.K):
                        self.env.step(action)

                    child_state = self._clone_state(self.env.unwrapped.get_state())
                    obs = self.env.unwrapped.get_obs()
                    obs_extra = obs["extra"]
                    child_key = self._state_key(obs_extra)
                    if child_key in closed_list:
                        continue

                    child_heuristic = self._get_heuristic(obs_extra)
                    child_g_value = current_node.g_value + (self.K * 0.01)

                    child_node = SearchNode(
                        child_state,
                        current_node.action_history + [act_idx],
                        child_g_value,
                        child_heuristic,
                        child_key,
                    )
                    heapq.heappush(open_list, child_node)

        tqdm.write("Max expansions reached. Returning trajectory for the closest node")
        self.env.unwrapped.set_state(root_state)
        return best_node.action_history

    def execute_plan(self, initial_state, plan_action_indicies, step_delay=0.0):
        self.env.unwrapped.set_state(initial_state)

        for act_idx in tqdm(plan_action_indicies, desc="Executing", unit="prim"):
            action = self.action_primitives[act_idx]

            for _ in range(self.K):
                self.env.step(action)
                self.env.render()
                if step_delay > 0:
                    time.sleep(step_delay)
