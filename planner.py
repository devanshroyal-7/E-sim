import heapq
import math
import time
from typing import Union

import numpy as np
from tqdm import tqdm

# Body-frame T corners (COM 0.0375 already subtracted), matching ManiSkill PushT.
TEE_LANDMARKS_XY = np.array(
    [
        [-0.1, -0.0125],
        [0.1, -0.0125],
        [-0.1, -0.0625],
        [0.1, -0.0625],
        [-0.025, 0.1375],
        [0.025, 0.1375],
        [-0.025, -0.0125],
        [0.025, -0.0125],
    ],
    dtype=np.float64,
)

G_STEP_COST = 0.03
H_WEIGHT = 5.0
_G_EPS = 1e-12


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


class OpenListPolicy:
    """Strategy for open/closed duplicate handling during search."""

    def should_expand(self, node: SearchNode, closed: set) -> bool:
        raise NotImplementedError

    def on_expand(self, node: SearchNode, closed: set) -> None:
        raise NotImplementedError

    def should_push_child(self, child: SearchNode, closed: set) -> bool:
        raise NotImplementedError


class BaseAStarPolicy(OpenListPolicy):
    """Closed-set A*: skip closed keys on pop and when generating children."""

    def should_expand(self, node: SearchNode, closed: set) -> bool:
        return node.state_key not in closed

    def on_expand(self, node: SearchNode, closed: set) -> None:
        closed.add(node.state_key)

    def should_push_child(self, child: SearchNode, closed: set) -> bool:
        return child.state_key not in closed


class BestGLazyDiscardPolicy(OpenListPolicy):
    """Best-g map with lazy stale discard on pop and reopen on better g."""

    def __init__(self, root_key, root_g: float = 0.0):
        self.best_g = {root_key: root_g}

    def should_expand(self, node: SearchNode, closed: set) -> bool:
        best = self.best_g.get(node.state_key, math.inf)
        if node.g_value > best + _G_EPS:
            return False
        if node.state_key in closed:
            return False
        return True

    def on_expand(self, node: SearchNode, closed: set) -> None:
        closed.add(node.state_key)
        prev = self.best_g.get(node.state_key, math.inf)
        if node.g_value < prev:
            self.best_g[node.state_key] = node.g_value

    def should_push_child(self, child: SearchNode, closed: set) -> bool:
        prev = self.best_g.get(child.state_key, math.inf)
        if child.g_value >= prev - _G_EPS:
            return False
        self.best_g[child.state_key] = child.g_value
        closed.discard(child.state_key)
        return True


def resolve_open_policy(
    open_policy: Union[str, OpenListPolicy],
    root_key=None,
    root_g: float = 0.0,
) -> OpenListPolicy:
    if isinstance(open_policy, OpenListPolicy):
        return open_policy
    if open_policy == "base":
        return BaseAStarPolicy()
    if open_policy == "best-g":
        if root_key is None:
            raise ValueError("root_key is required for best-g open policy")
        return BestGLazyDiscardPolicy(root_key, root_g=root_g)
    raise ValueError(
        f"Unknown open_policy {open_policy!r}; expected 'base', 'best-g', or OpenListPolicy"
    )


class SimPlanner:
    def __init__(self, env, K_substeps, step_size):
        self.env = env
        self.K = K_substeps
        self.step_size = step_size
        self.action_primitives = self._build_action_primitives(step_size)

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

    def _state_key(self, obs_extra, decimals=3):
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
        tcp = self._to_numpy(obs_extra["tcp_pose"])
        tcp_to_obj = float(np.linalg.norm(tcp[:, :2] - obj[:, :2]))

        goal_p = self._to_numpy(self.env.unwrapped.goal_tee.pose.p).reshape(-1)
        goal_q = self._to_numpy(self.env.unwrapped.goal_tee.pose.q).reshape(-1)
        goal_pose = np.concatenate([goal_p, goal_q])

        current = self._landmarks_world_xy(obj)
        goal = self._landmarks_world_xy(goal_pose)
        pose_err = float(np.mean(np.linalg.norm(current - goal, axis=-1)))
        return pose_err + 0.25 * tcp_to_obj

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

    def plan(
        self,
        max_expansions=200,
        threshold_value=None,
        open_policy: Union[str, OpenListPolicy] = "base",
    ):
        if threshold_value is None:
            threshold_value = float(self.env.unwrapped.intersection_thresh)

        root_state = self._clone_state(self.env.unwrapped.get_state())

        obs = self.env.unwrapped.get_obs()
        obs_extra = obs["extra"]

        root_key = self._state_key(obs_extra)
        root_inter = self._intersection()
        root_heuristic = self._get_heuristic(obs_extra)
        root_node = SearchNode(
            root_state, [], 0.0, root_heuristic, root_key, intersection=root_inter
        )

        policy = resolve_open_policy(open_policy, root_key=root_key, root_g=0.0)

        open_list = []
        closed_list = set()
        heapq.heappush(open_list, root_node)

        best_inter_node = root_node
        best_h_node = root_node
        expansions = 0

        with tqdm(total=max_expansions, desc="Planning", unit="exp") as pbar:
            while expansions < max_expansions:
                if not open_list:
                    tqdm.write("Open list is empty, returning best path found")
                    break

                current_node = heapq.heappop(open_list)
                if not policy.should_expand(current_node, closed_list):
                    continue
                policy.on_expand(current_node, closed_list)
                expansions += 1
                pbar.update(1)

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
                    child_inter = self._intersection()
                    child_heuristic = self._get_heuristic(obs_extra)
                    child_g_value = current_node.g_value + G_STEP_COST

                    child_node = SearchNode(
                        child_state,
                        current_node.action_history + [act_idx],
                        child_g_value,
                        child_heuristic,
                        child_key,
                        intersection=child_inter,
                    )
                    if not policy.should_push_child(child_node, closed_list):
                        continue

                    if self._is_better_inter_node(child_node, best_inter_node):
                        best_inter_node = child_node
                    if self._is_better_h_node(child_node, best_h_node):
                        best_h_node = child_node

                    heapq.heappush(open_list, child_node)

        result = self._fallback_plan_node(root_node, best_inter_node, best_h_node)
        tqdm.write(
            "Max expansions reached. Returning trajectory for the closest node "
            f"(best_intersection={best_inter_node.intersection:.4f}, "
            f"best_h={best_h_node.h_value:.4f}, "
            f"returned_depth={len(result.action_history)})"
        )
        self.env.unwrapped.set_state(root_state)
        return result.action_history

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
