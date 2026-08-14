"""The planner's result bundle and its npz round-trip."""
from dataclasses import dataclass

import numpy as np

from search_log import SearchLog


@dataclass
class PlanResult:
    actions: list
    expansion_xy: np.ndarray  # (N, 2) tee COM of each expanded node
    start_pose: np.ndarray
    goal_pose: np.ndarray
    trajectory_xy: np.ndarray  # (M, 2) tee COM along chosen plan, incl. start
    log: SearchLog = None


def save_plan_result(result: PlanResult, path):
    arrays = {
        "actions": np.asarray(result.actions),
        "expansion_xy": np.asarray(result.expansion_xy, dtype=np.float64),
        "start_pose": np.asarray(result.start_pose, dtype=np.float64),
        "goal_pose": np.asarray(result.goal_pose, dtype=np.float64),
        "trajectory_xy": np.asarray(result.trajectory_xy, dtype=np.float64),
    }
    if result.log is not None:
        arrays.update(result.log.npz_dict())
    np.savez_compressed(path, **arrays)


def load_plan_result(path) -> PlanResult:
    data = np.load(path)
    return PlanResult(
        actions=list(data["actions"]),
        expansion_xy=data["expansion_xy"],
        start_pose=data["start_pose"],
        goal_pose=data["goal_pose"],
        trajectory_xy=data["trajectory_xy"],
        log=SearchLog.from_npz(data),
    )
