"""A resumed search must expand exactly what one uninterrupted run would.

Splitting a run in two and comparing the expansion log column by column is the
only check that really covers the restore: the open list is a heap array, so
any drift in its order, in the closed set, or in the best-g table shows up as a
different node being popped a few expansions later.
"""
import gymnasium as gym
import mani_skill.envs  # noqa: F401  (registers PushT-v1)
import numpy as np
import pytest

from planner import SimPlanner
from search_checkpoint import Checkpointer, fingerprint, load_checkpoint
from search_recorder import SearchRecorder

ENV_KWARGS = dict(obs_mode="state_dict", control_mode="pd_ee_delta_pose")
K_SUBSTEPS = 10
STEP_SIZE = 0.2
THRESHOLD = 0.90
SPLIT, TOTAL = 60, 120


def _plan(max_expansions, checkpoint_path, resume_path=None):
    env = gym.make("PushT-v1", **ENV_KWARGS, render_mode=None)
    try:
        env.reset(seed=0)
        planner = SimPlanner(env, K_substeps=K_SUBSTEPS, step_size=STEP_SIZE)
        search_id = fingerprint(planner, THRESHOLD)
        resume = (
            load_checkpoint(resume_path, search_id) if resume_path is not None else None
        )
        recorder = SearchRecorder()
        return planner.plan(
            max_expansions=max_expansions,
            threshold_value=THRESHOLD,
            recorder=recorder,
            resume=resume,
            checkpointer=Checkpointer(checkpoint_path, search_id, recorder),
        )
    finally:
        env.close()


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("resume")
    _plan(SPLIT, tmp / "split.npz")
    return {
        "resumed": _plan(TOTAL, tmp / "resumed.npz", resume_path=tmp / "split.npz"),
        "single": _plan(TOTAL, tmp / "single.npz"),
    }


@pytest.mark.parametrize(
    "column", ["f", "g", "h", "parent", "depth", "obj_x", "obj_y", "intersection"]
)
def test_expansion_log_matches_single_run(runs, column):
    resumed = runs["resumed"].log.expansions[column]
    single = runs["single"].log.expansions[column]
    assert len(resumed) == len(single) == TOTAL
    np.testing.assert_array_equal(resumed, single)


def test_plan_matches_single_run(runs):
    assert runs["resumed"].actions == runs["single"].actions


def test_edge_log_matches_single_run(runs):
    resumed = runs["resumed"].log.edges
    single = runs["single"].log.edges
    for column in ("parent", "action", "cost", "dh", "novelty"):
        np.testing.assert_array_equal(resumed[column], single[column])


def test_edited_heuristic_is_refused():
    """The reason the fingerprint hashes source: a resumed open list scored by
    the old heuristic would be ordered by numbers the new one never produced."""
    env = gym.make("PushT-v1", **ENV_KWARGS, render_mode=None)
    try:
        env.reset(seed=0)

        class EditedHeuristic(SimPlanner):
            def _get_heuristic(self, obs_extra):
                return 2.0 * super()._get_heuristic(obs_extra)

        before = fingerprint(SimPlanner(env, K_SUBSTEPS, STEP_SIZE), THRESHOLD)
        after = fingerprint(EditedHeuristic(env, K_SUBSTEPS, STEP_SIZE), THRESHOLD)
        assert before["code"] != after["code"]
        assert {k: v for k, v in before.items() if k != "code"} == {
            k: v for k, v in after.items() if k != "code"
        }
    finally:
        env.close()


def test_counters_match_single_run(runs):
    resumed = runs["resumed"].log.summary
    single = runs["single"].log.summary
    for key in ("pushes", "pops", "pops_skipped_closed", "peak_open", "distinct_keys"):
        assert resumed[key] == single[key], key
