"""Instrumentation for the RRT planner, kept out of the search itself.

Mirrors search_recorder.SearchRecorder's hook pattern, but shaped for RRT's loop
(sample target -> nearest neighbor -> expand -> insert) instead of A*'s open/closed
list. NullRRTRecorder implements the same hooks as no-ops.
"""
import time
from contextlib import contextmanager

import numpy as np
from tqdm import tqdm

from search_log import ColumnLog, SearchLog

EXPANSION_COLUMNS = (
    "g",
    "h",
    "intersection",
    "parent",
    "obj_x",
    "obj_y",
    "obj_yaw",
    "tcp_x",
    "tcp_y",
    "rel_x",
    "rel_y",
    "best_h",
    "best_intersection",
    "tree_size",
    "elapsed",
)


class RRTRecorder:
    """Records one search. Reusing an instance across searches resets it."""

    def __init__(self):
        self._reset()

    def _reset(self):
        self.expansion_log = ColumnLog(EXPANSION_COLUMNS)
        self.params = {}
        self.max_iters = 0
        self.root = None
        self._t0 = time.perf_counter()
        self._bar = None

    @contextmanager
    def run(self, max_iters, params, root):
        self._reset()
        self.max_iters = max_iters
        self.params = dict(params)
        self.root = root
        self._t0 = time.perf_counter()
        with tqdm(total=max_iters, desc="RRT Planning", unit="iter") as bar:
            self._bar = bar
            try:
                yield self
            finally:
                self._bar = None

    def _add_row(self, node, parent_idx, best_h_node, best_inter_node, tree_size):
        obj_x, obj_y, obj_yaw = node.key
        self.expansion_log.add(
            g=node.g_value,
            h=node.h_value,
            intersection=node.intersection,
            parent=parent_idx,
            obj_x=obj_x,
            obj_y=obj_y,
            obj_yaw=obj_yaw,
            tcp_x=node.tcp_xy[0],
            tcp_y=node.tcp_xy[1],
            rel_x=node.rel_xy[0],
            rel_y=node.rel_xy[1],
            best_h=best_h_node.h_value,
            best_intersection=best_inter_node.intersection,
            tree_size=tree_size,
            elapsed=time.perf_counter() - self._t0,
        )

    def log_root(self, root_node):
        """Row 0 of the expansion table, so `parent` indices are self-consistent
        (they index into this table, and the root's own children point at 0)."""
        self._add_row(root_node, -1, root_node, root_node, 1)

    def inserted(self, node, parent_idx, best_h_node, best_inter_node, tree_size):
        if self._bar is not None:
            self._bar.update(1)
        self._add_row(node, parent_idx, best_h_node, best_inter_node, tree_size)
        if self._bar is not None:
            self._bar.set_postfix(
                best_inter=f"{best_inter_node.intersection:.4f}",
                best_h=f"{best_h_node.h_value:.4f}",
                tree=tree_size,
            )

    def goal_reached(self, node, tree_size):
        tqdm.write(
            f"Goal reached in {tree_size - 1} iterations "
            f"(intersection={node.intersection:.4f})"
        )

    def exhausted(self, best_inter_node, best_h_node, returned_depth):
        tqdm.write(
            "Max iters reached. Returning trajectory for the closest node "
            f"(best_intersection={best_inter_node.intersection:.4f}, "
            f"best_h={best_h_node.h_value:.4f}, "
            f"returned_depth={returned_depth})"
        )

    def expansion_xy(self):
        """Tee COM of every logged node, in insertion order (root included)."""
        if not len(self.expansion_log):
            return np.zeros((0, 2), dtype=np.float64)
        columns = self.expansion_log.arrays()
        return np.stack([columns["obj_x"], columns["obj_y"]], axis=1).astype(
            np.float64
        )

    def finish(self, result_node, goal_reached, tree_size):
        return SearchLog(
            expansions=self.expansion_log.arrays(),
            edges={},
            summary={
                "iterations": tree_size - 1,
                "max_iters": self.max_iters,
                "wall_time_s": time.perf_counter() - self._t0,
                "goal_reached": int(goal_reached),
                "root_h": self.root.h_value,
                "root_intersection": self.root.intersection,
                **self.params,
            },
        )


class NullRRTRecorder:
    """Same hooks as `RRTRecorder`, recording nothing."""

    @contextmanager
    def run(self, max_iters, params, root):
        yield self

    def log_root(self, root_node):
        pass

    def inserted(self, node, parent_idx, best_h_node, best_inter_node, tree_size):
        pass

    def goal_reached(self, node, tree_size):
        pass

    def exhausted(self, best_inter_node, best_h_node, returned_depth):
        pass

    def expansion_xy(self):
        return np.zeros((0, 2), dtype=np.float64)

    def finish(self, result_node, goal_reached, tree_size):
        return None
