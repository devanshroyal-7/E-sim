"""Instrumentation for the RRT planner, kept out of the search itself.

Mirrors search_recorder.SearchRecorder's hook pattern, but shaped for RRT's loop
(sample target -> nearest neighbor -> expand -> insert) instead of A*'s open/closed
list. NullRRTRecorder implements the same hooks as no-ops.

The recorder also owns the run's lifecycle in the other direction: it traps
SIGINT and raises `stop_requested` instead of unwinding, so Ctrl-C ends the
search at an iteration boundary rather than mid-insert, where the tree and the
expansion log would disagree.
"""
import signal
import time
from contextlib import contextmanager

import numpy as np
from tqdm import tqdm

from search_log import ColumnLog, SearchLog

EXPANSION_COLUMNS = (
    "intersection",
    "parent",
    "obj_x",
    "obj_y",
    "obj_yaw",
    "tcp_x",
    "tcp_y",
    "rel_x",
    "rel_y",
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
        self.stop_requested = False
        self._t0 = time.perf_counter()
        self._elapsed_offset = 0.0
        self._resumed = False
        self._bar = None
        self.reposition_attempts = 0
        self.reposition_successes = 0

    def resume_from(self, checkpoint):
        """Reopen a checkpoint's table so this run appends to it: row indices
        stay valid and the report covers both runs."""
        state = checkpoint.recorder
        self.expansion_log = ColumnLog.from_arrays(EXPANSION_COLUMNS, state["expansions"])
        self._elapsed_offset = float(state["counters"]["elapsed_s"])
        self.reposition_attempts = int(state["counters"].get("reposition_attempts", 0))
        self.reposition_successes = int(state["counters"].get("reposition_successes", 0))
        self._resumed = True

    def checkpoint_state(self):
        """The half of a checkpoint the recorder owns."""
        return {
            "expansions": self.expansion_log.arrays(),
            "counters": {
                "elapsed_s": self._elapsed(),
                "reposition_attempts": self.reposition_attempts,
                "reposition_successes": self.reposition_successes,
            },
        }

    def _elapsed(self):
        return self._elapsed_offset + time.perf_counter() - self._t0

    @contextmanager
    def run(self, max_iters, params, root, initial_iters=0):
        """Own the progress bar and the interrupt for the search loop.

        `initial_iters` is the loop's own iteration count (expand attempts +
        reposition attempts), so the bar advances 1:1 with the loop's own
        `iters < max_iters` condition instead of lagging behind whenever a
        stretch of iterations goes to repositions rather than expansions."""
        if self._resumed:
            self._resumed = False
        else:
            self._reset()
        self.max_iters = max_iters
        self.params = dict(params)
        self.root = root
        self.stop_requested = False
        self._t0 = time.perf_counter()
        with tqdm(
            total=max_iters,
            initial=initial_iters,
            desc="RRT Planning",
            unit="iter",
        ) as bar:
            self._bar = bar
            try:
                with self._stop_on_sigint():
                    yield self
            finally:
                self._bar = None

    @contextmanager
    def _stop_on_sigint(self):
        def handler(signum, frame):
            self.stop_requested = True
            # Hand SIGINT back, so a second Ctrl-C aborts immediately.
            signal.signal(signal.SIGINT, previous)
            tqdm.write(
                "interrupt received, stopping after this iteration "
                "(Ctrl-C again to abort now)"
            )

        try:
            previous = signal.signal(signal.SIGINT, handler)
        except ValueError:
            # Not the main thread, so leave interrupt handling alone.
            yield
            return
        try:
            yield
        finally:
            signal.signal(signal.SIGINT, previous)

    def _add_row(self, node, parent_idx, best_inter_node, tree_size):
        obj_x, obj_y, obj_yaw = node.key
        self.expansion_log.add(
            intersection=node.intersection,
            parent=parent_idx,
            obj_x=obj_x,
            obj_y=obj_y,
            obj_yaw=obj_yaw,
            tcp_x=node.tcp_xy[0],
            tcp_y=node.tcp_xy[1],
            rel_x=node.rel_xy[0],
            rel_y=node.rel_xy[1],
            best_intersection=best_inter_node.intersection,
            tree_size=tree_size,
            elapsed=self._elapsed(),
        )

    def log_root(self, root_node):
        """Row 0 of the expansion table, so `parent` indices are self-consistent
        (they index into this table, and the root's own children point at 0)."""
        self._add_row(root_node, -1, root_node, 1)

    def inserted(self, node, parent_idx, best_inter_node, tree_size):
        if self._bar is not None:
            self._bar.update(1)
        self._add_row(node, parent_idx, best_inter_node, tree_size)
        if self._bar is not None:
            self._bar.set_postfix(
                best_inter=f"{best_inter_node.intersection:.4f}",
                tree=tree_size,
            )

    def repositioned(self, node, node_idx, success):
        if self._bar is not None:
            self._bar.update(1)
        self.reposition_attempts += 1
        if success:
            self.reposition_successes += 1

    def goal_reached(self, node, tree_size, iters):
        tqdm.write(
            f"Goal reached in {iters} iterations ({tree_size - 1} expansions) "
            f"(intersection={node.intersection:.4f})"
        )

    def exhausted(self, best_inter_node, returned_depth):
        tqdm.write(
            "Max iters reached. Returning trajectory for the closest node "
            f"(best_intersection={best_inter_node.intersection:.4f}, "
            f"returned_depth={returned_depth})"
        )

    def interrupted(self, best_inter_node, returned_depth):
        tqdm.write(
            "Interrupted. Returning trajectory for the closest node so far "
            f"(best_intersection={best_inter_node.intersection:.4f}, "
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

    def finish(self, result_node, goal_reached, tree_size, iters):
        return SearchLog(
            expansions=self.expansion_log.arrays(),
            edges={},
            summary={
                "iterations": iters,
                "expansions": tree_size - 1,
                "reposition_attempts": self.reposition_attempts,
                "reposition_successes": self.reposition_successes,
                "max_iters": self.max_iters,
                "wall_time_s": self._elapsed(),
                "goal_reached": int(goal_reached),
                "root_intersection": self.root.intersection,
                **self.params,
            },
        )


class NullRRTRecorder:
    """Same hooks as `RRTRecorder`, recording nothing."""

    stop_requested = False

    @contextmanager
    def run(self, max_iters, params, root):
        yield self

    def resume_from(self, checkpoint):
        pass

    def checkpoint_state(self):
        return {"expansions": {}, "counters": {"elapsed_s": 0.0}}

    def log_root(self, root_node):
        pass

    def inserted(self, node, parent_idx, best_inter_node, tree_size):
        pass

    def repositioned(self, node, node_idx, success):
        pass

    def goal_reached(self, node, tree_size, iters):
        pass

    def exhausted(self, best_inter_node, returned_depth):
        pass

    def interrupted(self, best_inter_node, returned_depth):
        pass

    def expansion_xy(self):
        return np.zeros((0, 2), dtype=np.float64)

    def finish(self, result_node, goal_reached, tree_size, iters):
        return None
