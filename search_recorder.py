"""Instrumentation for the sim planner search, kept out of the search itself.

The planner drives a recorder through a handful of hooks -- one per search
event -- and the recorder turns those into the two tables described in
`search_log`, the heap counters, the progress bar, and the console messages.
Everything that is only interesting to the report (per-edge `dh`, `df`,
displacements, contact, novelty) is derived here from the parent/child pose
features, so the planner never computes a quantity it does not use.

`NullRecorder` implements the same hooks as no-ops, so a search can run with no
bar, no tables, and no allocation.
"""
import time
from contextlib import contextmanager

import numpy as np
from tqdm import tqdm

from geometry import wrap_pi
from search_log import (
    EDGE_CLOSED,
    EDGE_COLUMNS,
    EDGE_DUP_BETTER,
    EDGE_DUP_WORSE,
    EDGE_NEW,
    EXPANSION_COLUMNS,
    ColumnLog,
    SearchLog,
)

# An edge counts as "moved the T" past these thresholds.
CONTACT_XY_EPS = 1e-4
CONTACT_YAW_EPS = np.deg2rad(0.05)


class SearchRecorder:
    """Records one search. Reusing an instance across searches resets it."""

    def __init__(self):
        self._reset()

    def _reset(self):
        self.expansion_log = ColumnLog(EXPANSION_COLUMNS)
        self.edge_log = ColumnLog(EDGE_COLUMNS)
        self.params = {}
        self.max_expansions = 0
        self.root = None
        self.primitives_xy = np.zeros((0, 2))
        self.pops_skipped_closed = 0
        self.pushes = 0
        self.peak_open = 0
        self._t0 = time.perf_counter()
        self._bar = None
        # Set by `expanded`, so `generated` can describe an edge with child-side
        # values only.
        self._parent_node = None
        self._parent_features = None

    @contextmanager
    def run(self, max_expansions, params, root, primitives_xy):
        """Own the progress bar for the duration of the search loop."""
        self._reset()
        self.max_expansions = max_expansions
        self.params = dict(params)
        self.root = root
        self.primitives_xy = primitives_xy
        # The root is already on the open list by the time the loop starts.
        self.pushes = 1
        self.peak_open = 1
        self._t0 = time.perf_counter()
        with tqdm(total=max_expansions, desc="Planning", unit="exp") as bar:
            self._bar = bar
            try:
                yield self
            finally:
                self._bar = None

    def pop_closed(self):
        self.pops_skipped_closed += 1

    def expanded(
        self, node, features, best_h_node, best_inter_node, open_size, closed_size
    ):
        """One popped-and-expanded node. Heap counters are a snapshot taken
        before this node's children are generated."""
        self._parent_node = node
        self._parent_features = features
        if self._bar is not None:
            self._bar.update(1)

        self.expansion_log.add(
            f=node.f_value,
            g=node.g_value,
            h=node.h_value,
            depth=len(node.action_history),
            intersection=node.intersection,
            parent=node.parent_expansion,
            obj_x=features["obj_xy"][0],
            obj_y=features["obj_xy"][1],
            obj_yaw=features["yaw"],
            tcp_x=features["tcp_xy"][0],
            tcp_y=features["tcp_xy"][1],
            rel_x=features["rel_xy"][0],
            rel_y=features["rel_xy"][1],
            best_h=best_h_node.h_value,
            best_intersection=best_inter_node.intersection,
            open_size=open_size,
            closed_size=closed_size,
            pushes=self.pushes,
            elapsed=time.perf_counter() - self._t0,
        )

        if self._bar is not None:
            self._bar.set_postfix(
                best_inter=f"{best_inter_node.intersection:.4f}",
                inter_depth=len(best_inter_node.action_history),
                best_h=f"{best_h_node.h_value:.4f}",
                h_depth=len(best_h_node.action_history),
                open=open_size,
            )

    def generated(
        self,
        action,
        features,
        edge_cost,
        g,
        h,
        intersection,
        closed,
        prior_g,
    ):
        """One generated successor of the node last passed to `expanded`."""
        parent = self._parent_node
        parent_features = self._parent_features

        obj_disp = float(
            np.linalg.norm(features["obj_xy"] - parent_features["obj_xy"])
        )
        dyaw = float(wrap_pi(features["yaw"] - parent_features["yaw"]))
        delta_h = h - parent.h_value

        self.edge_log.add(
            parent=len(self.expansion_log) - 1,
            action=action,
            depth=len(parent.action_history) + 1,
            cost=edge_cost,
            dh=delta_h,
            df=edge_cost + self.params["h_weight"] * delta_h,
            dintersection=intersection - parent.intersection,
            obj_disp=obj_disp,
            dyaw=dyaw,
            tcp_disp=float(
                np.linalg.norm(features["tcp_xy"] - parent_features["tcp_xy"])
            ),
            contact=int(obj_disp > CONTACT_XY_EPS or abs(dyaw) > CONTACT_YAW_EPS),
            novelty=self._novelty(closed, prior_g, g),
        )

    @staticmethod
    def _novelty(closed, prior_g, g):
        if closed:
            return EDGE_CLOSED
        if prior_g is None:
            return EDGE_NEW
        return EDGE_DUP_BETTER if g < prior_g else EDGE_DUP_WORSE

    def pushed(self, open_size):
        self.pushes += 1
        self.peak_open = max(self.peak_open, open_size)

    def open_exhausted(self):
        tqdm.write("Open list is empty, returning best path found")

    def goal_reached(self, node):
        tqdm.write(
            f"Goal reached in {len(self.expansion_log)} expansions "
            f"(intersection={node.intersection:.4f})"
        )

    def exhausted(self, result_node, best_inter_node, best_h_node):
        tqdm.write(
            "Max expansions reached. Returning trajectory for the closest node "
            f"(best_intersection={best_inter_node.intersection:.4f}, "
            f"best_h={best_h_node.h_value:.4f}, "
            f"returned_depth={len(result_node.action_history)})"
        )

    def expansion_xy(self):
        """Tee COM of every expanded node, in expansion order."""
        if not len(self.expansion_log):
            return np.zeros((0, 2), dtype=np.float64)
        columns = self.expansion_log.arrays()
        return np.stack([columns["obj_x"], columns["obj_y"]], axis=1).astype(
            np.float64
        )

    def finish(self, result_node, goal_reached, distinct_keys):
        n_expansions = len(self.expansion_log)
        return SearchLog(
            expansions=self.expansion_log.arrays(),
            edges=self.edge_log.arrays(),
            summary={
                "expansions": n_expansions,
                "edges": len(self.edge_log),
                "max_expansions": self.max_expansions,
                # Every pop is either discarded as closed or expanded.
                "pops": n_expansions + self.pops_skipped_closed,
                "pops_skipped_closed": self.pops_skipped_closed,
                "pushes": self.pushes,
                "peak_open": self.peak_open,
                "distinct_keys": distinct_keys,
                "wall_time_s": time.perf_counter() - self._t0,
                "goal_reached": int(goal_reached),
                "returned_depth": len(result_node.action_history),
                "root_h": self.root.h_value,
                "root_intersection": self.root.intersection,
                **self.params,
            },
            primitives_xy=np.asarray(self.primitives_xy, dtype=np.float64),
        )


class NullRecorder:
    """Same hooks as `SearchRecorder`, recording nothing."""

    @contextmanager
    def run(self, max_expansions, params, root, primitives_xy):
        yield self

    def pop_closed(self):
        pass

    def expanded(
        self, node, features, best_h_node, best_inter_node, open_size, closed_size
    ):
        pass

    def generated(
        self, action, features, edge_cost, g, h, intersection, closed, prior_g
    ):
        pass

    def pushed(self, open_size):
        pass

    def open_exhausted(self):
        pass

    def goal_reached(self, node):
        pass

    def exhausted(self, result_node, best_inter_node, best_h_node):
        pass

    def expansion_xy(self):
        return np.zeros((0, 2), dtype=np.float64)

    def finish(self, result_node, goal_reached, distinct_keys):
        return None
