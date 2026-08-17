"""Per-expansion / per-edge instrumentation for the sim planner search.

The search writes two tables. `expansions` has one row per popped-and-expanded
node, in expansion order, so row i is the i-th expansion and `parent` indexes
back into the same table (root is -1). `edges` has one row per generated
successor, including the ones immediately discarded because their state key was
already closed, with `parent` indexing into the expansion table.

Everything that can be derived from those two tables is derived in
`format_report`; `summary` only carries counters the tables cannot recover
(heap traffic, wall time, search parameters).
"""
from dataclasses import dataclass, field

import numpy as np

# How an edge related to the states the search had already seen.
EDGE_NEW = 0
EDGE_CLOSED = 1
EDGE_DUP_WORSE = 2
EDGE_DUP_BETTER = 3
EDGE_NOVELTY_NAMES = {
    EDGE_NEW: "new key",
    EDGE_CLOSED: "already closed",
    EDGE_DUP_WORSE: "duplicate, no better g",
    EDGE_DUP_BETTER: "duplicate, better g",
}

EXPANSION_COLUMNS = (
    "f",
    "g",
    "h",
    "depth",
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
    "open_size",
    "closed_size",
    "pushes",
    "elapsed",
)

EDGE_COLUMNS = (
    "parent",
    "action",
    "depth",
    "cost",
    "dh",
    "df",
    "dintersection",
    "obj_disp",
    "dyaw",
    "tcp_disp",
    "contact",
    "novelty",
)

PLATEAU_TOL = 1e-6


class ColumnLog:
    """Append-only named columns, materialised as numpy arrays at the end."""

    def __init__(self, names):
        self.columns = {name: [] for name in names}

    @classmethod
    def from_arrays(cls, names, columns):
        """Reopen a materialised log for appending, as a resumed search does."""
        log = cls(names)
        if not columns:
            return log
        if set(columns) != set(log.columns):
            missing = sorted(set(log.columns) - set(columns))
            unknown = sorted(set(columns) - set(log.columns))
            raise KeyError(f"column mismatch (missing={missing}, unknown={unknown})")
        for name in log.columns:
            log.columns[name] = list(columns[name])
        return log

    def add(self, **values):
        if values.keys() != self.columns.keys():
            missing = sorted(set(self.columns) - set(values))
            unknown = sorted(set(values) - set(self.columns))
            raise KeyError(f"column mismatch (missing={missing}, unknown={unknown})")
        for name, value in values.items():
            self.columns[name].append(value)

    def arrays(self):
        return {name: np.asarray(values) for name, values in self.columns.items()}

    def __len__(self):
        for values in self.columns.values():
            return len(values)
        return 0


@dataclass
class SearchLog:
    expansions: dict  # column name -> (N,) array, one row per expansion
    edges: dict  # column name -> (E,) array, one row per generated successor
    summary: dict  # scalar counters and search parameters
    primitives_xy: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))

    def npz_dict(self):
        out = {f"exp__{k}": v for k, v in self.expansions.items()}
        out.update({f"edge__{k}": v for k, v in self.edges.items()})
        out.update({f"sum__{k}": np.asarray(v) for k, v in self.summary.items()})
        out["prim__xy"] = np.asarray(self.primitives_xy, dtype=np.float64)
        return out

    @classmethod
    def from_npz(cls, data):
        def section(prefix):
            return {
                k[len(prefix) :]: data[k] for k in data.files if k.startswith(prefix)
            }

        expansions = section("exp__")
        if not expansions:
            return None
        return cls(
            expansions=expansions,
            edges=section("edge__"),
            summary={k: v.item() for k, v in section("sum__").items()},
            primitives_xy=data["prim__xy"]
            if "prim__xy" in data.files
            else np.zeros((0, 2)),
        )


def _pct(part, whole):
    return 100.0 * part / whole if whole else 0.0


def _longest_run(mask):
    """Length of the longest run of True in a boolean array."""
    best = run = 0
    for flag in np.asarray(mask, dtype=bool):
        run = run + 1 if flag else 0
        best = max(best, run)
    return best


def _last_improvement(values):
    """Index of the last strict decrease in `values` (0 if it never improves)."""
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2:
        return 0
    drops = np.flatnonzero(np.diff(values) < 0.0)
    return int(drops[-1]) + 1 if drops.size else 0


def _hist_lines(counts, label, per_line=8):
    entries = [f"{label}={i}: {int(c)}" for i, c in enumerate(counts) if c]
    return [
        "    " + "  ".join(entries[i : i + per_line])
        for i in range(0, len(entries), per_line)
    ]


def format_report(log, actions=None, plateau_tol=PLATEAU_TOL):
    """Human-readable diagnostic dump of a finished search."""
    if log is None:
        return "no search log recorded"

    exp = log.expansions
    edges = log.edges
    s = log.summary
    n_exp = len(exp.get("f", ()))
    n_edge = len(edges.get("df", ()))
    if n_exp == 0:
        return "search log is empty (no expansions)"

    lines = ["", "=" * 72, "search diagnostics", "=" * 72]

    # --- totals -----------------------------------------------------------
    wall = float(s.get("wall_time_s", 0.0))
    rate = n_exp / wall if wall > 0 else float("nan")
    lines += [
        "totals",
        f"    expansions            {n_exp} / {int(s.get('max_expansions', 0))}"
        f"  ({wall:.1f} s, {rate:.1f} exp/s)",
        f"    edges generated       {n_edge}",
        f"    heap pushes / pops    {int(s.get('pushes', 0))} / {int(s.get('pops', 0))}"
        f"  ({int(s.get('pops_skipped_closed', 0))} pops discarded as closed)",
        f"    peak open list        {int(s.get('peak_open', 0))}",
        f"    distinct keys seen    {int(s.get('distinct_keys', 0))}",
    ]

    # --- progress ---------------------------------------------------------
    best_h = np.asarray(exp["best_h"], dtype=np.float64)
    best_inter = np.asarray(exp["best_intersection"], dtype=np.float64)
    stall = n_exp - 1 - _last_improvement(best_h)
    lines += [
        "progress",
        f"    h            root {float(s.get('root_h', np.nan)):.4f}"
        f" -> best {best_h[-1]:.4f}  (last improved {stall} expansions ago)",
        f"    intersection root {float(s.get('root_intersection', np.nan)):.4f}"
        f" -> best {best_inter[-1]:.4f}  (threshold"
        f" {float(s.get('threshold', np.nan)):.4f},"
        f" reached: {'yes' if s.get('goal_reached') else 'no'})",
        f"    returned plan depth   {int(s.get('returned_depth', 0))}",
    ]

    df = np.asarray(edges.get("df", ()), dtype=np.float64)
    dh = np.asarray(edges.get("dh", ()), dtype=np.float64)
    cost = np.asarray(edges.get("cost", ()), dtype=np.float64)
    dinter = np.asarray(edges.get("dintersection", ()), dtype=np.float64)
    contact = np.asarray(edges.get("contact", ()), dtype=bool)
    parents = np.asarray(edges.get("parent", ()), dtype=np.int64)

    # --- heuristic geometry ----------------------------------------------
    if n_edge:
        plateau = int(np.count_nonzero(np.abs(df) <= plateau_tol))
        negative = int(np.count_nonzero(df < -plateau_tol))
        f_vals = np.asarray(exp["f"], dtype=np.float64)
        running_max = np.maximum.accumulate(f_vals)
        violations = int(np.count_nonzero(f_vals < running_max - plateau_tol))
        max_drop = float(np.max(running_max - f_vals)) if n_exp else 0.0
        lines += [
            "heuristic geometry  (df = edge cost + w*dh, per generated edge)",
            f"    df     mean {df.mean():+.5f}  median {np.median(df):+.5f}"
            f"  min {df.min():+.5f}  max {df.max():+.5f}",
            f"    dh     mean {dh.mean():+.5f}  min {dh.min():+.5f}"
            f"  max {dh.max():+.5f}"
            f"  ({_pct(np.count_nonzero(dh < -plateau_tol), n_edge):.1f}% reduce h)",
            # A* can only discriminate between successors when df varies: a
            # constant df (zero or not) leaves it expanding the level blindly.
            f"    df spread  std {df.std():.5f}"
            f"  iqr {np.subtract(*np.percentile(df, [75, 25])):.5f}"
            f"  within {plateau_tol:g} of median:"
            f" {_pct(np.count_nonzero(np.abs(df - np.median(df)) <= plateau_tol), n_edge):.1f}%",
            f"    plateau edges |df| <= {plateau_tol:g}:  {plateau}"
            f" ({_pct(plateau, n_edge):.1f}%)",
            f"    inconsistent edges df < 0:      {negative}"
            f" ({_pct(negative, n_edge):.1f}%)",
            f"    popped f monotonicity: {violations} violations,"
            f" max drop {max_drop:.5f}",
        ]

    # --- object motion ----------------------------------------------------
    if n_edge:
        obj_disp = np.asarray(edges["obj_disp"], dtype=np.float64)
        dyaw = np.asarray(edges["dyaw"], dtype=np.float64)
        tcp_disp = np.asarray(edges["tcp_disp"], dtype=np.float64)
        per_exp_contact = np.bincount(parents[contact], minlength=n_exp)
        touched = contact.any()
        lines += [
            "object motion",
            f"    edges that moved the T          {int(contact.sum())}"
            f" ({_pct(contact.sum(), n_edge):.1f}%)",
            f"    expansions with >=1 such edge   "
            f"{int(np.count_nonzero(per_exp_contact))}"
            f" ({_pct(np.count_nonzero(per_exp_contact), n_exp):.1f}%)",
            f"    longest no-contact run          "
            f"{_longest_run(per_exp_contact == 0)} expansions",
            f"    on contact edges: |dxy| mean "
            f"{(obj_disp[contact].mean() if touched else 0.0) * 1000:.2f} mm,"
            f" max {(obj_disp.max() if n_edge else 0.0) * 1000:.2f} mm,"
            f" |dyaw| mean "
            f"{np.rad2deg(np.abs(dyaw[contact]).mean()) if touched else 0.0:.2f} deg",
            # step_size is a normalised delta-pose command, not a distance, so
            # this is the only place the achieved travel per primitive shows up.
            f"    tcp travel per primitive  mean {tcp_disp.mean() * 1000:.2f} mm,"
            f" max {tcp_disp.max() * 1000:.2f} mm"
            f"  (step_size {float(s.get('step_size', np.nan)):.3f}"
            f" x {int(s.get('k_substeps', 0))} substeps)",
        ]

    # --- redundancy -------------------------------------------------------
    obj_bins = np.stack(
        [
            np.round(np.asarray(exp["obj_x"], dtype=np.float64), 2),
            np.round(np.asarray(exp["obj_y"], dtype=np.float64), 2),
        ],
        axis=1,
    )
    yaw_res = float(s.get("yaw_res_deg", 5.0))
    yaw_bin = np.floor(np.rad2deg(np.asarray(exp["obj_yaw"])) / yaw_res)
    pose_bins = np.column_stack([obj_bins, yaw_bin])
    n_xy = len(np.unique(obj_bins, axis=0))
    n_pose = len(np.unique(pose_bins, axis=0))
    lines += [
        "state-space redundancy  (expanded nodes only)",
        f"    distinct object xy bins (1 cm)          {n_xy}",
        f"    distinct object poses (1 cm, {yaw_res:.0f} deg)      {n_pose}",
        f"    expansions per distinct object pose     {n_exp / max(n_pose, 1):.1f}"
        "   <- tcp shuffling per T pose",
    ]

    # --- depth ------------------------------------------------------------
    depth = np.asarray(exp["depth"], dtype=np.int64)
    max_generated = (
        int(np.max(np.asarray(edges["depth"], dtype=np.int64))) if n_edge else 0
    )
    lines += [
        f"depth of expanded nodes  (max expanded {int(depth.max())},"
        f" max generated {max_generated})",
    ]
    lines += _hist_lines(np.bincount(depth), "d")

    # --- per primitive ----------------------------------------------------
    if n_edge:
        action = np.asarray(edges["action"], dtype=np.int64)
        n_prim = int(action.max()) + 1
        prim_xy = np.asarray(log.primitives_xy, dtype=np.float64)
        if actions is not None and len(actions):
            plan_counts = np.bincount(
                np.asarray(actions, dtype=np.int64), minlength=n_prim
            )
        else:
            plan_counts = np.zeros(n_prim, dtype=np.int64)
        lines += [
            "per primitive",
            "    idx   dx      dy    |  count  contact%   mean cost"
            "    mean dh   mean dinter   in plan",
        ]
        for a in range(n_prim):
            sel = action == a
            count = int(sel.sum())
            if count == 0:
                continue
            dx, dy = (prim_xy[a] if a < len(prim_xy) else (np.nan, np.nan))
            lines.append(
                f"    {a:<3d} {dx:+.3f} {dy:+.3f}  | {count:6d}"
                f"  {_pct(contact[sel].sum(), count):7.1f}"
                f"   {cost[sel].mean():+.5f}"
                f"   {dh[sel].mean():+.5f}"
                f"   {dinter[sel].mean():+.6f}"
                f"   {int(plan_counts[a]) if a < len(plan_counts) else 0:7d}"
            )

    # --- edge fate --------------------------------------------------------
    if n_edge:
        novelty = np.asarray(edges["novelty"], dtype=np.int64)
        lines.append("edge fate")
        for code, name in EDGE_NOVELTY_NAMES.items():
            count = int(np.count_nonzero(novelty == code))
            lines.append(
                f"    {name:<24s} {count:7d} ({_pct(count, n_edge):5.1f}%)"
            )

    lines.append("=" * 72)
    return "\n".join(lines)
