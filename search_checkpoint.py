"""Save a running search to disk and pick it up again in a later process.

A checkpoint is the search's open list, closed set and best-g table, together
with the recorder's log, flattened into one npz. Nothing has to be re-expanded
to resume: every open node already carries the full sim state `env.set_state`
needs, and the open list is a heapq array, so storing it in order and reading
it back without re-heapifying continues the exact expansion sequence a single
long run would have produced.

`Checkpointer` is the hook the planner calls, in the same shape as
`SearchRecorder`: a `tick` per expansion that usually does nothing, a `save`,
and a `NullCheckpointer` that does neither. The node class is passed into
`restore` rather than imported, so this module depends on nothing in the
planner.

Resuming a search whose cost model or heuristic changed since the checkpoint
was written would leave stale g and h on tens of thousands of open nodes and
silently corrupt the ordering, so `fingerprint` pins down everything the
ordering depends on -- including the source of the heuristic itself -- and
`load_checkpoint` refuses on any mismatch.
"""
import hashlib
import inspect
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from geometry import TEE_LANDMARKS_XY, to_numpy

# Methods whose source is hashed into the fingerprint: between them they define
# g, h and the state key, which is everything that decides expansion order.
FINGERPRINTED_METHODS = (
    "_get_heuristic",
    "_get_action_cost",
    "_state_key",
    "_pose_features",
)

# Fingerprint fields that only warn on mismatch. The threshold is read once per
# pop as a goal test and never enters the ordering, so an open list stays valid
# under a new one.
SOFT_FIELDS = frozenset({"threshold"})

STATE_KEY_WIDTH = 5  # (rel_x, rel_y, obj_x, obj_y, yaw_bin)


class CheckpointMismatch(ValueError):
    """The checkpoint came from a search that would order nodes differently."""


# --------------------------------------------------------------------------- #
# fingerprint
# --------------------------------------------------------------------------- #
def fingerprint(planner, threshold):
    """Everything a resumed search has to agree with the checkpoint on.

    Reads the env's current state as the search root, so call this while the
    env is still at its post-reset pose, not mid-search.
    """
    params = dict(planner._search_params(float(threshold)))
    source = "".join(
        inspect.getsource(getattr(type(planner), name))
        for name in FINGERPRINTED_METHODS
    )
    params.update(
        goal_pose=_hash(planner.goal_pose),
        root_state=_hash(to_numpy(planner.env.unwrapped.get_state())),
        primitives=_hash(planner.action_primitives),
        tee_landmarks=_hash(TEE_LANDMARKS_XY),
        code=hashlib.sha1(source.encode()).hexdigest(),
    )
    return params


def _hash(array):
    return hashlib.sha1(
        np.ascontiguousarray(to_numpy(array), dtype=np.float64).tobytes()
    ).hexdigest()


def _check_fingerprint(saved, current, force):
    hard, soft = [], []
    for key in sorted(set(saved) | set(current)):
        before, after = saved.get(key), current.get(key)
        if before == after:
            continue
        (soft if key in SOFT_FIELDS else hard).append((key, before, after))

    for key, before, after in soft:
        print(f"warning: {key} changed since the checkpoint ({before} -> {after})")

    if not hard:
        return
    diff = "\n".join(f"    {key}: {before} -> {after}" for key, before, after in hard)
    if force:
        print(f"warning: resuming despite a changed search definition:\n{diff}")
        return
    raise CheckpointMismatch(
        "checkpoint was written by a different search, so its open list would "
        f"be ordered by stale values:\n{diff}\n"
        "    re-run without --resume, or pass --force to resume anyway"
    )


# --------------------------------------------------------------------------- #
# checkpoint
# --------------------------------------------------------------------------- #
@dataclass
class SearchCheckpoint:
    fingerprint: dict
    open_nodes: dict  # column -> array, one row per open slot, in heap order
    closed_keys: np.ndarray  # (M, 5)
    best_g_keys: np.ndarray  # (K, 5)
    best_g_values: np.ndarray  # (K,)
    incumbents: dict  # "h" / "inter" -> node snapshot
    expansions: int
    goal_reached: bool
    recorder: dict  # log tables and heap counters, opaque here

    def restore(self, node_cls, like=None):
        """Rebuild what `plan()` keeps in locals. `like` supplies the dtype and
        device for the restored sim states."""
        return (
            _unpack_open(self.open_nodes, node_cls, like),
            {_key(row) for row in self.closed_keys},
            {
                _key(row): float(g)
                for row, g in zip(self.best_g_keys, self.best_g_values)
            },
            _unpack_node(self.incumbents["h"], node_cls),
            _unpack_node(self.incumbents["inter"], node_cls),
            self.expansions,
        )


def _capture(expansions, open_list, closed_list, best_g_by_key, best_h, best_inter,
             recorder, fingerprint, goal_reached):
    best_g = list(best_g_by_key.items())
    return SearchCheckpoint(
        fingerprint=dict(fingerprint),
        open_nodes=_pack_open(open_list),
        closed_keys=_pack_keys(closed_list),
        best_g_keys=_pack_keys(key for key, _ in best_g),
        best_g_values=np.array([g for _, g in best_g], dtype=np.float64),
        incumbents={"h": _pack_node(best_h), "inter": _pack_node(best_inter)},
        expansions=int(expansions),
        goal_reached=bool(goal_reached),
        recorder=recorder.checkpoint_state(),
    )


def _pack_open(open_list):
    lengths = [len(node.action_history) for node in open_list]
    return {
        "sim_state": (
            np.stack([to_numpy(node.sim_state).reshape(-1) for node in open_list])
            if open_list
            else np.zeros((0, 0))
        ),
        "key": _pack_keys(node.state_key for node in open_list),
        "g": np.array([node.g_value for node in open_list], dtype=np.float64),
        "h": np.array([node.h_value for node in open_list], dtype=np.float64),
        "intersection": np.array(
            [node.intersection for node in open_list], dtype=np.float64
        ),
        "parent": np.array(
            [node.parent_expansion for node in open_list], dtype=np.int64
        ),
        "actions_flat": np.fromiter(
            (a for node in open_list for a in node.action_history),
            dtype=np.int16,
            count=sum(lengths),
        ),
        "actions_offsets": np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
    }


def _unpack_open(columns, node_cls, like):
    offsets = columns["actions_offsets"]
    flat = columns["actions_flat"]
    return [
        node_cls(
            _as_sim_state(columns["sim_state"][i], like),
            [int(a) for a in flat[offsets[i] : offsets[i + 1]]],
            float(columns["g"][i]),
            float(columns["h"][i]),
            _key(columns["key"][i]),
            intersection=float(columns["intersection"][i]),
            parent_expansion=int(columns["parent"][i]),
        )
        for i in range(len(columns["g"]))
    ]


def _pack_node(node):
    """The incumbents are only ever read for their plan and their scores, so
    their sim state is not worth storing."""
    return {
        "actions": np.asarray(node.action_history, dtype=np.int16),
        "scores": np.array(
            [node.g_value, node.h_value, node.intersection, node.parent_expansion],
            dtype=np.float64,
        ),
    }


def _unpack_node(packed, node_cls):
    g, h, intersection, parent = packed["scores"]
    return node_cls(
        None,
        [int(a) for a in packed["actions"]],
        float(g),
        float(h),
        None,
        intersection=float(intersection),
        parent_expansion=int(parent),
    )


def _pack_keys(keys):
    keys = list(keys)
    if not keys:
        return np.zeros((0, STATE_KEY_WIDTH), dtype=np.float64)
    return np.asarray(keys, dtype=np.float64)


def _key(row):
    """Rebuild a state key tuple. The leading values are rounded floats and the
    last is a yaw bin index, so this round-trips through float64 exactly."""
    return (*row[:-1].tolist(), int(row[-1]))


def _as_sim_state(row, like):
    tensor = torch.as_tensor(np.ascontiguousarray(row).reshape(1, -1))
    if hasattr(like, "device"):
        tensor = tensor.to(device=like.device, dtype=like.dtype)
    return tensor


# --------------------------------------------------------------------------- #
# npz round-trip
# --------------------------------------------------------------------------- #
def save_checkpoint(checkpoint, path):
    arrays = {
        "closed_keys": checkpoint.closed_keys,
        "best_g_keys": checkpoint.best_g_keys,
        "best_g_values": checkpoint.best_g_values,
        "ctr__expansions": np.asarray(checkpoint.expansions),
        "ctr__goal_reached": np.asarray(int(checkpoint.goal_reached)),
    }
    arrays.update({f"fp__{k}": np.asarray(v) for k, v in checkpoint.fingerprint.items()})
    arrays.update({f"open__{k}": v for k, v in checkpoint.open_nodes.items()})
    for name, packed in checkpoint.incumbents.items():
        arrays.update({f"inc__{name}__{k}": v for k, v in packed.items()})
    recorder = checkpoint.recorder
    arrays.update({f"rec_exp__{k}": v for k, v in recorder["expansions"].items()})
    arrays.update({f"rec_edge__{k}": v for k, v in recorder["edges"].items()})
    arrays.update(
        {f"rec_ctr__{k}": np.asarray(v) for k, v in recorder["counters"].items()}
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # A part-written autosave must not clobber the last good checkpoint.
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
    return path


def load_checkpoint(path, fingerprint=None, force=False):
    data = np.load(Path(path), allow_pickle=False)

    def section(prefix):
        return {k[len(prefix) :]: data[k] for k in data.files if k.startswith(prefix)}

    saved = {k: v.item() for k, v in section("fp__").items()}
    if fingerprint is not None:
        _check_fingerprint(saved, fingerprint, force)

    counters = {k: v.item() for k, v in section("ctr__").items()}
    return SearchCheckpoint(
        fingerprint=saved,
        open_nodes=section("open__"),
        closed_keys=data["closed_keys"],
        best_g_keys=data["best_g_keys"],
        best_g_values=data["best_g_values"],
        incumbents={name: section(f"inc__{name}__") for name in ("h", "inter")},
        expansions=int(counters["expansions"]),
        goal_reached=bool(counters["goal_reached"]),
        recorder={
            "expansions": section("rec_exp__"),
            "edges": section("rec_edge__"),
            "counters": {k: v.item() for k, v in section("rec_ctr__").items()},
        },
    )


# --------------------------------------------------------------------------- #
# planner-facing hooks
# --------------------------------------------------------------------------- #
class Checkpointer:
    """Writes the search to `path` every `every` expansions, and on demand."""

    def __init__(self, path, fingerprint, recorder, every=0):
        self.path = Path(path)
        self.fingerprint = fingerprint
        self.recorder = recorder
        self.every = int(every)
        self._last_saved = None

    def tick(self, expansions, open_list, closed_list, best_g_by_key, best_h,
             best_inter):
        if self._last_saved is None:
            # Anchor on where this run started, so a resume does not immediately
            # rewrite the checkpoint it was just loaded from.
            self._last_saved = expansions
        elif self.every and expansions - self._last_saved >= self.every:
            self.save(
                expansions, open_list, closed_list, best_g_by_key, best_h, best_inter
            )

    def save(self, expansions, open_list, closed_list, best_g_by_key, best_h,
             best_inter, goal_reached=False):
        checkpoint = _capture(
            expansions,
            open_list,
            closed_list,
            best_g_by_key,
            best_h,
            best_inter,
            self.recorder,
            self.fingerprint,
            goal_reached,
        )
        save_checkpoint(checkpoint, self.path)
        self._last_saved = expansions
        tqdm.write(
            f"checkpoint: {expansions} expansions, {len(open_list)} open -> "
            f"{self.path} ({self.path.stat().st_size / 1e6:.1f} MB)"
        )


class NullCheckpointer:
    """Same hooks as `Checkpointer`, writing nothing."""

    def tick(self, expansions, open_list, closed_list, best_g_by_key, best_h,
             best_inter):
        pass

    def save(self, expansions, open_list, closed_list, best_g_by_key, best_h,
             best_inter, goal_reached=False):
        pass
