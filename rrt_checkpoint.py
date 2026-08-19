"""Save a running RRT search to disk and pick it up again in a later process.

RRT has no open/closed split: every node in the tree, not just some frontier,
can be picked as a nearest neighbor for a future expansion, so a checkpoint
stores the whole tree table rather than a subset of it. Each row already
carries the sim_state `env.set_state()` needs and its parent's row index, so
resuming needs no re-expansion -- nearest-neighbor search and the two
incumbents (best_inter_node, best_h_node) all fall out of the restored table
directly, and any node's full action path is just a walk up `parent`.

`Checkpointer` is the hook the planner calls, in the same shape as
`RRTRecorder`: a `tick` per iteration that usually does nothing, a `save`, and
a `NullCheckpointer` that does neither.

Resuming a search whose heuristic or geometry changed since the checkpoint was
written would silently corrupt nearest-neighbor search and every node's g/h,
so `fingerprint` pins down everything scoring depends on and `load_checkpoint`
refuses on any mismatch.
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
# g, h and the nearest-neighbor metric, which is everything that decides how
# the tree grows and which nodes look best.
FINGERPRINTED_METHODS = (
    "_heuristic",
    "_landmark_dist",
    "_pose_features",
    "_expand",
)

# Fingerprint fields that only warn on mismatch. The threshold is read once per
# insert as a goal test and never enters scoring, so a tree stays valid under
# a new one.
SOFT_FIELDS = frozenset({"threshold"})


class CheckpointMismatch(ValueError):
    """The checkpoint came from a search that would score/grow the tree differently."""


# --------------------------------------------------------------------------- #
# fingerprint
# --------------------------------------------------------------------------- #
def fingerprint(planner, threshold):
    """Everything a resumed search has to agree with the checkpoint on.

    Reads the env's current state as the search root, so call this while the
    env is still at its post-reset pose, not mid-search.
    """
    params = {
        "threshold": float(threshold),
        "step_size": planner.step_size,
        "k_substeps": planner.K,
        "n_candidates": planner.n_candidates,
        "angle_jitter_scale": planner.angle_jitter_scale,
        "goal_bias": planner.goal_bias,
    }
    source = "".join(
        inspect.getsource(getattr(type(planner), name)) for name in FINGERPRINTED_METHODS
    )
    params.update(
        goal_pose=_hash(np.array(planner._goal_xytheta())),
        root_state=_hash(to_numpy(planner.env.unwrapped.get_state())),
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
        "checkpoint was written by a different search, so its tree would be "
        f"scored and grown differently:\n{diff}\n"
        "    re-run without --resume, or pass --force to resume anyway"
    )


# --------------------------------------------------------------------------- #
# checkpoint
# --------------------------------------------------------------------------- #
@dataclass
class RRTCheckpoint:
    fingerprint: dict
    nodes: dict  # column -> array, one row per tree node, in insertion order
    best_inter_idx: int
    best_h_idx: int
    count: int
    goal_reached: bool
    recorder: dict  # log table and counters, opaque here

    def restore(self, node_cls, like=None):
        """Rebuild what `plan()` keeps in locals: the node list, the
        nearest-neighbor arrays, and the two incumbents. `like` supplies the
        dtype and device for the restored sim states."""
        nodes = _unpack_nodes(self.nodes, node_cls, like)
        keys_xy = np.zeros((len(nodes), 2), dtype=np.float64)
        keys_theta = np.zeros((len(nodes),), dtype=np.float64)
        for i, node in enumerate(nodes):
            keys_xy[i] = node.key[:2]
            keys_theta[i] = node.key[2]
        return (
            nodes,
            keys_xy,
            keys_theta,
            nodes[self.best_inter_idx],
            nodes[self.best_h_idx],
        )


def _capture(nodes, best_inter_node, best_h_node, recorder, fingerprint, goal_reached):
    index_of = {id(node): i for i, node in enumerate(nodes)}
    return RRTCheckpoint(
        fingerprint=dict(fingerprint),
        nodes=_pack_nodes(nodes),
        best_inter_idx=index_of[id(best_inter_node)],
        best_h_idx=index_of[id(best_h_node)],
        count=len(nodes),
        goal_reached=bool(goal_reached),
        recorder=recorder.checkpoint_state(),
    )


def _pack_nodes(nodes):
    index_of = {id(node): i for i, node in enumerate(nodes)}
    action_dim = next(
        (len(n.action_from_parent) for n in nodes if n.action_from_parent is not None),
        0,
    )
    zero_action = np.zeros(action_dim, dtype=np.float32)
    return {
        "sim_state": np.stack([to_numpy(n.sim_state).reshape(-1) for n in nodes]),
        "parent": np.array(
            [index_of[id(n.parent)] if n.parent is not None else -1 for n in nodes],
            dtype=np.int64,
        ),
        "action": np.stack(
            [
                zero_action
                if n.action_from_parent is None
                else np.asarray(n.action_from_parent, dtype=np.float32)
                for n in nodes
            ]
        ),
        "key": np.array([n.key for n in nodes], dtype=np.float64),
        "g": np.array([n.g_value for n in nodes], dtype=np.float64),
        "h": np.array([n.h_value for n in nodes], dtype=np.float64),
        "intersection": np.array([n.intersection for n in nodes], dtype=np.float64),
        "tcp_xy": np.stack([np.asarray(n.tcp_xy, dtype=np.float64) for n in nodes]),
        "rel_xy": np.stack([np.asarray(n.rel_xy, dtype=np.float64) for n in nodes]),
    }


def _unpack_nodes(columns, node_cls, like):
    n = len(columns["g"])
    parents = columns["parent"]
    nodes = [None] * n
    for i in range(n):
        parent_idx = int(parents[i])
        parent = nodes[parent_idx] if parent_idx >= 0 else None
        action = None if parent_idx < 0 else columns["action"][i]
        nodes[i] = node_cls(
            _as_sim_state(columns["sim_state"][i], like),
            parent,
            action,
            tuple(columns["key"][i].tolist()),
            float(columns["g"][i]),
            float(columns["h"][i]),
            float(columns["intersection"][i]),
            columns["tcp_xy"][i].astype(np.float64),
            columns["rel_xy"][i].astype(np.float64),
        )
    return nodes


def _as_sim_state(row, like):
    tensor = torch.as_tensor(np.ascontiguousarray(row).reshape(1, -1))
    # numpy arrays can carry a `.device` attribute too (array API compat), so
    # check for an actual torch.Tensor rather than hasattr(like, "device").
    if isinstance(like, torch.Tensor):
        tensor = tensor.to(device=like.device, dtype=like.dtype)
    return tensor


# --------------------------------------------------------------------------- #
# npz round-trip
# --------------------------------------------------------------------------- #
def save_checkpoint(checkpoint, path):
    arrays = {
        "best_inter_idx": np.asarray(checkpoint.best_inter_idx),
        "best_h_idx": np.asarray(checkpoint.best_h_idx),
        "ctr__count": np.asarray(checkpoint.count),
        "ctr__goal_reached": np.asarray(int(checkpoint.goal_reached)),
    }
    arrays.update({f"fp__{k}": np.asarray(v) for k, v in checkpoint.fingerprint.items()})
    arrays.update({f"node__{k}": v for k, v in checkpoint.nodes.items()})
    recorder = checkpoint.recorder
    arrays.update({f"rec_exp__{k}": v for k, v in recorder["expansions"].items()})
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
    return RRTCheckpoint(
        fingerprint=saved,
        nodes=section("node__"),
        best_inter_idx=int(data["best_inter_idx"]),
        best_h_idx=int(data["best_h_idx"]),
        count=int(counters["count"]),
        goal_reached=bool(counters["goal_reached"]),
        recorder={
            "expansions": section("rec_exp__"),
            "counters": {k: v.item() for k, v in section("rec_ctr__").items()},
        },
    )


# --------------------------------------------------------------------------- #
# planner-facing hooks
# --------------------------------------------------------------------------- #
class Checkpointer:
    """Writes the search to `path` every `every` tree nodes, and on demand."""

    def __init__(self, path, fingerprint, recorder, every=0):
        self.path = Path(path)
        self.fingerprint = fingerprint
        self.recorder = recorder
        self.every = int(every)
        self._last_saved = None

    def tick(self, nodes, best_inter_node, best_h_node):
        count = len(nodes)
        if self._last_saved is None:
            # Anchor on where this run started, so a resume does not immediately
            # rewrite the checkpoint it was just loaded from.
            self._last_saved = count
        elif self.every and count - self._last_saved >= self.every:
            self.save(nodes, best_inter_node, best_h_node)

    def save(self, nodes, best_inter_node, best_h_node, goal_reached=False):
        checkpoint = _capture(
            nodes, best_inter_node, best_h_node, self.recorder, self.fingerprint, goal_reached
        )
        save_checkpoint(checkpoint, self.path)
        self._last_saved = len(nodes)
        tqdm.write(
            f"checkpoint: {len(nodes)} nodes -> {self.path} "
            f"({self.path.stat().st_size / 1e6:.1f} MB)"
        )


class NullCheckpointer:
    """Same hooks as `Checkpointer`, writing nothing."""

    def tick(self, nodes, best_inter_node, best_h_node):
        pass

    def save(self, nodes, best_inter_node, best_h_node, goal_reached=False):
        pass
