"""Save a running RRT search to disk and pick it up again in a later process.

RRT has no open/closed split: every node in the tree, not just some frontier,
can be picked as a nearest neighbor for a future expansion, so a checkpoint
stores the whole tree table rather than a subset of it. Each row already
carries the sim_state `env.set_state()` needs and its parent's row index, so
resuming needs no re-expansion -- nearest-neighbor search and the incumbent
(best_inter_node) both fall out of the restored table directly, and any
node's full action path is just a walk up `parent`.

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

from geometry import TEE_LANDMARKS_XY, ContactRecord, to_numpy

# Methods whose source is hashed into the fingerprint: between them they define
# the nearest-neighbor metric and the candidate tiebreak, which is everything
# that decides how the tree grows and which nodes look best.
FINGERPRINTED_METHODS = (
    "_pose_features",
    "_expand",
    "_reposition",
    "_choose_contact",
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
        "reposition_prob": planner.reposition_prob,
        "contact_standoff": planner.contact_standoff,
        "reposition_lift_dz_total": planner.reposition_lift_dz_total,
        "obj_pose_tolerance": planner.obj_pose_tolerance,
        "contact_xy_tolerance": planner.contact_xy_tolerance,
        "max_extra_contacts": planner.max_extra_contacts,
        "max_translate_retries": planner.max_translate_retries,
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
    count: int
    iters: int  # total loop iterations spent (extends + repositions), not just tree size
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
            self.iters,
        )


def _capture(nodes, best_inter_node, iters, recorder, fingerprint, goal_reached):
    index_of = {id(node): i for i, node in enumerate(nodes)}
    max_extra_contacts = int(fingerprint["max_extra_contacts"])
    # 1 lift + up to max_translate_retries move batches + 1 lower -- the most
    # actions a single successful _reposition() can produce.
    max_reposition_actions = 2 + int(fingerprint["max_translate_retries"])
    return RRTCheckpoint(
        fingerprint=dict(fingerprint),
        nodes=_pack_nodes(nodes, max_extra_contacts, max_reposition_actions),
        best_inter_idx=index_of[id(best_inter_node)],
        count=len(nodes),
        iters=int(iters),
        goal_reached=bool(goal_reached),
        recorder=recorder.checkpoint_state(),
    )


def _pack_nodes(nodes, max_extra_contacts, max_reposition_actions):
    index_of = {id(node): i for i, node in enumerate(nodes)}
    action_dim = next(
        (len(n.action_from_parent) for n in nodes if n.action_from_parent is not None),
        0,
    )
    zero_action = np.zeros(action_dim, dtype=np.float32)
    state_dim = to_numpy(nodes[0].sim_state).reshape(-1).shape[0]

    extra_count = np.array(
        [min(len(n.extra_contacts), max_extra_contacts) for n in nodes], dtype=np.int64
    )
    extra_sim_state = np.zeros((len(nodes), max_extra_contacts, state_dim), dtype=np.float64)
    extra_tcp_xy = np.zeros((len(nodes), max_extra_contacts, 2), dtype=np.float64)
    extra_rel_xy = np.zeros((len(nodes), max_extra_contacts, 2), dtype=np.float64)
    extra_source_index = np.zeros((len(nodes), max_extra_contacts), dtype=np.int64)
    extra_action_count = np.zeros((len(nodes), max_extra_contacts), dtype=np.int64)
    extra_actions = np.zeros(
        (len(nodes), max_extra_contacts, max_reposition_actions, max(action_dim, 1)),
        dtype=np.float32,
    )
    for i, n in enumerate(nodes):
        for j, contact in enumerate(n.extra_contacts[:max_extra_contacts]):
            extra_sim_state[i, j] = to_numpy(contact.sim_state).reshape(-1)
            extra_tcp_xy[i, j] = np.asarray(contact.tcp_xy, dtype=np.float64)
            extra_rel_xy[i, j] = np.asarray(contact.rel_xy, dtype=np.float64)
            extra_source_index[i, j] = contact.source_index
            n_actions = min(len(contact.actions), max_reposition_actions)
            extra_action_count[i, j] = n_actions
            for k, action in enumerate(contact.actions[:n_actions]):
                extra_actions[i, j, k] = np.asarray(action, dtype=np.float32)

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
        "intersection": np.array([n.intersection for n in nodes], dtype=np.float64),
        "tcp_xy": np.stack([np.asarray(n.tcp_xy, dtype=np.float64) for n in nodes]),
        "rel_xy": np.stack([np.asarray(n.rel_xy, dtype=np.float64) for n in nodes]),
        "parent_contact_index": np.array(
            [n.parent_contact_index for n in nodes], dtype=np.int64
        ),
        "extra_count": extra_count,
        "extra_sim_state": extra_sim_state,
        "extra_tcp_xy": extra_tcp_xy,
        "extra_rel_xy": extra_rel_xy,
        "extra_source_index": extra_source_index,
        "extra_action_count": extra_action_count,
        "extra_actions": extra_actions,
    }


def _unpack_nodes(columns, node_cls, like):
    n = len(columns["intersection"])
    parents = columns["parent"]
    nodes = [None] * n
    for i in range(n):
        parent_idx = int(parents[i])
        parent = nodes[parent_idx] if parent_idx >= 0 else None
        action = None if parent_idx < 0 else columns["action"][i]

        extra_contacts = []
        if "extra_count" in columns:
            has_actions = "extra_action_count" in columns
            for j in range(int(columns["extra_count"][i])):
                source_index = (
                    int(columns["extra_source_index"][i, j])
                    if "extra_source_index" in columns
                    else 0
                )
                if has_actions:
                    n_actions = int(columns["extra_action_count"][i, j])
                    actions = [
                        columns["extra_actions"][i, j, k] for k in range(n_actions)
                    ]
                else:
                    actions = []
                extra_contacts.append(
                    ContactRecord(
                        _as_sim_state(columns["extra_sim_state"][i, j], like),
                        columns["extra_tcp_xy"][i, j].astype(np.float64),
                        columns["extra_rel_xy"][i, j].astype(np.float64),
                        source_index=source_index,
                        actions=actions,
                    )
                )
        parent_contact_index = (
            int(columns["parent_contact_index"][i])
            if "parent_contact_index" in columns
            else 0
        )

        nodes[i] = node_cls(
            _as_sim_state(columns["sim_state"][i], like),
            parent,
            action,
            tuple(columns["key"][i].tolist()),
            float(columns["intersection"][i]),
            columns["tcp_xy"][i].astype(np.float64),
            columns["rel_xy"][i].astype(np.float64),
            parent_contact_index=parent_contact_index,
            extra_contacts=extra_contacts,
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
        "ctr__count": np.asarray(checkpoint.count),
        "ctr__iters": np.asarray(checkpoint.iters),
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
    count = int(counters["count"])
    return RRTCheckpoint(
        fingerprint=saved,
        nodes=section("node__"),
        best_inter_idx=int(data["best_inter_idx"]),
        count=count,
        # Older checkpoints (pre-reposition) never recorded iterations
        # separately from tree size -- they were the same thing.
        iters=int(counters.get("iters", count)),
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

    def tick(self, nodes, best_inter_node, iters):
        if self._last_saved is None:
            # Anchor on where this run started, so a resume does not immediately
            # rewrite the checkpoint it was just loaded from.
            self._last_saved = iters
        elif self.every and iters - self._last_saved >= self.every:
            self.save(nodes, best_inter_node, iters)

    def save(self, nodes, best_inter_node, iters, goal_reached=False):
        checkpoint = _capture(
            nodes, best_inter_node, iters, self.recorder, self.fingerprint, goal_reached
        )
        save_checkpoint(checkpoint, self.path)
        self._last_saved = iters
        tqdm.write(
            f"checkpoint: {len(nodes)} nodes -> {self.path} "
            f"({self.path.stat().st_size / 1e6:.1f} MB)"
        )


class NullCheckpointer:
    """Same hooks as `Checkpointer`, writing nothing."""

    def tick(self, nodes, best_inter_node, iters):
        pass

    def save(self, nodes, best_inter_node, iters, goal_reached=False):
        pass
