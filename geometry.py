"""T-shape geometry and pose helpers shared by the planner and the plots."""
from collections import namedtuple

import numpy as np

# An alternative way to be at a node's T pose: the same object pose, reached
# with the TCP at a different contact point (via a lift/translate/lower
# reposition move rather than a push).
#
# source_index/actions record provenance for plan extraction: source_index is
# which contact (in the owning node's `contacts()` list, 0 = primary) this one
# was reached from, and actions is the ordered list of action vectors (each
# meant to be applied for K env.step() calls, same convention as
# RRTNode.action_from_parent) that got the TCP there. The primary contact
# needs neither -- it's already where the node landed -- so both default to
# "no move needed".
ContactRecord = namedtuple(
    "ContactRecord",
    ["sim_state", "tcp_xy", "rel_xy", "source_index", "actions"],
    defaults=(None, ()),
)

# Body-frame T corners (COM 0.0375 already subtracted), matching ManiSkill PushT.
TEE_LANDMARKS_XY = np.array(
    [
        [0.1, -0.0125],
        [0.1, -0.0625],
        [-0.1, -0.0625],
        [-0.1, -0.0125],
        [-0.025, -0.0125],
        [-0.025, 0.1375],
        [0.025, 0.1375],
        [0.025, -0.0125],
    ],
    dtype=np.float64,
)


# Farthest a T corner ever gets from the T's COM -- a cheap "is the TCP still
# near the block" radius.
TEE_CIRCUMRADIUS = float(np.linalg.norm(TEE_LANDMARKS_XY, axis=-1).max())


def to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def yaw_from_quat(quat):
    quat = to_numpy(quat)
    return 2.0 * np.arctan2(quat[..., 3], quat[..., 0])


def wrap_pi(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def landmarks_world_xy(pose):
    """The 8 T corners in world xy for a (x, y, z, qw, qx, qy, qz) pose."""
    pose = to_numpy(pose).reshape(-1)
    xy = pose[:2]
    yaw = float(yaw_from_quat(pose[3:7]))
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    return TEE_LANDMARKS_XY @ rot.T + xy


# Consecutive landmark pairs (wrapping around) form the T's boundary edges.
_EDGE_STARTS = TEE_LANDMARKS_XY
_EDGE_ENDS = np.roll(TEE_LANDMARKS_XY, -1, axis=0)
_EDGE_LENGTHS = np.linalg.norm(_EDGE_ENDS - _EDGE_STARTS, axis=-1)
_EDGE_WEIGHTS = _EDGE_LENGTHS / _EDGE_LENGTHS.sum()

# The T's outline is concave (it has a re-entrant corner at the stem), so
# "which side of an edge is outward" can't be decided per-edge by comparing
# against a single interior reference point -- that test is only valid for
# convex shapes. The winding order is a single global property of the whole
# polygon instead: rotating each edge's direction by -90 degrees gives the
# outward normal for every edge consistently if the vertices run clockwise
# (+90 degrees if counterclockwise), verified below via the shoelace formula.
_SIGNED_AREA = 0.5 * np.sum(
    TEE_LANDMARKS_XY[:, 0] * _EDGE_ENDS[:, 1] - _EDGE_ENDS[:, 0] * TEE_LANDMARKS_XY[:, 1]
)
_OUTWARD_ROTATION_SIGN = -1.0 if _SIGNED_AREA < 0 else 1.0


def sample_contact_point_body_xy(standoff, rng=None):
    """A random point just outside the T's boundary, in the T's body frame.

    Samples uniformly over the T's perimeter (edge-length-weighted, not
    edge-index-weighted), then pushes the point outward by `standoff` along
    that edge's outward normal so a contact placed here is just clear of the
    T rather than exactly on its surface.
    """
    if rng is None:
        rng = np.random
    edge_idx = rng.choice(len(_EDGE_WEIGHTS), p=_EDGE_WEIGHTS)
    t = rng.uniform(0.0, 1.0)
    a, b = _EDGE_STARTS[edge_idx], _EDGE_ENDS[edge_idx]
    point = (1.0 - t) * a + t * b

    edge_dir = b - a
    normal = _OUTWARD_ROTATION_SIGN * np.array([edge_dir[1], -edge_dir[0]], dtype=np.float64)
    norm = np.linalg.norm(normal)
    if norm > 0:
        normal = normal / norm

    return point + standoff * normal
