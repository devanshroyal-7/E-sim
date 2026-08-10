"""T-shape geometry and pose helpers shared by the planner and the plots."""
import numpy as np

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
