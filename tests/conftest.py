"""Shared pytest fixtures for hermetic ACRONYM-shaped trees."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest


MINIMAL_OBJ = """\
# minimal triangle
v 0 0 0
v 1 0 0
v 0 1 0
f 1 2 3
"""


def _write_grasp_h5(path: Path, *, n_success: int = 2, n_fail: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    transforms = np.eye(4, dtype=np.float64)[None, :, :].repeat(n_success + n_fail, axis=0)
    for i in range(transforms.shape[0]):
        transforms[i, 0, 3] = float(i)
    success = np.array([1] * n_success + [0] * n_fail, dtype=np.int64)
    with h5py.File(path, "w") as f:
        f.create_dataset("grasps/transforms", data=transforms)
        f.create_dataset("grasps/qualities/flex/object_in_gripper", data=success)
        f.create_dataset("object/scale", data=np.float64(0.1))
        f.create_dataset("object/density", data=np.float64(1000.0))
        f.create_dataset("object/mass", data=np.float64(0.2))


@pytest.fixture
def fake_acronym_root(tmp_path: Path) -> Path:
    """Tiny ACRONYM-shaped tree with one Plate mesh/grasp and split JSON."""
    root = tmp_path / "acronym"
    category = "Plate"
    mesh_hash = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    scale = "0.100000"
    mesh_dir = root / "meshes" / category
    mesh_dir.mkdir(parents=True)
    (mesh_dir / f"{mesh_hash}.obj").write_text(MINIMAL_OBJ)

    grasp_name = f"{category}_{mesh_hash}_{scale}"
    _write_grasp_h5(root / "grasps" / f"{grasp_name}.h5")

    splits = {
        "train": [f"{grasp_name}.json"],
        "test": [f"{grasp_name}.json"],
    }
    split_dir = root / "splits"
    split_dir.mkdir(parents=True)
    (split_dir / f"{category}.json").write_text(json.dumps(splits))
    return root
