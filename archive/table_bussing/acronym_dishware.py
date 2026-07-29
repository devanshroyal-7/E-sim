"""ACRONYM dishware catalog: meshes, grasps, and a curated 3+2 selection.

Public surface (intentional wide module — import from here):

- **Types/constants:** ``DishwareObject``, ``SelectionEntry``, ``SplitName``,
  ``ACTIVE_CATEGORIES``, ``SELECTION_PATH``, ``FULL_MANIFEST_CACHE_PATH``
- **Full catalog:** ``build_dishware_manifest`` (optionally caches to
  ``FULL_MANIFEST_CACHE_PATH``), ``load_full_manifest`` / ``save_full_manifest``
- **Curated selection:** ``curate_selection``, ``ensure_selection``,
  ``save_selection`` / ``load_selection``, ``resolve_selection``
- **Runtime pool:** ``ensure_curated_dishware``, ``filter_dishware_objects``,
  ``sample_objects``
- **Physics/geometry:** ``load_object_physics``, ``load_grasps``,
  ``compute_object_footprint``, ``mesh_bounds``
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Sequence, TypedDict
import warnings

import h5py
import numpy as np
import trimesh

DATA_DIR = Path(__file__).resolve().parent / "data"
# Curated subset vendored in-repo (5 objects per active category).
DEFAULT_ACRONYM_ROOT = DATA_DIR / "acronym"
# Editable curated subset (3 train + 2 test per category). Swap mesh_hash/scale here.
SELECTION_PATH = DATA_DIR / "dishware_selection.json"
# Full ACRONYM dishware index cache (not the curated subset).
FULL_MANIFEST_CACHE_PATH = DATA_DIR / "acronym_dishware_manifest.json"

N_TRAIN = 3
N_TEST = 2
N_PER_CATEGORY = N_TRAIN + N_TEST

# Categories kept in the curated 5-object pool. WineGlass/Pan/... stay available
# via the full catalog helpers when you want them later.
ACTIVE_CATEGORIES: list[str] = [
    "Plate",
    "Bowl",
    "Cup",
    "Mug",
]

# Indexed for later experiments, but not part of the default 5-object selection.
EXTRA_CATEGORIES: list[str] = [
    "WineGlass",
    "Pan",
    "Teacup",
    "Teapot",
]

DISHWARE_CATEGORIES: list[str] = ACTIVE_CATEGORIES + EXTRA_CATEGORIES
REQUIRED_CATEGORIES: tuple[str, ...] = ("Plate", "Bowl")

SplitName = Literal["train", "test", "all"]

_ENTRY_RE = re.compile(
    r"^(?P<cat>[A-Za-z0-9]+)_(?P<hash>[0-9a-fA-F]+)_(?P<scale>[0-9.eE+\-]+)\.json$"
)


@dataclass(frozen=True)
class DishwareObject:
    category: str
    mesh_hash: str
    scale: float
    mesh_path: str
    grasp_path: str
    split: SplitName
    rank: int = 0
    density: float | None = None
    mass: float | None = None

    @property
    def key(self) -> str:
        return f"{self.category}_{self.mesh_hash}_{self.scale}"


class SelectionEntry(TypedDict):
    mesh_hash: str
    scale: str
    split: str
    rank: int


def resolve_acronym_root(acronym_root: os.PathLike | str | None = None) -> Path:
    if acronym_root is not None:
        root = Path(acronym_root).expanduser().resolve()
    else:
        env = os.environ.get("ACRONYM_ROOT")
        root = Path(env).expanduser().resolve() if env else DEFAULT_ACRONYM_ROOT
    if not root.is_dir():
        raise FileNotFoundError(f"ACRONYM root not found: {root}")
    return root


def _parse_split_entry(entry: str) -> tuple[str, str, float]:
    m = _ENTRY_RE.match(entry)
    if m is None:
        raise ValueError(f"Unrecognized ACRONYM split entry: {entry}")
    return m.group("cat"), m.group("hash"), float(m.group("scale"))


def _read_optional_h5_scalars(
    grasp_path: Path,
) -> tuple[float | None, float | None, float | None]:
    """Return (scale, density, mass) from grasp H5 when present."""
    try:
        with h5py.File(grasp_path, "r") as f:
            scale = float(f["object/scale"][()]) if "object/scale" in f else None
            density = float(f["object/density"][()]) if "object/density" in f else None
            mass = float(f["object/mass"][()]) if "object/mass" in f else None
            return scale, density, mass
    except OSError as exc:
        warnings.warn(
            f"Failed to read grasp metadata from {grasp_path}: {exc}",
            stacklevel=2,
        )
        return None, None, None


def save_full_manifest(
    objects: Sequence[DishwareObject],
    path: Path = FULL_MANIFEST_CACHE_PATH,
) -> Path:
    """Persist a full (or partial) ACRONYM dishware catalog scan to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "categories": sorted({o.category for o in objects}),
        "objects": [asdict(o) for o in objects],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def load_full_manifest(path: Path = FULL_MANIFEST_CACHE_PATH) -> list[DishwareObject]:
    """Load a previously cached ACRONYM dishware catalog scan."""
    with open(path) as f:
        payload = json.load(f)
    return [DishwareObject(**row) for row in payload["objects"]]


def build_dishware_manifest(
    acronym_root: os.PathLike | str | None = None,
    categories: Sequence[str] = DISHWARE_CATEGORIES,
    read_h5_metadata: bool = False,
    *,
    cache_path: Path | None = FULL_MANIFEST_CACHE_PATH,
    rebuild_cache: bool = False,
) -> list[DishwareObject]:
    """Build dishware entries from official ACRONYM split JSONs.

    Entries whose mesh ``.obj`` or grasp ``.h5`` is missing are skipped with a
    warning (the full ACRONYM tree is often incomplete locally).

    When ``cache_path`` is set and ``read_h5_metadata`` is false, successful
    full-category scans are written to ``FULL_MANIFEST_CACHE_PATH`` by default
    and reused on later calls (filter to ``categories``).
    """
    cats = list(categories)
    if (
        cache_path is not None
        and cache_path.is_file()
        and not rebuild_cache
        and not read_h5_metadata
    ):
        cached = load_full_manifest(cache_path)
        allowed = set(cats)
        filtered = [o for o in cached if o.category in allowed]
        have = {o.category for o in cached}
        if set(cats) <= have:
            return filtered

    root = resolve_acronym_root(acronym_root)
    objects: list[DishwareObject] = []

    for category in cats:
        split_path = root / "splits" / f"{category}.json"
        if not split_path.is_file():
            warnings.warn(f"Missing ACRONYM split file, skipping category: {split_path}")
            continue
        with open(split_path) as f:
            splits = json.load(f)

        for split_name in ("train", "test"):
            for entry in splits.get(split_name, []):
                try:
                    cat, mesh_hash, scale = _parse_split_entry(entry)
                except ValueError as exc:
                    warnings.warn(f"Skipping bad split entry {entry!r}: {exc}")
                    continue
                if cat != category:
                    continue

                mesh_path = root / "meshes" / category / f"{mesh_hash}.obj"
                grasp_stem = entry.replace(".json", "")
                grasp_path = root / "grasps" / f"{grasp_stem}.h5"
                if not mesh_path.is_file() or not grasp_path.is_file():
                    warnings.warn(
                        f"Skipping {category}/{mesh_hash}: missing mesh or grasp "
                        f"(mesh={mesh_path.is_file()}, grasp={grasp_path.is_file()})"
                    )
                    continue

                density = None
                mass = None
                if read_h5_metadata:
                    h5_scale, density, mass = _read_optional_h5_scalars(grasp_path)
                    if h5_scale is not None:
                        scale = h5_scale

                objects.append(
                    DishwareObject(
                        category=category,
                        mesh_hash=mesh_hash,
                        scale=scale,
                        mesh_path=str(mesh_path),
                        grasp_path=str(grasp_path),
                        split=split_name,
                        density=density,
                        mass=mass,
                    )
                )

    objects.sort(key=lambda o: (o.category, o.split, o.mesh_hash, o.scale))
    if (
        cache_path is not None
        and not read_h5_metadata
        and set(cats) >= set(DISHWARE_CATEGORIES)
    ):
        save_full_manifest(objects, path=cache_path)
    return objects


def _unique_by_mesh(objects: Sequence[DishwareObject]) -> list[DishwareObject]:
    """Keep one instance per mesh hash (lexicographically smallest scale)."""
    best: dict[str, DishwareObject] = {}
    for obj in objects:
        prev = best.get(obj.mesh_hash)
        if prev is None or (obj.mesh_hash, obj.scale) < (prev.mesh_hash, prev.scale):
            best[obj.mesh_hash] = obj
    return sorted(best.values(), key=lambda o: (o.mesh_hash, o.scale))


def _bowl_interior_content_frac(mesh_path: os.PathLike | str) -> float:
    """Fraction of vertices in the mid-height central region (proxy for contents).

    Empty bowl shells put almost no geometry there; bowls with food/utensils score
    much higher. Used only to prefer empty Bowl meshes during curation.
    """
    mesh = trimesh.load(str(mesh_path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if vertices.size == 0:
        # Invalid/empty mesh is not "empty dishware"; sort it last when preferring low scores.
        return float("inf")
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    extent = np.maximum(maxs - mins, 1e-12)
    cx = 0.5 * (mins[0] + maxs[0])
    cy = 0.5 * (mins[1] + maxs[1])
    rx, ry = 0.28 * extent[0], 0.28 * extent[1]
    z_lo = mins[2] + 0.35 * extent[2]
    z_hi = mins[2] + 0.80 * extent[2]
    interior = (
        (np.abs(vertices[:, 0] - cx) < rx)
        & (np.abs(vertices[:, 1] - cy) < ry)
        & (vertices[:, 2] > z_lo)
        & (vertices[:, 2] < z_hi)
    )
    return float(interior.mean())


def _prefer_empty_bowls(objects: Sequence[DishwareObject]) -> list[DishwareObject]:
    """Stable-sort bowls so empty (low interior content) meshes come first."""
    scored = [
        (_bowl_interior_content_frac(o.mesh_path), o.mesh_hash, o.scale, o)
        for o in objects
    ]
    scored.sort(key=lambda t: (t[0], t[1], t[2]))
    return [t[3] for t in scored]


def _entry_dict(obj: DishwareObject, split: str, rank: int) -> SelectionEntry:
    scale_str = Path(obj.grasp_path).stem.rsplit("_", 1)[-1]
    return {
        "mesh_hash": obj.mesh_hash,
        "scale": scale_str,
        "split": split,
        "rank": rank,
    }


def curate_selection(
    full_catalog: Sequence[DishwareObject],
    categories: Sequence[str] = ACTIVE_CATEGORIES,
    n_train: int = N_TRAIN,
    n_test: int = N_TEST,
) -> dict[str, list[SelectionEntry]]:
    """Pick ``n_train`` + ``n_test`` unique meshes per category (deterministic).

    Train slots come from the official train split. Test slots prefer official
    test meshes whose hash is not among the chosen train meshes; if ACRONYM's
    test split does not have enough distinct meshes, remaining train meshes
    are used and labeled ``test`` in our selection.

    For ``Bowl`` and ``Cup``, pools are sorted to prefer empty meshes (no
    food/utensils baked into the geometry).
    """
    selection: dict[str, list[dict]] = {}
    for category in categories:
        train_pool = _unique_by_mesh(
            [o for o in full_catalog if o.category == category and o.split == "train"]
        )
        test_pool = _unique_by_mesh(
            [o for o in full_catalog if o.category == category and o.split == "test"]
        )
        if category in ("Bowl", "Cup"):
            train_pool = _prefer_empty_bowls(train_pool)
            test_pool = _prefer_empty_bowls(test_pool)

        if len(train_pool) < n_train:
            raise RuntimeError(
                f"{category}: need {n_train} unique train meshes, found {len(train_pool)}"
            )

        chosen_train = train_pool[:n_train]
        chosen_train_hashes = {o.mesh_hash for o in chosen_train}

        # Prefer official test meshes not already in the curated train set.
        test_candidates = [o for o in test_pool if o.mesh_hash not in chosen_train_hashes]
        # Fill shortfall from leftover train meshes.
        if len(test_candidates) < n_test:
            leftovers = [
                o for o in train_pool[n_train:] if o.mesh_hash not in chosen_train_hashes
            ]
            # Also allow official test meshes that duplicate unchosen train hashes.
            seen = {o.mesh_hash for o in test_candidates}
            for o in leftovers + test_pool:
                if o.mesh_hash in chosen_train_hashes or o.mesh_hash in seen:
                    continue
                test_candidates.append(o)
                seen.add(o.mesh_hash)
                if len(test_candidates) >= n_test:
                    break

        if len(test_candidates) < n_test:
            raise RuntimeError(
                f"{category}: need {n_test} unique test meshes after curation, "
                f"found {len(test_candidates)}"
            )

        chosen = [
            _entry_dict(obj, "train", rank)
            for rank, obj in enumerate(chosen_train, start=1)
        ]
        chosen.extend(
            _entry_dict(obj, "test", rank)
            for rank, obj in enumerate(test_candidates[:n_test], start=1)
        )
        selection[category] = chosen
    return selection


def save_selection(
    selection: dict[str, list[SelectionEntry]],
    path: Path = SELECTION_PATH,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_train": N_TRAIN,
        "n_test": N_TEST,
        "categories": list(selection.keys()),
        "objects": selection,
        "notes": (
            "Edit mesh_hash/scale to swap assets. "
            "train ranks 1-3 = regular usage; test ranks 1-2 = new-env testing. "
            "Run: python render_dishware_gallery.py"
        ),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def load_selection(path: Path = SELECTION_PATH) -> dict[str, list[SelectionEntry]]:
    with open(path) as f:
        payload = json.load(f)
    objects = payload["objects"]
    for category, entries in objects.items():
        for i, entry in enumerate(entries):
            if "rank" not in entry:
                raise ValueError(
                    f"selection {path} {category}[{i}] missing required field 'rank'"
                )
    return objects


def resolve_selection(
    selection: dict[str, list[SelectionEntry]],
    acronym_root: os.PathLike | str | None = None,
) -> list[DishwareObject]:
    """Turn selection JSON entries into DishwareObjects with absolute paths."""
    root = resolve_acronym_root(acronym_root)
    objects: list[DishwareObject] = []
    for category, entries in selection.items():
        for entry in entries:
            mesh_hash = entry["mesh_hash"]
            scale = float(entry["scale"])
            split = entry["split"]
            if split not in ("train", "test"):
                raise ValueError(
                    f"selection entry split must be 'train' or 'test', got {split!r}"
                )
            rank = int(entry["rank"])
            mesh_path = root / "meshes" / category / f"{mesh_hash}.obj"
            grasp_name = f"{category}_{mesh_hash}_{entry['scale']}.h5"
            # Filename may use the float's original string form; try a few variants.
            grasp_path = root / "grasps" / grasp_name
            if not grasp_path.is_file():
                # Fall back: find any matching hash grasp and prefer closest scale.
                candidates = sorted((root / "grasps").glob(f"{category}_{mesh_hash}_*.h5"))
                if not candidates:
                    raise FileNotFoundError(
                        f"Missing grasp for {category}/{mesh_hash} under {root / 'grasps'}"
                    )
                grasp_path = min(
                    candidates,
                    key=lambda p: abs(float(p.stem.rsplit("_", 1)[-1]) - scale),
                )
                scale = float(grasp_path.stem.rsplit("_", 1)[-1])
            if not mesh_path.is_file():
                raise FileNotFoundError(f"Missing mesh: {mesh_path}")
            objects.append(
                DishwareObject(
                    category=category,
                    mesh_hash=mesh_hash,
                    scale=scale,
                    mesh_path=str(mesh_path),
                    grasp_path=str(grasp_path),
                    split=split,
                    rank=rank,
                )
            )
    objects.sort(key=lambda o: (o.category, o.split, o.rank, o.mesh_hash))
    return objects


def ensure_selection(
    acronym_root: os.PathLike | str | None = None,
    *,
    selection_path: Path = SELECTION_PATH,
    categories: Sequence[str] = ACTIVE_CATEGORIES,
    rebuild: bool = False,
) -> dict[str, list[SelectionEntry]]:
    """Load curated selection, creating a deterministic default if missing."""
    if selection_path.is_file() and not rebuild:
        return load_selection(selection_path)

    full = build_dishware_manifest(
        acronym_root=acronym_root, categories=categories, read_h5_metadata=False
    )
    selection = curate_selection(full, categories=categories)
    save_selection(selection, path=selection_path)
    return selection


def ensure_curated_dishware(
    acronym_root: os.PathLike | str | None = None,
    *,
    selection_path: Path = SELECTION_PATH,
    rebuild: bool = False,
    categories: Sequence[str] | None = None,
) -> list[DishwareObject]:
    """Load the curated dishware pool, creating ``selection_path`` if needed.

    May write ``data/dishware_selection.json`` when the file is missing or
    ``rebuild=True``. ``categories`` must be a subset of ``ACTIVE_CATEGORIES``;
    the on-disk selection is always curated for the full active set, then
    filtered to the requested subset.
    """
    cats = list(categories) if categories is not None else list(ACTIVE_CATEGORIES)
    unknown = [c for c in cats if c not in ACTIVE_CATEGORIES]
    if unknown:
        raise ValueError(
            "categories not in ACTIVE_CATEGORIES (use full-catalog helpers for "
            f"extras like WineGlass/Pan): {unknown}"
        )
    selection = ensure_selection(
        acronym_root=acronym_root,
        selection_path=selection_path,
        categories=ACTIVE_CATEGORIES,
        rebuild=rebuild,
    )
    filtered = {c: selection[c] for c in cats if c in selection}
    missing_requested = [c for c in cats if c not in filtered]
    if missing_requested:
        raise RuntimeError(
            f"Requested categories missing from selection file: {missing_requested}"
        )
    objects = resolve_selection(filtered, acronym_root=acronym_root)
    return objects


def filter_dishware_objects(
    objects: Sequence[DishwareObject],
    *,
    categories: Sequence[str] | None = None,
    split: SplitName = "train",
) -> list[DishwareObject]:
    out = list(objects)
    if categories is not None:
        allowed = set(categories)
        out = [o for o in out if o.category in allowed]
    if split != "all":
        out = [o for o in out if o.split == split]
    return out


def sample_objects(
    objects: Sequence[DishwareObject],
    counts: dict[str, int],
    *,
    split: SplitName = "train",
    rng: np.random.Generator | None = None,
) -> list[DishwareObject]:
    """Sample dishware instances for the requested per-category counts."""
    rng = rng or np.random.default_rng()
    pool = filter_dishware_objects(objects, split=split)
    by_cat: dict[str, list[DishwareObject]] = {}
    for obj in pool:
        by_cat.setdefault(obj.category, []).append(obj)

    chosen: list[DishwareObject] = []
    for category, n in counts.items():
        if n <= 0:
            continue
        available = by_cat.get(category, [])
        if not available:
            raise ValueError(
                f"No ACRONYM objects for category={category!r} split={split!r}"
            )
        idxs = rng.integers(0, len(available), size=n)
        chosen.extend(available[int(i)] for i in idxs)
    return chosen


def load_grasps(
    grasp_path: os.PathLike | str,
    *,
    success_only: bool = True,
) -> np.ndarray:
    """Load grasp transforms ``(K, 4, 4)`` in the object frame."""
    with h5py.File(grasp_path, "r") as f:
        transforms = np.asarray(f["grasps/transforms"], dtype=np.float64)
        if success_only:
            success = np.asarray(
                f["grasps/qualities/flex/object_in_gripper"], dtype=np.int64
            )
            transforms = transforms[success == 1]
    return transforms


def load_object_physics(grasp_path: os.PathLike | str) -> dict[str, float]:
    with h5py.File(grasp_path, "r") as f:
        out: dict[str, float] = {}
        if "object/scale" in f:
            out["scale"] = float(f["object/scale"][()])
        if "object/density" in f:
            out["density"] = float(f["object/density"][()])
        if "object/mass" in f:
            out["mass"] = float(f["object/mass"][()])
        return out


def mesh_bounds(
    mesh_path: os.PathLike | str, scale: float
) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bounds of the scaled mesh: ``(mins, maxs)`` each shape ``(3,)``."""
    mesh = trimesh.load(str(mesh_path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    vertices = np.asarray(mesh.vertices, dtype=np.float64) * float(scale)
    return vertices.min(axis=0), vertices.max(axis=0)


def xy_radius_from_bounds(mins: np.ndarray, maxs: np.ndarray) -> float:
    half = 0.5 * (maxs[:2] - mins[:2])
    return float(np.linalg.norm(half))


@dataclass(frozen=True)
class ObjectSpawnMetrics:
    """Shared spawn geometry/physics for ManiSkill dishware actors."""

    scale: float
    density: float
    spawn_z: float
    xy_radius: float
    center_xy: tuple[float, float]
    mins: tuple[float, float, float]
    maxs: tuple[float, float, float]


def compute_object_footprint(obj: DishwareObject) -> ObjectSpawnMetrics:
    """Resolve scale/density and AABB-derived spawn metrics for ``obj``.

    Used by both the table-bussing env and the gallery renderer so mesh physics
    and footprint policy live in one place on the catalog boundary.
    """
    physics = load_object_physics(obj.grasp_path)
    scale = float(physics.get("scale", obj.scale))
    density = float(physics.get("density", 1000.0))
    if not np.isfinite(density) or density <= 0:
        density = 1000.0

    mins_arr, maxs_arr = mesh_bounds(obj.mesh_path, scale)
    mins = (float(mins_arr[0]), float(mins_arr[1]), float(mins_arr[2]))
    maxs = (float(maxs_arr[0]), float(maxs_arr[1]), float(maxs_arr[2]))
    spawn_z = float(-mins[2])
    xy_radius = float(xy_radius_from_bounds(mins_arr, maxs_arr))
    center_xy = (
        float(0.5 * (mins[0] + maxs[0])),
        float(0.5 * (mins[1] + maxs[1])),
    )
    return ObjectSpawnMetrics(
        scale=scale,
        density=density,
        spawn_z=spawn_z,
        xy_radius=xy_radius,
        center_xy=center_xy,
        mins=mins,
        maxs=maxs,
    )


def print_selection_summary(objects: Sequence[DishwareObject]) -> None:
    by_cat: dict[str, list[DishwareObject]] = {}
    for o in objects:
        by_cat.setdefault(o.category, []).append(o)
    for cat in ACTIVE_CATEGORIES:
        rows = by_cat.get(cat, [])
        if not rows:
            continue
        rows = sorted(rows, key=lambda o: (o.split != "train", o.rank))
        print(f"{cat} ({len(rows)} objects):")
        for o in rows:
            print(
                f"  [{o.split:5} rank={o.rank}] hash={o.mesh_hash[:12]}… "
                f"scale={o.scale:.6g}"
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Overwrite data/dishware_selection.json with a fresh deterministic pick",
    )
    args = parser.parse_args()
    objs = ensure_curated_dishware(rebuild=args.rebuild)
    print(f"Selection file: {SELECTION_PATH}")
    print(f"Curated objects: {len(objs)} ({N_TRAIN} train + {N_TEST} test per category)")
    print_selection_summary(objs)
