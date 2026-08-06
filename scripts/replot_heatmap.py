"""Redraw the expansion heatmap from a cached plan result, without re-planning.

Examples:
  uv run scripts/replot_heatmap.py
  uv run scripts/replot_heatmap.py --cell-size 0.002
  uv run scripts/replot_heatmap.py --out /tmp/heatmap.png
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from planner import PlanResult
from viz_search import plot_expansion_heatmap, results_path

DEFAULT_NPZ_PATH = Path(__file__).resolve().parents[1] / "last_plan_result.npz"


def _load_result(path: Path) -> PlanResult:
    data = np.load(path)
    return PlanResult(
        actions=[int(a) for a in data["actions"]],
        expansion_xy=data["expansion_xy"],
        start_pose=data["start_pose"],
        goal_pose=data["goal_pose"],
        trajectory_xy=data["trajectory_xy"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--npz",
        type=Path,
        default=DEFAULT_NPZ_PATH,
        help=f"Plan result archive (default: {DEFAULT_NPZ_PATH.name})",
    )
    parser.add_argument(
        "--cell-size",
        type=float,
        default=0.003,
        help="Heatmap cell size in metres (default: 0.003)",
    )
    parser.add_argument(
        "--no-annotate",
        action="store_true",
        help="Omit the per-cell expansion counts",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output image path (default: timestamped file in results/)",
    )
    args = parser.parse_args()

    if not args.npz.exists():
        parser.error(f"{args.npz} not found; run main.py first to produce it")

    result = _load_result(args.npz)
    out_path = args.out if args.out is not None else results_path(
        "expansion_heatmap_replot"
    )
    saved = plot_expansion_heatmap(
        result,
        out_path,
        cell_size=args.cell_size,
        annotate=not args.no_annotate,
    )
    print(
        f"Replotted {len(result.expansion_xy)} expansions from {args.npz.name} "
        f"at cell_size={args.cell_size}"
    )
    print(f"Saved expansion heatmap to {saved}")


if __name__ == "__main__":
    main()
