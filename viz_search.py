"""Plot expansion heatmap with start/goal T poses and chosen trajectory."""
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle
from matplotlib.patheffects import withStroke

from planner import TEE_LANDMARKS_XY, PlanResult

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def results_path(stem, ext=".png", when=None):
    """results/<stem>_YYYYmmdd-HHMMSS<ext>, directory created on demand."""
    stamp = (when or datetime.now()).strftime("%Y%m%d-%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / f"{stem}_{stamp}{ext}"


def _yaw_from_quat(quat):
    quat = np.asarray(quat, dtype=np.float64).reshape(-1)
    return float(2.0 * np.arctan2(quat[3], quat[0]))


def _landmarks_world_xy(pose):
    pose = np.asarray(pose, dtype=np.float64).reshape(-1)
    xy = pose[:2]
    yaw = _yaw_from_quat(pose[3:7])
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    return TEE_LANDMARKS_XY @ rot.T + xy


def _draw_tee(ax, pose, *, facecolor, edgecolor, label=None, alpha=0.22, zorder=5, lw=1.8):
    landmarks = _landmarks_world_xy(pose)
    poly = Polygon(
        landmarks,
        closed=True,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=lw,
        alpha=alpha,
        label=label,
        zorder=zorder,
    )
    ax.add_patch(poly)

    xy = pose[:2]
    yaw = _yaw_from_quat(pose[3:7])
    tick_len = 0.04
    ax.plot(xy[0], xy[1], "+", color=edgecolor, markersize=8, zorder=zorder + 2)
    ax.add_patch(
        FancyArrowPatch(
            (xy[0], xy[1]),
            (xy[0] + tick_len * np.cos(yaw), xy[1] + tick_len * np.sin(yaw)),
            arrowstyle="-|>",
            mutation_scale=10,
            color=edgecolor,
            linewidth=1.2,
            zorder=zorder + 2,
        )
    )
    return landmarks


def _axis_limits(points, pad=0.05):
    all_xy = np.vstack(points)
    return (
        all_xy[:, 0].min() - pad,
        all_xy[:, 0].max() + pad,
        all_xy[:, 1].min() - pad,
        all_xy[:, 1].max() + pad,
    )


def _aligned_edges(lo, hi, cell_size):
    """Cell edges snapped to a global cell_size lattice, covering [lo, hi]."""
    start = np.floor(lo / cell_size) * cell_size
    stop = np.ceil(hi / cell_size) * cell_size
    n_cells = max(int(round((stop - start) / cell_size)), 1)
    return start + cell_size * np.arange(n_cells + 1)


def _draw_trajectory(ax, trajectory_xy, *, lw=2.4, halo=5.0, star=120, label=None):
    if len(trajectory_xy) < 2:
        return
    ax.plot(
        trajectory_xy[:, 0],
        trajectory_xy[:, 1],
        color="white",
        linewidth=halo,
        solid_capstyle="round",
        zorder=6,
    )
    ax.plot(
        trajectory_xy[:, 0],
        trajectory_xy[:, 1],
        color="black",
        linewidth=lw,
        solid_capstyle="round",
        zorder=7,
        label=label,
    )
    ax.scatter(
        trajectory_xy[0, 0],
        trajectory_xy[0, 1],
        c="white",
        s=70,
        zorder=9,
        marker="o",
        edgecolors="black",
        linewidths=1.2,
    )
    ax.scatter(
        trajectory_xy[-1, 0],
        trajectory_xy[-1, 1],
        c="black",
        s=star,
        zorder=9,
        marker="*",
    )


def plot_expansion_heatmap(
    result: PlanResult,
    out_path=None,
    cell_size=0.003,
    annotate=True,
    max_annot_cells=400,
):
    """Save a two-panel expansion figure.

    Left: start/goal T, plan trajectory, and a box marking the region the search
    actually touched. Right: expansions per `cell_size` cell of tee COM xy, so
    states that share an xy but differ in orientation stack into one cell.
    """
    when = datetime.now()
    out_path = Path(out_path) if out_path is not None else results_path(
        "expansion_heatmap", when=when
    )
    expansion_xy = np.asarray(result.expansion_xy, dtype=np.float64).reshape(-1, 2)
    trajectory_xy = np.asarray(result.trajectory_xy, dtype=np.float64).reshape(-1, 2)

    start_lm = _landmarks_world_xy(result.start_pose)
    goal_lm = _landmarks_world_xy(result.goal_pose)

    # Heat region spans everything the search touched, plan trajectory included.
    region_pts = [p for p in (expansion_xy, trajectory_xy) if len(p) > 0]
    region_xy = np.vstack(region_pts) if region_pts else np.zeros((1, 2))
    xedges = _aligned_edges(region_xy[:, 0].min(), region_xy[:, 0].max(), cell_size)
    yedges = _aligned_edges(region_xy[:, 1].min(), region_xy[:, 1].max(), cell_size)

    if len(expansion_xy) > 0:
        H, _, _ = np.histogram2d(
            expansion_xy[:, 0], expansion_xy[:, 1], bins=[xedges, yedges]
        )
    else:
        H = np.zeros((len(xedges) - 1, len(yedges) - 1), dtype=np.float64)

    vmax = float(H.max())
    norm = LogNorm(vmin=1, vmax=vmax) if vmax > 1 else Normalize(vmin=0, vmax=1)
    cmap = mpl.colormaps["magma_r"].copy()
    cmap.set_bad("#eeeeee")

    fig, (ax_ctx, ax_hm) = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)

    # --- Panel 1: workspace context -------------------------------------
    _draw_tee(
        ax_ctx,
        result.start_pose,
        facecolor="#4C78A8",
        edgecolor="#1F4E79",
        label="start T",
    )
    _draw_tee(
        ax_ctx,
        result.goal_pose,
        facecolor="#59A14F",
        edgecolor="#2E7D32",
        label="goal T",
    )
    _draw_trajectory(ax_ctx, trajectory_xy, label="trajectory")
    ax_ctx.add_patch(
        Rectangle(
            (xedges[0], yedges[0]),
            xedges[-1] - xedges[0],
            yedges[-1] - yedges[0],
            fill=False,
            edgecolor="#444444",
            linestyle="--",
            linewidth=1.5,
            zorder=10,
            label="heat region",
        )
    )

    ctx_pts = [start_lm, goal_lm, np.array([[xedges[0], yedges[0]], [xedges[-1], yedges[-1]]])]
    if len(trajectory_xy) > 0:
        ctx_pts.append(trajectory_xy)
    xmin, xmax, ymin, ymax = _axis_limits(ctx_pts)
    ax_ctx.set_xlim(xmin, xmax)
    ax_ctx.set_ylim(ymin, ymax)
    ax_ctx.set_aspect("equal", adjustable="box")
    ax_ctx.set_xlabel("x (m)")
    ax_ctx.set_ylabel("y (m)")
    ax_ctx.grid(True, linestyle="--", alpha=0.35)
    ax_ctx.legend(loc="best", framealpha=0.9)
    traj_disp = (
        float(np.linalg.norm(trajectory_xy[-1] - trajectory_xy[0]))
        if len(trajectory_xy) > 1
        else 0.0
    )
    ax_ctx.set_title(f"workspace context (plan len={len(result.actions)}, traj Δ={traj_disp:.3f} m)")

    # --- Panel 2: expansion count per cell -------------------------------
    mesh = ax_hm.pcolormesh(
        xedges,
        yedges,
        np.ma.masked_where(H.T == 0, H.T),
        cmap=cmap,
        norm=norm,
        shading="flat",
        edgecolors="white",
        linewidth=0.5,
    )
    cbar = fig.colorbar(mesh, ax=ax_hm, fraction=0.046, pad=0.04)
    cbar.set_label("expansions (log scale)" if vmax > 1 else "expansions")

    n_x, n_y = H.shape
    occupied = int(np.count_nonzero(H))
    if annotate and n_x * n_y <= max_annot_cells:
        xcenters = 0.5 * (xedges[:-1] + xedges[1:])
        ycenters = 0.5 * (yedges[:-1] + yedges[1:])
        fontsize = float(np.clip(90.0 / max(n_x, n_y), 4.5, 11.0))
        for i, xc in enumerate(xcenters):
            for j, yc in enumerate(ycenters):
                count = H[i, j]
                if count == 0:
                    continue
                dark_cell = norm(count) > 0.55
                fg, bg = ("white", "black") if dark_cell else ("black", "white")
                ax_hm.text(
                    xc,
                    yc,
                    f"{int(count)}",
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    color=fg,
                    zorder=12,
                    # Counts sit on top of the trajectory line; outline keeps them legible.
                    path_effects=[withStroke(linewidth=1.6, foreground=bg)],
                )

    _draw_trajectory(ax_hm, trajectory_xy, lw=1.6, halo=3.0, star=90)
    ax_hm.set_xlim(xedges[0], xedges[-1])
    ax_hm.set_ylim(yedges[0], yedges[-1])
    ax_hm.set_aspect("equal", adjustable="box")
    ax_hm.set_xlabel("x (m)")
    ax_hm.set_ylabel("y (m)")
    ax_hm.set_title(
        f"expansions per {cell_size * 1000:.0f} mm cell "
        f"({n_x}x{n_y} grid, {occupied} occupied)"
    )

    fig.suptitle(
        f"Expansion heatmap — {len(expansion_xy)} expansions — "
        f"{when.strftime('%Y-%m-%d %H:%M:%S')}"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
