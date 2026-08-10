"""Plot expansion heatmap with start/goal T poses and chosen trajectory."""
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle
from matplotlib.patheffects import withStroke

from geometry import TEE_LANDMARKS_XY, landmarks_world_xy, yaw_from_quat
from plan_io import PlanResult

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def results_path(stem, ext=".png", when=None):
    """results/<stem>_YYYYmmdd-HHMMSS<ext>, directory created on demand."""
    stamp = (when or datetime.now()).strftime("%Y%m%d-%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / f"{stem}_{stamp}{ext}"


def _draw_tee(ax, pose, *, facecolor, edgecolor, label=None, alpha=0.22, zorder=5, lw=1.8):
    landmarks = landmarks_world_xy(pose)
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
    yaw = float(yaw_from_quat(pose[3:7]))
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


def _count_norm(H):
    """Log norm for counts, degrading gracefully when every cell has one hit."""
    vmax = float(H.max()) if H.size else 0.0
    norm = LogNorm(vmin=1, vmax=vmax) if vmax > 1 else Normalize(vmin=0, vmax=1)
    return norm, vmax


def _heat_mesh(ax, H, xedges, yedges, cmap, norm):
    return ax.pcolormesh(
        xedges,
        yedges,
        np.ma.masked_where(H.T == 0, H.T),
        cmap=cmap,
        norm=norm,
        shading="flat",
        edgecolors="white",
        linewidth=0.5,
    )


def _count_colorbar(fig, ax, mesh, vmax):
    cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("expansions (log scale)" if vmax > 1 else "expansions")
    return cbar


def _empty_panel(ax, message):
    ax.text(
        0.5,
        0.5,
        message,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        color="#777777",
    )
    ax.set_xticks([])
    ax.set_yticks([])


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


def _panel_tcp_object_frame(fig, ax, rel_xy, cell_size, cmap):
    """Expansions binned by TCP position in the T's body frame.

    Object xy hides most of the search: nodes that share a T pose but differ in
    where the pusher sits collapse into one cell there and spread out here.
    """
    ax.set_xlabel("x in T frame (m)")
    ax.set_ylabel("y in T frame (m)")
    if rel_xy is None or len(rel_xy) == 0:
        _empty_panel(ax, "no per-expansion TCP log\n(re-run the planner to record it)")
        ax.set_title("TCP in object frame")
        return

    xedges = _aligned_edges(rel_xy[:, 0].min(), rel_xy[:, 0].max(), cell_size)
    yedges = _aligned_edges(rel_xy[:, 1].min(), rel_xy[:, 1].max(), cell_size)
    H, _, _ = np.histogram2d(rel_xy[:, 0], rel_xy[:, 1], bins=[xedges, yedges])
    norm, vmax = _count_norm(H)
    mesh = _heat_mesh(ax, H, xedges, yedges, cmap, norm)
    _count_colorbar(fig, ax, mesh, vmax)

    ax.add_patch(
        Polygon(
            TEE_LANDMARKS_XY,
            closed=True,
            fill=False,
            edgecolor="#1F4E79",
            linewidth=2.0,
            zorder=6,
            label="T (body frame)",
        )
    )
    ax.plot(0.0, 0.0, "+", color="#1F4E79", markersize=9, zorder=7)

    xmin, xmax, ymin, ymax = _axis_limits([TEE_LANDMARKS_XY, rel_xy], pad=0.02)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", framealpha=0.9)
    ax.set_title(
        f"TCP in object frame — expansions per {cell_size * 1000:.0f} mm cell "
        f"({int(np.count_nonzero(H))} occupied)"
    )


def _panel_expansion_order(fig, ax, expansion_xy, trajectory_xy, xedges, yedges):
    """Same tee COM points as the count heatmap, coloured by when they were expanded."""
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    if len(expansion_xy) == 0:
        _empty_panel(ax, "no expansions recorded")
        ax.set_title("expansion order")
        return

    scatter = ax.scatter(
        expansion_xy[:, 0],
        expansion_xy[:, 1],
        c=np.arange(len(expansion_xy)),
        cmap="viridis",
        s=9,
        linewidths=0,
        zorder=4,
    )
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("expansion index")

    _draw_trajectory(ax, trajectory_xy, lw=1.6, halo=3.0, star=90)
    ax.set_xlim(xedges[0], xedges[-1])
    ax.set_ylim(yedges[0], yedges[-1])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.set_title("expansion order (dark = early, bright = late)")


def plot_expansion_heatmap(
    result: PlanResult,
    out_path=None,
    cell_size=0.003,
    annotate=True,
    max_annot_cells=400,
    rel_cell_size=0.005,
):
    """Save a 2x2 expansion figure.

    Top left: start/goal T, plan trajectory, and a box marking the region the
    search actually touched. Top right: expansions per `cell_size` cell of tee
    COM xy, so states that share an xy but differ in orientation stack into one
    cell. Bottom left: the same expansions binned by TCP position in the T's
    body frame. Bottom right: tee COM coloured by expansion index.
    """
    when = datetime.now()
    out_path = Path(out_path) if out_path is not None else results_path(
        "expansion_heatmap", when=when
    )
    expansion_xy = np.asarray(result.expansion_xy, dtype=np.float64).reshape(-1, 2)
    trajectory_xy = np.asarray(result.trajectory_xy, dtype=np.float64).reshape(-1, 2)

    start_lm = landmarks_world_xy(result.start_pose)
    goal_lm = landmarks_world_xy(result.goal_pose)

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

    norm, vmax = _count_norm(H)
    cmap = mpl.colormaps["magma_r"].copy()
    cmap.set_bad("#eeeeee")

    fig, axes = plt.subplots(2, 2, figsize=(14, 13.5), constrained_layout=True)
    (ax_ctx, ax_hm), (ax_rel, ax_order) = axes

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
    mesh = _heat_mesh(ax_hm, H, xedges, yedges, cmap, norm)
    _count_colorbar(fig, ax_hm, mesh, vmax)

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

    # --- Panel 3: TCP in object frame ------------------------------------
    rel_xy = None
    if result.log is not None and "rel_x" in result.log.expansions:
        rel_xy = np.column_stack(
            [result.log.expansions["rel_x"], result.log.expansions["rel_y"]]
        ).astype(np.float64)
    _panel_tcp_object_frame(fig, ax_rel, rel_xy, rel_cell_size, cmap)

    # --- Panel 4: expansion order ----------------------------------------
    _panel_expansion_order(fig, ax_order, expansion_xy, trajectory_xy, xedges, yedges)

    fig.suptitle(
        f"Expansion heatmap — {len(expansion_xy)} expansions — "
        f"{when.strftime('%Y-%m-%d %H:%M:%S')}"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
