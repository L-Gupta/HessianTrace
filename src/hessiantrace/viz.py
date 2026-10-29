"""Visualization: 3D/contour loss surfaces with optimizer trajectories
overlaid, and summary charts for the statistical comparison.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)

from .surfaces import Surface

OPTIMIZER_COLORS = {
    "SGD": "#d62728",
    "SGD+Momentum": "#ff7f0e",
    "Adam": "#1f77b4",
    "RMSprop": "#2ca02c",
}


def _grid(surface: Surface, resolution: int):
    lo, hi = surface.bounds
    xs = np.linspace(lo, hi, resolution)
    ys = np.linspace(lo, hi, resolution)
    xx, yy = np.meshgrid(xs, ys)
    pts = np.stack([xx.ravel(), yy.ravel()], axis=-1)
    zz = surface.loss_np(pts).reshape(xx.shape)
    return xx, yy, zz


def plot_surface_3d(surface: Surface, resolution: int = 120, log_scale: bool = True) -> Figure:
    if surface.dim != 2:
        raise ValueError("3D surface plot only supported for 2D surfaces")
    xx, yy, zz = _grid(surface, resolution)
    z_plot = np.log1p(zz) if log_scale else zz

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(xx, yy, z_plot, cmap="viridis", linewidth=0, antialiased=True, alpha=0.9)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("log(1 + loss)" if log_scale else "loss")
    ax.set_title(f"{surface.name} loss surface")
    fig.tight_layout()
    return fig


def plot_trajectories(
    surface: Surface,
    trajectories_by_optimizer: dict[str, dict[int, np.ndarray]],
    resolution: int = 200,
    log_scale: bool = True,
) -> Figure:
    """2D contour of the surface with optimizer trajectories overlaid,
    color-coded by optimizer.
    """
    if surface.dim != 2:
        raise ValueError("Contour trajectory plot only supported for 2D surfaces")
    xx, yy, zz = _grid(surface, resolution)
    z_plot = np.log1p(zz) if log_scale else zz

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.contourf(xx, yy, z_plot, levels=40, cmap="viridis", alpha=0.85)

    for opt_name, traj_dict in trajectories_by_optimizer.items():
        color = OPTIMIZER_COLORS.get(opt_name, None)
        for i, traj in traj_dict.items():
            ax.plot(
                traj[:, 0], traj[:, 1], color=color, linewidth=1.0, alpha=0.8,
                label=opt_name if i == 0 else None,
            )
            ax.scatter(traj[-1, 0], traj[-1, 1], color=color, s=15, zorder=5)
            ax.scatter(traj[0, 0], traj[0, 1], color=color, s=10, marker="x", zorder=5)

    if surface.known_minima is not None:
        ax.scatter(
            surface.known_minima[:, 0], surface.known_minima[:, 1],
            color="white", edgecolor="black", marker="*", s=180, zorder=6,
            label="known minimum",
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Optimizer trajectories on {surface.name}")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_convergence_rates(summary_df, surface_key: str) -> Figure:
    """Stacked bar chart of stationary-point classification rates per
    optimizer, for one surface.
    """
    sub = summary_df[summary_df["surface"] == surface_key].sort_values("optimizer")
    optimizers = sub["optimizer"].tolist()
    categories = [
        ("local_min_rate", "local min", "#2ca02c"),
        ("saddle_rate", "saddle", "#d62728"),
        ("local_max_rate", "local max", "#9467bd"),
        ("degenerate_rate", "degenerate", "#7f7f7f"),
        ("not_converged_rate", "not converged", "#c7c7c7"),
    ]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bottom = np.zeros(len(sub))
    for col, label, color in categories:
        values = sub[col].to_numpy()
        ax.bar(optimizers, values, bottom=bottom, label=label, color=color)
        bottom += values

    ax.set_ylabel("fraction of runs")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Stationary-point classification: {surface_key}")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


def plot_steps_distribution(classified_df, surface_key: str) -> Figure:
    """Boxplot of steps-to-converge among runs that reached a genuine
    local minimum, grouped by optimizer.
    """
    sub = classified_df[
        (classified_df["surface"] == surface_key) & (classified_df["classification"] == "local_min")
    ]
    optimizers = sorted(sub["optimizer"].unique())
    data = [sub.loc[sub["optimizer"] == o, "steps_to_converge"].to_numpy() for o in optimizers]
    colors = [OPTIMIZER_COLORS.get(o, "#333333") for o in optimizers]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bp = ax.boxplot(data, tick_labels=optimizers, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel("steps to converge")
    ax.set_title(f"Convergence speed to a local min: {surface_key}")
    fig.tight_layout()
    return fig
