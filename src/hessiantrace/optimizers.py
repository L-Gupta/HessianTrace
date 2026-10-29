"""Optimizer sweep: run many independent random initializations against a
loss surface simultaneously (batched as rows of one TF variable) and log
per-run convergence outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
import tensorflow as tf

from .surfaces import Surface

# Reasonable default learning rates per optimizer family. Rosenbrock's
# curvature (~O(b)=100) is far steeper than the other two surfaces, so it
# gets its own smaller rates to avoid immediate divergence; gradients are
# also clipped by norm as a second safety net so every optimizer is at
# least numerically comparable rather than blowing up to NaN. Clipping is
# done per row in run_sweep, not via Keras's clipnorm: clipnorm takes the
# norm of the whole (n_inits, dim) gradient tensor, which would let one
# steep run shrink every other run's step and couple runs that are meant
# to be independent.
DEFAULT_LR = {
    "rosenbrock": {"SGD": 1e-3, "SGD+Momentum": 1e-3, "Adam": 5e-2, "RMSprop": 2e-2},
    "saddle_quartic": {"SGD": 2e-2, "SGD+Momentum": 1e-2, "Adam": 5e-2, "RMSprop": 5e-2},
    "rastrigin": {"SGD": 1e-2, "SGD+Momentum": 5e-3, "Adam": 3e-2, "RMSprop": 3e-2},
}

CLIP_NORM = 5.0


def _lr_schedule(lr: float, max_steps: int) -> tf.keras.optimizers.schedules.LearningRateSchedule:
    """Exponential decay applied identically across optimizers. Without
    decay, adaptive methods (Adam/RMSprop) take a fixed-size step forever
    and can hover indefinitely near a minimum without ever crossing a
    strict gradient-norm threshold -- decay lets every optimizer actually
    settle, so "steps to converge" measures real convergence speed rather
    than an artifact of a constant step size.
    """
    return tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr,
        decay_steps=max(1, max_steps // 4),
        decay_rate=0.5,
        staircase=False,
    )


def make_optimizer(name: str, lr: float, max_steps: int) -> tf.keras.optimizers.Optimizer:
    schedule = _lr_schedule(lr, max_steps)
    if name == "SGD":
        return tf.keras.optimizers.SGD(learning_rate=schedule)
    if name == "SGD+Momentum":
        return tf.keras.optimizers.SGD(learning_rate=schedule, momentum=0.9)
    if name == "Adam":
        return tf.keras.optimizers.Adam(learning_rate=schedule)
    if name == "RMSprop":
        return tf.keras.optimizers.RMSprop(learning_rate=schedule)
    raise ValueError(f"unknown optimizer {name!r}")


OPTIMIZER_NAMES = ["SGD", "SGD+Momentum", "Adam", "RMSprop"]


@dataclass
class SweepResult:
    runs: pd.DataFrame              # one row per initialization
    trajectories: dict[int, np.ndarray]  # run_id -> (steps_logged, dim) positions, subset only


def run_sweep(
    surface: Surface,
    surface_key: str,
    optimizer_name: str,
    n_inits: int = 100,
    max_steps: int = 3000,
    grad_tol: float = 1e-3,
    lr: Optional[float] = None,
    seed: int = 0,
    n_trajectories_kept: int = 8,
    trajectory_stride: int = 5,
) -> SweepResult:
    if lr is None:
        lr = DEFAULT_LR[surface_key][optimizer_name]

    x0 = surface.sample_inits(n_inits, seed=seed)
    x = tf.Variable(x0, dtype=tf.float64)
    optimizer = make_optimizer(optimizer_name, lr, max_steps)

    steps_to_converge = np.full(n_inits, -1, dtype=np.int64)
    converged_mask = np.zeros(n_inits, dtype=bool)

    n_keep = min(n_trajectories_kept, n_inits)
    traj_logs = {i: [x0[i].copy()] for i in range(n_keep)}

    for step in range(1, max_steps + 1):
        with tf.GradientTape() as tape:
            per_row_loss = surface.loss_fn(x)
            total = tf.reduce_sum(per_row_loss)
        grad = tape.gradient(total, x)
        optimizer.apply_gradients([(tf.clip_by_norm(grad, CLIP_NORM, axes=[-1]), x)])

        grad_norm = tf.norm(grad, axis=-1).numpy()
        newly_converged = (~converged_mask) & (grad_norm < grad_tol)
        steps_to_converge[newly_converged] = step
        converged_mask |= newly_converged

        if step % trajectory_stride == 0:
            x_np = x.numpy()
            for i in range(n_keep):
                traj_logs[i].append(x_np[i].copy())

        if converged_mask.all():
            break

    x_final = x.numpy()
    final_loss = surface.loss_np(x_final)
    final_grad_norm = grad_norm  # from last iteration

    rows = []
    for i in range(n_inits):
        rows.append(
            {
                "surface": surface_key,
                "optimizer": optimizer_name,
                "run_id": i,
                "seed": seed,
                "lr": lr,
                "x0": x0[i],
                "x_final": x_final[i],
                "loss_final": float(final_loss[i]),
                "grad_norm_final": float(final_grad_norm[i]),
                "converged": bool(converged_mask[i]),
                "steps_to_converge": int(steps_to_converge[i]) if converged_mask[i] else max_steps,
            }
        )
    runs = pd.DataFrame(rows)
    trajectories = {i: np.stack(v) for i, v in traj_logs.items()}
    return SweepResult(runs=runs, trajectories=trajectories)


def run_full_grid(
    surfaces: dict[str, Surface],
    optimizer_names: list[str] = OPTIMIZER_NAMES,
    n_inits: int = 100,
    max_steps: int = 3000,
    grad_tol: float = 1e-3,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict[tuple[str, str], dict[int, np.ndarray]]]:
    """Run every (surface, optimizer) pair and return a single concatenated
    runs DataFrame plus a dict of trajectories keyed by (surface, optimizer).
    """
    all_runs = []
    all_traj = {}
    for surface_key, surface in surfaces.items():
        for opt_name in optimizer_names:
            result = run_sweep(
                surface,
                surface_key,
                opt_name,
                n_inits=n_inits,
                max_steps=max_steps,
                grad_tol=grad_tol,
                seed=seed,
            )
            all_runs.append(result.runs)
            all_traj[(surface_key, opt_name)] = result.trajectories
    return pd.concat(all_runs, ignore_index=True), all_traj
