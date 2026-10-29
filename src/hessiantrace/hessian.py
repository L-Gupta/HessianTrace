"""Hessian computation and stationary-point classification -- the
empirical mirror of first/second-order stationary point (FOSP/SOSP)
theory: a converged point is a genuine local min only if it is a FOSP
(near-zero gradient, checked during the sweep) *and* a SOSP (positive
semi-definite Hessian), rather than a saddle masquerading as convergence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import tensorflow as tf

from .surfaces import Surface

CLASS_MIN = "local_min"
CLASS_MAX = "local_max"
CLASS_SADDLE = "saddle"
CLASS_DEGENERATE = "degenerate"
CLASS_NOT_CONVERGED = "not_converged"


def compute_hessian(surface: Surface, x: np.ndarray) -> np.ndarray:
    """Exact Hessian at a single point via nested autodiff (forward-over-
    reverse: outer tape differentiates the gradient produced by the inner
    tape), not finite differences.
    """
    x_var = tf.Variable(np.asarray(x, dtype=np.float64))
    with tf.GradientTape() as tape2:
        with tf.GradientTape() as tape1:
            loss = surface.loss_fn(x_var[tf.newaxis, :])[0]
        grad = tape1.gradient(loss, x_var)
    hessian = tape2.jacobian(grad, x_var)
    return hessian.numpy()


def classify_point(hessian: np.ndarray, tol: float = 1e-4) -> dict:
    """Classify a stationary point from its Hessian eigenvalues.

    Symmetrizes first to cancel float round-off (the true Hessian of a
    smooth scalar function is symmetric by Clairaut's theorem).
    """
    h_sym = (hessian + hessian.T) / 2.0
    eigvals = np.linalg.eigh(h_sym)[0]
    n_pos = int(np.sum(eigvals > tol))
    n_neg = int(np.sum(eigvals < -tol))
    n = len(eigvals)

    if n_pos == n:
        label = CLASS_MIN
    elif n_neg == n:
        label = CLASS_MAX
    elif n_pos > 0 and n_neg > 0:
        label = CLASS_SADDLE
    else:
        label = CLASS_DEGENERATE  # a near-zero eigenvalue with no opposite sign

    return {
        "classification": label,
        "eigenvalues": eigvals,
        "min_eig": float(eigvals.min()),
        "max_eig": float(eigvals.max()),
    }


def classify_dataframe(
    surfaces: dict[str, Surface],
    runs: pd.DataFrame,
    tol: float = 1e-4,
    max_grad_norm_for_classification: float = 1.0,
) -> pd.DataFrame:
    """Add Hessian-based classification columns to a copy of a sweep runs
    DataFrame, keyed off each row's ``surface`` column and ``x_final``.

    Eigenvalue-sign classification is only meaningful *at* a stationary
    point, so it is gated on ``grad_norm_final`` rather than on the
    sweep's strict ``converged`` flag. The two are not interchangeable:
    Rosenbrock's Hessian is indefinite almost everywhere above its valley
    floor, so a run that is still substantially descending (SGD on
    Rosenbrock: raw gradient norms of 1e2-1e3) must not be run through
    the eigenvalue test, or it gets mislabeled "saddle" despite never
    having actually stopped there. Conversely, a run can fail the
    sweep's *strict* convergence threshold yet already sit almost
    exactly at a minimum -- RMSprop/Adam rescale updates by the running
    RMS gradient, so their raw gradient norm can plateau around ~0.5 near
    a highly curved minimum (e.g. Rosenbrock's Hessian eigenvalues run to
    ~800, so grad-norm 0.5 implies a positional error of only ~6e-4)
    while never dropping below a strict absolute threshold. Runs with
    ``grad_norm_final`` under ``max_grad_norm_for_classification`` are
    therefore still classified via their Hessian even if ``converged`` is
    False; only runs still far from any stationary point are labeled
    ``not_converged``. In this project's data there is a multiple-orders-
    of-magnitude gap between the two populations (~1 vs ~1e2+), so the
    exact cutoff value is not sensitive within a wide range.
    """
    out = runs.copy()
    classifications, min_eigs, max_eigs = [], [], []
    for surface_key, x_final, grad_norm_final in zip(
        out["surface"], out["x_final"], out["grad_norm_final"]
    ):
        if grad_norm_final > max_grad_norm_for_classification:
            classifications.append(CLASS_NOT_CONVERGED)
            min_eigs.append(float("nan"))
            max_eigs.append(float("nan"))
            continue
        h = compute_hessian(surfaces[surface_key], x_final)
        result = classify_point(h, tol=tol)
        classifications.append(result["classification"])
        min_eigs.append(result["min_eig"])
        max_eigs.append(result["max_eig"])
    out["classification"] = classifications
    out["min_eig"] = min_eigs
    out["max_eig"] = max_eigs
    return out
