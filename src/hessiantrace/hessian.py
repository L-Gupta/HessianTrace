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


def compute_gradient_and_hessian(surface: Surface, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact gradient and Hessian at a single point via nested autodiff
    (reverse-over-reverse: outer tape differentiates the gradient produced
    by the inner tape), not finite differences.
    """
    x_var = tf.Variable(np.asarray(x, dtype=np.float64))
    with tf.GradientTape() as tape2:
        with tf.GradientTape() as tape1:
            loss = surface.loss_fn(x_var[tf.newaxis, :])[0]
        grad = tape1.gradient(loss, x_var)
    hessian = tape2.jacobian(grad, x_var)
    return grad.numpy(), hessian.numpy()


def compute_hessian(surface: Surface, x: np.ndarray) -> np.ndarray:
    return compute_gradient_and_hessian(surface, x)[1]


def newton_step_length(grad: np.ndarray, hessian: np.ndarray, eig_floor: float = 1e-4) -> float:
    """Length of the Newton step ``|H|^-1 g``: the local-quadratic estimate
    of how far ``x`` is from the nearest stationary point, of any type.

    Uses absolute eigenvalues so the estimate is a distance even when H is
    indefinite (near a saddle), and floors them at ``eig_floor`` so a
    gradient component along a flat direction yields a large step (not
    stationary) rather than being silently dropped as a pseudo-inverse
    would.
    """
    h_sym = (hessian + hessian.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(h_sym)
    coeffs = (eigvecs.T @ grad) / np.maximum(np.abs(eigvals), eig_floor)
    return float(np.linalg.norm(coeffs))


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
    max_newton_step: float = 0.05,
) -> pd.DataFrame:
    """Add Hessian-based classification columns to a copy of a sweep runs
    DataFrame, keyed off each row's ``surface`` column and ``x_final``.

    Eigenvalue-sign classification is only meaningful *at* a stationary
    point, so it is gated on the Newton-step length ``|H|^-1 g`` at
    ``x_final`` -- an estimate of the distance to the nearest stationary
    point -- rather than on the sweep's ``converged`` flag or the raw
    gradient norm. Raw gradient norm is a poor proxy for distance because
    it scales with curvature: RMSprop near Rosenbrock's sharply curved
    minimum can sit at gradient norm ~0.5 while within ~0.02 of it, while
    plain SGD crawling along the flat valley floor has a similar gradient
    norm yet is ~0.7 away. Gating on the raw gradient norm labels the
    latter ``local_min`` (the valley-floor Hessian is positive definite)
    even though Rosenbrock's only stationary point is at (1, 1).

    Runs whose Newton step exceeds ``max_newton_step`` are labeled
    ``not_converged``. The gradient is recomputed at ``x_final`` rather
    than taken from ``grad_norm_final``, which the sweep records one
    update earlier.
    """
    out = runs.copy()
    classifications, min_eigs, max_eigs, newton_steps = [], [], [], []
    for surface_key, x_final in zip(out["surface"], out["x_final"]):
        g, h = compute_gradient_and_hessian(surfaces[surface_key], x_final)
        step = newton_step_length(g, h, eig_floor=tol)
        newton_steps.append(step)
        if step > max_newton_step:
            classifications.append(CLASS_NOT_CONVERGED)
            min_eigs.append(float("nan"))
            max_eigs.append(float("nan"))
            continue
        result = classify_point(h, tol=tol)
        classifications.append(result["classification"])
        min_eigs.append(result["min_eig"])
        max_eigs.append(result["max_eig"])
    out["classification"] = classifications
    out["min_eig"] = min_eigs
    out["max_eig"] = max_eigs
    out["newton_step"] = newton_steps
    return out
