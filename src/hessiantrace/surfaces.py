"""Synthetic non-convex loss surfaces, implemented as differentiable
TensorFlow functions so gradients/Hessians come from autodiff rather than
hand-derived formulas.

Each surface exposes a batched ``loss_fn``: given a tensor of shape
``(..., dim)`` it returns per-row losses of shape ``(...,)``. Because rows
are independent, a whole sweep of initial points can be optimized in one
TF graph by summing the per-row losses before differentiating -- the
gradient of that sum w.r.t. the batched variable is exactly the stack of
per-row gradients.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import tensorflow as tf


@dataclass
class Surface:
    name: str
    dim: int
    bounds: tuple[float, float]
    loss_fn: Callable[[tf.Tensor], tf.Tensor]
    known_minima: Optional[np.ndarray] = None
    description: str = ""

    def sample_inits(self, n: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        lo, hi = self.bounds
        return rng.uniform(lo, hi, size=(n, self.dim)).astype(np.float64)

    def loss_np(self, x: np.ndarray) -> np.ndarray:
        return self.loss_fn(tf.constant(x, dtype=tf.float64)).numpy()


def rosenbrock(dim: int = 2, a: float = 1.0, b: float = 100.0) -> Surface:
    """Narrow curved valley; global minimum at all-ones with loss 0.

    Classic torture test for plain gradient descent -- the valley floor is
    extremely ill-conditioned (curvature ratio ~O(b)), so vanilla SGD tends
    to zig-zag or crawl while momentum/adaptive methods track the curve.
    """
    if dim < 2:
        raise ValueError("Rosenbrock requires dim >= 2")

    def loss_fn(x: tf.Tensor) -> tf.Tensor:
        x = tf.cast(x, tf.float64)
        head = x[..., :-1]
        tail = x[..., 1:]
        term = b * tf.square(tail - tf.square(head)) + tf.square(a - head)
        return tf.reduce_sum(term, axis=-1)

    return Surface(
        name=f"Rosenbrock-{dim}D",
        dim=dim,
        bounds=(-2.0, 2.0),
        loss_fn=loss_fn,
        known_minima=np.ones((1, dim)),
        description=(
            "Narrow curved valley (a=1, b=100). Global minimum at "
            f"x=(1,...,1) [{dim} dims] with loss 0. Tests an optimizer's "
            "ability to navigate ill-conditioned curvature."
        ),
    )


def saddle_quartic(dim: int = 2, n_saddle: Optional[int] = None) -> Surface:
    """Generalization of f(x, y) = x^2 - y^2, made bounded below by a
    quartic term on the "saddle" coordinates so gradient descent has
    somewhere to actually land instead of diverging to -infinity.

    Coordinates split into two groups:
      * "bowl" coordinates (first ``dim - n_saddle``): pure x^2, pinning
        the stationary structure to those axes being 0 at every critical
        point.
      * "saddle" coordinates (last ``n_saddle``): -y^2 + 0.5*y^4, a
        symmetric double well with a saddle at y=0 and minima at y=+-1.

    The origin is always a genuine saddle point (mixed Hessian signs);
    the 2*n_saddle points where each saddle coordinate is +-1 (bowl
    coordinates 0) are genuine local minima.
    """
    if n_saddle is None:
        n_saddle = max(1, dim // 2)
    if not (0 < n_saddle <= dim):
        raise ValueError("n_saddle must be in (0, dim]")
    n_bowl = dim - n_saddle

    def loss_fn(x: tf.Tensor) -> tf.Tensor:
        x = tf.cast(x, tf.float64)
        bowl = x[..., :n_bowl]
        saddle = x[..., n_bowl:]
        bowl_term = tf.reduce_sum(tf.square(bowl), axis=-1)
        saddle_term = tf.reduce_sum(
            -tf.square(saddle) + 0.5 * tf.square(tf.square(saddle)), axis=-1
        )
        return bowl_term + saddle_term

    minima = np.zeros((2 * n_saddle, dim))
    for i in range(n_saddle):
        minima[2 * i, n_bowl + i] = 1.0
        minima[2 * i + 1, n_bowl + i] = -1.0

    return Surface(
        name=f"SaddleQuartic-{dim}D",
        dim=dim,
        bounds=(-1.5, 1.5),
        loss_fn=loss_fn,
        known_minima=minima,
        description=(
            f"Generalized x^2 - y^2 saddle ({n_bowl} bowl dims + {n_saddle} "
            "quartic-regularized saddle dims). Saddle point at the origin, "
            f"{2 * n_saddle} true local minima at each saddle coordinate "
            "= +-1. Direct empirical mirror of FOSP/SOSP classification."
        ),
    )


def rastrigin(dim: int = 2, a: float = 10.0) -> Surface:
    """Highly multi-modal surface: a quadratic bowl overlaid with cosine
    ripples, producing many symmetric local minima. Global minimum at the
    origin with loss 0.
    """

    def loss_fn(x: tf.Tensor) -> tf.Tensor:
        x = tf.cast(x, tf.float64)
        term = tf.square(x) - a * tf.cos(2.0 * np.pi * x)
        return a * dim + tf.reduce_sum(term, axis=-1)

    return Surface(
        name=f"Rastrigin-{dim}D",
        dim=dim,
        bounds=(-5.12, 5.12),
        loss_fn=loss_fn,
        known_minima=np.zeros((1, dim)),
        description=(
            f"Quadratic bowl (A={a}) with cosine ripples added: many "
            "regularly-spaced local minima surround the global minimum "
            "at the origin. Tests whether an optimizer gets trapped."
        ),
    )


def default_surfaces(dim: int = 2) -> dict[str, Surface]:
    return {
        "rosenbrock": rosenbrock(dim=dim),
        "saddle_quartic": saddle_quartic(dim=dim),
        "rastrigin": rastrigin(dim=dim),
    }
