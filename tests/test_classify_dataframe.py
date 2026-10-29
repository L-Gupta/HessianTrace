"""Regression tests for the Newton-step gating in classify_dataframe: a run
that hasn't actually reached a stationary point must not be labeled
min/max/saddle based on curvature at wherever it happens to be sitting.
"""

import numpy as np
import pandas as pd

from hessiantrace.hessian import classify_dataframe
from hessiantrace.surfaces import rosenbrock, saddle_quartic


def _row(surface_key, x_final, converged=False):
    return {
        "surface": surface_key,
        "optimizer": "test",
        "run_id": 0,
        "seed": 0,
        "lr": 0.01,
        "x0": np.array([0.0, 0.0]),
        "x_final": np.asarray(x_final, dtype=float),
        "loss_final": 0.0,
        "grad_norm_final": 0.0,  # deliberately wrong: must not be trusted
        "converged": converged,
        "steps_to_converge": 100,
    }


def _classify(surface_key, surface, x_final):
    runs = pd.DataFrame([_row(surface_key, x_final)])
    return classify_dataframe({surface_key: surface}, runs).loc[0]


def test_far_from_stationary_point_is_not_converged_even_if_curvature_is_definite():
    # Hessian at (0.5, 1) is diag(2, 4), positive definite, but the
    # gradient (1, 0) puts the nearest stationary point 0.5 away.
    out = _classify("saddle_quartic", saddle_quartic(dim=2), [0.5, 1.0])
    assert out["classification"] == "not_converged"


def test_rosenbrock_indefinite_region_above_valley_is_not_misclassified_saddle():
    # Well above the valley floor (y - x^2 large), Rosenbrock's Hessian is
    # indefinite even though this point is nowhere near a stationary
    # point -- must not be reported as a "saddle".
    out = _classify("rosenbrock", rosenbrock(dim=2), [0.0, 2.0])
    assert out["classification"] == "not_converged"


def test_rosenbrock_valley_floor_with_small_gradient_is_not_local_min():
    # On the valley floor y = x^2 the gradient is small (norm 0.6 here)
    # and the Hessian is positive definite, yet the only stationary point
    # is (1, 1). A raw grad-norm gate labels this "local_min".
    out = _classify("rosenbrock", rosenbrock(dim=2), [0.7, 0.49])
    assert out["classification"] == "not_converged"
    assert out["newton_step"] > 0.05


def test_near_sharp_minimum_with_nonzero_gradient_is_local_min():
    # Close to Rosenbrock's minimum, high curvature makes the raw gradient
    # norm large relative to the actual distance -- still a local min.
    out = _classify("rosenbrock", rosenbrock(dim=2), [1.0005, 1.001])
    assert out["classification"] == "local_min"


def test_exact_stationary_point_is_classified_regardless_of_converged_flag():
    out = _classify("saddle_quartic", saddle_quartic(dim=2), [0.0, 1.0])
    assert out["classification"] == "local_min"
    assert out["newton_step"] < 1e-12
