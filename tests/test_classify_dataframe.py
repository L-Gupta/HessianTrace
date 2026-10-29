"""Regression tests for the grad-norm gating in classify_dataframe: a run
that hasn't actually reached a stationary point must not be labeled
min/max/saddle based on curvature at wherever it happens to be sitting.
"""

import numpy as np
import pandas as pd

from hessiantrace.hessian import classify_dataframe
from hessiantrace.surfaces import rosenbrock, saddle_quartic


def _row(surface_key, x_final, grad_norm_final, converged):
    return {
        "surface": surface_key,
        "optimizer": "test",
        "run_id": 0,
        "seed": 0,
        "lr": 0.01,
        "x0": np.array([0.0, 0.0]),
        "x_final": np.asarray(x_final),
        "loss_final": 0.0,
        "grad_norm_final": grad_norm_final,
        "converged": converged,
        "steps_to_converge": 100,
    }


def test_far_from_stationary_point_is_not_converged_even_if_curvature_is_definite():
    surf = saddle_quartic(dim=2)
    # A large gradient norm should force "not_converged" regardless of
    # what the local Hessian at that (non-stationary) point looks like.
    runs = pd.DataFrame([_row("saddle_quartic", [0.0, 1.0], 50.0, False)])
    out = classify_dataframe({"saddle_quartic": surf}, runs)
    assert out.loc[0, "classification"] == "not_converged"


def test_rosenbrock_indefinite_region_above_valley_is_not_misclassified_saddle():
    surf = rosenbrock(dim=2)
    # Well above the valley floor (y - x^2 large), Rosenbrock's Hessian is
    # indefinite even though this point is nowhere near a stationary
    # point -- must not be reported as a "saddle" just because the run
    # never converged there.
    runs = pd.DataFrame([_row("rosenbrock", [0.0, 2.0], 300.0, False)])
    out = classify_dataframe({"rosenbrock": surf}, runs)
    assert out.loc[0, "classification"] == "not_converged"


def test_small_grad_norm_but_unconverged_flag_is_still_classified():
    surf = saddle_quartic(dim=2)
    # converged=False (missed the strict threshold) but grad_norm_final
    # is tiny -- should still be classified via the Hessian, not dropped
    # into "not_converged".
    runs = pd.DataFrame([_row("saddle_quartic", [0.0, 1.0], 1e-6, False)])
    out = classify_dataframe({"saddle_quartic": surf}, runs)
    assert out.loc[0, "classification"] == "local_min"
