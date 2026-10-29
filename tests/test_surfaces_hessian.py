"""Sanity checks: known analytic stationary points classify as expected.
Run with: python -m pytest tests/ (from within the project venv).
"""

import numpy as np

from hessiantrace.hessian import classify_point, compute_hessian
from hessiantrace.surfaces import rastrigin, rosenbrock, saddle_quartic


def test_rosenbrock_global_min_classifies_as_min():
    surf = rosenbrock(dim=2)
    h = compute_hessian(surf, np.array([1.0, 1.0]))
    result = classify_point(h)
    assert result["classification"] == "local_min"


def test_saddle_quartic_origin_is_saddle():
    surf = saddle_quartic(dim=2)
    h = compute_hessian(surf, np.array([0.0, 0.0]))
    result = classify_point(h)
    assert result["classification"] == "saddle"


def test_saddle_quartic_known_minima_are_minima():
    surf = saddle_quartic(dim=2)
    for point in surf.known_minima:
        h = compute_hessian(surf, point)
        result = classify_point(h)
        assert result["classification"] == "local_min", f"point {point} -> {result['classification']}"


def test_rastrigin_origin_is_global_min():
    surf = rastrigin(dim=2)
    h = compute_hessian(surf, np.array([0.0, 0.0]))
    result = classify_point(h)
    assert result["classification"] == "local_min"
    assert surf.loss_np(np.array([[0.0, 0.0]]))[0] == 0.0


def test_rastrigin_neighboring_ripple_is_local_min_not_global():
    surf = rastrigin(dim=2)
    # Rastrigin ripples repeat every integer; (1, 0) is a local min with
    # strictly positive loss, distinct from the global min at the origin.
    h = compute_hessian(surf, np.array([1.0, 0.0]))
    result = classify_point(h)
    assert result["classification"] == "local_min"
    assert surf.loss_np(np.array([[1.0, 0.0]]))[0] > 0.0
