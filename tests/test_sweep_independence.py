"""Regression test: runs in a batched sweep must be independent -- a run's
outcome cannot depend on which other initializations share its batch.
Guards against batch-wide operations (e.g. Keras's ``clipnorm``, which
clips by the norm of the whole (n_inits, dim) gradient tensor).
"""

import numpy as np
import pytest

from hessiantrace.optimizers import OPTIMIZER_NAMES, run_sweep
from hessiantrace.surfaces import rosenbrock


@pytest.mark.parametrize("optimizer_name", OPTIMIZER_NAMES)
def test_batched_run_matches_solo_run(optimizer_name):
    # Rosenbrock from [-2, 2]^2 has raw gradient norms in the hundreds,
    # so the CLIP_NORM safety net is active for many rows at once.
    batch_surf = rosenbrock(dim=2)
    batch = run_sweep(
        batch_surf, "rosenbrock", optimizer_name, n_inits=20, max_steps=200, seed=0
    ).runs

    x0 = batch.loc[0, "x0"]
    solo_surf = rosenbrock(dim=2)
    solo_surf.sample_inits = lambda n, seed: x0[np.newaxis, :].copy()
    solo = run_sweep(
        solo_surf, "rosenbrock", optimizer_name, n_inits=1, max_steps=200, seed=0
    ).runs

    np.testing.assert_allclose(batch.loc[0, "x_final"], solo.loc[0, "x_final"], rtol=1e-9, atol=1e-12)
