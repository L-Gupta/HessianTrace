"""End-to-end pipeline: run the full optimizer sweep grid, classify every
converged point via its Hessian, and compute the statistical comparison.
Used by both the notebook and ``scripts/run_experiment.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import stats as stats_mod
from .hessian import classify_dataframe
from .optimizers import OPTIMIZER_NAMES, run_full_grid
from .surfaces import Surface, default_surfaces


@dataclass
class ExperimentResult:
    surfaces: dict[str, Surface]
    runs: pd.DataFrame                 # raw sweep output
    classified: pd.DataFrame           # runs + Hessian classification
    summary: pd.DataFrame              # per (surface, optimizer) stats
    trajectories: dict


def run_experiment(
    dim: int = 2,
    n_inits: int = 100,
    max_steps: int = 3000,
    grad_tol: float = 1e-3,
    hessian_tol: float = 1e-4,
    seed: int = 0,
    optimizer_names: list[str] = OPTIMIZER_NAMES,
) -> ExperimentResult:
    surfaces = default_surfaces(dim=dim)
    runs, trajectories = run_full_grid(
        surfaces,
        optimizer_names=optimizer_names,
        n_inits=n_inits,
        max_steps=max_steps,
        grad_tol=grad_tol,
        seed=seed,
    )
    classified = classify_dataframe(surfaces, runs, tol=hessian_tol)
    summary = stats_mod.convergence_summary(classified)
    return ExperimentResult(
        surfaces=surfaces,
        runs=runs,
        classified=classified,
        summary=summary,
        trajectories=trajectories,
    )
