# HessianTrace

Empirical analysis of optimizer behavior on non-convex loss landscapes.

Several optimizers (SGD, SGD+Momentum, Adam, RMSprop) are dropped onto
synthetic non-convex loss surfaces from many random starting points. Every
converged point is classified via its Hessian eigenvalues -- local minimum,
local maximum, or saddle -- and the optimizers are compared statistically on
how often and how quickly they reach a genuine local minimum rather than
stalling at a saddle point.

This is the empirical companion to first-order/second-order stationary
point (FOSP/SOSP) theory: a small gradient norm alone doesn't distinguish a
minimum from a saddle, so `hessiantrace` computes the second-order
information (via nested `tf.GradientTape` autodiff + `numpy.linalg.eigh`)
needed to tell them apart.

## Surfaces

- **Rosenbrock** -- narrow, ill-conditioned curved valley. Global min at
  `(1, 1)`.
- **Saddle-quartic** -- generalized `x^2 - y^2`, made bounded below by a
  quartic term. Genuine saddle at the origin, true local minima at
  `(0, +-1)`.
- **Rastrigin** -- quadratic bowl with cosine ripples; many regularly
  spaced local minima around the global min at the origin.

## Setup

```powershell
py -3.12 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\pip install -e .
```

TensorFlow does not yet support Python 3.14 on Windows, hence pinning the
venv to 3.12.

## Usage

Run the full sweep from the command line and save results to `results/`:

```powershell
.venv\Scripts\python scripts\run_experiment.py --n-inits 100 --max-steps 3000
```

Or open the walkthrough notebook (register the venv as a Jupyter kernel
first):

```powershell
.venv\Scripts\python -m ipykernel install --user --name hessiantrace
.venv\Scripts\jupyter notebook notebooks\report.ipynb
```

Run tests:

```powershell
.venv\Scripts\python -m pytest tests/
```

## Structure

```
src/hessiantrace/
  surfaces.py     TensorFlow-differentiable loss surfaces
  optimizers.py   batched optimizer sweep runner
  hessian.py      Hessian computation + stationary-point classification
  stats.py        bootstrap CIs, pairwise t-tests across optimizers
  viz.py          surface/trajectory/summary plots
  pipeline.py     end-to-end experiment orchestration
scripts/run_experiment.py   CLI entry point, saves results/*.csv
notebooks/report.ipynb      full walkthrough + writeup
tests/                      sanity checks on known analytic stationary points
```
