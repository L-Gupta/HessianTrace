"""Run the full optimizer sweep grid and save results to results/.

Usage:
    python scripts/run_experiment.py [--dim 2] [--n-inits 100] [--max-steps 3000] [--seed 0]
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from hessiantrace.pipeline import run_experiment

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--n-inits", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--grad-tol", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(
        f"Running sweep: dim={args.dim} n_inits={args.n_inits} "
        f"max_steps={args.max_steps} grad_tol={args.grad_tol} seed={args.seed}"
    )
    result = run_experiment(
        dim=args.dim,
        n_inits=args.n_inits,
        max_steps=args.max_steps,
        grad_tol=args.grad_tol,
        seed=args.seed,
    )

    RESULTS_DIR.mkdir(exist_ok=True)
    result.classified.to_csv(RESULTS_DIR / "classified_runs.csv", index=False)
    result.summary.to_csv(RESULTS_DIR / "summary.csv", index=False)
    with open(RESULTS_DIR / "trajectories.pkl", "wb") as f:
        pickle.dump(result.trajectories, f)

    print(f"Saved {len(result.classified)} runs to {RESULTS_DIR}")
    print(result.summary.to_string(index=False))


if __name__ == "__main__":
    main()
