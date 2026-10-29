"""Statistical comparison across optimizers: convergence-rate summaries,
bootstrap confidence intervals, and pairwise significance tests on
steps-to-converge.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from .hessian import CLASS_MIN


def bootstrap_ci(
    data: np.ndarray,
    stat_fn=np.mean,
    n_boot: int = 10_000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI. Returns (point_estimate, low, high)."""
    data = np.asarray(data)
    point = stat_fn(data)
    if len(data) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot_stats = np.empty(n_boot)
    n = len(data)
    for b in range(n_boot):
        sample = data[rng.integers(0, n, size=n)]
        boot_stats[b] = stat_fn(sample)
    alpha = (1.0 - ci) / 2.0
    low, high = np.quantile(boot_stats, [alpha, 1.0 - alpha])
    return float(point), float(low), float(high)


def convergence_summary(classified: pd.DataFrame, ci: float = 0.95, n_boot: int = 5000) -> pd.DataFrame:
    """Per (surface, optimizer): rate of landing at each stationary-point
    type (with bootstrap CI on the local-min rate), plus steps-to-converge
    stats restricted to runs that actually reached a genuine local min.
    """
    rows = []
    for (surface, optimizer), group in classified.groupby(["surface", "optimizer"]):
        n = len(group)
        is_min = (group["classification"] == CLASS_MIN).to_numpy()
        rate, lo, hi = bootstrap_ci(is_min.astype(float), np.mean, n_boot=n_boot, ci=ci, seed=0)

        min_steps = group.loc[group["classification"] == CLASS_MIN, "steps_to_converge"]
        rows.append(
            {
                "surface": surface,
                "optimizer": optimizer,
                "n_runs": n,
                "local_min_rate": rate,
                "local_min_rate_ci_low": lo,
                "local_min_rate_ci_high": hi,
                "saddle_rate": float((group["classification"] == "saddle").mean()),
                "local_max_rate": float((group["classification"] == "local_max").mean()),
                "degenerate_rate": float((group["classification"] == "degenerate").mean()),
                "not_converged_rate": float((group["classification"] == "not_converged").mean()),
                "mean_steps_to_min": float(min_steps.mean()) if len(min_steps) else float("nan"),
                "median_steps_to_min": float(min_steps.median()) if len(min_steps) else float("nan"),
                "std_steps_to_min": float(min_steps.std()) if len(min_steps) > 1 else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values(["surface", "optimizer"]).reset_index(drop=True)


def pairwise_steps_ttest(
    classified: pd.DataFrame, surface: str, only_local_min: bool = True
) -> pd.DataFrame:
    """Welch's t-test on steps-to-converge between every pair of optimizers
    for a given surface, restricted (by default) to runs that landed at a
    genuine local minimum -- comparing "speed to the right answer" rather
    than mixing in saddle-trapped runs.
    """
    sub = classified[classified["surface"] == surface]
    if only_local_min:
        sub = sub[sub["classification"] == CLASS_MIN]

    rows = []
    optimizers = sorted(sub["optimizer"].unique())
    for a, b in combinations(optimizers, 2):
        steps_a = sub.loc[sub["optimizer"] == a, "steps_to_converge"].to_numpy(dtype=float)
        steps_b = sub.loc[sub["optimizer"] == b, "steps_to_converge"].to_numpy(dtype=float)
        if len(steps_a) < 2 or len(steps_b) < 2:
            t_stat, p_value = float("nan"), float("nan")
        else:
            t_stat, p_value = scipy_stats.ttest_ind(steps_a, steps_b, equal_var=False)
        rows.append(
            {
                "surface": surface,
                "optimizer_a": a,
                "optimizer_b": b,
                "n_a": len(steps_a),
                "n_b": len(steps_b),
                "mean_a": float(np.mean(steps_a)) if len(steps_a) else float("nan"),
                "mean_b": float(np.mean(steps_b)) if len(steps_b) else float("nan"),
                "t_stat": float(t_stat),
                "p_value": float(p_value),
                "significant_p05": bool(p_value < 0.05) if not np.isnan(p_value) else False,
            }
        )
    return pd.DataFrame(rows)


def pairwise_rate_diff_ci(
    classified: pd.DataFrame,
    surface: str,
    ci: float = 0.95,
    n_boot: int = 5000,
    seed: int = 0,
) -> pd.DataFrame:
    """Bootstrap CI on the difference in local-min rate between every pair
    of optimizers for a given surface. A CI that excludes 0 indicates a
    significant difference in reliability (not just speed).
    """
    sub = classified[classified["surface"] == surface]
    optimizers = sorted(sub["optimizer"].unique())
    rng = np.random.default_rng(seed)

    rows = []
    for a, b in combinations(optimizers, 2):
        is_min_a = (sub.loc[sub["optimizer"] == a, "classification"] == CLASS_MIN).to_numpy()
        is_min_b = (sub.loc[sub["optimizer"] == b, "classification"] == CLASS_MIN).to_numpy()
        na, nb = len(is_min_a), len(is_min_b)
        diffs = np.empty(n_boot)
        for i in range(n_boot):
            sa = is_min_a[rng.integers(0, na, size=na)].mean() if na else np.nan
            sb = is_min_b[rng.integers(0, nb, size=nb)].mean() if nb else np.nan
            diffs[i] = sa - sb
        point = is_min_a.mean() - is_min_b.mean()
        alpha = (1.0 - ci) / 2.0
        lo, hi = np.quantile(diffs, [alpha, 1.0 - alpha])
        rows.append(
            {
                "surface": surface,
                "optimizer_a": a,
                "optimizer_b": b,
                "rate_diff": float(point),
                "ci_low": float(lo),
                "ci_high": float(hi),
                "significant": bool(lo > 0 or hi < 0),
            }
        )
    return pd.DataFrame(rows)
