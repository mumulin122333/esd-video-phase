"""Case-level bootstrap and paired tests (plan section 5, 'statistics')."""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np


def bootstrap_ci(case_scores: Mapping[str, float], n_boot: int = 10_000, alpha: float = 0.05,
                 seed: int = 42) -> tuple[float, float, float]:
    """Mean of per-case scores with a percentile CI from resampling cases."""
    v = np.asarray([case_scores[k] for k in sorted(case_scores)], dtype=float)
    if len(v) == 0:
        raise ValueError("no cases")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    boots = v[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(v.mean()), float(lo), float(hi)


def bootstrap_ci_pooled(cases: Sequence[str], statistic: Callable[[np.ndarray], float],
                        n_boot: int = 10_000, alpha: float = 0.05, seed: int = 42) -> tuple[float, float, float]:
    """CI for a statistic computed on pooled rows, resampling whole cases.

    ``statistic`` receives an index array into the original rows.
    """
    cases = np.asarray(cases)
    uniq = np.unique(cases)
    rows_by_case = {c: np.flatnonzero(cases == c) for c in uniq}
    rng = np.random.default_rng(seed)
    point = statistic(np.arange(len(cases)))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        boots[b] = statistic(np.concatenate([rows_by_case[c] for c in pick]))
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi)


def paired_bootstrap(a: Mapping[str, float], b: Mapping[str, float], n_boot: int = 10_000,
                     seed: int = 42) -> dict:
    """Mean difference a-b over shared cases, percentile CI and two-sided bootstrap p."""
    if set(a) != set(b):
        raise ValueError("paired comparison needs the same cases")
    keys = sorted(a)
    d = np.asarray([a[k] - b[k] for k in keys], dtype=float)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return {"mean_diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "p_bootstrap": float(min(1.0, p)), "n_cases": len(d)}


def wilcoxon_paired(a: Mapping[str, float], b: Mapping[str, float]) -> dict:
    from scipy.stats import wilcoxon

    if set(a) != set(b):
        raise ValueError("paired comparison needs the same cases")
    keys = sorted(a)
    x = np.asarray([a[k] for k in keys]); y = np.asarray([b[k] for k in keys])
    if np.allclose(x, y):
        return {"statistic": float("nan"), "p_value": 1.0, "n_cases": len(keys)}
    r = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
    return {"statistic": float(r.statistic), "p_value": float(r.pvalue), "n_cases": len(keys)}
