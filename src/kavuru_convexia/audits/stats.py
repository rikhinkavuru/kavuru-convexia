"""Bootstrap confidence intervals for the audit metrics.

The whole point of this harness is that an evaluative verdict is an *estimate*
with variance, so its summary statistics deserve the same treatment: headline
numbers are reported with a percentile bootstrap CI, and the resampling unit is
chosen to match the estimand ("generalize to a new asset"):

* **Reproducibility** — a *hierarchical* two-stage bootstrap (resample assets,
  then the N runs within each drawn asset), because a per-asset std/flip-rate
  from only N=8 runs is itself noisy and that noise must propagate.
* **Robustness** — a one-stage cluster bootstrap on assets, keeping each asset's
  fixed perturbation vector intact (the perturbations are a designed factor, not
  a sample, so they are not resampled).
* **Calibration** — a case bootstrap on the labeled assets; single-class
  resamples (undefined AUROC) are discarded and the discard fraction reported.
* **Conflict** — n is tiny (the conflicted subset), so *no* CI is reported; the
  raw per-asset values are shown instead. (Handled in the conflict module.)

Percentile endpoints are clipped to a metric's valid range so a bounded statistic
never reports an impossible interval. B defaults to 2000, seeded for reproducibility.
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import numpy as np

from .. import config

CI = tuple[float, float, float]  # (point, lo, hi)
Clip = Optional[tuple[Optional[float], Optional[float]]]

DEFAULT_N_BOOT = 2000


def _summarize(point: float, boots: list[float], level: float, clip: Clip) -> CI:
    if not boots:
        return (point, float("nan"), float("nan"))
    lo = float(np.percentile(boots, (1 - level) / 2 * 100))
    hi = float(np.percentile(boots, (1 + level) / 2 * 100))
    if clip is not None:
        clo, chi = clip
        if clo is not None:
            lo = max(clo, lo)
        if chi is not None:
            hi = min(chi, hi)
    return (float(point), lo, hi)


def bootstrap_ci(
    n: int,
    stat_fn: Callable[[np.ndarray], Optional[float]],
    *,
    n_boot: int = DEFAULT_N_BOOT,
    level: float = 0.95,
    seed: int = config.SEED,
    clip: Clip = None,
) -> CI:
    """Percentile bootstrap CI for a statistic over ``n`` resampled rows."""
    pv = stat_fn(np.arange(n))
    point = float(pv) if pv is not None and np.isfinite(pv) else float("nan")
    if n < 2:
        return (point, point, point)
    rng = np.random.default_rng(seed)
    boots: list[float] = []
    for _ in range(n_boot):
        v = stat_fn(rng.integers(0, n, size=n))
        if v is not None and np.isfinite(v):
            boots.append(float(v))
    return _summarize(point, boots, level, clip)


def mean_ci(values: Sequence[float], *, clip: Clip = None, **kw) -> CI:
    """CI on the mean of per-unit values (one-stage resample of the units)."""
    arr = np.asarray(values, dtype=float)
    return bootstrap_ci(len(arr), lambda idx: float(arr[idx].mean()) if len(idx) else float("nan"),
                        clip=clip, **kw)


def hierarchical_mean_ci(
    items: Sequence[Any],
    metric_fn: Callable[[Any, np.ndarray], float],
    n_runs_fn: Callable[[Any], int],
    *,
    n_boot: int = DEFAULT_N_BOOT,
    level: float = 0.95,
    seed: int = config.SEED,
    clip: Clip = None,
) -> CI:
    """Two-stage bootstrap: resample items (outer), then runs within each (inner).

    ``metric_fn(item, run_idx)`` recomputes an item's per-item metric on the
    resampled run indices; the replicate statistic is the mean across drawn items.
    """
    n = len(items)
    point = float(np.mean([metric_fn(it, np.arange(n_runs_fn(it))) for it in items]))
    if n < 2:
        return (point, point, point)
    rng = np.random.default_rng(seed)
    boots: list[float] = []
    for _ in range(n_boot):
        vals = []
        for ai in rng.integers(0, n, size=n):
            it = items[ai]
            m = n_runs_fn(it)
            vals.append(metric_fn(it, rng.integers(0, m, size=m)))
        boots.append(float(np.mean(vals)))
    return _summarize(point, boots, level, clip)


def metric_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_boot: int = DEFAULT_N_BOOT,
    level: float = 0.95,
    seed: int = config.SEED,
    clip: Clip = None,
) -> tuple[CI, float]:
    """CI on a paired (y_true, y_prob) metric, plus the fraction of dropped resamples.

    Resamples the labeled assets; a resample whose metric is undefined (e.g. AUROC
    on a single-class draw) is dropped, and the drop fraction is returned so the
    caller can flag an untrustworthy interval.
    """
    yt, yp = np.asarray(y_true, dtype=float), np.asarray(y_prob, dtype=float)
    n = len(yt)
    try:
        point = float(metric_fn(yt, yp))
    except Exception:  # noqa: BLE001
        point = float("nan")
    if n < 2:
        return (point, point, point), 0.0
    rng = np.random.default_rng(seed)
    boots: list[float] = []
    dropped = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            v = float(metric_fn(yt[idx], yp[idx]))
        except Exception:  # noqa: BLE001 — single-class / degenerate resample
            dropped += 1
            continue
        if np.isfinite(v):
            boots.append(v)
        else:
            dropped += 1
    return _summarize(point, boots, level, clip), dropped / n_boot


def fmt_ci(ci: Optional[Sequence[float]], digits: int = 3) -> str:
    """Render a CI ([lo, hi] or (point, lo, hi)) as 'lo-hi' or 'point [lo, hi]'."""
    if ci is None:
        return "n/a"
    vals = list(ci)
    if len(vals) == 2:
        lo, hi = vals
        return f"[{lo:.{digits}f}, {hi:.{digits}f}]" if np.isfinite(lo) and np.isfinite(hi) else "n/a"
    p, lo, hi = vals
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return f"{p:.{digits}f}"
    return f"{p:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"
