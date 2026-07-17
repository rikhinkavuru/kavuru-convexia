"""Shared helpers for the audit modules: concurrent evaluation and statistics."""
from __future__ import annotations

import itertools
from concurrent.futures import ThreadPoolExecutor
from typing import Hashable, Iterable, Optional, Sequence

import numpy as np

from .. import config
from ..assets import Asset
from ..evaluator import AssetEvaluator, Verdict
from ..logutil import get_logger

logger = get_logger(__name__)

Job = tuple[Hashable, Asset, str]  # (result key, asset, cache_tag)


def evaluate_batch(
    evaluator: AssetEvaluator,
    jobs: Sequence[Job],
    *,
    temperature: Optional[float] = None,
    max_workers: int = config.MAX_CONCURRENCY,
) -> dict[Hashable, Verdict]:
    """Evaluate many (asset, cache_tag) jobs concurrently, keyed by their result key.

    LLM calls are I/O bound, so a thread pool gives a large speedup; the on-disk
    cache means a re-run replays instantly. Distinct cache tags guarantee the
    reproducibility audit draws independent samples rather than one cached reply.
    """
    results: dict[Hashable, Verdict] = {}

    def _run(job: Job) -> tuple[Hashable, Verdict]:
        key, asset, tag = job
        return key, evaluator.evaluate(asset, temperature=temperature, cache_tag=tag)

    workers = max(1, min(max_workers, len(jobs)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for key, verdict in pool.map(_run, jobs):
            results[key] = verdict
    return results


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def jaccard(a: set, b: set) -> float:
    """Jaccard overlap; two empty sets are treated as identical (1.0)."""
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def mean_pairwise_jaccard(sets: Sequence[set]) -> float:
    """Mean Jaccard over all unordered pairs; 1.0 for fewer than two sets."""
    if len(sets) < 2:
        return 1.0
    pairs = [jaccard(a, b) for a, b in itertools.combinations(sets, 2)]
    return float(np.mean(pairs))


def modal(values: Iterable) -> object:
    """Most frequent value; ties resolved by first appearance (stable)."""
    seq = list(values)
    counts: dict[object, int] = {}
    for v in seq:
        counts[v] = counts.get(v, 0) + 1
    return max(counts, key=lambda k: (counts[k], -seq.index(k)))


def flip_rate(recommendations: Sequence[str]) -> float:
    """Fraction of recommendations disagreeing with the modal recommendation."""
    if not recommendations:
        return 0.0
    m = modal(recommendations)
    return sum(1 for r in recommendations if r != m) / len(recommendations)


def iqr(values: Sequence[float]) -> float:
    """Interquartile range (75th - 25th percentile)."""
    if len(values) < 2:
        return 0.0
    return float(np.subtract(*np.percentile(values, [75, 25])))


def clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))
