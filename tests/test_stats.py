"""Tests for the bootstrap-CI helpers."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from kavuru_convexia.audits.stats import (
    bootstrap_ci,
    fmt_ci,
    hierarchical_mean_ci,
    mean_ci,
    metric_ci,
)


def test_mean_ci_brackets_the_mean_and_orders():
    values = list(np.linspace(0.0, 1.0, 40))
    point, lo, hi = mean_ci(values)
    assert lo <= point <= hi
    assert abs(point - 0.5) < 1e-9


def test_mean_ci_clips_to_range():
    point, lo, hi = mean_ci([0.0, 0.0, 0.0, 0.02], clip=(0.0, 1.0))
    assert lo >= 0.0 and hi <= 1.0


def test_single_unit_has_degenerate_ci():
    point, lo, hi = bootstrap_ci(1, lambda idx: 0.5)
    assert point == lo == hi == 0.5


def test_hierarchical_point_equals_mean_of_per_item_metric():
    # Two "assets", each with run-level values; per-item metric = mean of runs.
    items = [{"v": np.array([0.1, 0.3])}, {"v": np.array([0.5, 0.7])}]
    point, lo, hi = hierarchical_mean_ci(
        items, lambda it, ri: float(it["v"][ri].mean()), lambda it: len(it["v"]))
    assert abs(point - 0.4) < 1e-9  # mean of per-item means (0.2, 0.6)
    assert lo <= point <= hi


def test_metric_ci_reports_discard_fraction_on_single_class_resamples():
    # Mostly one class -> many resamples are single-class -> AUROC undefined -> dropped.
    y_true = np.array([1.0] * 9 + [0.0])
    y_prob = np.array([0.8] * 9 + [0.1])
    (_point, lo, hi), discard = metric_ci(y_true, y_prob, roc_auc_score, clip=(0.0, 1.0))
    assert 0.0 <= discard <= 1.0
    assert discard > 0.0  # some resamples drew only the majority class


def test_fmt_ci_handles_pair_and_triple():
    assert fmt_ci([0.1, 0.2]).startswith("[")
    assert "[" in fmt_ci((0.15, 0.1, 0.2))
    assert fmt_ci(None) == "n/a"
