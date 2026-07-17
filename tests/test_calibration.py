"""The calibration audit must reward a calibrated evaluator and flag an overconfident one."""
from __future__ import annotations

import numpy as np
from _stubs import OverconfidentStub, WellCalibratedStub

from kavuru_convexia import config
from kavuru_convexia.assets import load_historical_assets
from kavuru_convexia.audits.calibration import audit_calibration, expected_calibration_error

HIST = load_historical_assets()


# --------------------------------------------------------------------------
# ECE implementation
# --------------------------------------------------------------------------
def test_ece_is_zero_when_perfectly_calibrated():
    # Predictions equal to the observed frequency in each bin -> ECE 0.
    y_true = np.array([1.0, 1.0, 0.0, 0.0])
    y_prob = np.array([1.0, 1.0, 0.0, 0.0])
    ece, curve = expected_calibration_error(y_true, y_prob, n_bins=10)
    assert ece == 0.0
    assert sum(b["count"] for b in curve) == 4


def test_ece_matches_hand_computation():
    # All four in one bin (mean_pred 0.5), observed 0.5 -> gap 0.
    # Shift preds to 0.9 with observed 0.5 -> ECE 0.4.
    y_true = np.array([1.0, 0.0, 1.0, 0.0])
    y_prob = np.array([0.9, 0.9, 0.9, 0.9])
    ece, _ = expected_calibration_error(y_true, y_prob, n_bins=10)
    assert ece == 0.4


# --------------------------------------------------------------------------
# The audit fires correctly
# --------------------------------------------------------------------------
def test_well_calibrated_stub_scores_well():
    result = audit_calibration(WellCalibratedStub(), HIST)
    assert result.requires_labels and not result.production_usable
    assert result.status in ("pass", "warn")
    assert result.metrics["auroc"] > 0.9  # separates outcomes cleanly


def test_overconfident_stub_is_flagged():
    good = audit_calibration(WellCalibratedStub(), HIST)
    bad = audit_calibration(OverconfidentStub(), HIST)
    assert bad.status == "fail"
    assert bad.metrics["ece"] >= config.ECE_FAIL
    assert bad.metrics["ece"] > good.metrics["ece"]
    assert bad.flags


def test_bins_capped_to_sample_count():
    # 12 labeled assets -> at most 6 bins, so bins are not sparser than the data.
    result = audit_calibration(WellCalibratedStub(), HIST)
    assert result.metrics["n_bins"] <= len(HIST) // 2


def test_blinded_calibration_is_distinct_check():
    result = audit_calibration(WellCalibratedStub(), HIST, blind_identity=True)
    assert result.name == "calibration_blinded"
    assert result.metrics["blind_identity"] == 1.0
    assert not result.production_usable
