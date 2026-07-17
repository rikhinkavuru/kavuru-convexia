"""The robustness audit must catch verdict drift under semantics-preserving edits."""
from __future__ import annotations

from _stubs import ConstantStub, OrderSensitiveStub

from kavuru_convexia.assets import build_synthetic_assets
from kavuru_convexia.audits.robustness import (
    audit_robustness,
    neutralize_entities,
    reformat_text,
    reorder_evidence,
)

ASSETS = build_synthetic_assets()[:3]


# --------------------------------------------------------------------------
# Perturbations are genuinely semantics-preserving
# --------------------------------------------------------------------------
def test_reorder_preserves_evidence_set():
    a = ASSETS[0]
    r = reorder_evidence(a)
    assert set(r.evidence_ids) == set(a.evidence_ids)
    assert r.evidence_ids != a.evidence_ids  # order actually changed
    assert {e.text for e in r.evidence} == {e.text for e in a.evidence}


def test_neutralize_strips_the_name():
    a = build_synthetic_assets()[0]
    n = neutralize_entities(a)
    assert n.name is None
    assert set(n.evidence_ids) == set(a.evidence_ids)  # evidence identity intact


def test_reformat_keeps_evidence_ids():
    a = ASSETS[0]
    f = reformat_text(a)
    assert set(f.evidence_ids) == set(a.evidence_ids)


# --------------------------------------------------------------------------
# The audit fires correctly
# --------------------------------------------------------------------------
def test_robust_evaluator_passes():
    result = audit_robustness(ConstantStub(), ASSETS)
    assert result.status == "pass"
    assert result.metrics["mean_abs_drift"] == 0.0
    assert result.metrics["rec_change_rate"] == 0.0


def test_order_sensitive_evaluator_fails():
    result = audit_robustness(OrderSensitiveStub(), ASSETS)
    assert result.status == "fail"
    assert result.metrics["max_abs_drift"] > 0.1  # reorder/neutralize move it a lot
    assert result.flags
