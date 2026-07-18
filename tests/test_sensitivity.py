"""The evidence-sensitivity audit must localize a single point of failure."""
from __future__ import annotations

from _stubs import AdverseGatedStub, ConstantStub, DecoupledSpofStub

from kavuru_convexia.assets import build_synthetic_assets

# The efficacy-vs-tox asset carries a strong-adverse tox snippet.
CONFLICT = [a for a in build_synthetic_assets() if a.id == "SYN-CONFLICT-EFFICACY-TOX"]


def test_spof_localized_to_the_adverse_snippet():
    from kavuru_convexia.audits.sensitivity import audit_evidence_sensitivity

    result = audit_evidence_sensitivity(AdverseGatedStub(), CONFLICT, k=2)
    assert result.metrics["spof_rate"] == 1.0
    per = result.detail["per_asset"]["SYN-CONFLICT-EFFICACY-TOX"]
    assert per["spof"] is True
    assert per["dominant_type"] == "tox"  # removing the tox snippet flips the call
    assert result.flags


def test_no_spof_for_an_evidence_insensitive_evaluator():
    from kavuru_convexia.audits.sensitivity import audit_evidence_sensitivity

    result = audit_evidence_sensitivity(ConstantStub(), CONFLICT, k=2)
    assert result.metrics["spof_rate"] == 0.0
    assert result.status == "pass"


def test_spof_detected_when_flipping_snippet_is_not_the_dominant_one():
    from kavuru_convexia.audits.sensitivity import audit_evidence_sensitivity

    result = audit_evidence_sensitivity(DecoupledSpofStub(), CONFLICT, k=2)
    per = result.detail["per_asset"]["SYN-CONFLICT-EFFICACY-TOX"]
    assert per["spof"] is True
    # The flip comes from the preclinical snippet, not the max-influence (tox) one.
    assert per["spof_type"] == "preclinical"
    assert per["dominant_type"] == "tox"
