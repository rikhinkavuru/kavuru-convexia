"""The conflict audit must catch anchoring and missing acknowledgment."""
from __future__ import annotations

import pytest
from _stubs import GoodConflictStub, PrimacyStub

from kavuru_convexia import config
from kavuru_convexia.assets import build_synthetic_assets
from kavuru_convexia.audits.conflict import audit_conflict, heuristic_ack_judge

CONFLICTED = [a for a in build_synthetic_assets() if a.has_planted_conflict]


def test_heuristic_ack_judge():
    assert heuristic_ack_judge("Strong efficacy is weighed against a serious tox signal.", "")
    assert not heuristic_ack_judge("The asset looks promising and should advance.", "")


def test_good_conflict_evaluator_passes():
    result = audit_conflict(GoodConflictStub(), CONFLICTED, n_consistency=4)
    assert result.status == "pass"
    assert result.metrics["acknowledgment_rate"] == 1.0
    assert result.metrics["max_anchoring_swing"] == pytest.approx(0.0, abs=1e-9)


def test_primacy_following_evaluator_fails_anchoring_and_ack():
    result = audit_conflict(PrimacyStub(), CONFLICTED, n_consistency=4)
    assert result.status == "fail"
    # Swapping the conflicting pair flips this stub's PoS by a large margin.
    assert result.metrics["max_anchoring_swing"] >= config.ANCHORING_POS_SWING_FAIL
    # And its bland rationale acknowledges nothing.
    assert result.metrics["acknowledgment_rate"] == 0.0
    assert result.flags


def test_requires_conflicted_assets():
    controls = [a for a in build_synthetic_assets() if not a.has_planted_conflict]
    with pytest.raises(ValueError):
        audit_conflict(GoodConflictStub(), controls)


def test_judge_panel_records_disagreement():
    # A panel that splits 2-1 on every asset must surface a disagreement signal.
    def split_panel(rationale, tension):
        return {"acknowledges": True, "votes": {"strict": False, "neutral": True, "lenient": True},
                "split": True}

    result = audit_conflict(GoodConflictStub(), CONFLICTED, n_consistency=3, ack_judge=split_panel)
    assert result.metrics["judge_disagreement_rate"] == 1.0
    assert result.metrics["ack_rate__strict"] == 0.0
    assert result.metrics["ack_rate__lenient"] == 1.0
