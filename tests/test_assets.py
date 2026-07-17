"""Tests for the asset + evidence schema and the bundled datasets."""
from __future__ import annotations

import pytest

from kavuru_convexia.assets import (
    DIRECTIONS,
    EVIDENCE_TYPES,
    STRENGTHS,
    Asset,
    EvidenceSnippet,
    build_synthetic_assets,
    load_historical_assets,
)


# --------------------------------------------------------------------------
# Schema validation
# --------------------------------------------------------------------------
def test_evidence_snippet_rejects_bad_vocab():
    with pytest.raises(ValueError):
        EvidenceSnippet(id="x", type="not_a_type", text="t", direction="supportive", strength="weak")
    with pytest.raises(ValueError):
        EvidenceSnippet(id="x", type="tox", text="t", direction="good", strength="weak")
    with pytest.raises(ValueError):
        EvidenceSnippet(id="x", type="tox", text="t", direction="adverse", strength="huge")


def test_asset_rejects_empty_and_duplicate_evidence():
    ev = EvidenceSnippet(id="e1", type="tox", text="t", direction="adverse", strength="weak")
    with pytest.raises(ValueError):
        Asset(id="a", description="d", evidence=[])
    with pytest.raises(ValueError):
        Asset(id="a", description="d", evidence=[ev, ev])  # duplicate id


def test_asset_round_trips_through_dict():
    for asset in load_historical_assets() + build_synthetic_assets():
        assert Asset.from_dict(asset.to_dict()).to_dict() == asset.to_dict()


# --------------------------------------------------------------------------
# Historical calibration set
# --------------------------------------------------------------------------
def test_historical_assets_are_labeled_and_balanced():
    hist = load_historical_assets()
    assert len(hist) == 20
    assert all(a.kind == "historical" for a in hist)
    assert all(a.true_outcome is not None for a in hist)
    # A balanced label set keeps calibration base rates meaningful.
    n_success = sum(a.true_outcome for a in hist)
    assert n_success == 10 and len(hist) - n_success == 10


def test_historical_assets_are_sourced_and_verified():
    for a in load_historical_assets():
        assert a.sources, f"{a.id} has no citation"
        assert all(s.url.startswith("http") for s in a.sources)
        ver = a.meta.get("verification", {})
        # Every outcome was adversarially re-verified and leakage-checked.
        assert ver.get("outcome_confirmed") is True, f"{a.id} outcome not confirmed"
        assert ver.get("leakage_check") is True, f"{a.id} leaks its outcome"


def test_historical_evidence_uses_valid_vocab():
    for a in load_historical_assets():
        for e in a.evidence:
            assert e.type in EVIDENCE_TYPES
            assert e.direction in DIRECTIONS
            assert e.strength in STRENGTHS


# --------------------------------------------------------------------------
# Synthetic controlled set
# --------------------------------------------------------------------------
def test_synthetic_assets_have_no_labels():
    syn = build_synthetic_assets()
    assert all(a.true_outcome is None for a in syn)
    assert all(a.kind == "synthetic" for a in syn)


def test_synthetic_population_split():
    syn = build_synthetic_assets()
    conflicted = [a for a in syn if a.has_planted_conflict]
    borderline = [a for a in syn if a.is_borderline]
    controls = [a for a in syn if not a.has_planted_conflict and not a.is_borderline]
    assert len(conflicted) == 4
    assert len(borderline) == 2
    assert len(controls) == 2
    # Conflict and borderline are mutually exclusive tags.
    assert not any(a.has_planted_conflict and a.is_borderline for a in syn)
    # Each conflicted asset names its conflicting evidence pair and carries at
    # least one supportive and one adverse snippet (a genuine tension).
    for a in conflicted:
        pair = a.meta.get("conflict_pair", [])
        assert len(pair) == 2 and all(pid in a.evidence_ids for pid in pair)
        dirs = {e.direction for e in a.evidence}
        assert "supportive" in dirs and "adverse" in dirs


def test_synthetic_controls_are_internally_coherent():
    controls = [
        a for a in build_synthetic_assets()
        if not a.has_planted_conflict and not a.is_borderline
    ]
    for a in controls:
        dirs = {e.direction for e in a.evidence}
        # A control is uniformly supportive or uniformly adverse — no mixed signal.
        assert dirs in ({"supportive"}, {"adverse"}), f"{a.id} is not conflict-free"
