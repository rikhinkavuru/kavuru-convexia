"""Tests for the Verdict type and the reference / external evaluators."""
from __future__ import annotations

import pytest

from kavuru_convexia.assets import build_synthetic_assets
from kavuru_convexia.evaluator import (
    ExternalAdapter,
    ReferenceAgent,
    Verdict,
    verdicts_to_json,
)
from kavuru_convexia.llm import LLMClient


ASSET = build_synthetic_assets()[0]  # efficacy-vs-tox conflict


class _FixedClient:
    """A stand-in LLMClient that always returns a canned response string."""

    def __init__(self, response: str) -> None:
        self.response = response

    def complete(self, system: str, user: str, *, temperature: float, cache_tag: str = "") -> str:
        return self.response


# --------------------------------------------------------------------------
# Verdict
# --------------------------------------------------------------------------
def test_verdict_from_dict_normalizes_fields():
    v = Verdict.from_dict(
        {"asset_id": "a", "pos_score": "0.4", "recommendation": "ADVANCE",
         "cited_evidence_ids": "e1; e2 | e3"}
    )
    assert v.recommendation == "advance"
    assert v.cited_evidence_ids == ["e1", "e2", "e3"]


def test_verdict_from_dict_defaults_bad_recommendation():
    v = Verdict.from_dict({"asset_id": "a", "pos_score": 0.9, "recommendation": "buy"})
    assert v.recommendation == "investigate"


# --------------------------------------------------------------------------
# ReferenceAgent
# --------------------------------------------------------------------------
def test_reference_agent_offline_is_deterministic_and_valid():
    agent = ReferenceAgent(client=LLMClient(offline=True))
    v1 = agent.evaluate(ASSET, cache_tag="x")
    v2 = agent.evaluate(ASSET, cache_tag="x")
    assert v1.pos_score == v2.pos_score  # same tag -> identical
    assert 0.0 <= v1.pos_score <= 1.0
    assert v1.recommendation in ("advance", "pass", "investigate")
    # Citations must reference real evidence ids only.
    assert set(v1.cited_evidence_ids) <= set(ASSET.evidence_ids)


def test_reference_agent_distinct_tags_vary():
    agent = ReferenceAgent(client=LLMClient(offline=True))
    scores = {agent.evaluate(ASSET, cache_tag=f"t{i}").pos_score for i in range(6)}
    assert len(scores) > 1  # offline stub exhibits tag-driven dispersion


def test_reference_agent_clamps_and_filters_hallucinated_citations():
    agent = ReferenceAgent(
        client=_FixedClient(
            '{"pos_score": 1.7, "recommendation": "advance", "rationale": "r",'
            ' "cited_evidence_ids": ["NONEXISTENT", "' + ASSET.evidence_ids[0] + '"]}'
        )
    )
    v = agent.evaluate(ASSET)
    assert v.pos_score == 1.0  # clamped into [0, 1]
    assert v.cited_evidence_ids == [ASSET.evidence_ids[0]]  # hallucinated id dropped


def test_reference_agent_degrades_on_malformed_json():
    agent = ReferenceAgent(client=_FixedClient("the asset looks risky, I'd pass"))
    v = agent.evaluate(ASSET)
    assert v.parse_error is not None
    assert v.recommendation == "investigate" and v.pos_score == 0.5


def test_extract_json_takes_first_object_not_trailing_brace():
    # A valid verdict followed by prose containing a stray brace must still parse.
    eid = ASSET.evidence_ids[0]
    agent = ReferenceAgent(client=_FixedClient(
        f'{{"pos_score":0.8,"recommendation":"advance","rationale":"r","cited_evidence_ids":["{eid}"]}}'
        "\n\nNote: see step {2} for details."
    ))
    v = agent.evaluate(ASSET)
    assert v.parse_error is None
    assert v.pos_score == 0.8 and v.recommendation == "advance"


def test_reference_agent_rejects_nan_pos():
    agent = ReferenceAgent(client=_FixedClient('{"pos_score": NaN, "recommendation": "advance"}'))
    v = agent.evaluate(ASSET)
    assert v.parse_error is not None  # non-finite is a parse failure, not a 0.0 verdict


def test_from_dict_clamps_and_rejects_nonfinite():
    assert Verdict.from_dict({"asset_id": "a", "pos_score": 1.7}).pos_score == 1.0
    assert Verdict.from_dict({"asset_id": "a", "pos_score": -0.3}).pos_score == 0.0
    with pytest.raises(ValueError):
        Verdict.from_dict({"asset_id": "a", "pos_score": float("nan")})


# --------------------------------------------------------------------------
# ExternalAdapter
# --------------------------------------------------------------------------
def test_external_adapter_serves_runs_in_order_then_cycles():
    records = [
        {"asset_id": ASSET.id, "pos_score": 0.2, "recommendation": "pass"},
        {"asset_id": ASSET.id, "pos_score": 0.8, "recommendation": "advance"},
    ]
    adapter = ExternalAdapter.from_records(records)
    assert adapter.evaluate(ASSET).pos_score == 0.2
    assert adapter.evaluate(ASSET).pos_score == 0.8
    assert adapter.evaluate(ASSET).pos_score == 0.2  # cycles


def test_external_adapter_raises_on_unknown_asset():
    adapter = ExternalAdapter.from_records([{"asset_id": "other", "pos_score": 0.5}])
    with pytest.raises(KeyError):
        adapter.evaluate(ASSET)


def test_external_adapter_json_round_trip(tmp_path):
    v = Verdict(asset_id=ASSET.id, pos_score=0.42, recommendation="investigate", rationale="r")
    path = tmp_path / "verdicts.json"
    verdicts_to_json([v], path)
    adapter = ExternalAdapter.from_json(path)
    assert adapter.evaluate(ASSET).pos_score == 0.42
