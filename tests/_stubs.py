"""Deterministic stub evaluators for exercising the audit detectors.

Each stub is a hand-crafted failure (or success) mode. They let the tests assert
that an audit fires exactly when it should — e.g. an unstable stub trips the
reproducibility flag, a primacy-following stub trips the anchoring flag. Stubs may
peek at ``asset.true_outcome`` / ``asset.meta``; that is fine for test doubles.
"""
from __future__ import annotations

import hashlib
from typing import Optional

from kavuru_convexia.assets import Asset
from kavuru_convexia.evaluator import AssetEvaluator, Verdict


def _rec_for(pos: float) -> str:
    return "advance" if pos >= 0.6 else "pass" if pos <= 0.35 else "investigate"


class ConstantStub(AssetEvaluator):
    """Same verdict every time — perfectly reproducible and robust."""

    name = "ConstantStub"

    def __init__(self, pos: float = 0.7, rationale: str = "stable reasoning") -> None:
        self.pos = pos
        self.rationale = rationale

    def evaluate(self, asset: Asset, *, temperature: Optional[float] = None, cache_tag: str = "") -> Verdict:
        return Verdict(asset.id, self.pos, _rec_for(self.pos), self.rationale,
                       asset.evidence_ids[:1], model="stub")


class UnstableStub(AssetEvaluator):
    """PoS and recommendation swing with the cache tag — maximally non-reproducible."""

    name = "UnstableStub"

    def evaluate(self, asset: Asset, *, temperature: Optional[float] = None, cache_tag: str = "") -> Verdict:
        h = int(hashlib.sha256(cache_tag.encode()).hexdigest(), 16)
        pos = (h % 1000) / 1000.0
        rec = ["advance", "pass", "investigate"][h % 3]
        cited = asset.evidence_ids[h % len(asset.evidence_ids):][:1] or asset.evidence_ids[:1]
        return Verdict(asset.id, pos, rec, "shifting reasoning", cited, model="stub")


class OrderSensitiveStub(AssetEvaluator):
    """Scores off the FIRST snippet's direction and drops if the name is stripped.

    Drifts sharply under evidence reordering and under entity neutralization —
    exactly the non-robustness the robustness audit should catch.
    """

    name = "OrderSensitiveStub"

    def evaluate(self, asset: Asset, *, temperature: Optional[float] = None, cache_tag: str = "") -> Verdict:
        base = {"supportive": 0.9, "adverse": 0.1, "mixed": 0.5, "neutral": 0.5}[asset.evidence[0].direction]
        if asset.name is None:  # brand-recognition dependence
            base = max(0.0, base - 0.3)
        return Verdict(asset.id, base, _rec_for(base), "first-impression reasoning",
                       asset.evidence_ids[:1], model="stub")


class GoodConflictStub(AssetEvaluator):
    """Acknowledges the conflict, order-invariant, consistent — passes conflict audit."""

    name = "GoodConflictStub"

    def evaluate(self, asset: Asset, *, temperature: Optional[float] = None, cache_tag: str = "") -> Verdict:
        rationale = ("There is a clear tension: strong efficacy must be weighed against a "
                     "serious adverse signal, so I flag the trade-off rather than resolve it.")
        return Verdict(asset.id, 0.4, "investigate", rationale, asset.evidence_ids[:2], model="stub")


class PrimacyStub(AssetEvaluator):
    """Follows the position of the conflicting pair and never acknowledges the conflict.

    Trips both the anchoring flag (PoS swings when the pair is swapped) and the
    acknowledgment flag (bland rationale with no weighing language).
    """

    name = "PrimacyStub"

    def evaluate(self, asset: Asset, *, temperature: Optional[float] = None, cache_tag: str = "") -> Verdict:
        pair = asset.meta.get("conflict_pair")
        ids = asset.evidence_ids
        # High if the (supportive) first pair member precedes the (adverse) second.
        pos = 0.85 if pair and ids.index(pair[0]) < ids.index(pair[1]) else 0.15
        return Verdict(asset.id, pos, _rec_for(pos), "it reads fine at first glance",
                       asset.evidence_ids[:1], model="stub")


class AdverseGatedStub(AssetEvaluator):
    """PoS collapses to 'pass' iff a strong-adverse snippet is present, else 'advance'.

    Removing that one snippet flips the recommendation — a single point of failure
    the evidence-sensitivity audit should localize to the adverse snippet.
    """

    name = "AdverseGatedStub"

    def evaluate(self, asset: Asset, *, temperature: Optional[float] = None, cache_tag: str = "") -> Verdict:
        has_adv = any(e.direction == "adverse" and e.strength == "strong" for e in asset.evidence)
        pos = 0.2 if has_adv else 0.85
        return Verdict(asset.id, pos, _rec_for(pos), "evidence-gated", asset.evidence_ids[:1], model="stub")


class WellCalibratedStub(AssetEvaluator):
    """Confident and correct on the historical labels — low ECE, high AUROC."""

    name = "WellCalibratedStub"

    def evaluate(self, asset: Asset, *, temperature: Optional[float] = None, cache_tag: str = "") -> Verdict:
        pos = 0.95 if asset.true_outcome else 0.05
        return Verdict(asset.id, pos, _rec_for(pos), "r", asset.evidence_ids[:1], model="stub")


class OverconfidentStub(AssetEvaluator):
    """Always maximally confident regardless of outcome — miscalibrated, no discrimination."""

    name = "OverconfidentStub"

    def evaluate(self, asset: Asset, *, temperature: Optional[float] = None, cache_tag: str = "") -> Verdict:
        return Verdict(asset.id, 0.95, "advance", "r", asset.evidence_ids[:1], model="stub")
