"""Asset + evidence dataset layer.

An :class:`Asset` bundles a drug-program description with a list of typed
:class:`EvidenceSnippet` objects and an optional ground-truth outcome. Two
populations live behind one schema:

* **Historical, known-outcome assets** (``kind="historical"``) — a small,
  hand-curated, source-cited set of real approved vs. discontinued drugs. Their
  ``true_outcome`` labels feed the (offline) calibration audit. The outcomes are
  hard facts confirmed against public sources and adversarially re-verified; the
  evidence snippets are curated editorial reconstructions of the pre-decision
  risk/benefit picture (never verbatim trial data, never leaking the outcome).
* **Synthetic controlled assets** (``kind="synthetic"``) — programmatically
  constructed to plant specific patterns, above all *conflicting evidence*
  (strong efficacy paired with a severe tox signal, positive science paired with
  blocking IP, and so on). They carry no ``true_outcome`` and drive the
  reproducibility / robustness / conflict audits.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from importlib import resources
from typing import Any, Literal, Optional

from .logutil import get_logger

logger = get_logger(__name__)

# Typed vocabularies. Kept as Literals (not Enums) so assets round-trip to plain
# JSON without custom encoders, which matters for the ExternalAdapter.
EvidenceType = Literal[
    "binding", "tox", "adme_pk", "immunogenicity", "preclinical", "ip_market"
]
Direction = Literal["supportive", "adverse", "mixed", "neutral"]
Strength = Literal["weak", "moderate", "strong"]

EVIDENCE_TYPES: tuple[EvidenceType, ...] = (
    "binding", "tox", "adme_pk", "immunogenicity", "preclinical", "ip_market",
)
DIRECTIONS: tuple[Direction, ...] = ("supportive", "adverse", "mixed", "neutral")
STRENGTHS: tuple[Strength, ...] = ("weak", "moderate", "strong")


@dataclass(frozen=True)
class Source:
    """A citation backing an asset's ground-truth outcome."""

    title: str
    url: str
    note: str = ""


@dataclass(frozen=True)
class EvidenceSnippet:
    """One typed piece of evidence about an asset, with direction and strength."""

    id: str
    type: EvidenceType
    text: str
    direction: Direction
    strength: Strength

    def __post_init__(self) -> None:
        if self.type not in EVIDENCE_TYPES:
            raise ValueError(f"evidence {self.id}: bad type {self.type!r}")
        if self.direction not in DIRECTIONS:
            raise ValueError(f"evidence {self.id}: bad direction {self.direction!r}")
        if self.strength not in STRENGTHS:
            raise ValueError(f"evidence {self.id}: bad strength {self.strength!r}")

    @property
    def is_adverse(self) -> bool:
        return self.direction == "adverse"


@dataclass
class Asset:
    """A drug asset: a description, typed evidence, and an optional outcome label."""

    id: str
    description: str
    evidence: list[EvidenceSnippet]
    true_outcome: Optional[bool] = None  # None for production/synthetic assets
    kind: Literal["historical", "synthetic"] = "synthetic"
    name: Optional[str] = None
    sources: list[Source] = field(default_factory=list)
    outcome_summary: str = ""
    outcome_confidence: str = ""
    caveats: str = ""
    # Free-form annotations: synthetic conflict metadata, historical verification.
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.evidence:
            raise ValueError(f"asset {self.id}: evidence must be non-empty")
        ids = [e.id for e in self.evidence]
        if len(ids) != len(set(ids)):
            raise ValueError(f"asset {self.id}: duplicate evidence ids {ids}")

    @property
    def evidence_ids(self) -> list[str]:
        return [e.id for e in self.evidence]

    @property
    def has_planted_conflict(self) -> bool:
        """True for synthetic assets deliberately built with contradictory evidence."""
        return bool(self.meta.get("conflict", False))

    @property
    def is_borderline(self) -> bool:
        """True for synthetic assets with genuinely ambiguous, balanced evidence."""
        return bool(self.meta.get("borderline", False))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Asset":
        evidence = [
            e if isinstance(e, EvidenceSnippet) else EvidenceSnippet(**e)
            for e in d["evidence"]
        ]
        sources = [
            s if isinstance(s, Source) else Source(**s) for s in d.get("sources", [])
        ]
        meta = dict(d.get("meta", {}))
        if "_verification" in d:  # fold historical provenance into meta
            meta["verification"] = d["_verification"]
        return cls(
            id=d["id"],
            description=d["description"],
            evidence=evidence,
            true_outcome=d.get("true_outcome"),
            kind=d.get("kind", "synthetic"),
            name=d.get("name"),
            sources=sources,
            outcome_summary=d.get("outcome_summary", ""),
            outcome_confidence=d.get("outcome_confidence", ""),
            caveats=d.get("caveats", ""),
            meta=meta,
        )


# ---------------------------------------------------------------------------
# Historical, known-outcome assets (calibration set)
# ---------------------------------------------------------------------------
def load_historical_assets() -> list[Asset]:
    """Load the curated, source-cited historical assets bundled with the package."""
    with resources.files("kavuru_convexia.data").joinpath(
        "historical_assets.json"
    ).open("r", encoding="utf-8") as fh:
        doc = json.load(fh)
    assets = [Asset.from_dict(a) for a in doc["assets"]]
    labeled = [a for a in assets if a.true_outcome is not None]
    logger.info(
        "loaded %d historical assets (%d labeled: %d success / %d failure)",
        len(assets),
        len(labeled),
        sum(a.true_outcome for a in labeled),
        sum(not a.true_outcome for a in labeled),
    )
    return assets


# ---------------------------------------------------------------------------
# Synthetic controlled assets (conflict-planted + controls)
# ---------------------------------------------------------------------------
def _ev(
    aid: str, n: int, etype: EvidenceType, direction: Direction, strength: Strength, text: str
) -> EvidenceSnippet:
    return EvidenceSnippet(id=f"{aid}-ev{n}", type=etype, direction=direction, strength=strength, text=text)


def build_synthetic_assets() -> list[Asset]:
    """Construct the synthetic asset population.

    Deterministic and hand-specified (not randomized) so each planted pattern is
    legible and every audit can assert against a known-correct answer. Four
    assets carry a deliberate internal conflict; two controls have none, so the
    conflict detector can be checked for false positives.
    """
    assets: list[Asset] = []

    a = "SYN-CONFLICT-EFFICACY-TOX"
    assets.append(Asset(
        id=a, kind="synthetic",
        name="Synthetic: potent efficacy vs. severe on-target-range hepatotoxicity",
        description=(
            "A small-molecule inhibitor of an oncogenic kinase under evaluation for a "
            "solid-tumor indication. The dossier pairs a strong efficacy story with a "
            "serious safety liability that emerges at the exposures required for that "
            "efficacy — the central go/no-go tension."
        ),
        evidence=[
            _ev(a, 1, "binding", "supportive", "strong", "Sub-nanomolar biochemical potency against the primary oncogenic target with a clean selectivity panel."),
            _ev(a, 2, "preclinical", "supportive", "strong", "Marked, durable tumor regression across multiple xenograft models at well-tolerated animal doses."),
            _ev(a, 3, "tox", "adverse", "strong", "Dose-dependent severe hepatotoxicity (transaminase elevations with hepatocellular necrosis) observed at the plasma exposures needed for the antitumor effect."),
            _ev(a, 4, "adme_pk", "supportive", "moderate", "Favorable oral bioavailability and a once-daily half-life with no major CYP liabilities."),
        ],
        meta={"conflict": True, "tension": "strong efficacy vs. severe on-target-range tox",
              "conflict_pair": [f"{a}-ev2", f"{a}-ev3"]},
    ))

    a = "SYN-CONFLICT-SCIENCE-IP"
    assets.append(Asset(
        id=a, kind="synthetic",
        name="Synthetic: compelling science vs. blocking intellectual property",
        description=(
            "A first-in-class mechanism with strong preclinical support whose commercial "
            "viability is threatened by a competitor's dominant patent estate — good "
            "science, questionable freedom to operate."
        ),
        evidence=[
            _ev(a, 1, "preclinical", "supportive", "strong", "Robust, reproducible in vivo efficacy in two disease-relevant models with a plausible translational path."),
            _ev(a, 2, "binding", "supportive", "moderate", "Confirmed target engagement with a mechanistically differentiated binding mode."),
            _ev(a, 3, "ip_market", "adverse", "strong", "A third party holds a broad, granted composition-of-matter patent that plausibly reads on this chemotype; freedom to operate is unclear and litigation risk is high."),
        ],
        meta={"conflict": True, "tension": "positive science vs. blocking IP / no freedom to operate",
              "conflict_pair": [f"{a}-ev1", f"{a}-ev3"]},
    ))

    a = "SYN-CONFLICT-EFFICACY-ADME"
    assets.append(Asset(
        id=a, kind="synthetic",
        name="Synthetic: compelling efficacy vs. prohibitive drug-like properties",
        description=(
            "A small-molecule candidate with compelling efficacy whose development is "
            "threatened by poor drug-like properties: very low oral bioavailability and "
            "rapid clearance make it doubtful that a therapeutic exposure can be reached "
            "in patients."
        ),
        evidence=[
            _ev(a, 1, "binding", "supportive", "strong", "Potent, selective engagement of the intended target with a clean off-target profile."),
            _ev(a, 2, "preclinical", "supportive", "strong", "Strong efficacy across disease models when adequate exposure is achieved by parenteral dosing."),
            _ev(a, 3, "adme_pk", "adverse", "strong", "Very low oral bioavailability and rapid metabolic clearance; projected human exposures fall well short of the efficacious range."),
        ],
        meta={"conflict": True, "tension": "compelling efficacy vs. prohibitive ADME / undeliverable exposure",
              "conflict_pair": [f"{a}-ev2", f"{a}-ev3"]},
    ))

    a = "SYN-CONFLICT-LATE-TOX"
    assets.append(Asset(
        id=a, kind="synthetic",
        name="Synthetic: strong profile with a serious safety signal listed last",
        description=(
            "A candidate whose dossier reads positively until a serious cardiovascular "
            "safety signal appears at the end — constructed to probe whether a verdict "
            "anchors on evidence ORDER rather than substance."
        ),
        evidence=[
            _ev(a, 1, "binding", "supportive", "strong", "Strong, selective target binding with a wide margin over off-targets."),
            _ev(a, 2, "preclinical", "supportive", "strong", "Consistent efficacy across models with a clear dose-response."),
            _ev(a, 3, "adme_pk", "supportive", "moderate", "Predictable human PK projection supporting convenient dosing."),
            _ev(a, 4, "tox", "adverse", "strong", "A serious pro-arrhythmic (QT-prolongation) cardiovascular signal at therapeutic exposures, flagged as a potential program-ending liability."),
        ],
        meta={"conflict": True, "tension": "strong overall profile vs. a serious late-listed safety signal",
              "conflict_pair": [f"{a}-ev2", f"{a}-ev4"], "anchoring_sensitive": True},
    ))

    a = "SYN-CONTROL-CLEAN-ADVANCE"
    assets.append(Asset(
        id=a, kind="synthetic",
        name="Synthetic control: uniformly supportive (no internal conflict)",
        description=(
            "A control asset whose evidence is uniformly positive, used to confirm the "
            "conflict detector does not fabricate a conflict where none exists."
        ),
        evidence=[
            _ev(a, 1, "binding", "supportive", "strong", "Excellent potency and selectivity."),
            _ev(a, 2, "preclinical", "supportive", "strong", "Strong, reproducible in vivo efficacy."),
            _ev(a, 3, "adme_pk", "supportive", "strong", "Clean PK/ADME with no metabolic liabilities."),
            _ev(a, 4, "ip_market", "supportive", "moderate", "Clear composition-of-matter position and a large addressable market."),
        ],
        meta={"conflict": False, "tension": "none (uniformly supportive control)"},
    ))

    a = "SYN-CONTROL-CLEAN-PASS"
    assets.append(Asset(
        id=a, kind="synthetic",
        name="Synthetic control: uniformly adverse (no internal conflict)",
        description=(
            "A control asset whose evidence is uniformly negative, used to confirm the "
            "conflict detector does not fire on a coherently weak (non-conflicted) profile."
        ),
        evidence=[
            _ev(a, 1, "tox", "adverse", "strong", "Severe, mechanism-based toxicity with no obvious therapeutic window."),
            _ev(a, 2, "preclinical", "adverse", "strong", "Failure to reproduce efficacy across disease-relevant models."),
            _ev(a, 3, "adme_pk", "adverse", "moderate", "Poor oral bioavailability and rapid clearance limiting achievable exposure."),
        ],
        meta={"conflict": False, "tension": "none (uniformly adverse control)"},
    ))

    # Borderline assets: genuinely ambiguous, balanced moderate/weak evidence with
    # no dominant signal. These sit near the decision boundary and are where a
    # verdict's reliability is most likely to degrade — exactly what to probe.
    a = "SYN-BORDERLINE-BALANCED"
    assets.append(Asset(
        id=a, kind="synthetic",
        name="Synthetic borderline: balanced moderate evidence, no dominant signal",
        description=(
            "An early-stage candidate whose dossier is genuinely ambiguous: modest, "
            "unreplicated efficacy set against a manageable but non-trivial safety signal "
            "and an equivocal PK picture. A rational reviewer would be close to indifferent."
        ),
        evidence=[
            _ev(a, 1, "binding", "supportive", "moderate", "Adequate but unexceptional target engagement with some off-target activity."),
            _ev(a, 2, "preclinical", "supportive", "moderate", "Modest efficacy in one animal model that was not clearly reproduced in a second."),
            _ev(a, 3, "tox", "adverse", "moderate", "A dose-related but reversible safety signal that narrows, without closing, the therapeutic window."),
            _ev(a, 4, "adme_pk", "mixed", "moderate", "Acceptable systemic exposure offset by a circulating metabolite of uncertain significance."),
        ],
        meta={"conflict": False, "borderline": True, "tension": "balanced / genuinely ambiguous"},
    ))

    a = "SYN-BORDERLINE-THIN"
    assets.append(Asset(
        id=a, kind="synthetic",
        name="Synthetic borderline: thin, mixed early evidence",
        description=(
            "A candidate with only thin, mixed early data: weak preclinical readouts, an "
            "open but crowded market, and a low-level immunogenicity flag. The evidence is "
            "too sparse to point clearly either way."
        ),
        evidence=[
            _ev(a, 1, "binding", "supportive", "moderate", "Confirmed binding to the intended target, potency roughly in the expected range."),
            _ev(a, 2, "preclinical", "mixed", "weak", "Early efficacy readouts are mixed and underpowered, with wide confidence intervals."),
            _ev(a, 3, "ip_market", "supportive", "weak", "Freedom to operate looks clear but the commercial space is crowded with fast followers."),
            _ev(a, 4, "immunogenicity", "adverse", "weak", "A low-level anti-drug-antibody signal of uncertain clinical relevance."),
        ],
        meta={"conflict": False, "borderline": True, "tension": "thin / underpowered evidence"},
    ))

    logger.info(
        "built %d synthetic assets (%d planted conflict, %d borderline, %d control)",
        len(assets),
        sum(a.has_planted_conflict for a in assets),
        sum(a.is_borderline for a in assets),
        sum(not a.has_planted_conflict and not a.is_borderline for a in assets),
    )
    return assets


def all_assets() -> list[Asset]:
    """Historical + synthetic assets, in a stable order."""
    return load_historical_assets() + build_synthetic_assets()
