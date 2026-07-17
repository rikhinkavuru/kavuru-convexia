"""Dataclasses for audit results and the aggregate reliability report.

A :class:`CheckResult` is the aggregate result of one audit module. An
:class:`AssetReliabilityEntry` is the per-verdict view (one asset), which is what
a reviewer acts on. A :class:`VerdictReliabilityReport` bundles both for a full
run and serializes to JSON.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

Status = Literal["pass", "warn", "fail"]

# fail dominates warn dominates pass when combining sub-results.
_STATUS_RANK = {"pass": 0, "warn": 1, "fail": 2}
_RANK_STATUS = {v: k for k, v in _STATUS_RANK.items()}


def worst_status(statuses: list[Status]) -> Status:
    """Return the most severe status in a list (pass < warn < fail)."""
    if not statuses:
        return "pass"
    return _RANK_STATUS[max(_STATUS_RANK[s] for s in statuses)]


@dataclass
class CheckResult:
    """The aggregate result of a single audit module."""

    name: str
    status: Status
    score: float  # 0..1 reliability sub-score for this dimension (1 = fully reliable)
    metrics: dict[str, float] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)  # per-asset + module-specific
    requires_labels: bool = False  # calibration needs outcome labels
    production_usable: bool = True  # can this run without ground truth, in production?

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AssetReliabilityEntry:
    """The per-verdict reliability view for one asset — the actionable unit."""

    asset_id: str
    kind: str
    reliability_score: float  # 0..1 over the production-usable signals for this asset
    status: Status
    recommendation: str  # human-readable gate verdict
    flags: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    name: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VerdictReliabilityReport:
    """Full reliability report for an evaluator over an asset set."""

    evaluator_name: str
    model: str
    checks: dict[str, CheckResult]
    entries: list[AssetReliabilityEntry]
    reliability_score: float  # weighted over production-usable checks
    overall_status: Status
    headline_flags: list[str]
    n_assets: int
    created: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluator_name": self.evaluator_name,
            "model": self.model,
            "created": self.created,
            "n_assets": self.n_assets,
            "reliability_score": self.reliability_score,
            "overall_status": self.overall_status,
            "headline_flags": self.headline_flags,
            "checks": {k: v.to_dict() for k, v in self.checks.items()},
            "entries": [e.to_dict() for e in self.entries],
        }

    def to_json(self, path: Path | str) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return p

    def summary_lines(self) -> list[str]:
        """A compact human-readable summary of the report."""
        lines = [
            f"Reliability report for {self.evaluator_name}",
            f"  overall: {self.overall_status.upper()}  "
            f"(reliability score {self.reliability_score:.2f} over {self.n_assets} assets)",
        ]
        for name, chk in self.checks.items():
            tag = "" if chk.production_usable else "  [offline: needs labels]"
            lines.append(f"  - {name:16s} {chk.status.upper():5s} score={chk.score:.2f}{tag}")
        for flag in self.headline_flags:
            lines.append(f"  ! {flag}")
        return lines
