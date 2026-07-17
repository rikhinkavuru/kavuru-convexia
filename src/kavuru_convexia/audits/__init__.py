"""Audit orchestration: run the modules and assemble a VerdictReliabilityReport.

The three ground-truth-free audits (reproducibility, robustness, conflict) form
the production-usable trust gate and drive the aggregate reliability score. The
calibration audit is attached when labeled assets are supplied but is explicitly
marked offline (it needs outcomes and cannot run in production).
"""
from __future__ import annotations

from datetime import datetime
from typing import Callable, Optional, Sequence

import numpy as np

from .. import config
from ..assets import Asset
from ..evaluator import AssetEvaluator
from ..logutil import get_logger
from ._common import clip01
from .calibration import audit_calibration
from .conflict import AckJudge, audit_conflict
from .reproducibility import audit_reproducibility
from .report_types import (
    AssetReliabilityEntry,
    CheckResult,
    VerdictReliabilityReport,
    worst_status,
)
from .robustness import audit_robustness

logger = get_logger(__name__)

__all__ = [
    "audit_agent",
    "audit_reproducibility",
    "audit_robustness",
    "audit_conflict",
    "audit_calibration",
    "CheckResult",
    "VerdictReliabilityReport",
    "AssetReliabilityEntry",
]

_RECOMMENDATION = {
    "fail": "Do not act on this verdict without human review.",
    "warn": "Corroborate before acting; reliability is marginal.",
    "pass": "Reliability checks passed; act with normal oversight.",
}


def _repro_subscore(d: dict) -> float:
    s_std = 1.0 - clip01(d["pos_std"] / config.POS_STD_FAIL)
    s_flip = 1.0 - clip01(d["flip_rate"] / config.FLIP_RATE_FAIL)
    s_rat = clip01(d["rationale_jaccard"])
    return float(np.mean([s_std, s_flip, s_rat]))


def _robust_subscore(d: dict) -> float:
    n_perts = max(1, len(d["per_perturbation"]))
    s_drift = 1.0 - clip01(d["mean_drift"] / config.POS_DRIFT_FAIL)
    s_rec = 1.0 - clip01(d["n_rec_changes"] / n_perts)
    return float(np.mean([s_drift, s_rec]))


def _conflict_subscore(d: dict) -> float:
    s_ack = 1.0 if d["acknowledges_conflict"] else 0.0
    s_anchor = 1.0 - clip01(d["anchoring_swing"] / config.ANCHORING_POS_SWING_FAIL)
    s_consist = 1.0 - clip01(d["consistency_flip_rate"] / config.FLIP_RATE_FAIL)
    return float(np.mean([s_ack, s_anchor, s_consist]))


def _build_entries(
    assets: Sequence[Asset], checks: dict[str, CheckResult]
) -> list[AssetReliabilityEntry]:
    repro = checks["reproducibility"].detail["per_asset"]
    robust = checks["robustness"].detail["per_asset"]
    conflict = checks["conflict"].detail["per_asset"] if "conflict" in checks else {}

    entries: list[AssetReliabilityEntry] = []
    for a in assets:
        components: list[tuple[float, float, str]] = []  # (weight, subscore, status)
        metrics: dict[str, float] = {}
        flags: list[str] = []

        rd = repro.get(a.id)
        if rd:
            components.append((config.WEIGHT_REPRODUCIBILITY, _repro_subscore(rd), rd["status"]))
            metrics |= {"pos_std": rd["pos_std"], "flip_rate": rd["flip_rate"],
                        "rationale_jaccard": rd["rationale_jaccard"]}
            if rd["flip_rate"] > 0:
                flags.append(f"recommendation flipped in {round(rd['flip_rate'] * checks['reproducibility'].detail['n'])}"
                             f"/{checks['reproducibility'].detail['n']} runs")
            if rd["pos_std"] >= config.POS_STD_WARN:
                flags.append(f"PoS dispersion across runs (std {rd['pos_std']:.3f})")
            if rd["rationale_jaccard"] < config.RATIONALE_JACCARD_WARN:
                flags.append(f"unstable rationale (cited-evidence Jaccard {rd['rationale_jaccard']:.2f})")

        bd = robust.get(a.id)
        if bd:
            components.append((config.WEIGHT_ROBUSTNESS, _robust_subscore(bd), bd["status"]))
            metrics |= {"robustness_max_drift": bd["max_drift"],
                        "robustness_mean_drift": bd["mean_drift"]}
            if bd["max_drift"] >= config.POS_DRIFT_WARN:
                flags.append(f"+{bd['max_drift']:.2f} PoS drift under {bd['worst_perturbation']}")
            if bd["n_rec_changes"]:
                flags.append(f"recommendation changed under {bd['n_rec_changes']} semantics-preserving edit(s)")

        cd = conflict.get(a.id)
        if cd:
            components.append((config.WEIGHT_CONFLICT, _conflict_subscore(cd), cd["status"]))
            metrics |= {"anchoring_swing": cd["anchoring_swing"],
                        "acknowledges_conflict": float(cd["acknowledges_conflict"])}
            if not cd["acknowledges_conflict"]:
                flags.append("did not acknowledge the planted evidence conflict")
            if cd["anchoring_swing"] >= config.ANCHORING_POS_SWING_WARN:
                flags.append(f"positional anchoring: {cd['anchoring_swing']:.2f} PoS swing on reorder")

        if not components:
            continue
        wsum = sum(w for w, _, _ in components)
        score = sum(w * s for w, s, _ in components) / wsum
        status = worst_status([st for _, _, st in components])
        entries.append(AssetReliabilityEntry(
            asset_id=a.id,
            name=a.name,
            kind=a.kind,
            reliability_score=round(score, 4),
            status=status,
            recommendation=_RECOMMENDATION[status],
            flags=flags,
            metrics={k: round(v, 4) for k, v in metrics.items()},
        ))
    return entries


def audit_agent(
    evaluator: AssetEvaluator,
    assets: Sequence[Asset],
    *,
    calibration_assets: Optional[Sequence[Asset]] = None,
    n: int = config.N_REPETITIONS,
    ack_judge: Optional[AckJudge] = None,
    paraphraser: Optional[Callable[[str], str]] = None,
    temperature: Optional[float] = None,
    created: Optional[str] = None,
) -> VerdictReliabilityReport:
    """Run all audits for an evaluator and assemble the reliability report.

    ``assets`` drives reproducibility / robustness / conflict (conflict uses the
    subset with a planted conflict). ``calibration_assets`` (labeled) adds the
    offline calibration check. Returns a per-verdict + aggregate report.
    """
    logger.info("auditing %s over %d assets", evaluator.name, len(assets))
    checks: dict[str, CheckResult] = {
        "reproducibility": audit_reproducibility(evaluator, assets, n=n, temperature=temperature),
        "robustness": audit_robustness(evaluator, assets, paraphraser=paraphraser, temperature=temperature),
    }
    conflicted = [a for a in assets if a.has_planted_conflict]
    if conflicted:
        checks["conflict"] = audit_conflict(
            evaluator, conflicted, n_consistency=n, ack_judge=ack_judge, temperature=temperature
        )
    if calibration_assets:
        checks["calibration"] = audit_calibration(evaluator, calibration_assets, temperature=temperature)

    entries = _build_entries(assets, checks)

    # Aggregate reliability score: weighted over the production-usable checks only.
    prod = {k: v for k, v in checks.items() if v.production_usable}
    weights = {"reproducibility": config.WEIGHT_REPRODUCIBILITY,
               "robustness": config.WEIGHT_ROBUSTNESS,
               "conflict": config.WEIGHT_CONFLICT}
    wsum = sum(weights[k] for k in prod)
    reliability_score = float(sum(weights[k] * prod[k].score for k in prod) / wsum) if wsum else 0.0
    overall_status = worst_status([v.status for v in prod.values()])

    headline_flags: list[str] = []
    for name, chk in checks.items():
        for flag in chk.flags[:3]:
            headline_flags.append(f"{name}: {flag}")

    return VerdictReliabilityReport(
        evaluator_name=evaluator.name,
        model=getattr(evaluator, "model", ""),
        checks=checks,
        entries=entries,
        reliability_score=round(reliability_score, 4),
        overall_status=overall_status,
        headline_flags=headline_flags,
        n_assets=len(assets),
        created=created or datetime.now().isoformat(timespec="seconds"),
    )
