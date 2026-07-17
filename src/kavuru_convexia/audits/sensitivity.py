"""Evidence-sensitivity audit (no labels; importance, not robustness).

Leave-one-evidence-out: re-score each asset with each single evidence snippet
removed and measure how far the PoS moves. Unlike the robustness audit, removing
evidence *does* change the meaning — so this is an **importance / fragility**
analysis, not a stability test. It answers: does the verdict hinge on one piece of
evidence (a single point of failure), and which one?

Per asset we take ``k`` base evaluations to estimate the native noise, form a
Bonferroni-widened band across the S snippets, and flag a snippet as a
single-point-of-failure only when removing it both (a) crosses the go/no-go
boundary and (b) moves PoS beyond that noise band. Cost is O(S) per asset.

Honest limits: single-snippet removal cannot catch a redundant *two-key* failure
(two snippets that only matter together), so "no SPOF found" is not "no fragility".
"""
from __future__ import annotations

from dataclasses import replace
from typing import Optional, Sequence

import numpy as np
from scipy.stats import norm

from .. import config
from ..assets import Asset
from ..evaluator import AssetEvaluator
from ..logutil import get_logger
from ._common import clip01, evaluate_batch, modal
from .report_types import CheckResult, worst_status
from .stats import mean_ci

logger = get_logger(__name__)

MIN_NOISE_BAND = 0.02  # floor so a deterministic evaluator is not flagged on trivial deltas


def _without(asset: Asset, drop_id: str) -> Asset:
    return replace(
        asset,
        evidence=[e for e in asset.evidence if e.id != drop_id],
        id=f"{asset.id}__drop-{drop_id}",
    )


def audit_evidence_sensitivity(
    evaluator: AssetEvaluator,
    assets: Sequence[Asset],
    *,
    k: int = 3,
    temperature: Optional[float] = None,
) -> CheckResult:
    """Audit how load-bearing each single evidence snippet is. Requires no labels."""
    logger.info("evidence-sensitivity: %d assets, k=%d base evals + leave-one-out", len(assets), k)
    jobs = []
    for a in assets:
        for i in range(k):
            jobs.append(((a.id, "base", i), a, f"sens-base{i}"))
        for e in a.evidence:
            jobs.append(((a.id, "drop", e.id), _without(a, e.id), "sens-drop"))
    verdicts = evaluate_batch(evaluator, jobs, temperature=temperature)

    per_asset: dict[str, dict] = {}
    for a in assets:
        base = [verdicts[(a.id, "base", i)] for i in range(k)]
        pos_full = float(np.mean([v.pos_score for v in base]))
        rec_full = modal([v.recommendation for v in base])
        sigma = float(np.std([v.pos_score for v in base]))
        n_snip = len(a.evidence)
        # Bonferroni band across the S snippets (two-sided, alpha=0.05), floored.
        # float() strips the numpy scalar so the report stays JSON-serializable.
        z = float(norm.ppf(1 - 0.05 / (2 * max(1, n_snip))))
        band = float(max(MIN_NOISE_BAND, sigma * z))

        deltas: dict[str, float] = {}
        rec_change: dict[str, bool] = {}
        for e in a.evidence:
            v = verdicts[(a.id, "drop", e.id)]
            deltas[e.id] = pos_full - v.pos_score  # signed: +ve => snippet raised PoS
            rec_change[e.id] = v.recommendation != rec_full
        abs_deltas = {eid: abs(d) for eid, d in deltas.items()}
        dominant = max(abs_deltas, key=abs_deltas.get)
        m_a = abs_deltas[dominant]
        total = sum(abs_deltas.values())
        concentration = m_a / total if total > 0 else 0.0
        # Single point of failure: the dominant snippet's removal flips the call AND
        # moves PoS beyond the noise band.
        spof = rec_change[dominant] and m_a > band
        dom_ev = next(e for e in a.evidence if e.id == dominant)
        per_asset[a.id] = {
            "pos_full": pos_full,
            "recommendation": rec_full,
            "noise_band": band,
            "max_influence": m_a,
            "dominant_snippet": dominant,
            "dominant_type": dom_ev.type,
            "dominant_signed_delta": deltas[dominant],
            "influence_concentration": concentration,
            "spof": bool(spof),
            "noise_limited": bool(m_a <= band),
        }

    m_values = [d["max_influence"] for d in per_asset.values()]
    spofs = [d["spof"] for d in per_asset.values()]
    spof_rate = float(np.mean(spofs))
    metrics = {
        "spof_rate": spof_rate,
        "mean_max_influence": float(np.mean(m_values)),
        "median_max_influence": float(np.median(m_values)),
        "mean_influence_concentration": float(np.mean([d["influence_concentration"] for d in per_asset.values()])),
    }
    metrics_ci = {"spof_rate": list(mean_ci(spofs, clip=(0.0, 1.0))[1:])}

    # Informational, not a hard gate: a verdict that hinges on one snippet is worth
    # knowing even when that snippet legitimately should dominate (e.g. severe tox).
    status = "warn" if spof_rate > 0 else "pass"
    score = 1.0 - clip01(spof_rate)
    flags = [
        f"[warn] {aid}: single-point-of-failure — removing `{d['dominant_snippet']}` "
        f"({d['dominant_type']}, Δ{d['dominant_signed_delta']:+.2f}) flips the recommendation"
        for aid, d in per_asset.items() if d["spof"]
    ]
    by_type: dict[str, int] = {}
    for d in per_asset.values():
        by_type[d["dominant_type"]] = by_type.get(d["dominant_type"], 0) + 1

    return CheckResult(
        name="evidence_sensitivity",
        status=status,
        score=score,
        metrics=metrics,
        metrics_ci=metrics_ci,
        flags=flags,
        detail={
            "per_asset": per_asset,
            "dominant_by_type": by_type,
            "note": ("Importance/fragility analysis, NOT robustness — removing evidence "
                     "changes meaning. Single-snippet removal cannot detect redundant "
                     "two-key failures, so 'no SPOF' is not 'no fragility'."),
        },
        requires_labels=False,
        production_usable=True,
    )
