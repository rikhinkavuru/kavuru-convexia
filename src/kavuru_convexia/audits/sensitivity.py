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
        if len(a.evidence) >= 2:  # a single-snippet asset cannot be left-one-out
            for e in a.evidence:
                jobs.append(((a.id, "drop", e.id), _without(a, e.id), "sens-drop"))
    verdicts = evaluate_batch(evaluator, jobs, temperature=temperature)

    per_asset: dict[str, dict] = {}
    for a in assets:
        base = [verdicts[(a.id, "base", i)] for i in range(k)]
        pos_full = float(np.mean([v.pos_score for v in base]))
        rec_full = modal([v.recommendation for v in base])
        n_snip = len(a.evidence)

        if n_snip < 2:
            # The whole verdict trivially rests on the sole snippet: 100% concentration.
            only = a.evidence[0]
            per_asset[a.id] = {
                "pos_full": pos_full, "recommendation": rec_full, "noise_band": float("nan"),
                "max_influence": float("nan"), "dominant_snippet": only.id, "dominant_type": only.type,
                "dominant_signed_delta": float("nan"), "influence_concentration": 1.0,
                "spof": True, "spof_snippet": only.id, "noise_limited": False, "single_evidence": True,
            }
            continue

        # sigma is a mean-vs-single-draw comparison, so the null SD of the delta is
        # sigma*sqrt(1 + 1/k); ddof=1 for the small-sample SD. Bonferroni over S snippets.
        sigma = float(np.std([v.pos_score for v in base], ddof=1)) if k > 1 else 0.0
        z = float(norm.ppf(1 - 0.05 / (2 * n_snip)))
        band = float(max(MIN_NOISE_BAND, sigma * (1.0 + 1.0 / k) ** 0.5 * z))

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
        # SPOF = ANY snippet whose removal flips the go/no-go beyond the noise band.
        # Because `recommendation` is an independent field (not a PoS threshold), a
        # rec-flip and a large |Δ| are decoupled, so we cannot only check `dominant`.
        spof_snips = [eid for eid, d in abs_deltas.items() if rec_change[eid] and d > band]
        dom_ev = next(e for e in a.evidence if e.id == dominant)
        sn = spof_snips[0] if spof_snips else None
        sn_ev = next((e for e in a.evidence if e.id == sn), None) if sn else None
        per_asset[a.id] = {
            "pos_full": pos_full,
            "recommendation": rec_full,
            "noise_band": band,
            "max_influence": m_a,
            "dominant_snippet": dominant,
            "dominant_type": dom_ev.type,
            "dominant_signed_delta": deltas[dominant],
            "influence_concentration": concentration,
            "spof": bool(spof_snips),
            "spof_snippet": sn,
            "spof_type": sn_ev.type if sn_ev else None,
            "spof_delta": deltas[sn] if sn else None,
            "noise_limited": bool(m_a <= band),
            "single_evidence": False,
        }

    m_values = [d["max_influence"] for d in per_asset.values()]  # may contain NaN (single-evidence)
    spofs = [d["spof"] for d in per_asset.values()]
    spof_rate = float(np.mean(spofs))
    metrics = {
        "spof_rate": spof_rate,
        "mean_max_influence": float(np.nanmean(m_values)),
        "median_max_influence": float(np.nanmedian(m_values)),
        "mean_influence_concentration": float(np.mean([d["influence_concentration"] for d in per_asset.values()])),
    }
    metrics_ci = {"spof_rate": list(mean_ci([float(s) for s in spofs], clip=(0.0, 1.0))[1:])}

    # Informational, not a hard gate: a verdict that hinges on one snippet is worth
    # knowing even when that snippet legitimately should dominate (e.g. severe tox).
    status = "warn" if spof_rate > 0 else "pass"
    score = 1.0 - clip01(spof_rate)
    flags = []
    for aid, d in per_asset.items():
        if not d["spof"]:
            continue
        if d.get("single_evidence"):
            flags.append(f"[warn] {aid}: single-evidence asset — the verdict rests entirely on `{d['spof_snippet']}`")
        else:
            flags.append(f"[warn] {aid}: single-point-of-failure — removing `{d['spof_snippet']}` "
                         f"({d['spof_type']}, Δ{d['spof_delta']:+.2f}) flips the recommendation")
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
