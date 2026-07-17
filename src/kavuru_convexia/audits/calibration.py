"""Calibration audit (labels REQUIRED; offline validation only).

On the historical known-outcome assets, compares predicted ``pos_score`` to the
realized binary outcome:

* **Reliability curve** — binned predicted probability vs. observed success frequency.
* **Expected Calibration Error (ECE)** — our own equal-width binning (documented
  below): the sample-weighted mean gap between predicted confidence and observed
  accuracy across bins.
* **Brier score** — mean squared error of the probabilistic prediction.
* **AUROC** — does the PoS score even separate successes from failures (discrimination)?

Because it needs outcome labels, this module CANNOT run in production; it is an
offline validation of the agent against history. The curated set is deliberately
balanced (~50% success), so calibration is judged against *its own* base rate; the
published pipeline base rate is reported only as an external sanity anchor.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score

from .. import config
from ..assets import Asset
from ..evaluator import AssetEvaluator
from ..logutil import get_logger
from ._common import clip01, evaluate_batch
from .report_types import CheckResult

logger = get_logger(__name__)


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int
) -> tuple[float, list[dict]]:
    """Equal-width-bin ECE and the per-bin reliability curve.

    Bins partition [0, 1] into ``n_bins`` equal-width intervals. For each non-empty
    bin we take the mean predicted probability and the observed success frequency;
    ECE is the sample-count-weighted mean of |mean_pred - observed| across bins.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # np.digitize with right=True puts p==1.0 in the top bin; clip index into range.
    idx = np.clip(np.digitize(y_prob, edges[1:-1], right=True), 0, n_bins - 1)
    total = len(y_prob)
    ece = 0.0
    curve: list[dict] = []
    for b in range(n_bins):
        mask = idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        mean_pred = float(y_prob[mask].mean())
        observed = float(y_true[mask].mean())
        ece += (count / total) * abs(mean_pred - observed)
        curve.append({
            "bin_lo": float(edges[b]), "bin_hi": float(edges[b + 1]),
            "mean_pred": mean_pred, "observed": observed, "count": count,
        })
    return float(ece), curve


def audit_calibration(
    evaluator: AssetEvaluator,
    assets: Sequence[Asset],
    *,
    n_bins: int = config.CALIBRATION_N_BINS,
    temperature: Optional[float] = None,
) -> CheckResult:
    """Audit predicted PoS against realized outcomes. Requires labels; offline only."""
    labeled = [a for a in assets if a.true_outcome is not None]
    if len(labeled) < 4:
        raise ValueError("calibration needs >=4 labeled assets")
    logger.info("calibration: %d labeled assets, %d bins", len(labeled), n_bins)

    jobs = [((a.id,), a, "calibration") for a in labeled]
    verdicts = evaluate_batch(evaluator, jobs, temperature=temperature)
    y_true = np.array([1.0 if a.true_outcome else 0.0 for a in labeled])
    y_prob = np.array([verdicts[(a.id,)].pos_score for a in labeled])

    ece, curve = expected_calibration_error(y_true, y_prob, n_bins)
    brier = float(brier_score_loss(y_true, y_prob))
    auroc = float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else float("nan")

    mean_pos = float(y_prob.mean())
    empirical_base_rate = float(y_true.mean())  # ~0.5 by construction of this set
    optimism_vs_set = mean_pos - empirical_base_rate  # systematic offset on this set

    status = "fail" if ece >= config.ECE_FAIL else "warn" if ece >= config.ECE_WARN else "pass"
    score = 1.0 - clip01(ece / config.ECE_FAIL)

    per_asset = {
        a.id: {"pos": float(verdicts[(a.id,)].pos_score), "true_outcome": bool(a.true_outcome),
               "name": a.name}
        for a in labeled
    }
    metrics = {
        "ece": ece,
        "brier": brier,
        "auroc": auroc,
        "mean_pos": mean_pos,
        "empirical_base_rate": empirical_base_rate,
        "published_base_rate": config.BASE_RATE_PHASE1_TO_APPROVAL,
        "optimism_vs_set": optimism_vs_set,
    }
    flags: list[str] = []
    if ece >= config.ECE_WARN:
        flags.append(f"[{status}] ECE {ece:.3f} (bins={n_bins}); predicted confidence "
                     "deviates from observed outcome frequency")
    if abs(optimism_vs_set) >= 0.10:
        direction = "over" if optimism_vs_set > 0 else "under"
        flags.append(f"mean PoS {mean_pos:.2f} vs this set's {empirical_base_rate:.2f} base rate "
                     f"=> systematic {direction}-confidence of {abs(optimism_vs_set):.2f}")
    if not np.isnan(auroc) and auroc < 0.7:
        flags.append(f"weak discrimination: AUROC {auroc:.2f} (PoS barely separates outcomes)")

    return CheckResult(
        name="calibration",
        status=status,
        score=score,
        metrics=metrics,
        flags=flags,
        detail={
            "per_asset": per_asset,
            "reliability_curve": curve,
            "base_rate_source": config.BASE_RATE_SOURCE,
            "note": ("Curated set is balanced (~50% success); calibration is judged "
                     "against its own base rate. The published pipeline base rate "
                     f"({config.BASE_RATE_PHASE1_TO_APPROVAL:.1%}) is an external anchor for "
                     "what a production PoS distribution should be sanity-checked against, "
                     "not a target for this balanced set."),
        },
        requires_labels=True,
        production_usable=False,
    )
