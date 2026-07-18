"""Reproducibility audit (no labels; production-usable).

Runs the evaluator ``n`` times on each asset and measures how much the verdict
moves under *identical* inputs. Three signals:

* **PoS dispersion** — standard deviation (and IQR) of the probability-of-success
  score across runs. A score that gates capital should barely move.
* **Recommendation flip-rate** — fraction of runs whose go/no-go disagrees with
  the modal recommendation. Flips are the most consequential instability.
* **Rationale stability** — mean pairwise Jaccard overlap of the cited-evidence
  sets. The *reasoning* can drift even when the number does not.

No ground truth is required, so this runs in production against live verdicts.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .. import config
from ..assets import Asset
from ..evaluator import AssetEvaluator
from ..logutil import get_logger
from ._common import clip01, evaluate_batch, flip_rate, iqr, mean_pairwise_jaccard, modal
from .report_types import CheckResult, Status, worst_status
from .stats import hierarchical_mean_ci

logger = get_logger(__name__)


def _asset_status(pos_std: float, fr: float, rationale_jaccard: float, parse_err_frac: float) -> Status:
    # A run that fails to produce a parseable verdict is itself unreproducible; a
    # majority of failed runs fails the asset, any failure warns.
    if pos_std >= config.POS_STD_FAIL or fr >= config.FLIP_RATE_FAIL or parse_err_frac >= 0.5:
        return "fail"
    if (
        pos_std >= config.POS_STD_WARN
        or fr >= config.FLIP_RATE_WARN
        or rationale_jaccard < config.RATIONALE_JACCARD_WARN
        or parse_err_frac > 0.0
    ):
        return "warn"
    return "pass"


def audit_reproducibility(
    evaluator: AssetEvaluator,
    assets: Sequence[Asset],
    *,
    n: int = config.N_REPETITIONS,
    temperature: Optional[float] = None,
) -> CheckResult:
    """Audit run-to-run stability of the evaluator's verdicts. Requires no labels."""
    logger.info("reproducibility: %d assets x %d repetitions", len(assets), n)
    jobs = [((a.id, i), a, f"rep{i}") for a in assets for i in range(n)]
    verdicts = evaluate_batch(evaluator, jobs, temperature=temperature)

    per_asset: dict[str, dict] = {}
    boot_items: list[dict] = []  # run-level data per asset for the hierarchical bootstrap
    for a in assets:
        runs = [verdicts[(a.id, i)] for i in range(n)]
        scores = [v.pos_score for v in runs]
        recs = [v.recommendation for v in runs]
        n_parse_errors = sum(v.parse_error is not None for v in runs)
        parse_err_frac = n_parse_errors / n
        # Rationale stability is measured over runs that actually produced a verdict,
        # so an evaluator that refuses/errors every run is NOT scored as "perfectly
        # stable" (all-empty citation sets would otherwise yield Jaccard 1.0).
        valid = [v for v in runs if v.parse_error is None]
        cited_sets = [set(v.cited_evidence_ids) for v in valid]
        rationale_jaccard = mean_pairwise_jaccard(cited_sets) if len(valid) >= 2 else 0.0
        pos_std = float(np.std(scores))  # population std across the n runs
        fr = flip_rate(recs)
        status = _asset_status(pos_std, fr, rationale_jaccard, parse_err_frac)
        per_asset[a.id] = {
            "scores": [float(s) for s in scores],  # raw per-run PoS (for variance plots)
            "pos_mean": float(np.mean(scores)),
            "pos_std": pos_std,
            "pos_iqr": iqr(scores),
            "pos_min": float(np.min(scores)),
            "pos_max": float(np.max(scores)),
            "flip_rate": fr,
            "modal_recommendation": modal(recs),  # deterministic tie-break
            "rationale_jaccard": rationale_jaccard,
            "n_parse_errors": n_parse_errors,
            "status": status,
        }
        boot_items.append({
            "scores": np.asarray(scores, dtype=float),
            "recs": recs,
            "cited": [set(v.cited_evidence_ids) for v in runs],
            "valid": [v.parse_error is None for v in runs],
            "n": n,
        })

    stds = [d["pos_std"] for d in per_asset.values()]
    frs = [d["flip_rate"] for d in per_asset.values()]
    rjs = [d["rationale_jaccard"] for d in per_asset.values()]
    metrics = {
        "n_repetitions": float(n),
        "pos_std_mean": float(np.mean(stds)),
        "pos_std_max": float(np.max(stds)),
        "flip_rate_mean": float(np.mean(frs)),
        "flip_rate_max": float(np.max(frs)),
        "rationale_jaccard_mean": float(np.mean(rjs)),
        "rationale_jaccard_min": float(np.min(rjs)),
        "parse_error_rate": float(
            np.mean([d["n_parse_errors"] for d in per_asset.values()]) / n
        ),
    }

    # Dimension score: average of four [0,1] sub-scores (1 = perfectly stable);
    # unparseable/refused runs drag the score down rather than being ignored.
    s_std = 1.0 - clip01(metrics["pos_std_mean"] / config.POS_STD_FAIL)
    s_flip = 1.0 - clip01(metrics["flip_rate_mean"] / config.FLIP_RATE_FAIL)
    s_rationale = clip01(metrics["rationale_jaccard_mean"])
    s_parse = 1.0 - clip01(metrics["parse_error_rate"])
    score = float(np.mean([s_std, s_flip, s_rationale, s_parse]))

    # Hierarchical bootstrap: resample assets AND the N runs within each drawn
    # asset, so the noise of an 8-run per-asset estimate propagates into the CI.
    def _jaccard_over(item: dict, ri: np.ndarray) -> float:
        # Jaccard is a pairwise U-statistic; naive with-replacement resampling would
        # create identical (self) pairs that score 1.0 and bias the CI upward. De-dup
        # the drawn run indices so the inner bootstrap only varies WHICH runs appear.
        valid = [item["cited"][j] for j in np.unique(ri) if item["valid"][j]]
        return mean_pairwise_jaccard(valid) if len(valid) >= 2 else 0.0

    n_runs = lambda it: it["n"]  # noqa: E731
    metrics_ci = {
        "pos_std_mean": list(hierarchical_mean_ci(
            boot_items, lambda it, ri: float(np.std(it["scores"][ri])), n_runs, clip=(0.0, 1.0))[1:]),
        "flip_rate_mean": list(hierarchical_mean_ci(
            boot_items, lambda it, ri: flip_rate([it["recs"][j] for j in ri]), n_runs, clip=(0.0, 1.0))[1:]),
        "rationale_jaccard_mean": list(hierarchical_mean_ci(
            boot_items, _jaccard_over, n_runs, clip=(0.0, 1.0))[1:]),
    }

    statuses = [d["status"] for d in per_asset.values()]
    status = worst_status(statuses)
    flags: list[str] = []
    for aid, d in sorted(per_asset.items(), key=lambda kv: -kv[1]["flip_rate"]):
        if d["status"] == "pass":
            continue
        parts = []
        if d["flip_rate"] > 0:
            parts.append(f"recommendation flipped in {round(d['flip_rate'] * n)}/{n} runs")
        if d["pos_std"] >= config.POS_STD_WARN:
            parts.append(f"PoS std {d['pos_std']:.3f}")
        if d["rationale_jaccard"] < config.RATIONALE_JACCARD_WARN:
            parts.append(f"rationale Jaccard {d['rationale_jaccard']:.2f}")
        flags.append(f"[{d['status']}] {aid}: " + "; ".join(parts))

    return CheckResult(
        name="reproducibility",
        status=status,
        score=score,
        metrics=metrics,
        metrics_ci=metrics_ci,
        flags=flags,
        detail={"per_asset": per_asset, "n": n},
        requires_labels=False,
        production_usable=True,
    )
