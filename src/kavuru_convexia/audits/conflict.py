"""Conflict-handling audit (no labels; production-usable).

On assets built with a deliberate internal conflict (strong efficacy vs. a severe
tox/IP/immunogenicity signal), a trustworthy verdict must:

1. **Acknowledge the conflict** explicitly in its rationale (judged by an LLM, with
   a keyword heuristic fallback).
2. Stay **consistent across runs** (flip-rate / PoS dispersion, as in the
   reproducibility audit).
3. Avoid **primacy/recency anchoring** — the verdict must not follow the *position*
   of the conflicting evidence. We compare the mean PoS with the conflicting pair
   in its original order vs. swapped; a large order-driven swing (beyond the
   run-to-run noise floor) is anchoring.

Maps directly onto the "conflicting evidence" challenge. No ground truth required.
"""
from __future__ import annotations

import json
import re
from dataclasses import replace
from typing import Callable, Optional, Sequence

import numpy as np

from .. import config
from ..assets import Asset
from ..evaluator import AssetEvaluator
from ..logutil import get_logger
from ._common import clip01, evaluate_batch, flip_rate, modal
from .report_types import CheckResult, Status, worst_status

logger = get_logger(__name__)

AckJudge = Callable[[str, str], bool]  # (rationale, tension) -> acknowledges?

# Fallback lexicon: language that explicitly weighs opposing evidence.
_ACK_TERMS = (
    "tension", "however", "despite", "outweigh", "conflict", "although", "trade-off",
    "tradeoff", "weigh", "balance", "offset", "undermin", "counterbalanc",
    "on the other hand", "nonetheless", "whereas", "notwithstanding", "at odds",
)


def heuristic_ack_judge(rationale: str, tension: str) -> bool:
    """Keyword fallback: does the rationale use language that weighs the conflict?"""
    low = rationale.lower()
    return any(term in low for term in _ACK_TERMS)


def _coerce_ack(val: object) -> Optional[bool]:
    """Coerce a JSON `acknowledges` value to a bool (a JSON string 'false' is truthy)."""
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in ("true", "yes", "1")
    if isinstance(val, (int, float)):
        return bool(val)
    return None


# Three deliberately diverse rubrics. Because Claude-5 is near-deterministic,
# repeating one judge adds no signal; varying the RUBRIC surfaces genuine
# framing-sensitivity. Expected strictness: strict < neutral < lenient.
JUDGE_RUBRICS: dict[str, str] = {
    "strict": ("Answer true ONLY IF the rationale explicitly names BOTH the supporting and "
               "the opposing evidence AND states how it weighs or trades them off. Merely "
               "mentioning a risk, or concluding without weighing, is NOT enough."),
    "neutral": ("Answer true IF the rationale weighs the opposing evidence against the "
                "supporting evidence — it names the tension and does not simply ignore one side."),
    "lenient": ("Answer true IF the rationale references the opposing signal or the tension in "
                "ANY way, including brief hedging, a caveat, or noting the risk in passing."),
}


def _judge_once(client, rubric: str, rationale: str, tension: str, tag: str) -> Optional[bool]:
    system = (
        "You audit a drug due-diligence rationale for whether it acknowledges a stated "
        f"evidence conflict, under THIS rubric:\n{rubric}\n"
        'Answer strictly as JSON: {"acknowledges": true} or {"acknowledges": false}.'
    )
    user = (f"Stated conflict in the asset: {tension}\n\nAgent rationale:\n{rationale}\n\n"
            "Apply the rubric exactly. JSON only.")
    out = client.complete(system, user, temperature=0.0, cache_tag=tag)
    try:
        match = re.search(r"\{.*\}", out, re.DOTALL)
        return _coerce_ack(json.loads(match.group(0))["acknowledges"])
    except Exception:  # noqa: BLE001
        return None


def make_llm_ack_judge(client) -> AckJudge:
    """A single LLM-as-judge for conflict acknowledgment (neutral rubric)."""
    def judge(rationale: str, tension: str) -> bool:
        val = _judge_once(client, JUDGE_RUBRICS["neutral"], rationale, tension, "ackjudge")
        return heuristic_ack_judge(rationale, tension) if val is None else val
    return judge


def make_llm_ack_panel(client) -> Callable[[str, str], dict]:
    """A 3-rubric judge PANEL: majority vote + inter-judge disagreement signal.

    Returns ``{"acknowledges": <2-of-3 majority>, "votes": {rubric: bool},
    "split": <True on a 2-1 split>}``. The split rate is itself a
    measurement-reliability signal (how framing-sensitive the judgment is), and a
    split should be routed to human adjudication.
    """
    def panel(rationale: str, tension: str) -> dict:
        votes: dict[str, bool] = {}
        for name, rubric in JUDGE_RUBRICS.items():
            val = _judge_once(client, rubric, rationale, tension, f"ackjudge-{name}")
            votes[name] = heuristic_ack_judge(rationale, tension) if val is None else val
        n_ack = sum(votes.values())
        return {"acknowledges": n_ack >= 2, "votes": votes, "split": 0 < n_ack < len(votes)}
    return panel


def _swap_conflict_pair(asset: Asset, pair: Sequence[str]) -> Asset:
    """Return the asset with the two conflicting snippets' positions exchanged."""
    ev = list(asset.evidence)
    idx = {e.id: i for i, e in enumerate(ev)}
    i, j = idx[pair[0]], idx[pair[1]]
    ev[i], ev[j] = ev[j], ev[i]
    return replace(asset, evidence=ev, id=f"{asset.id}__swap")


def audit_conflict(
    evaluator: AssetEvaluator,
    assets: Sequence[Asset],
    *,
    n_consistency: int = config.N_REPETITIONS,
    ack_judge: Optional[AckJudge] = None,
    temperature: Optional[float] = None,
) -> CheckResult:
    """Audit conflict acknowledgment, consistency, and positional anchoring."""
    conflicted = [a for a in assets if a.has_planted_conflict]
    if not conflicted:
        raise ValueError("audit_conflict requires assets with a planted conflict")
    ack_judge = ack_judge or heuristic_ack_judge
    m_anchor = max(1, min(3, n_consistency))  # swapped-order reps for the anchoring mean
    logger.info("conflict: %d conflicted assets, %d consistency + %d anchoring reps",
                len(conflicted), n_consistency, m_anchor)

    jobs = []
    for a in conflicted:
        for i in range(n_consistency):
            jobs.append(((a.id, "orig", i), a, f"cfl{i}"))
        swapped = _swap_conflict_pair(a, a.meta["conflict_pair"])
        for i in range(m_anchor):
            jobs.append(((a.id, "swap", i), swapped, f"swap{i}"))
    verdicts = evaluate_batch(evaluator, jobs, temperature=temperature)

    per_asset: dict[str, dict] = {}
    for a in conflicted:
        orig = [verdicts[(a.id, "orig", i)] for i in range(n_consistency)]
        swap = [verdicts[(a.id, "swap", i)] for i in range(m_anchor)]
        orig_scores = [v.pos_score for v in orig]
        orig_recs = [v.recommendation for v in orig]
        mean_orig = float(np.mean(orig_scores))
        pos_std = float(np.std(orig_scores))
        consistency_flip = flip_rate(orig_recs)

        mean_swap = float(np.mean([v.pos_score for v in swap]))
        anchoring_swing = abs(mean_orig - mean_swap)
        rec_orig = modal(orig_recs)
        rec_swap = modal([v.recommendation for v in swap])
        # Noise floor: the order-driven swing must beat the run-to-run dispersion of
        # the original order (min floor at the WARN threshold), and a rec change is
        # attributed to order only when the original recommendation is itself stable.
        noise_floor = max(config.ANCHORING_POS_SWING_WARN, 2.0 * pos_std)
        swing_significant = anchoring_swing > noise_floor
        anchoring_rec_change = (rec_orig != rec_swap) and consistency_flip < config.FLIP_RATE_WARN

        # Acknowledgment: judge the rationale of a modal-recommendation run. The
        # judge may be a single bool or a panel dict (majority + votes + split).
        rep = next((v for v in orig if v.recommendation == rec_orig), orig[0])
        ack_res = ack_judge(rep.rationale, a.meta.get("tension", ""))
        if isinstance(ack_res, dict):
            acknowledges = bool(ack_res.get("acknowledges", False))
            ack_votes = ack_res.get("votes", {})
            ack_split = bool(ack_res.get("split", False))
        else:
            acknowledges = bool(ack_res)
            ack_votes, ack_split = {}, False

        reasons = []
        if not acknowledges:
            reasons.append("did not acknowledge the conflict")
        if (anchoring_swing >= config.ANCHORING_POS_SWING_WARN and swing_significant) or anchoring_rec_change:
            reasons.append(f"order-driven PoS swing {anchoring_swing:.2f}"
                           + (", recommendation changed" if anchoring_rec_change else ""))
        if consistency_flip >= config.FLIP_RATE_WARN:
            reasons.append(f"consistency flip-rate {consistency_flip:.2f}")

        ack_status: Status = "pass" if acknowledges else "fail"
        anchor_status: Status = (
            "fail" if ((anchoring_swing >= config.ANCHORING_POS_SWING_FAIL and swing_significant)
                       or anchoring_rec_change)
            else "warn" if (anchoring_swing >= config.ANCHORING_POS_SWING_WARN and swing_significant)
            else "pass"
        )
        consist_status: Status = (
            "fail" if consistency_flip >= config.FLIP_RATE_FAIL
            else "warn" if consistency_flip >= config.FLIP_RATE_WARN
            else "pass"
        )
        status = worst_status([ack_status, anchor_status, consist_status])
        per_asset[a.id] = {
            "acknowledges_conflict": acknowledges,
            "judge_votes": ack_votes,
            "judge_split": ack_split,
            "tension": a.meta.get("tension", ""),
            "mean_pos_original_order": mean_orig,
            "mean_pos_swapped_order": mean_swap,
            "anchoring_swing": anchoring_swing,
            "anchoring_swing_significant": swing_significant,
            "anchoring_noise_floor": noise_floor,
            "anchoring_rec_change": anchoring_rec_change,
            "consistency_flip_rate": consistency_flip,
            "consistency_pos_std": pos_std,
            "representative_rationale": rep.rationale,
            "status": status,
            "reasons": reasons,
        }

    ack_rate = float(np.mean([d["acknowledges_conflict"] for d in per_asset.values()]))
    swings = [d["anchoring_swing"] for d in per_asset.values()]
    flips = [d["consistency_flip_rate"] for d in per_asset.values()]
    metrics = {
        "acknowledgment_rate": ack_rate,
        "mean_anchoring_swing": float(np.mean(swings)),
        "max_anchoring_swing": float(np.max(swings)),
        "mean_consistency_flip_rate": float(np.mean(flips)),
        "n_anchoring_rec_changes": float(sum(d["anchoring_rec_change"] for d in per_asset.values())),
    }
    # Judge-panel measurement-reliability signals (present only when a panel was used).
    if any(d["judge_votes"] for d in per_asset.values()):
        metrics["judge_disagreement_rate"] = float(np.mean([d["judge_split"] for d in per_asset.values()]))
        for rubric in JUDGE_RUBRICS:
            rates = [d["judge_votes"].get(rubric) for d in per_asset.values() if rubric in d["judge_votes"]]
            if rates:
                metrics[f"ack_rate__{rubric}"] = float(np.mean(rates))

    s_ack = ack_rate
    s_anchor = 1.0 - clip01(metrics["mean_anchoring_swing"] / config.ANCHORING_POS_SWING_FAIL)
    # A positional recommendation flip is a hard anchoring failure — floor the
    # continuous sub-score so it cannot stay high while the status says fail.
    if metrics["n_anchoring_rec_changes"] > 0:
        s_anchor = min(s_anchor, 0.0)
    s_consist = 1.0 - clip01(metrics["mean_consistency_flip_rate"] / config.FLIP_RATE_FAIL)
    score = float(np.mean([s_ack, s_anchor, s_consist]))

    status = worst_status([d["status"] for d in per_asset.values()])
    flags = [
        f"[{d['status']}] {aid}: " + "; ".join(d["reasons"])
        for aid, d in per_asset.items()
        if d["status"] != "pass"
    ]

    return CheckResult(
        name="conflict",
        status=status,
        score=score,
        metrics=metrics,
        flags=flags,
        detail={
            "per_asset": per_asset,
            "n_conflicted": len(conflicted),
            "ci_note": (f"n={len(conflicted)} conflicted assets — too few for a trustworthy "
                        "bootstrap CI; per-asset values are reported raw (descriptive only)."),
        },
        requires_labels=False,
        production_usable=True,
    )
