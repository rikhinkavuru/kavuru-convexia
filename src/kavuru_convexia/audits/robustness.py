"""Robustness audit (no labels; production-usable).

Applies *semantics-preserving* perturbations that must not change the answer, and
measures how far the verdict drifts:

* **reorder** — reverse the evidence order (content identical).
* **neutralize** — strip the drug's name/brand to neutral placeholders. This
  doubles as an anti-memorization probe: if a historical asset's PoS falls once
  "Keytruda" becomes "the candidate", the agent was scoring brand recognition,
  not the evidence.
* **reformat** — reformat numbers/units/punctuation (pure surface form).
* **paraphrase** — reword the description while preserving every fact (LLM-driven;
  skipped when no paraphraser is supplied, e.g. offline).

Large drift under any of these means the verdict is unreliable. No labels needed.
"""
from __future__ import annotations

import re
from dataclasses import replace
from typing import Callable, Optional, Sequence

import numpy as np

from .. import config
from ..assets import Asset, EvidenceSnippet
from ..evaluator import AssetEvaluator
from ..logutil import get_logger
from ._common import clip01, evaluate_batch
from .report_types import CheckResult, Status, worst_status

logger = get_logger(__name__)

Perturbation = tuple[str, Callable[[Asset], Asset]]
_GENERIC_NAME_WORDS = {"synthetic", "control", "borderline", "the", "and", "with", "vs"}


# ---------------------------------------------------------------------------
# Semantics-preserving perturbations (Asset -> Asset)
# ---------------------------------------------------------------------------
def reorder_evidence(asset: Asset) -> Asset:
    """Reverse the evidence order — same content, different sequence."""
    return replace(asset, evidence=list(reversed(asset.evidence)), id=f"{asset.id}__reorder")


def _name_tokens(name: str) -> list[str]:
    """Capitalized name/brand tokens worth neutralizing (drops generic words)."""
    toks = re.findall(r"[A-Za-z][A-Za-z0-9-]{3,}", name)
    return [t for t in toks if t.lower() not in _GENERIC_NAME_WORDS and t[0].isupper()]


def neutralize_entities(asset: Asset) -> Asset:
    """Replace the drug name/brand with a neutral placeholder everywhere it appears."""
    tokens = _name_tokens(asset.name) if asset.name else []

    def scrub(text: str) -> str:
        for tok in tokens:
            text = re.sub(rf"\b{re.escape(tok)}\b", "the candidate", text)
        return text

    new_evidence = [replace(e, text=scrub(e.text)) for e in asset.evidence]
    return replace(
        asset,
        name=None,
        description=scrub(asset.description),
        evidence=new_evidence,
        id=f"{asset.id}__neutralized",
    )


_REFORMATS: list[tuple[str, str]] = [
    (r"(\d+)\s*%", r"\1 percent"),
    (r"\b(\d+)\s*mg\b", r"\1-mg"),
    (r"\bonce-daily\b", "once daily"),
    (r"\btwice-daily\b", "twice daily"),
    (r"\be\.g\.,?", "for example"),
    (r"\bi\.e\.,?", "that is"),
]


def reformat_text(asset: Asset) -> Asset:
    """Reformat numbers/units/punctuation without changing meaning."""

    def fmt(text: str) -> str:
        for pat, rep in _REFORMATS:
            text = re.sub(pat, rep, text)
        return text

    new_evidence = [replace(e, text=fmt(e.text)) for e in asset.evidence]
    return replace(asset, description=fmt(asset.description), evidence=new_evidence, id=f"{asset.id}__reformat")


def make_llm_paraphraser(client) -> Callable[[str], str]:
    """A paraphraser backed by an LLM client (used for the `paraphrase` perturbation)."""
    system = (
        "You paraphrase text for a robustness test. Preserve every fact, number, "
        "entity, and claim exactly; change only wording and sentence structure. "
        "Do not add, remove, or soften any information. Output only the paraphrase."
    )

    def paraphrase(text: str) -> str:
        out = client.complete(system, f"Paraphrase this:\n\n{text}", temperature=0.0, cache_tag="paraphrase")
        return out.strip() or text

    return paraphrase


def paraphrase_description(paraphraser: Callable[[str], str]) -> Callable[[Asset], Asset]:
    def _fn(asset: Asset) -> Asset:
        return replace(asset, description=paraphraser(asset.description), id=f"{asset.id}__paraphrase")

    return _fn


def default_perturbations(paraphraser: Optional[Callable[[str], str]] = None) -> list[Perturbation]:
    """The standard perturbation set; includes paraphrase only if a paraphraser is given."""
    perts: list[Perturbation] = [
        ("reorder", reorder_evidence),
        ("neutralize", neutralize_entities),
        ("reformat", reformat_text),
    ]
    if paraphraser is not None:
        perts.append(("paraphrase", paraphrase_description(paraphraser)))
    return perts


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------
def _asset_status(max_drift: float, n_rec_changes: int, max_drift_warn_hit: bool) -> Status:
    if max_drift >= config.POS_DRIFT_FAIL or (n_rec_changes > 0 and max_drift_warn_hit):
        return "fail"
    if max_drift >= config.POS_DRIFT_WARN or n_rec_changes > 0:
        return "warn"
    return "pass"


def audit_robustness(
    evaluator: AssetEvaluator,
    assets: Sequence[Asset],
    *,
    perturbations: Optional[list[Perturbation]] = None,
    paraphraser: Optional[Callable[[str], str]] = None,
    temperature: Optional[float] = None,
) -> CheckResult:
    """Audit verdict drift under semantics-preserving edits. Requires no labels."""
    perturbations = perturbations if perturbations is not None else default_perturbations(paraphraser)
    pert_names = [name for name, _ in perturbations]
    logger.info("robustness: %d assets x %d perturbations (%s)",
                len(assets), len(perturbations), ", ".join(pert_names))

    jobs = [((a.id, "base"), a, "robust-base") for a in assets]
    for a in assets:
        for name, fn in perturbations:
            jobs.append(((a.id, name), fn(a), "robust"))
    verdicts = evaluate_batch(evaluator, jobs, temperature=temperature)

    per_asset: dict[str, dict] = {}
    all_drifts: list[float] = []
    rec_change_flags: list[bool] = []
    per_pert_drift: dict[str, list[float]] = {n: [] for n in pert_names}
    for a in assets:
        base = verdicts[(a.id, "base")]
        pert_detail: dict[str, dict] = {}
        drifts: dict[str, float] = {}
        n_rec_changes = 0
        for name in pert_names:
            v = verdicts[(a.id, name)]
            drift = abs(v.pos_score - base.pos_score)
            rec_changed = v.recommendation != base.recommendation
            drifts[name] = drift
            per_pert_drift[name].append(drift)
            all_drifts.append(drift)
            rec_change_flags.append(rec_changed)
            n_rec_changes += int(rec_changed)
            pert_detail[name] = {"pos": v.pos_score, "drift": drift, "rec_change": rec_changed,
                                 "recommendation": v.recommendation}
        max_drift = max(drifts.values())
        worst = max(drifts, key=drifts.get)
        status = _asset_status(max_drift, n_rec_changes, max_drift >= config.POS_DRIFT_WARN)
        per_asset[a.id] = {
            "baseline_pos": base.pos_score,
            "baseline_recommendation": base.recommendation,
            "max_drift": max_drift,
            "mean_drift": float(np.mean(list(drifts.values()))),
            "worst_perturbation": worst,
            "n_rec_changes": n_rec_changes,
            "per_perturbation": pert_detail,
            "status": status,
        }

    rec_change_rate = float(np.mean(rec_change_flags)) if rec_change_flags else 0.0
    metrics = {
        "mean_abs_drift": float(np.mean(all_drifts)) if all_drifts else 0.0,
        "max_abs_drift": float(np.max(all_drifts)) if all_drifts else 0.0,
        "rec_change_rate": rec_change_rate,
        **{f"mean_drift__{n}": float(np.mean(d)) for n, d in per_pert_drift.items() if d},
    }

    s_drift = 1.0 - clip01(metrics["mean_abs_drift"] / config.POS_DRIFT_FAIL)
    s_rec = 1.0 - clip01(rec_change_rate)
    score = float(np.mean([s_drift, s_rec]))

    status = worst_status([d["status"] for d in per_asset.values()])
    flags: list[str] = []
    for aid, d in sorted(per_asset.items(), key=lambda kv: -kv[1]["max_drift"]):
        if d["status"] == "pass":
            continue
        msg = f"[{d['status']}] {aid}: +{d['max_drift']:.2f} PoS drift under {d['worst_perturbation']}"
        if d["n_rec_changes"]:
            msg += f"; recommendation changed under {d['n_rec_changes']} perturbation(s)"
        flags.append(msg)

    return CheckResult(
        name="robustness",
        status=status,
        score=score,
        metrics=metrics,
        flags=flags,
        detail={"per_asset": per_asset, "perturbations": pert_names},
        requires_labels=False,
        production_usable=True,
    )
