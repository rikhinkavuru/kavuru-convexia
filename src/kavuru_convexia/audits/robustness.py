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
from .stats import mean_ci

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
            # Case-insensitive: the drug name recurs lowercased mid-sentence and in
            # generic form; missing those would leave the identity un-blinded.
            text = re.sub(rf"\b{re.escape(tok)}\b", "the candidate", text, flags=re.IGNORECASE)
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
def _asset_status(max_sig_drift: float, n_rec_changes: int) -> Status:
    # A recommendation reversal under an edit that MUST NOT change the answer is a
    # discrete failure on its own (once attributed to the edit, not native noise).
    if max_sig_drift >= config.POS_DRIFT_FAIL or n_rec_changes > 0:
        return "fail"
    if max_sig_drift >= config.POS_DRIFT_WARN:
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
    """Audit verdict drift under semantics-preserving edits. Requires no labels.

    Two base samples estimate the evaluator's native run-to-run noise for each
    asset; a perturbation's drift is only counted as "significant" once it clears
    that noise floor, and a recommendation change is attributed to the edit only if
    the base recommendation was itself stable across the two base samples.
    """
    perturbations = perturbations if perturbations is not None else default_perturbations(paraphraser)
    pert_names = [name for name, _ in perturbations]
    logger.info("robustness: %d assets x %d perturbations (%s)",
                len(assets), len(perturbations), ", ".join(pert_names))

    jobs = []
    for a in assets:
        jobs.append(((a.id, "base0"), a, "robust-base0"))
        jobs.append(((a.id, "base1"), a, "robust-base1"))  # control arm for the noise floor
        for name, fn in perturbations:
            jobs.append(((a.id, name), fn(a), "robust"))
    verdicts = evaluate_batch(evaluator, jobs, temperature=temperature)

    per_asset: dict[str, dict] = {}
    all_drifts: list[float] = []
    rec_change_flags: list[bool] = []
    per_pert_drift: dict[str, list[float]] = {n: [] for n in pert_names}
    for a in assets:
        base0, base1 = verdicts[(a.id, "base0")], verdicts[(a.id, "base1")]
        baseline_pos = (base0.pos_score + base1.pos_score) / 2.0
        native_noise = abs(base0.pos_score - base1.pos_score)
        # A perturbation must move PoS by more than the native noise (min floor at
        # the WARN threshold) to count. Base recommendation must agree across the two
        # control samples for a perturbation rec-change to be blamed on the edit.
        noise_floor = max(config.POS_DRIFT_WARN, 2.0 * native_noise)
        base_rec_stable = base0.recommendation == base1.recommendation

        pert_detail: dict[str, dict] = {}
        drifts: dict[str, float] = {}
        sig_drifts: dict[str, float] = {}
        n_rec_changes = 0
        for name in pert_names:
            v = verdicts[(a.id, name)]
            drift = abs(v.pos_score - baseline_pos)
            rec_changed = base_rec_stable and v.recommendation != base0.recommendation
            drifts[name] = drift
            sig_drifts[name] = drift if drift > noise_floor else 0.0
            per_pert_drift[name].append(drift)
            all_drifts.append(drift)
            rec_change_flags.append(rec_changed)
            n_rec_changes += int(rec_changed)
            pert_detail[name] = {"pos": v.pos_score, "drift": drift,
                                 "significant": drift > noise_floor, "rec_change": rec_changed,
                                 "recommendation": v.recommendation}
        max_drift = max(drifts.values())
        max_sig_drift = max(sig_drifts.values())
        worst = max(drifts, key=drifts.get)
        status = _asset_status(max_sig_drift, n_rec_changes)
        per_asset[a.id] = {
            "baseline_pos": baseline_pos,
            "baseline_recommendation": base0.recommendation,
            "native_noise": native_noise,
            "max_drift": max_drift,
            "max_significant_drift": max_sig_drift,
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

    # Cluster bootstrap on assets: keep each asset's fixed perturbation vector
    # intact (the perturbations are a designed factor, not a resampled sample).
    n_perts = max(1, len(pert_names))
    metrics_ci = {
        "mean_abs_drift": list(mean_ci([d["mean_drift"] for d in per_asset.values()], clip=(0.0, 1.0))[1:]),
        "rec_change_rate": list(mean_ci(
            [d["n_rec_changes"] / n_perts for d in per_asset.values()], clip=(0.0, 1.0))[1:]),
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
        metrics_ci=metrics_ci,
        flags=flags,
        detail={"per_asset": per_asset, "perturbations": pert_names},
        requires_labels=False,
        production_usable=True,
    )
