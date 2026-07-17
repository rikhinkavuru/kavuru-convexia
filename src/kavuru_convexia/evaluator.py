"""The pluggable evaluator interface and its reference implementations.

* :class:`Verdict` — the structured object every evaluator emits.
* :class:`AssetEvaluator` — the ABC that decouples the audits from *how* a
  verdict is produced.
* :class:`ReferenceAgent` — an LLM agent that turns an asset + its evidence into
  a structured verdict. It stands in for a production PoS/scientific agent.
* :class:`ExternalAdapter` — loads verdicts captured from an external system
  (e.g., a live playground) so they run through the identical audit pipeline.
"""
from __future__ import annotations

import csv
import json
import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from . import config
from .assets import Asset
from .llm import LLMClient
from .logutil import get_logger

logger = get_logger(__name__)

RECOMMENDATIONS: tuple[str, ...] = ("advance", "pass", "investigate")

DEFAULT_SYSTEM_PROMPT = (
    "You are a rigorous drug-asset due-diligence agent for a venture that acquires "
    "early-stage assets. Given an asset and its typed evidence, judge the asset's "
    "probability of clinical/regulatory success and issue a go/no-go recommendation. "
    "Weigh the evidence on its merits; a single serious safety or freedom-to-operate "
    "liability can outweigh strong efficacy. If the evidence conflicts, say so "
    "explicitly in your rationale.\n\n"
    "Respond with ONLY a JSON object, no prose around it, with exactly these keys:\n"
    '  "pos_score": a float in [0, 1] — probability of eventual success\n'
    '  "recommendation": one of "advance", "pass", "investigate"\n'
    '  "rationale": a concise justification (2-4 sentences)\n'
    '  "cited_evidence_ids": the list of evidence ids you actually relied on'
)


@dataclass
class Verdict:
    """An evaluator's structured judgment on a single asset."""

    asset_id: str
    pos_score: float
    recommendation: str  # one of RECOMMENDATIONS
    rationale: str
    cited_evidence_ids: list[str] = field(default_factory=list)
    model: str = ""
    temperature: Optional[float] = None
    raw: str = ""  # raw model text, retained for debugging
    parse_error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Verdict":
        cited = d.get("cited_evidence_ids", [])
        if isinstance(cited, str):  # tolerate delimited CSV cells
            cited = [c.strip() for c in re.split(r"[;|,]", cited) if c.strip()]
        rec = str(d.get("recommendation", "investigate")).lower().strip()
        return cls(
            asset_id=str(d["asset_id"]),
            pos_score=float(d["pos_score"]),
            recommendation=rec if rec in RECOMMENDATIONS else "investigate",
            rationale=str(d.get("rationale", "")),
            cited_evidence_ids=list(cited),
            model=str(d.get("model", "")),
            temperature=(float(d["temperature"]) if d.get("temperature") not in (None, "") else None),
            raw=str(d.get("raw", "")),
            parse_error=d.get("parse_error"),
        )


class AssetEvaluator(ABC):
    """Produces a :class:`Verdict` for an asset. The audits depend only on this."""

    name: str = "AssetEvaluator"

    @abstractmethod
    def evaluate(
        self, asset: Asset, *, temperature: Optional[float] = None, cache_tag: str = ""
    ) -> Verdict:
        """Return a verdict for ``asset``.

        ``temperature`` controls sampling stochasticity; ``cache_tag`` distinguishes
        otherwise-identical calls (e.g., repetition index) so the reproducibility
        audit draws independent samples that still cache/replay deterministically.
        (The Anthropic API exposes no request seed, so cross-run determinism comes
        from the disk cache, not a seed argument.)
        """


def _extract_json(text: str) -> dict[str, Any]:
    """Parse the first JSON object in a model response, tolerating stray prose."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("no JSON object found in response")
    return json.loads(match.group(0))


class ReferenceAgent(AssetEvaluator):
    """An LLM agent that emits a structured verdict from an asset's evidence."""

    def __init__(
        self,
        client: Optional[LLMClient] = None,
        *,
        model: str = config.ANTHROPIC_MODEL,
        temperature: float = config.REPRO_TEMPERATURE,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        self.client = client or LLMClient(model=model)
        self.model = model
        self.default_temperature = temperature
        self.system_prompt = system_prompt
        self.name = f"ReferenceAgent({model})"

    @staticmethod
    def render_user_prompt(asset: Asset) -> str:
        """Render an asset into the user prompt.

        The evidence-line format ``- [type/direction/strength] id: text`` is a
        contract shared with :mod:`kavuru_convexia.llm`'s offline parser.
        """
        lines = [f"ASSET {asset.id}"]
        if asset.name:
            lines.append(f"Name: {asset.name}")
        lines += [f"Description: {asset.description}", "", "EVIDENCE:"]
        for e in asset.evidence:
            lines.append(f"- [{e.type}/{e.direction}/{e.strength}] {e.id}: {e.text}")
        lines += ["", "Return the JSON verdict."]
        return "\n".join(lines)

    def evaluate(
        self, asset: Asset, *, temperature: Optional[float] = None, cache_tag: str = ""
    ) -> Verdict:
        temp = self.default_temperature if temperature is None else temperature
        user = self.render_user_prompt(asset)
        text = self.client.complete(
            self.system_prompt, user, temperature=temp, cache_tag=cache_tag or "0"
        )
        return self._parse(asset, text, temp)

    def _parse(self, asset: Asset, text: str, temperature: float) -> Verdict:
        valid_ids = set(asset.evidence_ids)
        error: Optional[str] = None
        try:
            obj = _extract_json(text)
            pos = min(1.0, max(0.0, float(obj["pos_score"])))
            rec = str(obj.get("recommendation", "investigate")).lower().strip()
            if rec not in RECOMMENDATIONS:
                rec = "investigate"
            rationale = str(obj.get("rationale", ""))
            # Keep only citations that reference real evidence — a hallucinated id
            # is itself a reliability signal, so we drop it rather than trust it.
            cited = [c for c in obj.get("cited_evidence_ids", []) if c in valid_ids]
        except Exception as exc:  # noqa: BLE001 — any malformed verdict degrades gracefully
            error = f"{type(exc).__name__}: {exc}"
            logger.warning("verdict parse failed for %s: %s", asset.id, error)
            pos, rec, rationale, cited = 0.5, "investigate", text.strip()[:500], []
        return Verdict(
            asset_id=asset.id,
            pos_score=pos,
            recommendation=rec,
            rationale=rationale,
            cited_evidence_ids=cited,
            model=self.model,
            temperature=temperature,
            raw=text,
            parse_error=error,
        )


class ExternalAdapter(AssetEvaluator):
    """Serve pre-captured verdicts (e.g., from a live playground) as an evaluator.

    Verdicts are grouped by ``asset_id`` in capture order. Successive
    ``evaluate`` calls for an asset return successive captured runs, which is
    exactly what the reproducibility audit needs. If more calls arrive than runs
    were captured, it cycles and warns rather than fabricating new verdicts.
    """

    def __init__(self, verdicts_by_asset: dict[str, list[Verdict]], name: str = "ExternalAdapter") -> None:
        if not verdicts_by_asset:
            raise ValueError("ExternalAdapter requires at least one captured verdict")
        self._by_asset = verdicts_by_asset
        self._cursor: dict[str, int] = {k: 0 for k in verdicts_by_asset}
        self.name = name

    @classmethod
    def from_records(cls, records: Iterable[dict[str, Any]], **kw: Any) -> "ExternalAdapter":
        grouped: dict[str, list[Verdict]] = {}
        for rec in records:
            v = Verdict.from_dict(rec)
            grouped.setdefault(v.asset_id, []).append(v)
        return cls(grouped, **kw)

    @classmethod
    def from_json(cls, path: Path | str, **kw: Any) -> "ExternalAdapter":
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        records = doc["verdicts"] if isinstance(doc, dict) else doc
        return cls.from_records(records, **kw)

    @classmethod
    def from_csv(cls, path: Path | str, **kw: Any) -> "ExternalAdapter":
        with Path(path).open(newline="", encoding="utf-8") as fh:
            return cls.from_records(list(csv.DictReader(fh)), **kw)

    def evaluate(
        self, asset: Asset, *, temperature: Optional[float] = None, cache_tag: str = ""
    ) -> Verdict:
        runs = self._by_asset.get(asset.id)
        if not runs:
            raise KeyError(f"ExternalAdapter has no captured verdict for asset {asset.id!r}")
        idx = self._cursor[asset.id]
        if idx >= len(runs):
            logger.warning(
                "ExternalAdapter: only %d captured runs for %s; cycling on request %d",
                len(runs), asset.id, idx + 1,
            )
        verdict = runs[idx % len(runs)]
        self._cursor[asset.id] = idx + 1
        return verdict


def verdicts_to_json(verdicts: Iterable[Verdict], path: Path | str) -> None:
    """Serialize verdicts to the JSON shape :meth:`ExternalAdapter.from_json` reads."""
    payload = {"verdicts": [v.to_dict() for v in verdicts]}
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
