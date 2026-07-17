"""Cached LLM client wrapping the Anthropic SDK, with a deterministic offline path.

Every completion is cached to disk keyed by a hash of ``(model, system, user,
temperature, max_tokens, cache_tag)``. The ``cache_tag`` is the crucial knob for
the reproducibility audit: repeated samples of the *same* prompt are keyed by a
distinct tag (the run index), so a fresh run makes N genuinely independent API
calls (real temperature-driven variance) while a re-run replays them byte-for-byte.

When no ``ANTHROPIC_API_KEY`` is set — or ``KAVURU_OFFLINE=1`` — the client uses a
self-contained deterministic pseudo-agent so the pipeline (and the test suite)
runs end-to-end without network. Offline output is derived only from the prompt
strings, so it never masquerades as a real model call in the cache metadata.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path

from . import config
from .logutil import get_logger

logger = get_logger(__name__)

# Shared contract: evidence is rendered into prompts as
#   - [type/direction/strength] evidence_id: text
# The offline pseudo-agent parses this exact shape, so keep the two in lockstep.
_EVIDENCE_LINE = re.compile(r"- \[(\w+)/(\w+)/(\w+)\]\s+(\S+):")
_STRENGTH_W = {"weak": 1.0, "moderate": 2.0, "strong": 3.0}
_DIRECTION_W = {"supportive": 1.0, "adverse": -1.0, "mixed": 0.0, "neutral": 0.0}


class LLMClient:
    """A disk-cached completion client with a real (Anthropic) and offline mode."""

    def __init__(
        self,
        model: str = config.ANTHROPIC_MODEL,
        *,
        cache_dir: Path | str = config.CACHE_DIR,
        offline: bool | None = None,
        max_tokens: int = config.MAX_TOKENS,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        want_offline = config.OFFLINE if offline is None else offline
        has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
        self.offline = want_offline or not has_key
        self._client = None
        if self.offline:
            reason = "KAVURU_OFFLINE set" if want_offline else "no ANTHROPIC_API_KEY"
            logger.info("LLMClient in OFFLINE mode (%s); using deterministic stub", reason)
        else:
            import anthropic  # imported lazily so offline/tests need no network stack

            # The SDK retries transient errors (429/5xx) with backoff internally.
            self._client = anthropic.Anthropic(max_retries=4)
            # Claude 5-family models reject an explicit `temperature`; we detect that
            # on first use and thereafter measure the model's *native* run-to-run
            # non-determinism (which is exactly the production behavior we audit).
            self._temperature_supported = True

    # -- caching ------------------------------------------------------------
    def _cache_key(self, system: str, user: str, temperature: float, cache_tag: str) -> str:
        blob = json.dumps(
            {
                "model": self.model,
                "system": system,
                "user": user,
                "temperature": round(temperature, 4),
                "max_tokens": self.max_tokens,
                "tag": cache_tag,
                "offline": self.offline,
            },
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    # -- public API ---------------------------------------------------------
    def complete(
        self, system: str, user: str, *, temperature: float, cache_tag: str = ""
    ) -> str:
        """Return a completion string, hitting the on-disk cache first."""
        key = self._cache_key(system, user, temperature, cache_tag)
        path = self._cache_path(key)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))["response"]

        if self.offline:
            text = self._offline_complete(user, temperature, cache_tag)
            source = "offline-stub"
        else:
            text = self._api_complete(system, user, temperature)
            source = self.model

        path.write_text(
            json.dumps(
                {"source": source, "temperature": temperature, "tag": cache_tag,
                 "system": system, "user": user, "response": text},
                indent=2,
            ),
            encoding="utf-8",
        )
        return text

    # -- real API ----------------------------------------------------------
    def _api_complete(self, system: str, user: str, temperature: float) -> str:
        """Call the Anthropic API, dropping `temperature` if the model rejects it."""
        import anthropic

        def _text(resp: object) -> str:
            return "".join(b.text for b in resp.content if b.type == "text")  # type: ignore[attr-defined]

        base = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        if self._temperature_supported:
            try:
                return _text(self._client.messages.create(temperature=temperature, **base))  # type: ignore[union-attr]
            except anthropic.BadRequestError as exc:
                if "temperature" not in str(exc).lower():
                    raise
                logger.info(
                    "model %s rejects `temperature`; measuring native non-determinism instead",
                    self.model,
                )
                self._temperature_supported = False
        return _text(self._client.messages.create(**base))  # type: ignore[union-attr]

    # -- offline pseudo-agent ----------------------------------------------
    def _offline_complete(self, user: str, temperature: float, cache_tag: str) -> str:
        """Deterministic verdict JSON derived purely from the prompt text.

        Scores evidence by direction x strength, then adds reproducible noise
        keyed by a hash of ``(user, cache_tag)`` scaled by temperature — so the
        stub exhibits realistic run-to-run dispersion (distinct tags),
        order-sensitivity (reordered text hashes differently), and conflict
        acknowledgment (when strong opposing evidence coexists).
        """
        found = _EVIDENCE_LINE.findall(user)
        raw = sum(_DIRECTION_W.get(d, 0.0) * _STRENGTH_W.get(s, 1.0) for _, d, s, _ in found)
        base = 1.0 / (1.0 + math.exp(-raw / 3.0))  # squash net evidence to (0, 1)

        h = int(hashlib.sha256(f"{user}|{cache_tag}".encode("utf-8")).hexdigest(), 16)
        noise_amp = 0.12 * (temperature / 0.7 if temperature else 0.0)
        noise = ((h % 10_000) / 10_000.0 - 0.5) * 2.0 * noise_amp
        pos = min(0.98, max(0.02, base + noise))

        if pos >= 0.60:
            rec = "advance"
        elif pos <= 0.35:
            rec = "pass"
        else:
            rec = "investigate"

        strong_sup = [eid for t, d, s, eid in found if d == "supportive" and s == "strong"]
        strong_adv = [eid for t, d, s, eid in found if d == "adverse" and s == "strong"]
        others = [eid for _, _, _, eid in found if eid not in strong_sup + strong_adv]
        # Rationale instability: the cited set's tail rotates with the tag.
        rotate = h % (len(others) + 1) if others else 0
        cited = strong_sup[:1] + strong_adv[:1] + (others[rotate:] + others[:rotate])[:1]
        cited = [c for c in dict.fromkeys(cited) if c]  # de-dup, drop empties

        conflicted = bool(strong_sup and strong_adv)
        if conflicted:
            rationale = (
                "The evidence is in tension: strong supportive signals must be weighed "
                "against a serious adverse finding, so this verdict acknowledges the "
                "conflict rather than resolving it cleanly."
            )
        else:
            rationale = "The evidence points consistently in one direction on balance."

        return json.dumps(
            {
                "pos_score": round(pos, 4),
                "recommendation": rec,
                "rationale": rationale,
                "cited_evidence_ids": cited,
            }
        )
