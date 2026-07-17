# Playground audit protocol

How to run the **reproducibility**, **robustness**, and **conflict** audits
against a live agent you can only reach through a web playground (e.g. Convexia's
public PoS agent), by capturing its verdicts and replaying them through the
identical pipeline via `ExternalAdapter`. Calibration is intentionally excluded
here — it needs ground-truth outcomes and is an offline check.

This protocol uses **only publicly available functionality**, captures outputs
**faithfully**, and keeps request volume **minimal**. It is a measurement
protocol, not a scraper: a handful of assets, a few repetitions each.

---

## 0. Ground rules

- Use only the public playground UI/API as intended. Do not attempt to bypass
  auth, rate limits, or access non-public endpoints.
- Capture what the agent actually returns, verbatim. Never edit a verdict to make
  a point — the whole exercise is about faithful measurement.
- Keep it small: **≤ 5 assets**, **≤ 5 repetitions** per submission, with a pause
  between calls. That is enough for the statistics here and does not stress the
  service.
- Record the date, the model/version string if the playground exposes one, and
  the exact submitted text alongside every capture.

## 1. Choose the assets

Submit a small, deliberate set:

1. **2–3 conflicted assets** — e.g. the bundled synthetic conflict assets
   (`SYN-CONFLICT-EFFICACY-TOX`, `SYN-CONFLICT-SCIENCE-IP`), which pair strong
   efficacy with a severe tox or IP liability. These drive the conflict audit.
2. **1–2 borderline assets** — `SYN-BORDERLINE-BALANCED` / `SYN-BORDERLINE-THIN`,
   where reliability is most likely to degrade.

Export the exact prompt text for each with the reference renderer so what you
paste into the playground matches what the harness reasons about:

```python
from kavuru_convexia.assets import build_synthetic_assets
from kavuru_convexia.evaluator import ReferenceAgent

for a in build_synthetic_assets():
    print("=" * 80, a.id, "=" * 80)
    print(ReferenceAgent.render_user_prompt(a))
```

## 2. Reproducibility & conflict-consistency captures

For each asset, submit the **same** prompt **N = 5** times (fresh session each
time so nothing is cached on their side) and record every verdict.

## 3. Robustness captures

For each asset, also submit the **semantics-preserving perturbations** the
robustness audit expects, capturing one verdict per variant. Generate the exact
variant texts locally so they are reproducible:

```python
from kavuru_convexia.assets import build_synthetic_assets
from kavuru_convexia.evaluator import ReferenceAgent
from kavuru_convexia.audits.robustness import reorder_evidence, neutralize_entities, reformat_text

asset = build_synthetic_assets()[0]
for name, fn in [("reorder", reorder_evidence), ("neutralize", neutralize_entities), ("reformat", reformat_text)]:
    print("#", name)
    print(ReferenceAgent.render_user_prompt(fn(asset)))
```

Submit each variant under the perturbed asset id it prints (e.g.
`SYN-CONFLICT-EFFICACY-TOX__reorder`) so the capture keys line up with what the
audit requests. (The `paraphrase` perturbation is optional here; skip it unless
you paraphrase the description with a fixed, recorded rewrite.)

## 4. Capture format

Save every verdict as one row/record. `ExternalAdapter` reads either JSON or CSV.

**JSON** (`captures.json`):

```json
{
  "verdicts": [
    {"asset_id": "SYN-CONFLICT-EFFICACY-TOX", "pos_score": 0.31, "recommendation": "pass",
     "rationale": "Strong efficacy but the hepatotoxicity at efficacious exposure dominates...",
     "cited_evidence_ids": ["SYN-CONFLICT-EFFICACY-TOX-ev2", "SYN-CONFLICT-EFFICACY-TOX-ev3"],
     "model": "convexia-pos-playground", "temperature": null},
    {"asset_id": "SYN-CONFLICT-EFFICACY-TOX", "pos_score": 0.28, "recommendation": "pass", "rationale": "...",
     "cited_evidence_ids": ["SYN-CONFLICT-EFFICACY-TOX-ev3"]}
  ]
}
```

**CSV** (`captures.csv`): columns `asset_id,pos_score,recommendation,rationale,cited_evidence_ids`
with `cited_evidence_ids` as a `;`-separated cell.

Rules:
- One record **per submission**, in the order you captured them. The adapter
  serves an asset's repeated runs in that order.
- `pos_score` in `[0, 1]`. If the playground reports a percentage, divide by 100.
- Put the perturbed submissions under their perturbed `asset_id`s.

## 5. Run the audits on the captures

```python
from kavuru_convexia.assets import build_synthetic_assets
from kavuru_convexia.evaluator import ExternalAdapter
from kavuru_convexia.audits import audit_reproducibility, audit_conflict

adapter = ExternalAdapter.from_json("captures.json")   # or .from_csv("captures.csv")
assets = [a for a in build_synthetic_assets() if a.id in {"SYN-CONFLICT-EFFICACY-TOX", "SYN-CONFLICT-SCIENCE-IP"}]

# N here must equal the number of repeated captures you saved per asset.
print("\n".join(audit_reproducibility(adapter, assets, n=5).flags) or "reproducibility: clean")
print("\n".join(audit_conflict(adapter, assets, n_consistency=5).flags) or "conflict: clean")
```

For robustness, the adapter must hold captures for each perturbed asset id; then
run `audit_robustness(adapter, assets)` with the default perturbations (drop
`paraphrase` unless captured). Because the adapter serves whatever you captured,
the audit runs **identically** to how it runs against the built-in reference
agent — the whole point of the adapter.

## 6. What you get

The same reproducibility / robustness / conflict metrics and per-verdict flags as
the reference-agent run, now computed on the live playground's real verdicts:
score dispersion, recommendation flip-rate, rationale-citation stability, drift
under semantics-preserving edits, conflict acknowledgment, and positional
anchoring — the production-usable trust gate, applied to their own agent.
