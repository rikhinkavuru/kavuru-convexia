# Methodology

How `kavuru-convexia` measures the reliability of an LLM-agent drug-asset verdict,
and — because a measurement instrument should quantify its own uncertainty — how it
puts confidence intervals on those measurements.

The premise (following the measurement-reliability framing of the accompanying
INFORMS work) is that an evaluative verdict is an **estimate with variance**, not a
fact. So the harness treats every summary statistic as an estimate and reports it
with a bootstrap CI, and it draws a hard line between checks that need ground truth
and checks that do not.

---

## 1. The unit of analysis

A **verdict** is `(pos_score ∈ [0,1], recommendation ∈ {advance, pass, investigate},
rationale, cited_evidence_ids)` produced by an `AssetEvaluator` for an **asset**
(a description + typed evidence snippets). Every audit operates on verdicts, so any
evaluator — the bundled `ReferenceAgent` or an `ExternalAdapter` replaying captured
playground verdicts — is audited identically.

## 2. The audits

| Audit | Question | Per-asset statistic | Labels? | In the gate? |
|---|---|---|---|---|
| Reproducibility | Same input, same verdict? | PoS std, recommendation flip-rate, cited-evidence Jaccard across N runs | no | yes |
| Robustness | Semantics-preserving edits don't move it? | max/mean \|ΔPoS\|, recommendation change under reorder / neutralize / reformat / paraphrase | no | yes |
| Conflict | Contradictions handled honestly? | acknowledgment (judge panel), cross-run consistency, positional anchoring | no | yes |
| Evidence sensitivity | Does the verdict hinge on one snippet? | max single-evidence PoS influence, single-point-of-failure | no | **no** (importance, not stability) |
| Calibration | Do PoS scores match outcomes? | ECE, Brier, AUROC vs. realized outcomes | **yes** | **no** (offline) |

**Why three of five run in production.** Reproducibility, robustness, and conflict
interrogate *behavior* (re-run, perturb, reorder) and never touch the outcome, so
they can gate a fresh verdict. Calibration compares to realized outcomes and is
therefore an offline validation only. Evidence sensitivity is production-usable but
measures *importance*, not stability, so it is reported beside the gate, not inside it.

## 3. Noise control

Modern models are near-deterministic, so a naive drift measurement conflates a real
effect with run-to-run noise. Each audit establishes a noise floor:

- **Robustness** takes two base evaluations per asset; the native noise is
  `|pos_base0 − pos_base1|`, and a perturbation's drift counts only if it exceeds
  `max(WARN_threshold, 2·noise)`. A recommendation change is attributed to the edit
  only when the base recommendation is itself stable across the two base samples.
- **Conflict anchoring** compares the mean PoS with the conflicting evidence pair in
  original vs. swapped position; the swing must exceed `max(WARN, 2·σ_runs)` (σ from
  the consistency runs), and an order-driven recommendation change is counted only
  when the consistency flip-rate is below the warning threshold.
- **Evidence sensitivity** uses `k` base evaluations to estimate σ and a
  Bonferroni-widened band `σ·Φ⁻¹(1 − 0.05/2S)` across the `S` snippets, floored so a
  deterministic evaluator is not flagged on trivial deltas.

## 4. Bootstrap confidence intervals

Percentile bootstrap, `B = 2000`, seeded. The resampling unit is chosen to match the
estimand ("generalize to a new asset"), not convenience:

- **Reproducibility — hierarchical (two-stage).** Resample the assets (outer) *and*
  the N runs within each drawn asset (inner). A per-asset std or flip-rate estimated
  from only N=8 runs is itself noisy; the inner resample propagates that within-asset
  sampling error, which an assets-only bootstrap would understate.
- **Robustness — cluster bootstrap on assets.** Each asset's fixed perturbation
  vector is kept intact; the perturbations are a designed factor, not a sample, so
  they are not resampled.
- **Calibration — case bootstrap on assets.** Brier is a smooth mean and its CI is
  trustworthy. AUROC drops single-class resamples (undefined) and the drop fraction
  is reported; ECE is a non-smooth binned plug-in — both are flagged unstable at this n.
- **Conflict — no CI.** The conflicted subset is tiny (n≈4); a bootstrap there
  fabricates precision the data do not contain, so the raw per-asset values are
  reported with an explicit "descriptive only, no CI" caveat.

Percentile endpoints are clipped to each metric's valid range so a bounded statistic
never reports an impossible interval, and every CI is reported alongside its `n`.

## 5. The conflict acknowledgment judge panel

Whether a rationale "acknowledges the conflict" is judged by an LLM — which has its
own non-determinism, the very thing this project studies. Because a near-deterministic
model gives near-identical answers on repeats, the panel varies the **rubric**, not
the sample: three judges (strict / neutral / lenient) vote, 2-of-3 majority decides,
and the **2-1 disagreement rate** is reported as a judge-framing-sensitivity signal.
The strict→lenient acknowledgment range brackets how much the conclusion depends on
where the bar is set. A single-judge fallback and a keyword heuristic keep the audit
running when the panel model is unavailable.

## 6. Aggregation

Each asset's verdict gets a reliability score = a weighted mean of the available
production-usable sub-scores (reproducibility 0.40, robustness 0.30, conflict 0.30;
renormalized when an asset carries no planted conflict). The report's aggregate score
is the **mean of the per-verdict scores** (coverage-correct — the conflict dimension
only covers the conflicted subset), carried with its own bootstrap CI. The overall
gate status is the worst status among the three stability dimensions; calibration and
evidence sensitivity never drive it.

## 7. Honesty boundaries

- **Calibration is offline.** It needs outcome labels and cannot gate a live verdict.
- **Memorization.** The historical drugs are famous, so a high AUROC may reflect the
  model recognizing known outcomes. The harness re-runs calibration with each drug's
  identity stripped and reports the AUROC change; but mechanism + indication still
  identify iconic drugs, so name-blinding is necessary, not sufficient — a clean read
  needs held-out assets.
- **Sensitivity ≠ robustness.** Removing evidence changes meaning; single-snippet
  removal cannot detect a redundant two-key failure, so "no single point of failure"
  is not "no fragility".
- **Curated evidence.** Snippets are faithful pre-decision reconstructions, not raw
  trial data; only the outcomes are hard, sourced facts.

## 8. Reproducibility of the harness itself

Global seeds; every LLM call cached to disk by prompt hash (so a re-run replays
byte-for-byte); the bootstrap RNG is seeded; the offline stub makes the whole pipeline
and the test suite runnable without a key. Every number in the README and the sample
report is produced by `make demo`, never hand-entered.
