# Verdict reliability report — ReferenceAgent(claude-sonnet-5)

- **Model:** `claude-sonnet-5`
- **Assets audited:** 20
- **Overall status:** **FAIL**  (reliability score **0.92** over the production-usable checks)
- **Generated:** 2026-07-17T18:43:43

## Audit summary

| audit | status | score | production-usable |
|---|---|---|---|
| reproducibility | WARN | 0.96 | yes |
| robustness | FAIL | 0.86 | yes |
| conflict | PASS | 0.93 | yes |
| calibration | WARN | 0.43 | no (needs labels) |

![reproducibility](../figures/reproducibility_variance.png)

![robustness](../figures/robustness_drift.png)

![conflict](../figures/conflict_table.png)

![calibration](../figures/calibration_curve.png)

## Headline flags

- reproducibility: [warn] verubecestat: PoS std 0.052
- robustness: [fail] preladenant: +0.10 PoS drift under reorder
- robustness: [warn] osimertinib-tagrisso: +0.10 PoS drift under reorder
- robustness: [fail] SYN-CONFLICT-EFFICACY-TOX: +0.07 PoS drift under neutralize; recommendation changed under 2 perturbation(s)
- calibration: [warn] ECE 0.113 (bins=10); predicted confidence deviates from observed outcome frequency

## Per-verdict reliability

| asset | kind | reliability | status | key flags |
|---|---|---|---|---|
| `verubecestat` | historical | 0.79 | WARN | PoS dispersion across runs (std 0.052); +0.07 PoS drift under reformat |
| `osimertinib-tagrisso` | historical | 0.84 | WARN | +0.10 PoS drift under reorder |
| `SYN-CONFLICT-EFFICACY-TOX` | synthetic | 0.86 | FAIL | +0.07 PoS drift under neutralize; recommendation changed under 2 semantics-preserving edit(s) |
| `SYN-CONFLICT-SCIENCE-IP` | synthetic | 0.87 | WARN | +0.06 PoS drift under reorder |
| `torcetrapib` | historical | 0.89 | PASS | — |
| `preladenant` | historical | 0.89 | FAIL | +0.10 PoS drift under reorder |
| `SYN-CONFLICT-EFFICACY-ADME` | synthetic | 0.90 | PASS | — |
| `semaglutide-ozempic` | historical | 0.91 | PASS | — |
| `sofosbuvir-sovaldi` | historical | 0.92 | PASS | — |
| `SYN-CONFLICT-LATE-TOX` | synthetic | 0.92 | PASS | — |
| `bapineuzumab` | historical | 0.94 | PASS | — |
| `nirmatrelvir-paxlovid` | historical | 0.94 | PASS | — |
| `SYN-BORDERLINE-BALANCED` | synthetic | 0.95 | PASS | — |
| `pembrolizumab-keytruda` | historical | 0.95 | PASS | — |
| `semagacestat` | historical | 0.96 | PASS | — |
| `SYN-BORDERLINE-THIN` | synthetic | 0.96 | WARN | +0.07 PoS drift under reorder |
| `SYN-CONTROL-CLEAN-ADVANCE` | synthetic | 0.97 | PASS | — |
| `fialuridine-fiau` | historical | 0.98 | PASS | — |
| `SYN-CONTROL-CLEAN-PASS` | synthetic | 0.99 | PASS | — |
| `imatinib-gleevec-cml-2001` | historical | 1.00 | PASS | — |

## Calibration note (offline)

> Curated set is balanced (~50% success); calibration is judged against its own base rate. The published pipeline base rate (7.9%) is an external anchor for what a production PoS distribution should be sanity-checked against, not a target for this balanced set.

Source for the pipeline base rate: BIO/Informa/QLS, Clinical Development Success Rates 2011-2020.
