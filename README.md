# kavuru-convexia

A reliability audit harness for LLM-agent drug-asset evaluations. Given an
agent's verdict on a drug asset — a probability-of-success (PoS) score, a
go/no-go recommendation, and a rationale — it measures whether that verdict is
**reproducible, calibrated, robust, and honest about conflicting evidence**.

> This README is finalized in the last build step with real numbers from
> `make demo`. See the build brief and `docs/` for the product proposal.

## Quickstart

```bash
make setup          # build .venv and install the package
export ANTHROPIC_API_KEY=...   # required for the reference agent
make demo           # run the full audit -> figures + report + notebook in outputs/
make test           # run the test suite (offline, no network)
```
