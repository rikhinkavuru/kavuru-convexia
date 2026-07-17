"""kavuru-convexia — a reliability audit harness for LLM-agent drug-asset verdicts.

Given an agent's verdict on a drug asset (a probability-of-success score, a
go/no-go recommendation, and a rationale), this package measures whether that
verdict is *reproducible*, *calibrated*, *robust*, and *honest about conflicting
evidence* — the failure modes that make an agentic acquisition verdict unsafe to
act on. See :mod:`kavuru_convexia.audits` for the individual audit modules.
"""
from __future__ import annotations

__version__ = "0.1.0"
