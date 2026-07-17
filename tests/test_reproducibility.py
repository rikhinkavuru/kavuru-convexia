"""The reproducibility audit must fire on an unstable evaluator, not a stable one."""
from __future__ import annotations

from _stubs import ConstantStub, UnstableStub

from kavuru_convexia import config
from kavuru_convexia.assets import build_synthetic_assets
from kavuru_convexia.audits.reproducibility import audit_reproducibility

ASSETS = build_synthetic_assets()[:3]


def test_stable_evaluator_passes():
    result = audit_reproducibility(ConstantStub(), ASSETS, n=6)
    assert result.status == "pass"
    assert result.metrics["flip_rate_mean"] == 0.0
    assert result.metrics["pos_std_mean"] < 1e-9  # identical runs (fp dust from np.std)
    assert result.score > 0.95


def test_unstable_evaluator_fails():
    result = audit_reproducibility(UnstableStub(), ASSETS, n=8)
    assert result.status == "fail"
    # A maximally unstable stub should trip the flip-rate and dispersion thresholds.
    assert result.metrics["flip_rate_max"] >= config.FLIP_RATE_FAIL
    assert result.metrics["pos_std_max"] >= config.POS_STD_FAIL
    assert result.flags  # human-readable flags emitted
    assert result.score < 0.5


def test_perfect_rationale_stability_when_citations_constant():
    result = audit_reproducibility(ConstantStub(), ASSETS, n=5)
    assert result.metrics["rationale_jaccard_mean"] == 1.0
