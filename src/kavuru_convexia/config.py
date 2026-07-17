"""Central configuration: seeds, model constants, thresholds, and paths.

This is the single source of truth for every tunable in the harness. The
repo-root ``config.py`` re-exports these names for convenience — importing from
either works. Every threshold carries a one-line rationale so a reviewer can
audit the choice rather than trust a magic number.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED: int = 7  # global seed for numpy/random; keeps the demo + tests deterministic


def set_global_seeds(seed: int = SEED) -> None:
    """Seed the Python and NumPy RNGs for a reproducible run."""
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# LLM / evaluator
# ---------------------------------------------------------------------------
# Reference-agent model. Overridable via env so the harness stays model-agnostic
# and a reviewer can point it at whatever model their key can reach.
ANTHROPIC_MODEL: str = os.environ.get("KAVURU_MODEL", "claude-sonnet-5")
# Cheaper model for the LLM-as-judge conflict-acknowledgment check.
JUDGE_MODEL: str = os.environ.get("KAVURU_JUDGE_MODEL", "claude-haiku-4-5-20251001")
MAX_TOKENS: int = 1024
# A non-zero temperature is REQUIRED for the reproducibility audit: it is exactly
# the run-to-run non-determinism that a capital-allocation verdict must not have,
# so we measure it head-on rather than hiding it behind temperature=0.
REPRO_TEMPERATURE: float = 0.7
# Repetitions per asset for the reproducibility / conflict-consistency audits.
N_REPETITIONS: int = 8

# Offline mode swaps the API for a deterministic stub agent (used by the test
# suite and by anyone without a key). Never let offline runs masquerade as real.
OFFLINE: bool = os.environ.get("KAVURU_OFFLINE", "0") == "1"

# ---------------------------------------------------------------------------
# Thresholds — reproducibility (no labels required; production-usable)
# ---------------------------------------------------------------------------
# PoS lives in [0, 1]. A standard deviation above these under *identical* inputs
# is unacceptable dispersion for a score that gates millions in capital.
POS_STD_WARN: float = 0.05
POS_STD_FAIL: float = 0.10
# Fraction of runs whose recommendation disagrees with the modal recommendation.
FLIP_RATE_WARN: float = 0.10
FLIP_RATE_FAIL: float = 0.25
# Mean pairwise Jaccard overlap of cited-evidence sets across runs; below this the
# stated rationale is not stable even when the score happens to be.
RATIONALE_JACCARD_WARN: float = 0.60

# ---------------------------------------------------------------------------
# Thresholds — robustness (no labels required; production-usable)
# ---------------------------------------------------------------------------
# |Δ PoS| under a semantics-preserving edit that must not change the answer.
POS_DRIFT_WARN: float = 0.05
POS_DRIFT_FAIL: float = 0.10

# ---------------------------------------------------------------------------
# Thresholds — calibration (labels REQUIRED; offline validation only)
# ---------------------------------------------------------------------------
CALIBRATION_N_BINS: int = 10  # equal-width bins for the reliability curve + ECE
ECE_WARN: float = 0.10
ECE_FAIL: float = 0.20

# ---------------------------------------------------------------------------
# Thresholds — conflict handling (no labels required; production-usable)
# ---------------------------------------------------------------------------
# On conflicted assets, a verdict that tracks snippet ORDER rather than substance
# is anchoring. A position-driven PoS swing above this fails the anchoring test.
ANCHORING_POS_SWING_WARN: float = 0.05
ANCHORING_POS_SWING_FAIL: float = 0.10

# ---------------------------------------------------------------------------
# Aggregate reliability score — weights over the three production-usable audits
# (calibration is excluded because it needs labels and cannot run in production).
# Weights sum to 1.0; reproducibility is weighted highest as the headline signal.
# ---------------------------------------------------------------------------
WEIGHT_REPRODUCIBILITY: float = 0.40
WEIGHT_ROBUSTNESS: float = 0.30
WEIGHT_CONFLICT: float = 0.30

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# config.py lives at src/kavuru_convexia/config.py -> parents[2] is the repo root.
REPO_ROOT: Path = Path(__file__).resolve().parents[2]
CACHE_DIR: Path = Path(os.environ.get("KAVURU_CACHE_DIR", str(REPO_ROOT / ".cache")))
OUTPUTS_DIR: Path = REPO_ROOT / "outputs"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"
REPORTS_DIR: Path = OUTPUTS_DIR / "reports"
DOCS_DIR: Path = REPO_ROOT / "docs"  # tracked copies of canonical figures for the README


def ensure_dirs() -> None:
    """Create all output/cache directories if they do not exist."""
    for directory in (CACHE_DIR, OUTPUTS_DIR, FIGURES_DIR, REPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
