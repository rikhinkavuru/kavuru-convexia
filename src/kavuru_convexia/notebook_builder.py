"""End-to-end demo: run the full audit and build/execute the story notebook.

``run_demo`` performs the real audit in-process (primary agent over all assets +
a cross-agent comparison), writes figures and the reliability report to
``outputs/``, and returns the report. ``main`` additionally assembles a readable
Jupyter notebook that walks the whole story and executes it (replaying the warm
cache) into ``outputs/demo.ipynb``. This module is the ``make demo`` entry point.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

from . import config, reporting
from .assets import build_synthetic_assets, load_historical_assets
from .audits import audit_agent
from .audits.conflict import make_llm_ack_judge
from .audits.report_types import VerdictReliabilityReport
from .audits.robustness import make_llm_paraphraser
from .config import ensure_dirs, set_global_seeds
from .evaluator import ReferenceAgent
from .llm import LLMClient
from .logutil import get_logger

logger = get_logger(__name__)


def run_demo(
    *, n: int = config.N_REPETITIONS, cross_agent: bool = True,
) -> tuple[VerdictReliabilityReport, dict[str, VerdictReliabilityReport]]:
    """Run the full reliability audit and write all artifacts to ``outputs/``."""
    set_global_seeds()
    ensure_dirs()

    historical = load_historical_assets()
    synthetic = build_synthetic_assets()
    all_assets = historical + synthetic

    agent = ReferenceAgent(model=config.ANTHROPIC_MODEL)
    judge_client = LLMClient(model=config.JUDGE_MODEL)
    ack_judge = make_llm_ack_judge(judge_client)
    # Paraphrase perturbation needs a real model; skip it when offline.
    paraphraser = None if judge_client.offline else make_llm_paraphraser(judge_client)

    logger.info("running primary audit: %s over %d assets", agent.name, len(all_assets))
    report = audit_agent(
        agent, all_assets, calibration_assets=historical, blind_calibration=True, n=n,
        ack_judge=ack_judge, paraphraser=paraphraser,
    )
    figures = reporting.save_report_figures(report, config.FIGURES_DIR)
    reporting.save_markdown_report(report, config.REPORTS_DIR, figures=figures)

    # A cross-agent comparison is only meaningful with a real model: offline, both
    # "agents" share the identical deterministic stub, so the ranking is a tie.
    comparison: dict[str, VerdictReliabilityReport] = {}
    if cross_agent and not agent.client.offline:
        # Fair head-to-head on the synthetic set (the primary agent's synthetic
        # runs are already cached from the full audit; only the second agent is new).
        primary_syn = audit_agent(agent, synthetic, n=n, ack_judge=ack_judge, paraphraser=paraphraser)
        second = ReferenceAgent(model=config.JUDGE_MODEL)
        second_syn = audit_agent(second, synthetic, n=n, ack_judge=ack_judge, paraphraser=paraphraser)
        comparison = {agent.model: primary_syn, second.model: second_syn}
        reporting.fig_agent_comparison(comparison, config.FIGURES_DIR / "agent_comparison.png")

    return report, comparison


# ---------------------------------------------------------------------------
# Notebook assembly
# ---------------------------------------------------------------------------
_INTRO_MD = """\
# kavuru-convexia — reliability audit of an LLM drug-asset agent

This notebook runs the full harness end to end: it builds the asset set, points a
reference LLM agent at it, and audits the agent's verdicts for **reproducibility**,
**robustness**, and **conflict-handling** (all ground-truth-free, hence usable in
production), plus **calibration** against historical outcomes (offline, since it
needs labels). It renders the figures and a per-verdict reliability report.

All numbers come from a real run against the configured model; there is nothing
hard-coded here.
"""

_CELLS: list[tuple[str, str]] = [
    ("md", _INTRO_MD),
    ("code", """\
import warnings; warnings.filterwarnings("ignore")
from IPython.display import Image, display
from kavuru_convexia import config, reporting
from kavuru_convexia.config import set_global_seeds, ensure_dirs
from kavuru_convexia.assets import load_historical_assets, build_synthetic_assets
from kavuru_convexia.evaluator import ReferenceAgent
from kavuru_convexia.llm import LLMClient
from kavuru_convexia.audits import audit_agent
from kavuru_convexia.audits.conflict import make_llm_ack_judge
from kavuru_convexia.audits.robustness import make_llm_paraphraser

set_global_seeds(); ensure_dirs()
historical = load_historical_assets()
synthetic = build_synthetic_assets()
all_assets = historical + synthetic
print(f"{len(historical)} historical (labeled) + {len(synthetic)} synthetic assets")"""),
    ("md", "## The reference agent\\n\\nA structured-verdict LLM agent standing in for a "
           "production PoS agent. A cheaper model backs the conflict-acknowledgment judge "
           "and the paraphrase perturbation."),
    ("code", """\
agent = ReferenceAgent(model=config.ANTHROPIC_MODEL)
judge_client = LLMClient(model=config.JUDGE_MODEL)
ack_judge = make_llm_ack_judge(judge_client)
paraphraser = None if judge_client.offline else make_llm_paraphraser(judge_client)
print("reference:", agent.name)"""),
    ("md", "## Run the audit\\n\\nReproducibility / robustness / conflict are label-free; "
           "calibration uses the historical outcomes and is marked offline."),
    ("code", """\
report = audit_agent(
    agent, all_assets, calibration_assets=historical, blind_calibration=True,
    n=config.N_REPETITIONS, ack_judge=ack_judge, paraphraser=paraphraser,
)
print("\\n".join(report.summary_lines()))"""),
    ("md", "## Figures"),
    ("code", """\
figures = reporting.save_report_figures(report, config.FIGURES_DIR)
for name in ("reproducibility", "robustness", "conflict", "calibration", "calibration_blinded"):
    if name in figures:
        display(Image(str(figures[name])))"""),
    ("md", "## Memorization check\\n\\nRe-score the historical drugs with their identity "
           "stripped. A large AUROC drop would mean the agent recognizes famous outcomes "
           "rather than reasoning from the evidence."),
    ("code", """\
cal, blind = report.checks.get("calibration"), report.checks.get("calibration_blinded")
if cal and blind:
    print(f"revealed : AUROC {cal.metrics['auroc']:.2f}  ECE {cal.metrics['ece']:.3f}")
    print(f"blinded  : AUROC {blind.metrics['auroc']:.2f}  ECE {blind.metrics['ece']:.3f}")
    print(f"AUROC change when blinded: {cal.metrics['auroc'] - blind.metrics['auroc']:+.2f}")"""),
    ("md", "## Per-verdict reliability\\n\\nEach asset's verdict gets a reliability score, a "
           "status, and human-readable flags — the actionable output of the gate."),
    ("code", """\
import pandas as pd
rows = [{"asset": e.asset_id, "kind": e.kind, "reliability": e.reliability_score,
         "status": e.status, "flags": "; ".join(e.flags) or "-"} for e in report.entries]
pd.set_option("display.max_colwidth", 80)
pd.DataFrame(rows).sort_values("reliability").reset_index(drop=True)"""),
    ("md", "## Cross-agent comparison\\n\\nThe gate's real job: rank candidate agents by "
           "reliability. Here the reference model vs. a cheaper one on the synthetic set."),
    ("code", """\
if agent.client.offline:
    print("Offline mode: skipping cross-agent comparison (both agents share the same stub).")
else:
    primary_syn = audit_agent(agent, synthetic, n=config.N_REPETITIONS, ack_judge=ack_judge, paraphraser=paraphraser)
    second = ReferenceAgent(model=config.JUDGE_MODEL)
    second_syn = audit_agent(second, synthetic, n=config.N_REPETITIONS, ack_judge=ack_judge, paraphraser=paraphraser)
    comparison = {agent.model: primary_syn, second.model: second_syn}
    display(Image(str(reporting.fig_agent_comparison(comparison, config.FIGURES_DIR / "agent_comparison.png"))))
    for m, r in comparison.items():
        print(f"{m}: reliability {r.reliability_score:.2f} ({r.overall_status})")"""),
    ("md", "## Save the report"),
    ("code", """\
md_path, json_path = reporting.save_markdown_report(report, config.REPORTS_DIR, figures=figures)
print("report:", md_path)
print("json:  ", json_path)"""),
]


def build_notebook():
    """Assemble (but do not execute) the demo notebook as an nbformat node."""
    import nbformat
    from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

    cells = [
        new_markdown_cell(src) if kind == "md" else new_code_cell(src)
        for kind, src in _CELLS
    ]
    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": _KERNEL_NAME, "display_name": "kavuru-convexia", "language": "python"}
    return nb


_KERNEL_NAME = "kavuru-convexia"


def _ensure_kernel() -> None:
    """Register a venv-local kernelspec pointing at THIS interpreter (idempotent).

    We register our own named spec rather than reuse a discovered `python3` — that
    could be a global kernel whose Python lacks this package.
    """
    subprocess.run(
        [sys.executable, "-m", "ipykernel", "install", "--sys-prefix",
         "--name", _KERNEL_NAME, "--display-name", "kavuru-convexia"],
        check=True, capture_output=True,
    )


def execute_notebook(out_path: Path | str, *, timeout: int = 1800) -> Path:
    """Build, execute, and write the demo notebook (replays the warm cache)."""
    import nbformat
    from nbclient import NotebookClient

    _ensure_kernel()
    nb = build_notebook()
    client = NotebookClient(nb, timeout=timeout, kernel_name=_KERNEL_NAME,
                            resources={"metadata": {"path": str(config.REPO_ROOT)}})
    client.execute()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, out_path)
    return out_path


def main() -> None:
    ensure_dirs()
    report, comparison = run_demo()
    print("\n".join(report.summary_lines()))
    for m, r in comparison.items():
        print(f"  cross-agent {m}: reliability {r.reliability_score:.2f} ({r.overall_status})")
    try:
        nb_path = execute_notebook(config.OUTPUTS_DIR / "demo.ipynb")
        print(f"executed notebook -> {nb_path}")
    except Exception as exc:  # noqa: BLE001 — report artifacts already saved; notebook is a bonus
        logger.warning("notebook execution skipped (%s: %s)", type(exc).__name__, exc)
    print(f"report -> {config.REPORTS_DIR / 'reliability_report.md'}")


if __name__ == "__main__":
    main()
