"""Figures and rendered reports for a VerdictReliabilityReport.

All plotting uses the non-interactive Agg backend so figures render headless
(CI, notebooks, ``make demo``). Each figure function writes a PNG and returns its
path; :func:`render_markdown_report` embeds them into a reviewer-facing report.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless; must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402

from . import config  # noqa: E402
from .audits.report_types import CheckResult, VerdictReliabilityReport  # noqa: E402
from .logutil import get_logger  # noqa: E402

logger = get_logger(__name__)

_STATUS_COLOR = {"pass": "#2e7d32", "warn": "#f39c12", "fail": "#c0392b"}
_ACCENT = "#2c3e50"


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight", "axes.titleweight": "bold"})


def _save(fig: plt.Figure, path: Path | str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_reproducibility_variance(repro: CheckResult, path: Path | str) -> Path:
    """Per-asset PoS spread across repeated runs, sorted by dispersion."""
    set_style()
    per = repro.detail["per_asset"]
    items = sorted(per.items(), key=lambda kv: kv[1]["pos_std"], reverse=True)
    labels = [aid for aid, _ in items]
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(labels) + 1.5))
    for y, (_, d) in enumerate(items):
        scores = d.get("scores", [d["pos_mean"]])
        ax.plot([min(scores), max(scores)], [y, y], color="#bdc3c7", lw=3, zorder=1, solid_capstyle="round")
        ax.scatter(scores, [y] * len(scores), color=_STATUS_COLOR[d["status"]], s=22, zorder=2, alpha=0.8)
        ax.scatter([d["pos_mean"]], [y], color=_ACCENT, marker="|", s=260, zorder=3, lw=2)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("Probability-of-success (PoS) across repeated runs")
    ax.set_title(f"Reproducibility: PoS dispersion per asset  "
                 f"(mean std {repro.metrics['pos_std_mean']:.3f}, "
                 f"max flip-rate {repro.metrics['flip_rate_max']:.0%})")
    ax.invert_yaxis()
    return _save(fig, path)


def fig_reliability_curve(cal: CheckResult, path: Path | str) -> Path:
    """Calibration reliability curve with base-rate anchors and headline metrics."""
    set_style()
    curve = cal.detail["reliability_curve"]
    fig, ax = plt.subplots(figsize=(6.2, 6))
    ax.plot([0, 1], [0, 1], ls="--", color="#95a5a6", label="perfect calibration")
    if curve:
        xs = [b["mean_pred"] for b in curve]
        ys = [b["observed"] for b in curve]
        sizes = [40 + 60 * b["count"] for b in curve]
        ax.plot(xs, ys, "-o", color=_ACCENT, markersize=0, lw=1.5, alpha=0.7)
        ax.scatter(xs, ys, s=sizes, color=_ACCENT, alpha=0.85, zorder=3, label="observed (size ~ n)")
    ax.axhline(cal.metrics["empirical_base_rate"], color="#2980b9", ls=":", lw=1.3,
               label=f"set base rate {cal.metrics['empirical_base_rate']:.2f}")
    ax.axvline(cal.metrics["published_base_rate"], color="#8e44ad", ls=":", lw=1.3,
               label=f"pipeline base rate {cal.metrics['published_base_rate']:.1%}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted PoS")
    ax.set_ylabel("Observed success frequency")
    ax.set_title(f"Calibration (offline): ECE {cal.metrics['ece']:.3f} · "
                 f"Brier {cal.metrics['brier']:.3f} · AUROC {cal.metrics['auroc']:.2f}")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    return _save(fig, path)


def fig_robustness_drift(robust: CheckResult, path: Path | str) -> Path:
    """Mean PoS drift per perturbation type, with WARN/FAIL thresholds."""
    set_style()
    perts = robust.detail["perturbations"]
    means = [robust.metrics.get(f"mean_drift__{p}", 0.0) for p in perts]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars = ax.bar(perts, means, color=_ACCENT, alpha=0.85)
    ax.axhline(config.POS_DRIFT_WARN, color="#f39c12", ls="--", lw=1.2, label=f"warn {config.POS_DRIFT_WARN}")
    ax.axhline(config.POS_DRIFT_FAIL, color="#c0392b", ls="--", lw=1.2, label=f"fail {config.POS_DRIFT_FAIL}")
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m, f"{m:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Mean |Δ PoS| under perturbation")
    ax.set_title(f"Robustness: verdict drift under semantics-preserving edits  "
                 f"(recommendation-change rate {robust.metrics['rec_change_rate']:.0%})")
    ax.legend(fontsize=8)
    return _save(fig, path)


def fig_conflict_table(conflict: CheckResult, path: Path | str) -> Path:
    """Conflict-handling table: acknowledgment, anchoring, consistency per asset."""
    set_style()
    per = conflict.detail["per_asset"]
    rows, cell_colors = [], []
    for aid, d in per.items():
        rows.append([
            aid,
            "yes" if d["acknowledges_conflict"] else "NO",
            f"{d['anchoring_swing']:.2f}",
            f"{d['consistency_flip_rate']:.2f}",
            d["status"].upper(),
        ])
        cell_colors.append(["white",
                            "#d5f5e3" if d["acknowledges_conflict"] else "#f5b7b1",
                            "white", "white",
                            _STATUS_COLOR[d["status"]] + "33"])
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(rows) + 1.4))
    ax.axis("off")
    cols = ["asset", "acknowledges", "anchoring swing", "consistency flip", "status"]
    tbl = ax.table(cellText=rows, colLabels=cols, cellColours=cell_colors, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.5)
    for c in range(len(cols)):
        tbl[0, c].set_facecolor(_ACCENT)
        tbl[0, c].set_text_props(color="white", fontweight="bold")
    ax.set_title(f"Conflict handling  (acknowledgment rate {conflict.metrics['acknowledgment_rate']:.0%}, "
                 f"max anchoring swing {conflict.metrics['max_anchoring_swing']:.2f})", pad=18)
    return _save(fig, path)


def fig_agent_comparison(reports: dict[str, VerdictReliabilityReport], path: Path | str) -> Path:
    """Grouped bars comparing reliability sub-scores across candidate agents."""
    set_style()
    dims = ["reproducibility", "robustness", "conflict"]
    labels = list(reports)
    x = np.arange(len(dims))
    width = 0.8 / max(1, len(labels))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    palette = sns.color_palette("deep", len(labels))
    for i, label in enumerate(labels):
        scores = [reports[label].checks[d].score if d in reports[label].checks else 0.0 for d in dims]
        ax.bar(x + i * width, scores, width, label=label, color=palette[i], alpha=0.9)
    ax.set_xticks(x + width * (len(labels) - 1) / 2)
    ax.set_xticklabels(dims)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("reliability sub-score (1 = fully reliable)")
    ax.set_title("Reliability by dimension across candidate agents")
    ax.legend(fontsize=9)
    return _save(fig, path)


def save_report_figures(report: VerdictReliabilityReport, figures_dir: Path | str) -> dict[str, Path]:
    """Generate every figure for a report; returns {name: path}. Missing checks skipped."""
    figures_dir = Path(figures_dir)
    figs: dict[str, Path] = {}
    if "reproducibility" in report.checks:
        figs["reproducibility"] = fig_reproducibility_variance(
            report.checks["reproducibility"], figures_dir / "reproducibility_variance.png")
    if "robustness" in report.checks:
        figs["robustness"] = fig_robustness_drift(
            report.checks["robustness"], figures_dir / "robustness_drift.png")
    if "conflict" in report.checks:
        figs["conflict"] = fig_conflict_table(
            report.checks["conflict"], figures_dir / "conflict_table.png")
    if "calibration" in report.checks:
        figs["calibration"] = fig_reliability_curve(
            report.checks["calibration"], figures_dir / "calibration_curve.png")
    logger.info("wrote %d figures to %s", len(figs), figures_dir)
    return figs


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
def _entries_table(report: VerdictReliabilityReport) -> list[str]:
    lines = ["| asset | kind | reliability | status | key flags |",
             "|---|---|---|---|---|"]
    for e in sorted(report.entries, key=lambda e: e.reliability_score):
        flags = "; ".join(e.flags[:2]) if e.flags else "—"
        lines.append(f"| `{e.asset_id}` | {e.kind} | {e.reliability_score:.2f} "
                     f"| {e.status.upper()} | {flags} |")
    return lines


def render_markdown_report(
    report: VerdictReliabilityReport, figures: Optional[dict[str, Path]] = None,
    *, figures_relative_to: Optional[Path] = None,
) -> str:
    """Render a reviewer-facing markdown report, embedding figures if provided."""
    def rel(p: Path) -> str:
        if figures_relative_to is not None:
            return os.path.relpath(Path(p), figures_relative_to)
        return str(p)

    L: list[str] = []
    L.append(f"# Verdict reliability report — {report.evaluator_name}")
    L.append("")
    L.append(f"- **Model:** `{report.model}`")
    L.append(f"- **Assets audited:** {report.n_assets}")
    L.append(f"- **Overall status:** **{report.overall_status.upper()}**  "
             f"(reliability score **{report.reliability_score:.2f}** over the production-usable checks)")
    L.append(f"- **Generated:** {report.created}")
    L.append("")

    L.append("## Audit summary")
    L.append("")
    L.append("| audit | status | score | production-usable |")
    L.append("|---|---|---|---|")
    for name, chk in report.checks.items():
        L.append(f"| {name} | {chk.status.upper()} | {chk.score:.2f} | "
                 f"{'yes' if chk.production_usable else 'no (needs labels)'} |")
    L.append("")

    for name, chk in report.checks.items():
        if figures and name in figures:
            L.append(f"![{name}]({rel(figures[name])})")
            L.append("")

    if report.headline_flags:
        L.append("## Headline flags")
        L.append("")
        for f in report.headline_flags:
            L.append(f"- {f}")
        L.append("")

    L.append("## Per-verdict reliability")
    L.append("")
    L.extend(_entries_table(report))
    L.append("")

    if "calibration" in report.checks:
        cal = report.checks["calibration"]
        L.append("## Calibration note (offline)")
        L.append("")
        L.append(f"> {cal.detail['note']}")
        L.append("")
        L.append(f"Source for the pipeline base rate: {cal.detail['base_rate_source']}.")
        L.append("")

    return "\n".join(L)


def save_markdown_report(
    report: VerdictReliabilityReport, out_dir: Path | str, *, figures: Optional[dict[str, Path]] = None,
) -> tuple[Path, Path]:
    """Write report.json and report.md (with embedded figures) into out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = report.to_json(out_dir / "reliability_report.json")
    md = render_markdown_report(report, figures, figures_relative_to=out_dir)
    md_path = out_dir / "reliability_report.md"
    md_path.write_text(md, encoding="utf-8")
    logger.info("wrote report to %s and %s", md_path, json_path)
    return md_path, json_path
