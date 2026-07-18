"""Command-line interface: ``kavuru-convexia <command>``.

    audit   Run the production-usable audits over captured verdicts (JSON/CSV)
    demo    Run the full reference-agent demo (needs ANTHROPIC_API_KEY)
    assets  List the bundled assets

``audit`` is the production entry point: it needs no API key and no labels — it
replays externally-captured verdicts (e.g. from a live playground) through the
reproducibility / robustness / conflict audits, reporting each with bootstrap CIs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__, config
from .assets import all_assets
from .audits import audit_conflict, audit_reproducibility, audit_robustness
from .audits.report_types import CheckResult
from .audits.stats import fmt_ci
from .evaluator import ExternalAdapter
from .logutil import get_logger

logger = get_logger(__name__)


def _print_check(chk: CheckResult) -> None:
    tag = "" if chk.production_usable else " [offline]"
    print(f"\n[{chk.status.upper()}] {chk.name}  (score {chk.score:.2f}){tag}")
    for key, val in chk.metrics.items():
        ci = chk.metrics_ci.get(key)
        suffix = f"  95% CI {fmt_ci(ci)}" if ci else ""
        print(f"    {key:26s} {val:.4f}{suffix}")
    for flag in chk.flags:
        print(f"    ! {flag}")


def cmd_audit(args: argparse.Namespace) -> int:
    from collections import defaultdict

    path = Path(args.captures)
    if not path.exists():
        print(f"error: captures file not found: {path}", file=sys.stderr)
        return 2

    def load() -> ExternalAdapter:  # fresh adapter per audit so cursors don't collide
        return ExternalAdapter.from_csv(path) if path.suffix.lower() == ".csv" else ExternalAdapter.from_json(path)

    depths = {aid: len(runs) for aid, runs in load()._by_asset.items()}  # noqa: SLF001
    by_id = {a.id: a for a in all_assets()}
    covered = [by_id[aid] for aid in depths if aid in by_id]
    if not covered:
        print("error: no captured asset ids match the bundled assets", file=sys.stderr)
        return 2
    reproducible = [a for a in covered if depths[a.id] >= 2]
    skipped = [a for a in covered if depths[a.id] < 2]
    print(f"Auditing {len(covered)} asset(s) from {path.name}")
    if skipped:
        print(f"  ({len(skipped)} asset(s) with <2 captures skipped for reproducibility)")

    ran = 0
    # Reproducibility per capture-depth group, so a shallow asset does not cap the
    # depth used for the well-captured ones.
    groups: dict[int, list] = defaultdict(list)
    for a in reproducible:
        groups[depths[a.id]].append(a)
    for depth, grp in sorted(groups.items()):
        try:
            _print_check(audit_reproducibility(load(), grp, n=depth))
            ran += 1
        except (KeyError, ValueError) as exc:
            logger.warning("reproducibility skipped (depth %d): %s", depth, exc)
    conflicted = [a for a in reproducible if a.has_planted_conflict]
    if conflicted:
        try:
            _print_check(audit_conflict(load(), conflicted, n_consistency=min(depths[a.id] for a in conflicted)))
            ran += 1
        except (KeyError, ValueError) as exc:
            logger.warning("conflict skipped (need orig+swap captures): %s", exc)
    try:
        _print_check(audit_robustness(load(), covered))
        ran += 1
    except (KeyError, ValueError) as exc:
        logger.warning("robustness skipped (need perturbed captures): %s", exc)

    if ran == 0:
        print("\nNo audit could run — check the capture format (see playground_protocol.md).")
        return 1
    print(f"\n{ran} audit(s) ran. See playground_protocol.md to capture perturbed/reordered runs "
          "for the robustness and conflict audits.")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from .notebook_builder import main as demo_main

    demo_main()
    return 0


def cmd_assets(args: argparse.Namespace) -> int:
    for a in all_assets():
        label = "labeled" if a.true_outcome is not None else "unlabeled"
        print(f"  {a.id:34s} {a.kind:10s} {label:9s} {len(a.evidence)} evidence")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kavuru-convexia",
                                description="Reliability audit harness for LLM drug-asset verdicts.")
    p.add_argument("--version", action="version", version=f"kavuru-convexia {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("audit", help="audit captured verdicts (JSON/CSV) — no API key needed")
    a.add_argument("captures", help="path to a captures .json or .csv (see playground_protocol.md)")
    a.set_defaults(func=cmd_audit)

    d = sub.add_parser("demo", help="run the full reference-agent demo (needs ANTHROPIC_API_KEY)")
    d.set_defaults(func=cmd_demo)

    s = sub.add_parser("assets", help="list the bundled assets")
    s.set_defaults(func=cmd_assets)
    return p


def main(argv: list[str] | None = None) -> int:
    config.ensure_dirs()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
