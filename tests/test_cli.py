"""Tests for the command-line interface (no API key needed)."""
from __future__ import annotations

import json

from kavuru_convexia.assets import build_synthetic_assets
from kavuru_convexia.cli import main
from kavuru_convexia.evaluator import ReferenceAgent, verdicts_to_json
from kavuru_convexia.llm import LLMClient


def _write_captures(path, n=5):
    agent = ReferenceAgent(client=LLMClient(offline=True))
    verdicts = [
        agent.evaluate(a, cache_tag=f"cap{i}")
        for a in build_synthetic_assets()[:2]
        for i in range(n)
    ]
    verdicts_to_json(verdicts, path)


def test_cli_audit_runs_reproducibility(tmp_path, capsys):
    captures = tmp_path / "captures.json"
    _write_captures(captures)
    rc = main(["audit", str(captures)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "reproducibility" in out
    assert "95% CI" in out  # CIs are reported


def test_cli_audit_missing_file_errors(tmp_path):
    assert main(["audit", str(tmp_path / "nope.json")]) == 2


def test_cli_assets_lists_bundled(capsys):
    assert main(["assets"]) == 0
    assert "historical" in capsys.readouterr().out
