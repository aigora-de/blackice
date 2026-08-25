# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""The ledger line is a contract between two modules, and nothing enforced it.

``on_epoch`` renders ``- [SEV/state] (persona) title @ loc`` into this run's
cross-epoch memory; the CLI writes the same findings out as JSON;
``load_prior_findings`` reads that JSON back on a later run (``--prior-findings``)
and renders the line *again*, from a second implementation.

So the two renderers must agree byte-for-byte, and until #19 gave them one
implementation nothing checked that they did. These are regression tests over the
whole round trip — ``on_epoch`` → the CLI's JSON → ``load_prior_findings`` —
rather than over either renderer alone, because agreeing is the property that
matters and either half can drift on its own.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from blackice.backends.claude_code import PanelSession, load_prior_findings
from blackice.engine import Finding, ReviewSpec, Severity


def _cli_json(findings):
    """The findings array exactly as ``blackice.main`` writes it under '--- JSON ---'."""
    return {"findings": [
        {"persona": f.persona, "severity": f.severity.name, "title": f.title,
         "file": f.file, "line": f.line, "open": f.counts_open}
        for f in findings]}


def _round_trip(tmp_path, findings) -> tuple[str, str]:
    """Render the same findings both ways: (this run's memory, a later run's seed)."""
    session = PanelSession(repo_root=tmp_path, spec=ReviewSpec(why="w", what="x"),
                           personas={}, base="")
    session.on_epoch(SimpleNamespace(ledger={f.key: f for f in findings}))

    path = tmp_path / "findings.json"
    path.write_text(json.dumps(_cli_json(findings)))

    return session.prior_summary, load_prior_findings(path)


@pytest.mark.parametrize("finding", [
    Finding("P1", "unbounded retry loop", Severity.BLOCKER, "retry", "runner.py", 120),
    Finding("P2", "stale threshold", Severity.NOTE, "calc", "calc.py", 7, verified=False),
    Finding("P3", "ruin-class data loss", Severity.UGLY, "drop", "a.py", 1),
    Finding("P4", "no location at all", Severity.NON_BLOCKING, "meta"),
    Finding("P5", "a file but no line", Severity.NOTE, "meta", "only.py"),
])
def test_the_two_renderers_agree(tmp_path, finding):
    epoch_memory, seeded_memory = _round_trip(tmp_path, [finding])

    assert epoch_memory == seeded_memory


def test_they_agree_over_a_whole_ledger(tmp_path):
    findings = [
        Finding("P1", "unbounded retry loop", Severity.BLOCKER, "retry", "runner.py", 120),
        Finding("P2", "stale threshold", Severity.NOTE, "calc", "calc.py", 7, verified=False),
        Finding("P3", "no location", Severity.UGLY, "meta"),
    ]

    epoch_memory, seeded_memory = _round_trip(tmp_path, findings)

    assert epoch_memory == seeded_memory
    assert len(epoch_memory.splitlines()) == 3


def test_the_line_still_looks_like_this(tmp_path):
    """One literal assertion, so the shared format cannot drift silently."""
    epoch_memory, seeded_memory = _round_trip(tmp_path, [
        Finding("P1", "unbounded retry loop", Severity.BLOCKER, "retry", "runner.py", 120)])

    assert epoch_memory == "- [BLOCKER/open] (P1) unbounded retry loop @ runner.py:120"
    assert seeded_memory == epoch_memory


def test_a_resolved_finding_reads_as_resolved_on_both_sides(tmp_path):
    epoch_memory, seeded_memory = _round_trip(tmp_path, [
        Finding("P2", "fixed since", Severity.BLOCKER, "x", "a.py", 3, verified=False)])

    assert epoch_memory == "- [BLOCKER/resolved] (P2) fixed since @ a.py:3"
    assert seeded_memory == epoch_memory
