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

The line carries per-finding *provenance* as well as state (#71): whether a
finding is the instrument's own diagnosis (``about_run``, #73) and whether the
call that produced it opened nothing (``ungrounded``, #70). Both cross the round
trip, which is what makes a contaminated artefact seed a later run contaminated
rather than clean (#13).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from kuang.backends.claude_code import PanelSession, load_prior_findings
from kuang.engine import (Finding, PersonaReport, PersonaStatus, ReviewSpec,
                          Severity)


def _cli_json(findings, ungrounded=()):
    """The findings array exactly as ``kuang.cli.main`` writes it under '--- JSON ---'.

    Kept faithful deliberately. It carried neither ``about_run`` (added by #73) nor
    ``ungrounded`` (#71) while claiming to mirror the record, which is the shape
    #87 was filed for — a fixture that claims a fidelity it does not have. The
    round trip cannot prove agreement about a key the fixture omits.
    """
    return {"findings": [
        {"persona": f.persona, "severity": f.severity.name, "title": f.title,
         "file": f.file, "line": f.line, "open": f.counts_open,
         "about_run": f.about_run, "ungrounded": f.key in set(ungrounded)}
        for f in findings]}


def _round_trip(tmp_path, findings, ungrounded=()) -> tuple[str, str]:
    """Render the same findings both ways: (this run's memory, a later run's seed).

    Drives the real ``on_epoch``, which is the left-hand end of the round trip the
    module docstring names — including the (persona, epoch) join it performs to
    decide which lines are ``[ungrounded]``.
    """
    session = PanelSession(repo_root=tmp_path, spec=ReviewSpec(why="w", what="x"),
                           personas={}, base="")
    marked = set(ungrounded)
    reports = [
        PersonaReport(persona="P1", status=PersonaStatus.CONTRIBUTED, turns=turns,
                      findings=[f for f in findings if (f.key in marked) is degraded])
        for turns, degraded in ((1, True), (7, False))]
    session.on_epoch(SimpleNamespace(
        ledger={f.key: f for f in findings},
        epochs=[SimpleNamespace(index=1, reports=reports)]))

    path = tmp_path / "findings.json"
    path.write_text(json.dumps(_cli_json(findings, ungrounded)))

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


def test_a_seed_whose_finding_has_no_line_key_still_loads(tmp_path):
    """REGRESSION for #25: ``f["line"]`` raised KeyError on that artefact.

    An artefact written by anything other than this CLI — or by a future version
    that omits an absent field — used to take the seeded re-run down at startup,
    which is the one moment the operator has the least context to interpret it.
    """
    path = tmp_path / "prior.json"
    path.write_text(json.dumps({"findings": [
        {"persona": "P1", "severity": "BLOCKER", "title": "t", "file": "a.py",
         "open": True}]}))

    assert load_prior_findings(path) == "- [BLOCKER/open] (P1) t @ a.py:None"


def test_there_is_exactly_one_renderer():
    """The property the round-trip depends on, stated directly."""
    from kuang import report
    from kuang.backends.claude_code import memory

    assert memory.ledger_line is report.ledger_line


# --- provenance crosses the round trip too (#71) ------------------------------

_MARKED = Finding("P1", "unbounded retry loop", Severity.BLOCKER, "retry",
                  "runner.py", 120)
_DIAGNOSIS = Finding("P1", "agent error: boom", Severity.NOTE, "meta",
                     about_run=True)


@pytest.mark.parametrize("finding, ungrounded", [
    (_MARKED, ()),
    (_MARKED, (_MARKED.key,)),
    (_DIAGNOSIS, ()),
    (_DIAGNOSIS, (_DIAGNOSIS.key,)),
], ids=["clean", "ungrounded", "about-the-run", "both"])
def test_the_two_renderers_agree_about_provenance(tmp_path, finding, ungrounded):
    """Every combination, because either half can drift on its own."""
    epoch_memory, seeded_memory = _round_trip(tmp_path, [finding], ungrounded)

    assert epoch_memory == seeded_memory


def test_an_ungrounded_line_still_looks_like_this(tmp_path):
    epoch_memory, seeded_memory = _round_trip(tmp_path, [_MARKED], (_MARKED.key,))

    assert epoch_memory == ("- [BLOCKER/open] (P1) unbounded retry loop "
                            "@ runner.py:120 [ungrounded]")
    assert seeded_memory == epoch_memory


def test_a_line_about_the_run_still_looks_like_this(tmp_path):
    epoch_memory, seeded_memory = _round_trip(tmp_path, [_DIAGNOSIS])

    assert epoch_memory == "- [NOTE/open] (P1) agent error: boom @ - [about the run]"
    assert seeded_memory == epoch_memory


def test_an_artefact_that_predates_the_marks_seeds_exactly_as_before(tmp_path):
    """A saved run from before #71 carries neither key; it must render unchanged."""
    path = tmp_path / "prior.json"
    path.write_text(json.dumps({"findings": [
        {"persona": "P1", "severity": "BLOCKER", "title": "t", "file": "a.py",
         "line": 3, "open": True}]}))

    assert load_prior_findings(path) == "- [BLOCKER/open] (P1) t @ a.py:3"
