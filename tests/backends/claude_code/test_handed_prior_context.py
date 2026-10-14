# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""The shipped checkpoint reads a COPY and must rebuild exactly the same prior context (#155).

``PanelSession.on_epoch`` rebuilds the next epoch's prior context from ``run.ledger`` and,
through ``ungrounded_keys``, from ``run.epochs[*].reports``. Since #155 it is handed a copy
the engine built, so a copy that dropped the reports, or a report's ``turns``, would
silently change what the next panel is told, with every test of the halt still green.
This is the fidelity row: the next epoch's memory, computed on what the checkpoint is
handed and on the live record at the same moment, must be identical, and must carry the
``[ungrounded]`` provenance the reports are the only source of.

GREEN ON MAIN, by design: main handed the live record, which is trivially identical. It is
the guard on a copy that is not faithful, and the mutation dropping ``reports`` turns it red.

Stdlib + stub seams only: no network, no ``claude`` subprocess.
"""

from __future__ import annotations

import pytest

import kuang.engine.loop as loop_module
from kuang.backends.claude_code.contract import parse_findings
from kuang.backends.claude_code.memory import epoch_summary, ungrounded_keys
from kuang.backends.claude_code.spawn import UNGROUNDED_TURNS
from kuang.engine import (EpochResult, Finding, HaltingSet, PanelConfig, PersonaReport,
                          PersonaStatus, ReviewRun, ReviewSpec, Severity, run)
from kuang.engine._handed import handed


def _spawn(persona, mandate, surface, epoch):  # noqa: ARG001
    """``grounded`` opened files; ``ungrounded`` answered in one turn, calling no tool."""
    turns = UNGROUNDED_TURNS if persona == "ungrounded" else 7
    return PersonaReport(persona=persona, verdict="NO", tokens=5, turns=turns,
                         status=PersonaStatus.CONTRIBUTED,
                         findings=[Finding(persona, f"{persona} at {epoch}", Severity.BLOCKER,
                                           f"{persona}-{epoch}", "a.py", epoch * 100)])


def _memory(review_run):
    return epoch_summary(review_run.ledger.values(), ungrounded_keys(review_run))


def test_the_checkpoint_builds_the_same_prior_context_from_its_copy(monkeypatch):
    made: list[ReviewRun] = []

    def construct():
        made.append(ReviewRun())
        return made[-1]
    monkeypatch.setattr(loop_module, "ReviewRun", construct)

    pairs: list[tuple[str, str]] = []
    run(ReviewSpec(why="w", what="x"), HaltingSet(max_epochs=3, stall_patience=5),
        PanelConfig(personas=[("grounded", "m"), ("ungrounded", "m")]),
        spawn=_spawn, gather=lambda epoch: "s", parallel=False,
        checkpoint=lambda handed: pairs.append((_memory(handed), _memory(made[0]))))

    assert len(pairs) == 3
    for from_copy, from_live in pairs:
        assert from_copy == from_live
        assert "[ungrounded]" in from_copy
    assert pairs[-1][0].count("[ungrounded]") == 3


@pytest.mark.parametrize("field", ["reports", "turns"])
def test_the_fixture_can_express_the_axis(field):
    """GREEN ON MAIN, by design: the fixture's own guard. Without the reports, or with every
    report's ``turns`` lost, the memory loses its provenance, so the test above can see a
    copy that drops either."""
    review_run = ReviewRun()
    report = _spawn("ungrounded", "m", "s", 1)
    review_run.ledger = {f.key: f for f in report.findings}
    review_run.epochs = [EpochResult(index=1, reports=[report], new_findings=[],
                                     open_blockers=0, open_uglies=0)]
    whole = _memory(review_run)
    if field == "reports":
        review_run.epochs[0].reports = []
    else:
        report.turns = 0

    assert "[ungrounded]" in whole
    assert "[ungrounded]" not in _memory(review_run)


_REPLIES = [
    "no contract at all",
    "```json\n{}\n```",
    "```json\n[]\n```",
    '```json\n{"verdict": "YES", "findings": []}\n```',
    '```json\n{"verdict": "NO", "findings": [{"title": "t", "severity": "UGLY", '
    '"claim_class": "c", "file": "a.py", "line": 7, "evidence": "e"}]}\n```',
    '```json\n{"verdict": "maybe", "findings": [{"title": 3, "severity": "SEVERE", '
    '"claim_class": null, "file": ["x"], "line": "~42"}, "not an object", null]}\n```',
    '```json\n{"verdict": ["NO"], "findings": {"title": "t"}}\n```',
    '```json\n{"verdict": "NO", "findings": [{"title": "t", "severity": "blocker", '
    '"claim_class": "c", "file": "None", "line": true}]}\n```',
]


@pytest.mark.parametrize("reply", _REPLIES)
def test_the_shipped_parser_never_produces_a_report_the_copy_refuses(reply):
    """The mirror image of the entry refusal (#155): a rule that refused what the SHIPPED
    backend produces would fail every persona of a healthy run closed. A sample of well-
    and ill-formed replies through the real parser, each copied, equal, and not the same
    object. GREEN ON MAIN only in the sense that main never copied: there is no refusal to
    fire. The pre-merge reviews fuzzed thousands more; this pins a sample in the suite."""
    report = parse_findings("p", reply)

    copy = handed(report)

    assert copy == report and copy is not report
