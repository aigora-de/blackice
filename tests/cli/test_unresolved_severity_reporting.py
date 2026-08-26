# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""A severity the panel could not resolve must reach the operator (#24).

Parsing it correctly is half the fix; the other half is that the run *says* it
happened. These drive ``kuang.cli.main`` end to end over a throwaway repo with
the subprocess boundary stubbed, so the whole real path runs — persona sourcing,
prompt assembly, ``parse_findings``, the engine loop, the printed summary and the
``--- JSON ---`` artefact — with nothing spawned and no network.

They are the tests that go red if the raw value is dropped anywhere between the
parser and the artefact, which is exactly where the old defect hid: the severity
was rewritten and no output disagreed with the run.
"""

from __future__ import annotations

import json

import pytest

from kuang.backends.claude_code import session as session_module
from kuang.cli import main


def _stub_claude(monkeypatch, findings_json: str) -> None:
    """Every persona replies with the same canned contract block."""
    def _fake(self, prompt, mandate, tools, model):  # noqa: ANN001, ARG001
        return f"I reviewed it.\n\n```json\n{findings_json}\n```\n", 0, None

    monkeypatch.setattr(session_module.PanelSession, "_run_claude", _fake)


def _run(repo, *extra) -> tuple[int, str]:
    return main(["--repo", str(repo), "--base", "HEAD~1", "--max-epochs", "1",
                 "--no-parallel", *extra])


def _artefact(out: str) -> dict:
    return json.loads(out.split("--- JSON ---")[-1])


@pytest.fixture
def undecided_panel(changed_repo, monkeypatch):
    """A panel whose every persona declares a tie it will not resolve."""
    _stub_claude(monkeypatch, json.dumps({"verdict": "YES", "findings": [
        {"title": "maybe ruin", "severity": "BLOCKER/UGLY", "claim_class": "ruin",
         "file": "a.py", "line": 3, "evidence": "could go either way"}]}))
    return changed_repo


def test_the_raw_value_and_the_persona_reach_the_operator(undecided_panel, capsys):
    _run(undecided_panel)
    out = capsys.readouterr().out

    assert "BLOCKER/UGLY" in out, "the operator never sees what the persona wrote"
    entries = _artefact(out)["unresolved_severities"]
    assert entries, "the artefact carries no record that a severity was unresolved"
    assert {e["raw"] for e in entries} == {"BLOCKER/UGLY"}
    assert all(e["epoch"] == 1 and e["persona"] for e in entries)


def test_the_finding_is_escalated_not_downgraded(undecided_panel, capsys):
    _run(undecided_panel)
    payload = _artefact(capsys.readouterr().out)

    assert payload["open_blockers"] > 0
    assert {f["severity"] for f in payload["findings"]} == {"BLOCKER"}


def test_an_unresolved_severity_blocks_convergence(undecided_panel, capsys):
    """Every persona voted YES; the run still must not report a good verdict.

    This is decision 1 as a property of the system rather than of the parser: a
    severity nobody could read cannot produce CONVERGED.
    """
    _run(undecided_panel)
    payload = _artefact(capsys.readouterr().out)

    assert payload["halt_reason"] != "converged"


def test_a_clean_run_says_nothing_about_unresolved_severities(changed_repo, capsys,
                                                              monkeypatch):
    """The new reporting is silent when there is nothing to report."""
    _stub_claude(monkeypatch, json.dumps({"verdict": "YES", "findings": [
        {"title": "fine", "severity": "NOTE", "claim_class": "x"}]}))

    _run(changed_repo)
    out = capsys.readouterr().out

    assert "unresolved" not in out.split("--- JSON ---")[0].lower()
    assert _artefact(out)["unresolved_severities"] == []
