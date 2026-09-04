# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""A verdict the panel could not read must reach the operator (#26).

The sibling of ``test_unresolved_severity_reporting`` (#24), and for the same
reason: refusing to count a value is half the fix, and the run *saying* it
refused is the other half. A verdict nobody could read is the one that decides
whether the run reports a good verdict, so absorbing it silently is how a run
lies about having been approved.

These drive ``kuang.cli.main`` end to end over a throwaway repo with the
subprocess boundary stubbed — persona sourcing, prompt assembly,
``parse_findings``, the engine loop, the printed summary and the ``--- JSON ---``
artefact — with nothing spawned and no network.
"""

from __future__ import annotations

import json

import pytest

from kuang.backends.claude_code import session as session_module
from kuang.backends.claude_code.spawn import CallResult
from kuang.backends.claude_code.contract import _TEMPLATE_BLOCK
from kuang.cli import main


def _stub_reply(monkeypatch, reply: str) -> None:
    """Every persona hands back the same canned reply text."""
    def _fake(self, prompt, mandate, tools, model):  # noqa: ANN001, ARG001
        return CallResult(reply, 0)

    monkeypatch.setattr(session_module.PanelSession, "_run_claude", _fake)


def _block(body: str) -> str:
    return f"I reviewed it.\n\n```json\n{body}\n```\n"


def _run(repo, *extra) -> tuple[int, str]:
    return main(["--repo", str(repo), "--base", "HEAD~1", "--max-epochs", "1",
                 "--no-parallel", *extra])


def _artefact(out: str) -> dict:
    return json.loads(out.split("--- JSON ---")[-1])


@pytest.fixture
def hedging_panel(changed_repo, monkeypatch):
    """A panel whose every persona writes a verdict outside the vocabulary."""
    _stub_reply(monkeypatch, _block(json.dumps(
        {"verdict": "SOUND-WITH-CONCERNS", "findings": []})))
    return changed_repo


def test_the_raw_verdict_and_the_persona_reach_the_operator(hedging_panel, capsys):
    _run(hedging_panel)
    out = capsys.readouterr().out

    assert "SOUND-WITH-CONCERNS" in out, "the operator never sees what was written"
    entries = _artefact(out)["unresolved_verdicts"]
    assert entries, "the artefact carries no record that a verdict was unread"
    assert {e["raw"] for e in entries} == {"SOUND-WITH-CONCERNS"}
    assert all(e["epoch"] == 1 and e["persona"] for e in entries)


def test_an_unread_verdict_cannot_produce_a_good_verdict(hedging_panel, capsys):
    """The property, not the parser: nothing was blocking and it still is not GOOD."""
    _run(hedging_panel)
    payload = _artefact(capsys.readouterr().out)

    assert payload["open_blockers"] == 0 and payload["open_uglies"] == 0
    assert payload["halt_reason"] != "converged"


def test_a_clean_run_says_nothing_about_unresolved_verdicts(changed_repo, capsys,
                                                            monkeypatch):
    """The new reporting is silent when there is nothing to report."""
    _stub_reply(monkeypatch, _block(json.dumps({"verdict": "YES", "findings": []})))

    _run(changed_repo)
    out = capsys.readouterr().out

    # An allow-list rather than a ban, since #82: a clean run now states what its
    # good verdict RESTS ON, which is a line about the agreement and not about a
    # verdict nobody could read. The claim is unchanged and still exact — the only
    # thing this console says about verdicts is that one line — and it stays
    # stronger than matching the section header alone, which a stray mention
    # elsewhere would slip past.
    console = out.split("--- JSON ---")[0].lower()
    mentions = [ln for ln in console.splitlines() if "verdict" in ln]
    assert all(ln.startswith("verdict rests on:") for ln in mentions), mentions
    assert _artefact(out)["unresolved_verdicts"] == []


def test_an_echoed_template_neither_hides_the_review_nor_approves_it(changed_repo,
                                                                     capsys,
                                                                     monkeypatch):
    """#26 end to end: the real finding survives and the run is not CONVERGED."""
    real = _block(json.dumps({"verdict": "NO", "findings": [
        {"title": "unbounded retry", "severity": "BLOCKER", "claim_class": "ruin",
         "file": "a.py", "line": 1, "evidence": "read it"}]}))
    _stub_reply(monkeypatch, real + "\nThe contract, for reference:\n"
                + f"```json\n{_TEMPLATE_BLOCK}\n```\n")

    _run(changed_repo)
    payload = _artefact(capsys.readouterr().out)

    titles = [f["title"] for f in payload["findings"]]
    assert "unbounded retry" in titles, "the real review was discarded"
    assert "..." not in titles, "the echoed placeholder reached the ledger"
    assert payload["halt_reason"] != "converged"
