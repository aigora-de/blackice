# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""A run must say whether the panel actually ran (#30).

The third instance of the doctrine ``test_unresolved_severity_reporting`` (#24)
and ``test_verdict_reporting`` (#26) established, and the one where the missing
data was not merely misread but **absent**: a persona that reviewed and found
nothing, one that was never spawned, and one that does not exist produced
byte-identical output. Status is therefore set at the source and reported for
every persona on every epoch, never derived from the ledger.

These drive ``kuang.cli.main`` end to end over a throwaway repo with the
subprocess boundary stubbed — persona sourcing, prompt assembly,
``parse_findings``, the engine loop, the printed summary and the ``--- JSON ---``
artefact — with nothing spawned and no network.

The panel here is **sourced from a CLAUDE.md**, not the distilled default. That is
deliberate: the measurement this issue was designed against ran on a repo with no
CLAUDE.md at all, so it said nothing about the panel an operator actually convenes.
"""

from __future__ import annotations

import json

import pytest

from kuang.backends.claude_code import load_personas
from kuang.backends.claude_code import session as session_module
from kuang.backends.claude_code.contract import _TEMPLATE_BLOCK
from kuang.cli import main

# Three experts, and no fourth: "completeness" and "ruin" appear in the bodies, so
# ``_ensure_specialists`` injects neither of its defaults. A small named panel
# makes "which persona" assertable without depending on the default set.
_CLAUDE_MD = """\
# A repo

# Resident Experts

## Analyst — Correctness

Does the change compute the right thing?

## Critic — Completeness

Find what everyone else missed; assume shared blind spots.

## Sentinel — Ruin

Hunt ruin-class hazards only.
"""

_NO_JSON = ("I read the diff and have nothing structured to say.", 0, None)


def _contract(verdict="YES", findings=(), extra="") -> tuple[str, int, None]:
    body = json.dumps({"verdict": verdict, "findings": list(findings)})
    return (f"I reviewed it.\n\n```json\n{body}\n```\n{extra}", 0, None)


_CLEAN = _contract()
_FINDING = _contract("NO", [{"title": "a real finding", "severity": "BLOCKER",
                            "claim_class": "correctness", "file": "a.py", "line": 1,
                            "evidence": "read it"}])


@pytest.fixture
def sourced_repo(changed_repo):
    """``changed_repo`` with a CLAUDE.md the panel is sourced from."""
    (changed_repo / "CLAUDE.md").write_text(_CLAUDE_MD)
    return changed_repo


def _stub(monkeypatch, replies: dict, *, default=_CLEAN, formatter=_NO_JSON) -> None:
    """Reply per persona, keyed off the mandate (which carries the name)."""
    def _fake(self, prompt, mandate, tools, model):  # noqa: ANN001, ARG001
        if "formatter" in mandate:
            return formatter
        for name, reply in replies.items():
            if f"You are {name} " in mandate:
                return reply
        return default

    monkeypatch.setattr(session_module.PanelSession, "_run_claude", _fake)


def _run(repo, *extra) -> int:
    return main(["--repo", str(repo), "--base", "HEAD~1", "--max-epochs", "1",
                 "--no-parallel", *extra])


def _artefact(out: str) -> dict:
    return json.loads(out.split("--- JSON ---")[-1])


def _status(payload: dict, persona: str) -> str:
    return next(r["status"] for r in payload["participation"]
                if r["persona"] == persona)


# --- the roster: everyone asked is accounted for, every epoch -----------------

def test_every_persona_asked_is_accounted_for(sourced_repo, capsys, monkeypatch):
    """The defect, inverted: no persona may be absent from the artefact."""
    _stub(monkeypatch, {})
    _run(sourced_repo)
    payload = _artefact(capsys.readouterr().out)

    roster = [p.name for p in load_personas(sourced_repo)[0]]
    assert payload["panel"]["personas"] == roster
    assert payload["panel"]["source"] == "CLAUDE.md"
    # Not a literal count: the property is one record per persona per epoch.
    assert len(payload["participation"]) == len(roster) * payload["epochs"]
    assert {r["persona"] for r in payload["participation"]} == set(roster)


def test_a_clean_run_still_reports_participation(sourced_repo, capsys, monkeypatch):
    """Always-on, unlike #24's and #26's sections — and here is why.

    Their absence is a complete claim ("no severity went unread"). "The panel ran
    in full" is the claim an operator most needs to trust, and no absent section
    can make it.
    """
    _stub(monkeypatch, {})
    _run(sourced_repo)
    out = capsys.readouterr().out

    assert "panel participation" in out
    assert all(r["status"] == "found_nothing"
               for r in _artefact(out)["participation"])


# --- the states, distinguished at the source ---------------------------------

def test_a_silent_persona_and_a_contributor_are_distinguishable(sourced_repo, capsys,
                                                                monkeypatch):
    _stub(monkeypatch, {"Analyst": _FINDING})
    _run(sourced_repo)
    payload = _artefact(capsys.readouterr().out)

    assert _status(payload, "Analyst") == "contributed"
    assert _status(payload, "Critic") == "found_nothing"


def test_a_failed_agent_is_not_silence(sourced_repo, capsys, monkeypatch):
    """The measured defect: an error envelope and an empty review looked alike."""
    _stub(monkeypatch, {"Critic": ("", 0, "agent error: error_max_turns: …")})
    _run(sourced_repo)
    payload = _artefact(capsys.readouterr().out)

    assert _status(payload, "Critic") == "agent_error"
    assert _status(payload, "Analyst") == "found_nothing"


def test_a_backend_failure_and_an_engine_failure_stay_apart(sourced_repo, capsys,
                                                            monkeypatch):
    """``agent error:`` (#29) and ``persona failed:`` (#25) are worded differently
    at the source precisely so this issue can tell them apart. Recovering the
    distinction by matching a finding title would be the same defect one layer up.
    """
    _stub(monkeypatch, {"Critic": ("", 0, "agent error: error_max_turns: …")})
    real_spawn = session_module.PanelSession.spawn

    def _spawn(self, persona, mandate, surface, epoch):  # noqa: ANN001
        if persona == "Sentinel":
            raise RuntimeError("boom")
        return real_spawn(self, persona, mandate, surface, epoch)

    monkeypatch.setattr(session_module.PanelSession, "spawn", _spawn)

    _run(sourced_repo)
    payload = _artefact(capsys.readouterr().out)

    assert _status(payload, "Critic") == "agent_error"
    assert _status(payload, "Sentinel") == "spawn_failed"


def test_an_unreadable_reply_is_not_silence(sourced_repo, capsys, monkeypatch):
    """A persona that replied but wrote no contract, whose retry also failed."""
    _stub(monkeypatch, {"Analyst": _NO_JSON})
    _run(sourced_repo)
    payload = _artefact(capsys.readouterr().out)

    assert _status(payload, "Analyst") == "unreadable"


def test_a_persona_that_echoed_the_template_is_not_a_contributor(sourced_repo, capsys,
                                                                 monkeypatch):
    """Status is computed from the REVIEW, before the meta findings are appended.

    An echoed template appends a ``meta`` NOTE (#26), so a status read off
    ``len(findings)`` would call a persona that found nothing a contributor.
    """
    _stub(monkeypatch, {"Analyst": _contract(
        extra=f"\nThe contract, for reference:\n```json\n{_TEMPLATE_BLOCK}\n```\n")})
    _run(sourced_repo)
    payload = _artefact(capsys.readouterr().out)

    assert _status(payload, "Analyst") == "found_nothing"


def test_a_dry_run_reports_that_nothing_was_spawned(sourced_repo, capsys, monkeypatch):
    """A dry run must not pass for a panel that reviewed."""
    _stub(monkeypatch, {})
    _run(sourced_repo, "--dry-run")
    payload = _artefact(capsys.readouterr().out)

    assert all(r["status"] == "not_spawned" for r in payload["participation"])


# --- the property, not the string --------------------------------------------

def test_a_degraded_run_cannot_be_read_as_a_complete_panel(sourced_repo, capsys,
                                                           monkeypatch):
    """#40's exit criterion as a property of the artefact rather than a substring.

    Nothing blocking is open and the run halts normally; the ONLY thing that says
    this panel did not run in full is the participation record. Asserted as the
    reader's own question — "may I read this ledger as a full pass?" — so a
    mutation that keeps a plausible-looking section cannot satisfy it.
    """
    _stub(monkeypatch, {"Critic": ("", 0, "agent error: error_max_turns: …")})
    _run(sourced_repo)
    payload = _artefact(capsys.readouterr().out)

    reviewed = {"contributed", "found_nothing"}
    assert not all(r["status"] in reviewed for r in payload["participation"]), \
        "a run in which a persona never reviewed reads as a complete panel"
    assert payload["open_blockers"] == 0 and payload["open_uglies"] == 0


def test_a_healthy_run_can_be_read_as_a_complete_panel(sourced_repo, capsys,
                                                       monkeypatch):
    """The other half of the property: it must be able to say yes, too."""
    _stub(monkeypatch, {"Analyst": _FINDING})
    _run(sourced_repo)
    payload = _artefact(capsys.readouterr().out)

    reviewed = {"contributed", "found_nothing"}
    assert all(r["status"] in reviewed for r in payload["participation"])
