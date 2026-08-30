# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""A run must say whether the reviewer called a tool at all (#70).

The fourth instance of the doctrine ``test_unresolved_severity_reporting`` (#24),
``test_verdict_reporting`` (#26), ``test_participation_reporting`` (#30) and
``test_permission_reporting`` (#67) established, and the fourth degradation state
none of the others can see: the process succeeded, ``is_error`` is false, the
tools were granted and nothing was refused, a well-formed contract came back — and
the reviewer answered from the prompt alone, having opened nothing.

Reported **inside the participation section** rather than as a section of its own.
A turn count is a fact about one spawn, which is exactly what a participation
record is; that section is already always-on, so #30's asymmetry argument is
satisfied without a fourth always-on block on every run.

These drive ``kuang.cli.main`` end to end over a throwaway repo with the
subprocess boundary stubbed. **A stubbed envelope proves the handler**: the turn
counts asserted on here are the live captures in
``backends/claude_code/test_spawn.py``, and the rule they feed is measured, not
assumed.
"""

from __future__ import annotations

import json

import pytest

from kuang.backends.claude_code import session as session_module
from kuang.backends.claude_code.spawn import CallResult
from kuang.cli import main

# Three experts, as ``test_participation_reporting`` uses: "completeness" and
# "ruin" appear in the bodies, so no default specialist is injected.
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

# The measured healthy range is 6-9 turns; the one-turn call is the defect.
_GROUNDED, _NO_TOOL = 7, 1


def _contract(verdict="YES", findings=(), *, turns=_GROUNDED) -> CallResult:
    body = json.dumps({"verdict": verdict, "findings": list(findings)})
    return CallResult(f"I reviewed it.\n\n```json\n{body}\n```", 0, None, (), turns)


_UGLY = [{"title": "unbounded loss on retry", "severity": "UGLY",
          "claim_class": "ruin", "file": "a.py", "line": 1, "evidence": "traced it"}]


@pytest.fixture
def sourced_repo(changed_repo):
    (changed_repo / "CLAUDE.md").write_text(_CLAUDE_MD)
    return changed_repo


def _stub(monkeypatch, replies: dict, *, default=None, formatter=None) -> None:
    default = default if default is not None else _contract()
    formatter = formatter if formatter is not None else CallResult("no json", 0)

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


def _turns(payload: dict, persona: str) -> int:
    return next(r["turns"] for r in payload["participation"]
                if r["persona"] == persona)


# --- the defect, and its mirror image ----------------------------------------

def test_a_persona_that_called_no_tool_is_named(sourced_repo, capsys, monkeypatch):
    """The acceptance criterion: per persona, whether the reviewer called a tool.

    Every other channel reports this run as healthy — nothing errored, nothing was
    refused, every persona contributed, and the panel voted.
    """
    _stub(monkeypatch, {"Critic": _contract(turns=_NO_TOOL)})
    _run(sourced_repo)
    out = capsys.readouterr().out
    payload = _artefact(out)

    assert _turns(payload, "Critic") == 1
    assert _turns(payload, "Analyst") == 7
    marked = [ln for ln in out.splitlines() if "CALLED NO TOOL" in ln]
    assert any("(Critic)" in ln for ln in marked), "the persona is not named"
    assert not any("(Analyst)" in ln for ln in marked), "and the grounded one is not"


def test_the_epoch_line_says_how_many_called_no_tool(sourced_repo, capsys,
                                                     monkeypatch):
    """The count and its meaning, once per epoch, above the per-persona marks.

    A seven-persona three-epoch run is 21 marked-or-unmarked lines, and the mark
    itself does not say what it means. This is the line that does, and it is the
    line a reader scanning the report actually sees.
    """
    _stub(monkeypatch, {"Critic": _contract(turns=_NO_TOOL),
                        "Sentinel": _contract(turns=_NO_TOOL)})
    _run(sourced_repo)
    epoch_line = next(ln for ln in capsys.readouterr().out.splitlines()
                      if ln.startswith("  epoch 1:"))

    assert "2 CALLED NO TOOL" in epoch_line, epoch_line
    assert "answered from the prompt alone" in epoch_line, "the mark is unexplained"

    # …and it must be able to say nothing, too.
    _stub(monkeypatch, {})
    _run(sourced_repo)
    clean = next(ln for ln in capsys.readouterr().out.splitlines()
                 if ln.startswith("  epoch 1:"))
    assert "CALLED NO TOOL" not in clean


def test_a_grounded_panel_is_not_flagged(sourced_repo, capsys, monkeypatch):
    """The mirror-image defect: a healthy run must claim nothing happened.

    The measured healthy range is 6-9 turns, and a rule that fires there would
    make the report worse than silence.
    """
    _stub(monkeypatch, {"Analyst": _contract(turns=9), "Critic": _contract(turns=6)})
    _run(sourced_repo)
    out = capsys.readouterr().out

    assert "CALLED NO TOOL" not in out
    assert all(r["turns"] >= 6 for r in _artefact(out)["participation"])
    assert "panel participation" in out, "and it still says so, always-on"


def test_a_turn_count_nobody_reported_is_not_read_as_either(sourced_repo, capsys,
                                                            monkeypatch):
    """0 is "the runtime did not say", and must read as neither health nor harm.

    The #30 defect one layer down: a backend's silence must not pass for a fact.
    """
    _stub(monkeypatch, {"Analyst": _contract(turns=0)})
    _run(sourced_repo)
    out = capsys.readouterr().out

    assert _turns(_artefact(out), "Analyst") == 0
    assert "CALLED NO TOOL" not in out
    line = next(ln for ln in out.splitlines() if "(Analyst)" in ln and "finding" in ln)
    assert "unreported" in line, f"silence passed for a fact: {line!r}"


# --- what did not happen must not be reported as if it had --------------------

def test_a_dry_run_reports_no_turn_degradation(sourced_repo, capsys, monkeypatch):
    """Nothing was spawned, so no turn was taken — and none may be implied.

    #30 separated ``NOT_SPAWNED`` from ``FOUND_NOTHING`` and #67's permissions
    section says "nothing was spawned" rather than "no tool call was refused", for
    exactly this reason.
    """
    _stub(monkeypatch, {})
    _run(sourced_repo, "--dry-run")
    out = capsys.readouterr().out
    payload = _artefact(out)

    assert all(r["turns"] == 0 for r in payload["participation"])
    assert all(r["status"] == "not_spawned" for r in payload["participation"])
    assert "CALLED NO TOOL" not in out
    assert "unreported" not in out, "a dry run suffered no degradation to report"


def test_an_errored_persona_is_not_also_reported_as_ungrounded(sourced_repo, capsys,
                                                               monkeypatch):
    """One failure, one heading. ``_API_ERROR`` is a real capture at ``num_turns: 1``.

    A call that failed has no review to be ungrounded, and #29 already reports it.
    Saying it twice under two headings would inflate one failure into two.
    """
    _stub(monkeypatch, {"Critic": CallResult(
        "", 0, "agent error: api_error: API Error: Connection refused", (), 1)})
    _run(sourced_repo)
    out = capsys.readouterr().out

    assert _turns(_artefact(out), "Critic") == 1, "the fact is still recorded"
    assert "CALLED NO TOOL" not in out, "but it is not a second degradation"
    assert "agent error" in out


# --- the vote: inherited from #24 / #26 / #67, not re-decided ------------------

def test_a_persona_that_called_no_tool_still_votes(sourced_repo, capsys, monkeypatch):
    """A model-controlled value that drives halting must not silently NOT count.

    #67 measured why the failure is symmetric: its starved persona voted NO and
    raised an UGLY, so discarding such a vote would have *unlatched* a
    circuit-breaker rather than merely losing a YES. Here both halves are asserted
    — the one-turn persona's YES is needed for quorum, and a one-turn UGLY still
    latches the breaker.
    """
    _stub(monkeypatch, {"Critic": _contract(turns=_NO_TOOL)})
    _run(sourced_repo)
    assert _artefact(capsys.readouterr().out)["halt_reason"] == "converged", \
        "its YES was discarded, so the panel silently fell short of quorum"

    _stub(monkeypatch, {"Critic": _contract("NO", _UGLY, turns=_NO_TOOL)})
    _run(sourced_repo)
    payload = _artefact(capsys.readouterr().out)

    assert payload["halt_reason"] == "escalate_ugly", "the breaker was unlatched"
    assert payload["open_uglies"] == 1


# --- the property, not the string --------------------------------------------

def test_an_ungrounded_run_cannot_be_read_as_a_grounded_panel(sourced_repo, capsys,
                                                              monkeypatch):
    """#40's exit criterion as a property of the artefact rather than a substring.

    Nothing blocking is open, the run converges, every persona contributed and
    nothing was refused: the ONLY thing that says this panel opened nothing is the
    turn count. Asserted as the reader's own question so a mutation that keeps a
    plausible-looking section cannot satisfy it.
    """
    _stub(monkeypatch, {}, default=_contract(turns=_NO_TOOL))
    _run(sourced_repo)
    payload = _artefact(capsys.readouterr().out)

    assert not all(r["turns"] > 1 for r in payload["participation"]), \
        "a panel that called no tool reads as one that reviewed the source"
    assert payload["halt_reason"] == "converged"
    assert payload["open_blockers"] == 0 and payload["open_uglies"] == 0
    assert all(r["status"] in {"contributed", "found_nothing"}
               for r in payload["participation"]), "and every other channel agrees"


def test_a_grounded_run_can_be_read_as_one(sourced_repo, capsys, monkeypatch):
    """The other half of the property: it must be able to say yes, too.

    Without this, a change that reported every persona as ungrounded would satisfy
    the assertion above — the easiest vacuous test there is.
    """
    _stub(monkeypatch, {})
    _run(sourced_repo)
    payload = _artefact(capsys.readouterr().out)

    assert all(r["turns"] > 1 for r in payload["participation"])
