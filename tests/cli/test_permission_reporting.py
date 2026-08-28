# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""A run must say what the panel could actually do (#67).

The fourth instance of the doctrine ``test_unresolved_severity_reporting`` (#24),
``test_verdict_reporting`` (#26) and ``test_participation_reporting`` (#30)
established, and the third degradation state: the process succeeded, ``is_error``
is false, a well-formed contract came back and the persona **voted** — having
reviewed with fewer tools than the panel believes it granted it.

Two facts are reported, and they are not the same fact:

* **what the panel granted** — a tool that is granted *and* deny-listed was never
  in the reviewer's session at all. Deterministic, known before any spawn, and the
  half that fires on the issue's own reproduction case;
* **what the agent was refused** — ``permission_denials`` from the result envelope,
  which is populated only for a tool present in the session but not allowed.

Measured end to end on agent CLI 2.1.246 before any of this was written: four
personas with every review tool deny-listed returned 43 findings and six UGLY,
latched the circuit-breaker, and were reported as ``epoch 1: 4 contributed`` under
a header line reading ``tools=['Read', 'Grep', 'Glob']`` — tools none of them had.

These drive ``kuang.cli.main`` end to end over a throwaway repo with the subprocess
boundary stubbed — persona sourcing, prompt assembly, ``parse_findings``, the engine
loop, the printed summary and the ``--- JSON ---`` artefact — with nothing spawned
and no network. The panel is sourced from a ``CLAUDE.md``, not the distilled
default, so the assertions are about the panel an operator actually convenes.
"""

from __future__ import annotations

import json

import pytest

from kuang.backends.claude_code import session as session_module
from kuang.backends.claude_code.permissions import DEFAULT_DISALLOWED_TOOLS
from kuang.backends.claude_code.spawn import CallResult
from kuang.cli import main

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

# Deny-listing everything a reviewer reads with. This is the CLI form of the
# issue's reproduction case, and it needs no source edit to trigger.
_DENY_EVERYTHING = ["Read", "Grep", "Glob", *DEFAULT_DISALLOWED_TOOLS]


def _contract(verdict="YES", findings=()) -> str:
    body = json.dumps({"verdict": verdict, "findings": list(findings)})
    return f"I reviewed it.\n\n```json\n{body}\n```"


@pytest.fixture
def sourced_repo(changed_repo):
    (changed_repo / "CLAUDE.md").write_text(_CLAUDE_MD)
    return changed_repo


def _stub(monkeypatch, denials: dict | None = None, *, text=None) -> None:
    """Reply per persona, keyed off the mandate (which carries the name).

    ``denials`` maps a persona name to the tools its call was refused, so a test
    can make one persona degraded and leave the rest healthy — which is what
    "*which* persona" has to mean.
    """
    denials = denials or {}
    reply = _contract() if text is None else text

    def _fake(self, prompt, mandate, tools, model):  # noqa: ANN001, ARG001
        refused: tuple[str, ...] = ()
        for name, tools_denied in denials.items():
            if f"You are {name} " in mandate:
                refused = tuple(tools_denied)
        return CallResult(text=reply, output_tokens=0, error=None,
                          denied_tools=refused)

    monkeypatch.setattr(session_module.PanelSession, "_run_claude", _fake)


def _run(repo, *extra) -> int:
    return main(["--repo", str(repo), "--base", "HEAD~1", "--max-epochs", "1",
                 "--no-parallel", *extra])


def _artefact(out: str) -> dict:
    return json.loads(out.split("--- JSON ---")[-1])


def _unavailable(payload: dict, persona: str) -> list[str]:
    return next(r["unavailable"] for r in payload["permissions"]["personas"]
                if r["persona"] == persona)


# --- the defect: tools the panel never had ----------------------------------

def test_a_panel_granted_deny_listed_tools_says_so(sourced_repo, capsys, monkeypatch):
    """The measured defect, inverted.

    Every review tool on both lists. The reviewers reply, contribute and vote, and
    before this the run's only mention of tools asserted it had them.
    """
    _stub(monkeypatch)
    _run(sourced_repo, "--disallow-tools", *_DENY_EVERYTHING)
    out = capsys.readouterr().out

    assert "DEGRADED" in out.split("panel permissions:")[1].split("\n\n")[0]
    payload = _artefact(out)
    for persona in ("Analyst", "Critic", "Sentinel"):
        assert _unavailable(payload, persona) == ["Read", "Grep", "Glob"]


def test_the_operator_is_warned_before_the_panel_is_paid_for(sourced_repo, capsys,
                                                             monkeypatch):
    """A pre-flight warning, not only a post-mortem.

    The contradiction is knowable from the flags alone, so an operator can still
    abort. Reporting it only in the summary means the whole panel has been paid for
    by the time anyone can act on it.
    """
    _stub(monkeypatch)
    _run(sourced_repo, "--disallow-tools", *_DENY_EVERYTHING)

    before_run = capsys.readouterr().out.split("=== HALT:")[0]
    assert "Read" in before_run and "not available" in before_run.lower()


def test_only_the_contradicted_persona_is_named(sourced_repo, capsys, monkeypatch):
    """Per persona, because ``--allow-tools`` and ``panel.yaml`` both vary them."""
    _stub(monkeypatch)
    _run(sourced_repo, "--allow-tools", "Read", "Grep", "--disallow-tools", "Grep")
    payload = _artefact(capsys.readouterr().out)

    assert _unavailable(payload, "Analyst") == ["Grep"]
    assert payload["permissions"]["personas"][0]["granted"] == ["Read", "Grep"]


# --- the mirror-image defect: a healthy run must claim nothing ---------------

def test_a_healthy_run_says_the_panel_had_its_tools(sourced_repo, capsys, monkeypatch):
    """Always on, and the asymmetry is the argument (#30's fifth design decision).

    #24's and #26's sections report *exceptions*, so their absence is a complete
    claim. This is the mirror image: "the panel had the tools it says it had" is
    the claim an operator most needs to trust, and no absent section can make it.
    """
    _stub(monkeypatch)
    _run(sourced_repo)
    out = capsys.readouterr().out

    assert "panel permissions:" in out
    assert "DEGRADED" not in out.split("panel permissions:")[1].split("\n\n")[0]
    payload = _artefact(out)
    assert all(r["unavailable"] == [] for r in payload["permissions"]["personas"])
    assert payload["permissions"]["refusals"] == []


def test_the_shipped_default_policy_reports_no_degradation(sourced_repo, capsys,
                                                           monkeypatch):
    """Getting this wrong makes every healthy run report degradation.

    The default policy is the run every operator gets, so this is the guard on the
    mirror image of the defect rather than on the defect.
    """
    _stub(monkeypatch)
    _run(sourced_repo)

    assert "DEGRADED" not in capsys.readouterr().out


# --- what the agent was refused ---------------------------------------------

def test_a_refused_tool_call_names_the_persona_and_the_tool(sourced_repo, capsys,
                                                            monkeypatch):
    """The issue's first acceptance criterion, on the shape that populates it."""
    _stub(monkeypatch, {"Critic": ["Bash"]})
    _run(sourced_repo)
    payload = _artefact(capsys.readouterr().out)

    assert payload["permissions"]["refusals"] == [
        {"epoch": 1, "persona": "Critic", "tools": ["Bash"]}]


def test_a_refused_tool_call_is_a_fact_not_a_degradation(sourced_repo, capsys,
                                                         monkeypatch):
    """Deny-by-default means an Edit/Write/Bash refusal is the system WORKING.

    Measured: a refused call can be the agent probing an unallowed tool and then
    completing the task another way. The run states what was refused; it does not
    decide what that meant. Classifying it as degradation would make every reviewer
    that reaches for a tool it was never granted look like a broken one.
    """
    _stub(monkeypatch, {"Critic": ["Bash"]})
    _run(sourced_repo)
    out = capsys.readouterr().out

    section = out.split("panel permissions:")[1].split("\n\n")[0]
    assert "Bash" in section
    assert "DEGRADED" not in section


def test_a_dry_run_claims_no_refusals(sourced_repo, capsys, monkeypatch):
    """Nothing was spawned, so "no tool call was refused" would be a true sentence
    that means the opposite of what an operator would read into it — the shape of
    #69, and the reason ``ReduceState`` separates ``DRY_RUN`` from ``RAN``.
    """
    _stub(monkeypatch)
    _run(sourced_repo, "--dry-run")
    out = capsys.readouterr().out

    section = out.split("panel permissions:")[1].split("\n\n")[0]
    assert "nothing was spawned" in section
    assert _artefact(out)["permissions"]["refusals"] == []


def test_the_deterministic_half_still_fires_in_a_dry_run(sourced_repo, capsys,
                                                         monkeypatch):
    """The half that needs no subprocess is the half a dry run can still report.

    Pre-flight confirmation is the dry run's only job, and "these flags give your
    panel no tools" is exactly the thing worth confirming before spending.
    """
    _stub(monkeypatch)
    _run(sourced_repo, "--dry-run", "--disallow-tools", *_DENY_EVERYTHING)
    payload = _artefact(capsys.readouterr().out)

    assert _unavailable(payload, "Analyst") == ["Read", "Grep", "Glob"]


# --- the vote, and the doctrine it follows ----------------------------------

def test_a_persona_denied_its_tools_still_casts_its_vote(sourced_repo, capsys,
                                                         monkeypatch):
    """#24 and #26 settled this shape and it is not reopened here.

    A model-controlled value that drives halting must not silently count **and must
    not silently not-count either**. Discarding the vote would move quorum for a
    reason nobody can see, which is the defect #26 fixed — and the measured case
    voted NO and raised an UGLY, so discarding it would silently *unlatch* a
    circuit-breaker. The failure is symmetric; the tool records and reports, and the
    human adjudicates. blackice is a synthesiser, never a judge.
    """
    _stub(monkeypatch)
    exit_code = _run(sourced_repo, "--disallow-tools", *_DENY_EVERYTHING)
    payload = _artefact(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["halt_reason"] == "converged", "the vote was counted, not dropped"
    assert all(r["status"] == "found_nothing" for r in payload["participation"])


# --- the property: a reader's own question, asked of the artefact ------------

def test_a_toolless_panel_cannot_be_read_as_a_grounded_review(sourced_repo, capsys,
                                                              monkeypatch):
    """*May I read these findings as verified against source?*

    Asked of a run where nothing blocking is open and the halt is ordinary, so the
    permissions block is the only thing in the artefact that can answer. Alone this
    would pass if every run claimed degradation, which is what the pair below is for.
    """
    _stub(monkeypatch)
    _run(sourced_repo, "--disallow-tools", *_DENY_EVERYTHING)
    payload = _artefact(capsys.readouterr().out)

    assert payload["halt_reason"] == "converged"
    assert all(r["status"] == "found_nothing" for r in payload["participation"])
    assert any(r["unavailable"] for r in payload["permissions"]["personas"]), (
        "a clean verdict from a panel that could not open a file must not read clean")


def test_a_grounded_review_can_be_read_as_one(sourced_repo, capsys, monkeypatch):
    """The half that forbids answering the question above with a constant."""
    _stub(monkeypatch)
    _run(sourced_repo)
    payload = _artefact(capsys.readouterr().out)

    assert payload["halt_reason"] == "converged"
    assert not any(r["unavailable"] for r in payload["permissions"]["personas"])
    assert payload["permissions"]["refusals"] == []


# --- the roster: the policy itself is on the record --------------------------

def test_the_artefact_records_the_policy_the_run_actually_used(sourced_repo, capsys,
                                                               monkeypatch):
    """An artefact read back cold cannot check a contradiction it cannot see.

    The same argument as #30's ``panel`` block: without the deny-list and the mode,
    "unavailable: []" is a claim the reader has no way to audit.
    """
    _stub(monkeypatch)
    _run(sourced_repo, "--permission-mode", "default")
    perms = _artefact(capsys.readouterr().out)["permissions"]

    assert perms["mode"] == "default"
    assert perms["denied"] == list(DEFAULT_DISALLOWED_TOOLS)
