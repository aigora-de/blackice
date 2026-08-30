# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""A run must say when the surface was not the one that was asked for (#69).

The fifth instance of the doctrine ``test_unresolved_severity_reporting`` (#24),
``test_verdict_reporting`` (#26), ``test_participation_reporting`` (#30),
``test_permission_reporting`` (#67) and ``test_turn_reporting`` (#70)
established, and the fifth degradation state none of the others can see: the
process succeeded, the tools were granted, nothing was refused, the reviewer
opened files and answered in a healthy number of turns — and it was handed less
than the operator named. It votes, and the run reports ``found_nothing`` under a
heading that says "participation".

**Surface-side.** Nothing here reads anything back from the subprocess: what a
persona was GIVEN is known in ``surface.py`` before a subprocess exists, which is
why these tests need no envelope capture and why the rule is exact rather than
measured.

Reported **inside the participation section**, beside the sentence that lies,
rather than as a fifth always-on block — #70's reason. The console carries the
exception; the artefact carries the fact on every run, which is what a cold read
needs. Saying "the surface was complete" on the console is #11's acceptance
criterion, not this issue's, and is deliberately left to it.

These drive ``kuang.cli.main`` end to end over a throwaway repo with the
subprocess boundary stubbed. Every surface below is assembled by the real
``surface.py``: a stub can hand a panel a surface the real code would never
produce, and designing a detector for one of those would be designing for the
stub.
"""

from __future__ import annotations

import json

import pytest

from kuang.backends.claude_code import session as session_module
from kuang.backends.claude_code.spawn import CallResult
from kuang.cli import main

# Three experts, as the sibling reporting tests use: "completeness" and "ruin"
# appear in the bodies, so no default specialist is injected.
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

_GROUNDED = 7  # the measured healthy turn count (#70)

_UGLY = [{"title": "unbounded loss on retry", "severity": "UGLY",
          "claim_class": "ruin", "file": "a.py", "line": 1, "evidence": "traced it"}]


def _contract(verdict="YES", findings=(), *, turns=_GROUNDED) -> CallResult:
    body = json.dumps({"verdict": verdict, "findings": list(findings)})
    return CallResult(f"I reviewed it.\n\n```json\n{body}\n```", 0, None, (), turns)


@pytest.fixture
def paths_repo(changed_repo, commit_all):
    """``changed_repo`` plus two more tracked files, so a cap can bite.

    The last commit is deliberately a **one-line** change, so ``--base HEAD~1``
    is the surface most likely to embarrass a rule that fires too eagerly.
    """
    (changed_repo / "CLAUDE.md").write_text(_CLAUDE_MD)
    (changed_repo / "b.py").write_text("B = 2\n" * 40)
    (changed_repo / "c.py").write_text("C = 3\n" * 40)
    commit_all(changed_repo, "more")
    (changed_repo / "a.py").write_text("def f():\n    return 3\n")
    commit_all(changed_repo, "one line")
    return changed_repo


def _stub(monkeypatch, replies: dict | None = None, *, default=None) -> None:
    replies = replies or {}
    default = default if default is not None else _contract()

    def _fake(self, prompt, mandate, tools, model):  # noqa: ANN001, ARG001
        for name, reply in replies.items():
            if f"You are {name} " in mandate:
                return reply
        return default

    monkeypatch.setattr(session_module.PanelSession, "_run_claude", _fake)


def _run(repo, *extra) -> int:
    return main(["--repo", str(repo), "--max-epochs", "1", "--no-parallel", *extra])


def _paths(repo, *extra) -> int:
    return _run(repo, "--paths", "a.py", "b.py", "c.py", *extra)


def _artefact(out: str) -> dict:
    return json.loads(out.split("--- JSON ---")[-1])


def _epoch_line(out: str) -> str:
    return next(ln for ln in out.splitlines() if ln.startswith("  epoch 1:"))


# --- the defect ---------------------------------------------------------------

def test_a_cap_breached_surface_is_named_in_the_run(paths_repo, capsys, monkeypatch):
    """The acceptance criterion, over the surface the real assembler produced.

    Every other channel reports this run as healthy: nothing errored, nothing was
    refused, every persona reviewed at a healthy turn count and voted, and the
    loop halted CONVERGED. Two of the three named files were never in the prompt.
    """
    _stub(monkeypatch)
    _paths(paths_repo, "--max-surface-bytes", "200")
    out = capsys.readouterr().out
    payload = _artefact(out)

    assert payload["surface"] == [{"epoch": 1, "mode": "paths", "omitted": 2,
                                   "truncated": False, "unreadable": 0,
                                   "unresolved": 0, "bounded": True,
                                   "degraded": True}]
    assert "SURFACE REDUCED" in out
    assert "2 file(s) omitted at the cap" in _epoch_line(out)
    # The state it must not be confused with is right there on the same lines.
    assert payload["halt_reason"] == "converged"
    assert all(r["status"] == "found_nothing" for r in payload["participation"])


def test_paths_that_matched_nothing_are_reported(paths_repo, capsys, monkeypatch):
    """The quietest of the three: the panel is told, in the surface; nobody else is."""
    _stub(monkeypatch)
    _run(paths_repo, "--paths", "a.py", "gone.py", "also_gone.py")
    out = capsys.readouterr().out

    assert _artefact(out)["surface"][0]["unresolved"] == 2
    assert "2 named path(s) matched no tracked file" in _epoch_line(out)


def test_a_file_cut_in_place_is_reported_as_its_own_loss(paths_repo, capsys,
                                                         monkeypatch):
    """Dropping whole files and cutting one mid-way are two losses, not one.

    The cap is a byte count and the unit of review is a file, so a reader is told
    which of the two happened rather than a single word covering both.
    """
    _stub(monkeypatch)
    _run(paths_repo, "--paths", "b.py", "--max-surface-bytes", "60")
    out = capsys.readouterr().out

    assert _artefact(out)["surface"][0]["truncated"] is True
    assert "1 file cut mid-way at the cap" in _epoch_line(out)
    assert "omitted at the cap" not in out, "nothing was dropped whole"


def test_the_epoch_line_says_what_was_lost_and_can_say_nothing(paths_repo, capsys,
                                                               monkeypatch):
    """The line that says what the mark MEANS, in both directions.

    A mark nobody explains is a mark a reader must reconstruct, and a mark that
    cannot fall silent is not a mark at all.
    """
    _stub(monkeypatch)
    _paths(paths_repo, "--max-surface-bytes", "200")
    cut = _epoch_line(capsys.readouterr().out)
    assert "SURFACE REDUCED" in cut, cut
    assert "3 found nothing" in cut, "and the tally it qualifies is still there"

    _stub(monkeypatch)
    _paths(paths_repo)
    assert "SURFACE REDUCED" not in _epoch_line(capsys.readouterr().out)


def test_a_file_that_could_not_be_read_is_reported_as_that(paths_repo, capsys,
                                                           commit_all, monkeypatch):
    """Named, reviewed by nobody, and not blamed on the cap.

    A binary file inside a named directory is a real reduction — the operator
    asked for those paths and one of them never reached a reviewer — and it is a
    different loss from the cap, so it is a different sentence.
    """
    (paths_repo / "logo.bin").write_bytes(b"\xff\xfe\x00\x01binary")
    commit_all(paths_repo, "binary")
    _stub(monkeypatch)
    _run(paths_repo, "--paths", "a.py", "logo.bin")
    out = capsys.readouterr().out

    assert _artefact(out)["surface"][0]["unreadable"] == 1
    assert "1 file(s) could not be read" in _epoch_line(out)
    assert "omitted at the cap" not in out, "and not blamed on the cap"


def test_a_reviewed_file_that_talks_about_the_cap_is_not_a_breach(
        paths_repo, capsys, commit_all, monkeypatch):
    """Why the run is told by the assembler and not by reading its own output.

    Recovering the breach by searching the surface for ``--- OMITTED`` or
    ``surface cap`` reads a fact off wording, and is wrong on the first file that
    discusses the cap. Not hypothetical: this project's own ``surface.py``
    contains both markers, and reviewing it in path mode is an ordinary run.
    """
    (paths_repo / "surface_ish.py").write_text(
        'NOTICE = "--- OMITTED (not shown) ---"\nWHY = "b.py: surface cap"\n'
        'CUT = "… [truncated at surface cap: b.py]"\n')
    commit_all(paths_repo, "markers")
    _stub(monkeypatch)
    _run(paths_repo, "--paths", "surface_ish.py")
    out = capsys.readouterr().out

    assert "SURFACE REDUCED" not in out
    assert _artefact(out)["surface"][0]["degraded"] is False


# --- the mirror image: a run that lost nothing must claim nothing -------------

def test_a_complete_path_surface_claims_nothing(paths_repo, capsys, monkeypatch):
    _stub(monkeypatch)
    _paths(paths_repo)
    out = capsys.readouterr().out

    assert "SURFACE REDUCED" not in out
    assert _artefact(out)["surface"][0]["degraded"] is False
    assert "panel participation" in out, "and the section is still always-on"


def test_a_healthy_diff_run_claims_nothing(paths_repo, capsys, monkeypatch):
    """A one-line diff is the surface most likely to embarrass an eager rule.

    Diff mode applies no cap at all (#27), so there is no breach to detect and
    none is invented — the artefact says the surface was unbounded rather than
    claiming a completeness nobody checked.
    """
    _stub(monkeypatch)
    _run(paths_repo, "--base", "HEAD~1")
    out = capsys.readouterr().out

    assert "SURFACE REDUCED" not in out
    assert _artefact(out)["surface"] == [{"epoch": 1, "mode": "diff", "omitted": 0,
                                          "truncated": False, "unreadable": 0,
                                          "unresolved": 0, "bounded": False,
                                          "degraded": False}]


def test_the_shipped_default_cap_does_not_fire_on_a_real_path_surface(
        paths_repo, capsys, monkeypatch):
    """The shipped default, exercised as shipped.

    If this ever goes red, every default path-mode run has started reporting a
    degradation it did not suffer — the mirror image of the defect, and the one
    that would make the report worse than silence.
    """
    _stub(monkeypatch)
    _run(paths_repo, "--paths", ".")
    out = capsys.readouterr().out

    assert "SURFACE REDUCED" not in out
    assert _artefact(out)["surface"][0]["degraded"] is False


# --- the property pair, deliberately -----------------------------------------

def test_a_reduced_surface_run_cannot_be_read_as_a_complete_one(paths_repo, capsys,
                                                                monkeypatch):
    _stub(monkeypatch)
    _paths(paths_repo, "--max-surface-bytes", "200")
    reduced = capsys.readouterr().out
    _stub(monkeypatch)
    _paths(paths_repo)
    whole = capsys.readouterr().out

    assert reduced != whole
    assert _artefact(reduced)["surface"] != _artefact(whole)["surface"]


def test_a_complete_run_is_not_reported_as_reduced(paths_repo, capsys, monkeypatch):
    """The pair's other half. "A degraded run cannot be read as a clean one"
    passes trivially if every run reports degraded, so this forbids that."""
    _stub(monkeypatch)
    _paths(paths_repo)
    out = capsys.readouterr().out

    assert "REDUCED" not in out
    assert not any(r["degraded"] for r in _artefact(out)["surface"])


# --- what it must not be confused with ---------------------------------------

def test_a_dry_run_reports_the_reduction(paths_repo, capsys, monkeypatch):
    """Nothing was spawned — and the surface was still gathered, and still cut.

    The opposite of #70's answer, on purpose. A turn count of 0 means the runtime
    did not say; this fact is ours, computed before any subprocess exists, and
    pre-flight confirmation is the dry run's only job.
    """
    _stub(monkeypatch)
    _paths(paths_repo, "--max-surface-bytes", "200", "--dry-run")
    out = capsys.readouterr().out
    payload = _artefact(out)

    assert payload["surface"][0]["degraded"] is True
    assert "SURFACE REDUCED" in out
    assert all(r["status"] == "not_spawned" for r in payload["participation"])
    assert "CALLED NO TOOL" not in out, "and nothing that was not measured"


def test_a_cut_surface_and_an_ungrounded_reviewer_are_two_marks(paths_repo, capsys,
                                                                monkeypatch):
    """One failure, one heading — and these are two failures, not one.

    #70's mark stays attachable to a persona on a cut surface: *the surface was
    reduced* and *the reviewer opened nothing* have different causes and either
    can happen without the other, so neither may absorb the other.
    """
    _stub(monkeypatch, {"Critic": _contract(turns=1)})
    _paths(paths_repo, "--max-surface-bytes", "200")
    out = capsys.readouterr().out

    assert "SURFACE REDUCED" in out
    assert "1 CALLED NO TOOL" in out
    marked = [ln for ln in out.splitlines() if "CALLED NO TOOL" in ln]
    assert any("(Critic)" in ln for ln in marked)


def test_a_persona_on_a_reduced_surface_still_votes(paths_repo, capsys, monkeypatch):
    """Inherited from #24, #26 and #67, not re-taken here.

    A model-controlled value that drives halting must not silently count and must
    not silently NOT-count either. #67's measurement is the argument: its starved
    persona voted NO and raised an UGLY, so discarding the vote would have
    UNLATCHED the circuit-breaker rather than merely losing a YES. The run
    records the reduction beside the vote and the human adjudicates.
    """
    # Half one: the YES votes still COUNT. Quorum is unanimity among whoever ran,
    # so a discarded vote would move it for a reason nobody can see (#26).
    _stub(monkeypatch)
    _paths(paths_repo, "--max-surface-bytes", "200")
    counted = _artefact(capsys.readouterr().out)
    assert counted["halt_reason"] == "converged", "a vote was silently discarded"
    assert counted["surface"][0]["degraded"] is True

    # Half two: and they still latch the breaker. #67's starved persona voted NO
    # and raised an UGLY, so discarding would have UNLATCHED it, not lost a YES.
    _stub(monkeypatch, {"Sentinel": _contract("NO", _UGLY, turns=_GROUNDED)})
    rc = _paths(paths_repo, "--max-surface-bytes", "200")
    payload = _artefact(capsys.readouterr().out)
    assert payload["halt_reason"] == "escalate_ugly", "the breaker still latched"
    assert payload["open_uglies"] == 1
    assert rc == 3


def test_the_artefact_states_the_surface_on_every_run(paths_repo, capsys,
                                                      monkeypatch):
    """Always-on where a cold read needs it: an absent key makes no claim."""
    _stub(monkeypatch)
    _run(paths_repo, "--base", "HEAD~1")
    payload = _artefact(capsys.readouterr().out)

    assert payload["surface"], "a healthy run says so too"
    assert payload["surface"][0]["mode"] == "diff"
