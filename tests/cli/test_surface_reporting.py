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

**Amended by #74.** #69 reported the loss **inside the participation section**,
beside the sentence that lies, and said nothing at all on a healthy run — leaving
"what was this composed of" to #11 and #74. #74 answers it: there is now an
always-on ``review surface:`` section stating what the panel was handed, and
#69's ``SURFACE REDUCED`` mark has **moved off the participation line into it**,
because leaving it on both makes two sections about one thing. Every #69
assertion below is preserved word for word and re-aimed at the surface line;
each converted test says so in its docstring. The always-on argument is #30's,
not #69's: *this is what the panel looked at* is a claim no absent section can
make, exactly as *the panel ran in full* is.

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


def _section(out: str, header: str) -> list[str]:
    """The lines of one report section, up to the blank line that ends it.

    Two sections now carry an ``epoch 1:`` line — what the panel was given and
    what it did with it — so a test that means one of them must say which.
    """
    lines = out.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith(header))
    end = next((i for i in range(start + 1, len(lines)) if not lines[i].strip()),
               len(lines))
    return lines[start:end]


def _surface_lines(out: str) -> list[str]:
    return _section(out, "review surface:")


def _surface_line(out: str) -> str:
    """Epoch 1's line in the surface section: what the panel was handed."""
    return next(ln for ln in _surface_lines(out) if ln.startswith("  epoch 1:"))


def _epoch_line(out: str) -> str:
    """Epoch 1's line in the participation section: what the panel did."""
    return next(ln for ln in _section(out, "panel participation:")
                if ln.startswith("  epoch 1:"))


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
                                   "degraded": True, "size": 127, "files": 1,
                                   "cap": 200, "refs": None,
                                   "paths": ["a.py", "b.py", "c.py"],
                                   "omitted_files": ["b.py", "c.py"],
                                   "truncated_file": None,
                                   "unreadable_files": [], "unresolved_paths": []}]
    assert "SURFACE REDUCED" in out
    # Converted by #74: the mark moved from the participation line onto the
    # surface line. The same sentence, one section along.
    assert "2 file(s) omitted at the cap" in _surface_line(out)
    # And #74's own criterion: the files that were dropped are NAMED.
    assert "omitted at the cap (2): b.py, c.py" in out
    # The state it must not be confused with is right there on the same lines.
    assert payload["halt_reason"] == "converged"
    assert all(r["status"] == "found_nothing" for r in payload["participation"])


def test_paths_that_matched_nothing_are_reported(paths_repo, capsys, monkeypatch):
    """The quietest of the three: the panel is told, in the surface; nobody else is."""
    _stub(monkeypatch)
    _run(paths_repo, "--paths", "a.py", "gone.py", "also_gone.py")
    out = capsys.readouterr().out

    assert _artefact(out)["surface"][0]["unresolved"] == 2
    # Converted by #74: asserted on the surface line, and the paths are named.
    assert "2 named path(s) matched no tracked file" in _surface_line(out)
    assert "matched no tracked file (2): gone.py, also_gone.py" in out


def test_a_file_cut_in_place_is_reported_as_its_own_loss(paths_repo, capsys,
                                                         monkeypatch):
    """Dropping whole files and cutting one mid-way are two losses, not one.

    The cap is a byte count and the unit of review is a file, so a reader is told
    which of the two happened rather than a single word covering both.
    """
    _stub(monkeypatch)
    _run(paths_repo, "--paths", "b.py", "--max-surface-bytes", "60")
    out = capsys.readouterr().out

    record = _artefact(out)["surface"][0]
    assert record["truncated"] is True
    # Converted by #74: asserted on the surface line, and the file is named.
    assert "1 file cut mid-way at the cap" in _surface_line(out)
    assert record["truncated_file"] == "b.py"
    assert "cut mid-way at the cap (1): b.py" in out
    assert "omitted at the cap" not in out, "nothing was dropped whole"


def test_the_epoch_line_says_what_was_lost_and_can_say_nothing(paths_repo, capsys,
                                                               monkeypatch):
    """The line that says what the mark MEANS, in both directions.

    A mark nobody explains is a mark a reader must reconstruct, and a mark that
    cannot fall silent is not a mark at all.
    """
    _stub(monkeypatch)
    _paths(paths_repo, "--max-surface-bytes", "200")
    out = capsys.readouterr().out
    cut = _surface_line(out)
    assert "SURFACE REDUCED" in cut, cut
    # Converted by #74: the tally the mark used to sit beside is one section
    # down; what the mark qualifies is now on the mark's own line.
    assert "3 found nothing" in _epoch_line(out), "and the tally is still there"
    assert "1 file(s), 127 bytes" in cut, "beside what the panel actually got"

    _stub(monkeypatch)
    _paths(paths_repo)
    assert "SURFACE REDUCED" not in _surface_line(capsys.readouterr().out)


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
    # Converted by #74: asserted on the surface line, and the file is named.
    assert "1 file(s) could not be read" in _surface_line(out)
    assert "could not be read (1): logo.bin" in out
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
    record = _artefact(out)["surface"][0]
    assert record == {"epoch": 1, "mode": "diff", "omitted": 0,
                      "truncated": False, "unreadable": 0, "unresolved": 0,
                      "bounded": False, "degraded": False,
                      "size": record["size"], "files": 1, "cap": None,
                      "refs": ["HEAD~1", "HEAD"], "paths": [],
                      "omitted_files": [], "truncated_file": None,
                      "unreadable_files": [], "unresolved_paths": []}
    assert record["size"] > 0, "and the size is a measurement, not a placeholder"


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

    Unchanged in substance by #74, and deliberately: both marks still appear on
    the same run. They now appear in two different sections, which is the point
    — what the panel was given, and what it did with it.
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


# --- what the panel was GIVEN, said on every run (#74) ------------------------
#
# #69 built the channel and reported only how the surface fell short, so two runs
# over entirely different files were byte-identical in their artefacts. These pin
# the composition: an always-on section on the console, and the same facts in the
# artefact, which is what an archived run is compared from.

def test_a_healthy_path_run_says_what_the_panel_was_given(paths_repo, capsys,
                                                          monkeypatch):
    """Always-on, and the claim no absent section can make (#30's argument).

    Nothing was lost here, so #69's channel says nothing at all — and "what was
    reviewed" is exactly the question a run read back cold has to answer.
    """
    _stub(monkeypatch)
    _paths(paths_repo)
    out = capsys.readouterr().out
    record = _artefact(out)["surface"][0]

    assert "review surface: paths — a.py b.py c.py | cap 200000 bytes" in out
    assert _surface_line(out) == (
        f"  epoch 1: 3 file(s), {record['size']} bytes"
        " — every named path was included")
    assert (record["files"], record["cap"]) == (3, 200_000)
    assert record["paths"] == ["a.py", "b.py", "c.py"]
    assert record["degraded"] is False


def test_a_healthy_diff_run_says_what_the_panel_was_given(paths_repo, capsys,
                                                          monkeypatch):
    """Diff mode is the default, and it must not claim a bound it never applied.

    #27 is the reason the closing sentence here is a different sentence from path
    mode's: nothing was dropped because nothing could be, not because the whole
    change fitted inside a cap somebody checked.
    """
    _stub(monkeypatch)
    _run(paths_repo, "--base", "HEAD~1")
    out = capsys.readouterr().out
    record = _artefact(out)["surface"][0]

    assert ("review surface: diff — HEAD~1...HEAD | "
            "no cap was applied (diff mode is unbounded)") in out
    assert _surface_line(out) == (
        f"  epoch 1: 1 file(s), {record['size']} bytes"
        " — nothing was dropped, and no cap was applied")
    assert "every named path was included" not in out, "a claim nobody checked"


def test_a_dry_run_says_what_it_would_have_reviewed(paths_repo, capsys, monkeypatch):
    """Pre-flight confirmation is the dry run's only job, and this is it.

    The composition is known before any subprocess exists — the opposite of a
    turn count (#70) — so "you named three files and one fitted" is available
    without spending anything. The wording is about the surface and never about
    what anyone did with it, so nothing here claims a degradation a dry run did
    not suffer.
    """
    _stub(monkeypatch)
    _paths(paths_repo, "--max-surface-bytes", "200", "--dry-run")
    out = capsys.readouterr().out
    payload = _artefact(out)

    assert "review surface: paths — a.py b.py c.py | cap 200 bytes" in out
    assert "1 file(s), 127 bytes" in _surface_line(out)
    assert "omitted at the cap (2): b.py, c.py" in out
    assert all(r["status"] == "not_spawned" for r in payload["participation"])
    assert "CALLED NO TOOL" not in out, "and nothing that was not measured"


def test_two_healthy_path_runs_over_different_files_are_told_apart(
        paths_repo, capsys, monkeypatch):
    """The regression, end to end. Both runs are healthy in every channel.

    On ``main`` these two artefacts were identical: same mode, same four zeroed
    loss counters, same ``bounded``. A run recording only how it fell short
    cannot say what it reviewed.
    """
    _stub(monkeypatch)
    _run(paths_repo, "--paths", "a.py")
    first = capsys.readouterr().out
    _stub(monkeypatch)
    _run(paths_repo, "--paths", "b.py")
    second = capsys.readouterr().out

    assert _artefact(first)["surface"][0]["degraded"] is False
    assert _artefact(second)["surface"][0]["degraded"] is False
    assert _artefact(first)["surface"] != _artefact(second)["surface"]
    assert _surface_lines(first) != _surface_lines(second)


def test_two_healthy_diff_runs_over_different_refs_are_told_apart(
        paths_repo, capsys, monkeypatch):
    """The same regression in the mode 11 of the 13 golden invocations use."""
    _stub(monkeypatch)
    _run(paths_repo, "--base", "HEAD~1")
    first = capsys.readouterr().out
    _stub(monkeypatch)
    _run(paths_repo, "--base", "HEAD~2")
    second = capsys.readouterr().out

    assert _artefact(first)["surface"][0]["refs"] == ["HEAD~1", "HEAD"]
    assert _artefact(second)["surface"][0]["refs"] == ["HEAD~2", "HEAD"]
    assert _artefact(first)["surface"] != _artefact(second)["surface"]


def test_a_surface_that_shrinks_between_epochs_is_not_a_degradation(
        paths_repo, capsys, monkeypatch):
    """The intended workflow, which must not read as a loss.

    The surface is re-gathered every epoch, and fixes landing at the gate make it
    SMALLER. A rule that read a shrinking surface as a reduction would fire on
    the panel working exactly as designed — the mirror image of #69's defect.
    """
    calls: list[int] = []

    def _fake(self, prompt, mandate, tools, model):  # noqa: ANN001, ARG001
        calls.append(1)
        if len(calls) == 3:  # the last call of epoch 1: the fix lands
            (paths_repo / "b.py").write_text("B = 2\n")
        return _contract("NO")

    monkeypatch.setattr(session_module.PanelSession, "_run_claude", _fake)
    main(["--repo", str(paths_repo), "--max-epochs", "2", "--no-parallel",
          "--paths", "a.py", "b.py"])
    out = capsys.readouterr().out
    records = _artefact(out)["surface"]

    assert len(records) == 2, "the surface is gathered once per epoch"
    assert records[1]["size"] < records[0]["size"], "the fix landed"
    assert [r["degraded"] for r in records] == [False, False]
    assert [r["files"] for r in records] == [2, 2]
    assert "REDUCED" not in out
    epoch_lines = [ln for ln in _surface_lines(out) if ln.startswith("  epoch")]
    assert len(epoch_lines) == 2 and epoch_lines[0] != epoch_lines[1]


def test_a_file_count_git_will_not_give_is_said_aloud(paths_repo, capsys,
                                                      monkeypatch):
    """A second git call must never kill the run it only describes (#70's shape).

    The count is unavailable, so the line says so rather than printing a 0 that
    reads like a measurement — and the size, which IS known, is still reported.
    """
    import subprocess

    from kuang.backends.claude_code import surface as surface_module
    real = subprocess.run

    def _fake_git(cmd, *a, **kw):  # noqa: ANN001, ANN002, ANN003
        if "--name-only" in cmd:
            return subprocess.CompletedProcess(cmd, 128, "", "fatal: no")
        return real(cmd, *a, **kw)

    monkeypatch.setattr(surface_module.subprocess, "run", _fake_git)
    _stub(monkeypatch)
    rc = _run(paths_repo, "--base", "HEAD~1")
    out = capsys.readouterr().out

    assert rc == 0, "the run is unharmed"
    assert "file count unreported" in _surface_line(out)
    assert _artefact(out)["surface"][0]["files"] is None
    assert "bytes" in _surface_line(out), "and what IS known is still said"


def test_the_included_files_are_counted_and_never_named(paths_repo, capsys,
                                                        monkeypatch):
    """The boundary #69's guard holds, restated where the artefact is written.

    The files the operator named and did NOT get are named; the files they got
    are counted. Naming everything included approaches shipping the surface
    itself into an artefact that gets pasted into a public repo's issues.
    """
    _stub(monkeypatch)
    _paths(paths_repo, "--max-surface-bytes", "200")
    payload = _artefact(capsys.readouterr().out)
    record = payload["surface"][0]

    assert record["files"] == 1, "what fitted is a count"
    assert record["omitted_files"] == ["b.py", "c.py"], "what did not is named"
    assert "def f():" not in json.dumps(record), "and no content crosses at all"


def test_the_reduction_is_marked_once_and_not_in_two_places(paths_repo, capsys,
                                                            monkeypatch):
    """#69's mark MOVED; it was not copied.

    Leaving it on the participation line as well would make two sections about
    one thing — the failure #74 names — and would let a reader double-count one
    degradation. The mark belongs on the line carrying the composition it
    qualifies.
    """
    _stub(monkeypatch)
    _paths(paths_repo, "--max-surface-bytes", "200")
    out = capsys.readouterr().out

    assert out.count("SURFACE REDUCED") == 1
    assert "SURFACE REDUCED" in _surface_line(out)
    assert "SURFACE REDUCED" not in _epoch_line(out)


def test_what_the_panel_was_given_is_reported_before_what_it_did(
        paths_repo, capsys, monkeypatch):
    """Order is the argument, so it is asserted rather than left to chance.

    A participation tally read before knowing what the panel was handed is a
    tally a reader has to revisit. Given, then done with it.
    """
    _stub(monkeypatch)
    _paths(paths_repo)
    out = capsys.readouterr().out

    assert out.index("review surface:") < out.index("panel participation:")
