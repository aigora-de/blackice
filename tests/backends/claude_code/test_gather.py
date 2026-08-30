# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""Surface-gathering tests: a failed ``git diff`` must never look like a review.

Regression tests. Before the fix, ``PanelSession.gather`` ran ``git diff`` with
``capture_output=True`` and used only ``stdout``, discarding both the return code
and stderr. A mistyped base ref therefore exited 128 with empty stdout, and the
panel was handed the literal string ``(empty diff)``: every persona found
nothing, voted YES, and the loop halted **CONVERGED**. The tool reported GOOD on
a review it had never performed.

The fix is to refuse to assemble a surface git could not produce — and it is
**two** guards, not one: a non-zero return code, and empty stdout. They are tested
apart on purpose (#87). Removing the return-code guard used to leave the bad-ref
test green, because a bad ref produces no stdout and the *empty* guard raised in its
place: the test passed on a mechanism it does not name, and this module's docstring
claimed a coverage it did not have. Each test below now says which guard it holds,
and ``test_a_git_failure_that_still_printed_something_is_refused`` covers the state
where only the return code can save the run.

The last section is the consequence of that refusal (#69). Because an *empty*
surface is refused in both modes, the surface failure that can still reach a panel
is one that is **present and is not the one that was asked for** — and what each
epoch's surface cost is now recorded on the session, one record per gather, in the
order they were gathered. Also regression tests.
"""

from __future__ import annotations

import pytest

import subprocess

from kuang.backends.claude_code import PanelSession, SurfaceError
from kuang.backends.claude_code import surface as surface_module
from kuang.cli import main
from kuang.engine import ReviewSpec


def _session(changed_repo, **kw) -> PanelSession:
    return PanelSession(repo_root=changed_repo, spec=ReviewSpec(why="w", what="x"),
                        personas={}, base="HEAD~1", head="HEAD", **kw)


# --- the defect ---------------------------------------------------------------

def test_a_bad_base_ref_raises_instead_of_yielding_an_empty_surface(changed_repo):
    """And raises for the RIGHT reason: git failed, not "the diff was empty".

    Both guards are live in this state, so asserting only that something was raised
    let the empty-stdout guard stand in for the return-code guard and kept the test
    green without it (#87). The distinction is not cosmetic — the two messages send
    an operator to different mistakes.
    """
    session = _session(changed_repo)
    session.base = "no-such-ref"

    with pytest.raises(SurfaceError) as excinfo:
        session.gather(1)

    assert "failed (" in str(excinfo.value), "the empty-diff guard stood in for it"


def test_a_git_failure_that_still_printed_something_is_refused(changed_repo,
                                                              monkeypatch):
    """Non-zero exit WITH output on stdout — the one state only this guard covers.

    Nothing exercised it before (#87). git can fail after writing a partial diff:
    a broken index, an unreadable object part-way through, a `dubious ownership`
    refusal. The empty-stdout guard cannot help there, and a partial diff presented
    as a whole one is exactly the review-that-never-happened this module exists for.

    The subprocess is stubbed because the state is hard to provoke reliably from a
    real repo, and this is characterisation of OUR handling, not of git.
    """
    def _fake_run(argv, **kw):  # noqa: ANN001, ARG001
        return subprocess.CompletedProcess(
            argv, 128,
            stdout="diff --git a/a.py b/a.py\n@@ -1 +1 @@\n-return 1\n",
            stderr="fatal: unable to read tree abc123")

    monkeypatch.setattr(surface_module.subprocess, "run", _fake_run)
    session = _session(changed_repo)

    with pytest.raises(SurfaceError) as excinfo:
        session.gather(1)

    message = str(excinfo.value)
    assert "failed (128)" in message
    assert "unable to read tree" in message


def test_the_failure_names_the_refs_and_carries_git_s_own_message(changed_repo):
    session = _session(changed_repo)
    session.base = "no-such-ref"

    with pytest.raises(SurfaceError) as excinfo:
        session.gather(1)

    message = str(excinfo.value)
    assert "no-such-ref" in message          # which ref the operator got wrong
    assert "unknown revision" in message     # git's diagnosis, previously discarded


def test_the_cli_reports_the_failure_and_does_not_report_a_verdict(changed_repo, capsys):
    rc = main(["--repo", str(changed_repo), "--base", "no-such-ref", "--dry-run"])

    captured = capsys.readouterr()
    assert rc != 0, "a run that never gathered a surface must not exit success"
    assert "no-such-ref" in captured.err
    assert "HALT" not in captured.out, "no halt verdict may be printed for a run that never ran"


# --- the happy paths stay exactly as they were --------------------------------

def test_a_real_diff_is_still_gathered(changed_repo):
    surface = _session(changed_repo).gather(1)

    assert "return 2" in surface


def test_an_empty_diff_raises_rather_than_being_reviewed(changed_repo):
    # Same ruin by a different road: git succeeds, the diff is empty, and the
    # panel unanimously approves a change it cannot see.
    session = _session(changed_repo)
    session.base = "HEAD"

    with pytest.raises(SurfaceError):
        session.gather(1)


def test_path_mode_with_no_reviewable_files_raises(changed_repo):
    session = _session(changed_repo, paths=["does_not_exist.py"])

    with pytest.raises(SurfaceError) as excinfo:
        session.gather(1)

    assert "does_not_exist.py" in str(excinfo.value)


def test_path_mode_does_not_touch_the_diff_path(changed_repo):
    """With a base ref that CANNOT resolve, so taking the diff path would raise.

    The previous form passed a valid ``base``, so a gather that ran ``git diff`` as
    well succeeded invisibly and the test held nothing its siblings did not (#87).
    The empty base is not a contrivance: ``cli.main`` passes ``base=args.base or ""``
    and path mode never requires ``--base``, so an empty ref is what a real
    path-mode run actually carries.
    """
    session = _session(changed_repo, paths=["a.py"])
    session.base = ""

    assert "--- FILE: a.py" in session.gather(1)


# --- what the run knows about the surface it gathered (#69) -------------------
#
# ``gather`` refuses an EMPTY surface (above), so the reachable failure is a
# surface that is PRESENT and is not the one the operator asked for. The record
# is kept here, on the session, one per epoch in order — the same place and the
# same shape as ``reduce_states`` (#30), and for the same reason: the engine's
# ``GatherSurface`` seam returns the surface text and nothing else, it never
# reads this and never branches on it, and widening it would oblige every
# implementation to produce a diagnostic about a cap it does not have.

def test_a_diff_mode_gather_records_that_no_cap_was_applied(changed_repo):
    session = _session(changed_repo)
    session.gather(1)

    assert len(session.surface_states) == 1
    rec = session.surface_states[0]
    assert rec.mode == "diff"
    assert rec.bounded is False, "diff mode ignores the cap entirely (#27)"
    assert rec.degraded is False, "and losing nothing is not a degradation"


def test_a_path_mode_gather_records_what_the_cap_dropped(changed_repo):
    (changed_repo / "b.py").write_text("B = 2\n" * 40)
    session = _session(changed_repo, paths=["a.py", "b.py"], max_surface_bytes=60)
    session.gather(1)

    rec = session.surface_states[0]
    assert rec.mode == "paths"
    assert rec.bounded is True
    assert rec.degraded is True
    assert rec.omitted == 1


def test_one_record_per_epoch_in_order(changed_repo):
    """The surface is re-gathered every epoch, so the record is per epoch.

    An epoch whose surface was cut and one whose surface was whole must not be
    collapsed into a single claim about the run.
    """
    (changed_repo / "b.py").write_text("B = 2\n" * 40)
    session = _session(changed_repo, paths=["a.py", "b.py"], max_surface_bytes=60)
    session.gather(1)
    session.max_surface_bytes = 100_000
    session.gather(2)

    assert [r.degraded for r in session.surface_states] == [True, False]


def test_a_refused_surface_records_nothing(changed_repo):
    """A run that stops is not a run that reviewed a reduced surface.

    ``SurfaceError`` ends the run before any persona exists, so there is no
    reduced surface to report and none is invented.
    """
    session = _session(changed_repo)
    session.base = "no-such-ref"
    with pytest.raises(SurfaceError):
        session.gather(1)

    assert session.surface_states == []
