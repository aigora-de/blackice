# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""Surface-gathering tests: a failed ``git diff`` must never look like a review.

Regression tests. Before the fix, ``PanelSession.gather`` ran ``git diff`` with
``capture_output=True`` and used only ``stdout``, discarding both the return code
and stderr. A mistyped base ref therefore exited 128 with empty stdout, and the
panel was handed the literal string ``(empty diff)``: every persona found
nothing, voted YES, and the loop halted **CONVERGED**. The tool reported GOOD on
a review it had never performed.

The fix is to refuse to assemble a surface git could not produce. These tests go
red if the return code is ignored again.
"""

from __future__ import annotations

import pytest

from blackice.backends.claude_code import PanelSession, SurfaceError
from blackice.cli import main
from blackice.engine import ReviewSpec


def _session(changed_repo, **kw) -> PanelSession:
    return PanelSession(repo_root=changed_repo, spec=ReviewSpec(why="w", what="x"),
                        personas={}, base="HEAD~1", head="HEAD", **kw)


# --- the defect ---------------------------------------------------------------

def test_a_bad_base_ref_raises_instead_of_yielding_an_empty_surface(changed_repo):
    session = _session(changed_repo)
    session.base = "no-such-ref"

    with pytest.raises(SurfaceError):
        session.gather(1)


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
    session = _session(changed_repo, paths=["a.py"])

    assert "--- FILE: a.py" in session.gather(1)
