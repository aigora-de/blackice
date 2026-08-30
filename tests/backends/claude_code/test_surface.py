# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""Tests for the path / whole-file review surface (issue #6, then #69).

These exercise ``surface.build_path_surface`` and its helper ``_expand_paths`` in
isolation — no ``claude`` subprocess, no network. A real git repo is created under
``tmp_path`` so directory expansion and ``.gitignore`` honouring are tested
against actual ``git ls-files`` behaviour, not a mock of it.

Two sections, and they pin different things. The first four pin the **surface
text** a persona is handed (#6): what is rendered, what the cap does to it, and
what a path that matched nothing looks like in it. The last section pins the
**record returned beside that text** (#69) — what the assembler knows about the
difference between the surface it was asked for and the surface it produced.
That difference was computed here all along and discarded, which is why a panel
handed one of the reduced surfaces below reported ``found_nothing`` and the run
halted ``converged``. Every test in both sections is a regression test.
"""

from __future__ import annotations

from dataclasses import fields

import pytest

from kuang.backends.claude_code.surface import (DIFF_SURFACE, SurfaceError,
                                                   _expand_paths, _render_file,
                                                   build_path_surface,
                                                   gather_diff)


# --- rendering: full content with citable line numbers ----------------------

def test_single_file_rendered_with_line_numbers(tmp_path):
    """A plain file (no git needed) is rendered whole, with 1-based line numbers."""
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    surface, _ = build_path_surface(tmp_path, ["calc.py"], max_bytes=10_000)
    assert "--- FILE: calc.py (2 lines) ---" in surface
    assert "1| def add(a, b):" in surface
    assert "2|     return a + b" in surface


# --- directory expansion via git ls-files -----------------------------------

def test_directory_expanded_via_git_ls_files(git_repo, commit_all):
    (git_repo / "pkg").mkdir()
    (git_repo / "pkg" / "a.py").write_text("A = 1\n")
    (git_repo / "pkg" / "b.py").write_text("B = 2\n")
    commit_all(git_repo)
    surface, _ = build_path_surface(git_repo, ["pkg"], max_bytes=10_000)
    assert "--- FILE: pkg/a.py" in surface
    assert "--- FILE: pkg/b.py" in surface
    assert "A = 1" in surface and "B = 2" in surface


def test_gitignored_file_excluded_from_directory(git_repo, commit_all):
    """`.gitignore` is honoured for free: an ignored (untracked) file is absent."""
    (git_repo / "pkg").mkdir()
    (git_repo / "pkg" / "keep.py").write_text("KEEP = 1\n")
    (git_repo / "pkg" / "secret.env").write_text("TOKEN=xyz\n")
    (git_repo / ".gitignore").write_text("*.env\n")
    commit_all(git_repo)
    surface, _ = build_path_surface(git_repo, ["pkg"], max_bytes=10_000)
    assert "keep.py" in surface
    assert "secret.env" not in surface
    assert "TOKEN=xyz" not in surface


def test_expand_paths_dedups_file_and_dir_overlap(git_repo, commit_all):
    (git_repo / "pkg").mkdir()
    (git_repo / "pkg" / "a.py").write_text("A = 1\n")
    commit_all(git_repo)
    files, missing = _expand_paths(git_repo, ["pkg", "pkg/a.py"])
    assert len(files) == 1
    assert missing == []


# --- the total-size cap: truncate, never silently drop ----------------------

def test_surface_cap_omits_and_names_dropped_file(git_repo, commit_all):
    (git_repo / "a.py").write_text("A = 1\n")
    (git_repo / "b.py").write_text("B = 2\n")
    commit_all(git_repo)
    cap = len(_render_file("a.py", "A = 1")) + 5  # room for a.py, not b.py
    surface, _ = build_path_surface(git_repo, ["a.py", "b.py"], max_bytes=cap)
    assert "A = 1" in surface
    assert "B = 2" not in surface
    assert "--- OMITTED" in surface
    assert "b.py: surface cap" in surface


def test_single_file_over_cap_is_truncated_in_place(tmp_path):
    """A lone file bigger than the cap is truncated (with a marker), not dropped."""
    (tmp_path / "big.py").write_text("\n".join(f"line{i}" for i in range(500)) + "\n")
    surface, _ = build_path_surface(tmp_path, ["big.py"], max_bytes=200)
    assert "truncated at surface cap" in surface
    assert "big.py" in surface
    assert len(surface) < 1_000  # the cap is actually honoured


# --- missing / untracked paths are surfaced, not swallowed -------------------

def test_missing_path_is_reported(git_repo, commit_all):
    (git_repo / "real.py").write_text("R = 1\n")
    commit_all(git_repo)
    surface, _ = build_path_surface(git_repo, ["real.py", "does_not_exist.py"], max_bytes=10_000)
    assert "R = 1" in surface
    assert "PATHS WITH NO TRACKED FILES" in surface
    assert "does_not_exist.py" in surface


def test_no_reviewable_files_raises(git_repo, commit_all):
    # A surface with nothing in it is an operator error, not a review: returning
    # a placeholder let the panel "approve" it and the loop halt CONVERGED.
    with pytest.raises(SurfaceError) as excinfo:
        build_path_surface(git_repo, ["nope.py"], max_bytes=10_000)
    assert "nope.py" in str(excinfo.value)

# --- what the assembler knows about what it dropped (#69) -------------------
#
# The cap and the missing-path notice have always been computed here and thrown
# away: ``build_path_surface`` returned a string, so a panel handed one of the
# three reduced surfaces below was indistinguishable, in the run's own output,
# from a panel handed everything it asked for. These pin the record it now
# returns beside the text. Regression tests, all of them.

def _cap_for(*rendered: str) -> int:
    """The exact cap that fits the given rendered files and nothing more."""
    return sum(len(r) for r in rendered)


def test_a_complete_surface_records_no_reduction(git_repo, commit_all):
    """The mirror image, at the source: nothing was lost, so nothing is claimed."""
    (git_repo / "a.py").write_text("A = 1\n")
    (git_repo / "b.py").write_text("B = 2\n")
    commit_all(git_repo)
    _, rec = build_path_surface(git_repo, ["a.py", "b.py"], max_bytes=10_000)
    assert rec.mode == "paths"
    assert (rec.omitted, rec.truncated, rec.unresolved) == (0, False, 0)
    assert rec.bounded is True
    assert rec.degraded is False


def test_files_dropped_at_the_cap_are_counted(git_repo, commit_all):
    (git_repo / "a.py").write_text("A = 1\n")
    (git_repo / "b.py").write_text("B = 2\n")
    (git_repo / "c.py").write_text("C = 3\n")
    commit_all(git_repo)
    cap = _cap_for(_render_file("a.py", "A = 1"))
    _, rec = build_path_surface(git_repo, ["a.py", "b.py", "c.py"], max_bytes=cap)
    assert rec.omitted == 2
    assert rec.truncated is False
    assert rec.degraded is True


def test_a_file_cut_in_place_is_recorded_as_truncated(tmp_path):
    """Cutting one file mid-way is a different loss from dropping whole files.

    The cap is a BYTE count and the unit of review is a FILE, so the assembler
    has two ways to lose content and they are recorded as two facts.
    """
    (tmp_path / "big.py").write_text("\n".join(f"line{i}" for i in range(500)) + "\n")
    _, rec = build_path_surface(tmp_path, ["big.py"], max_bytes=200)
    assert rec.truncated is True
    assert rec.omitted == 0
    assert rec.degraded is True


def test_a_cut_first_file_and_the_files_behind_it_are_both_recorded(git_repo,
                                                                    commit_all):
    """Both losses at once: nothing fits, so the first is cut and the rest go."""
    (git_repo / "a.py").write_text("A = 1\n" * 50)
    (git_repo / "b.py").write_text("B = 2\n")
    (git_repo / "c.py").write_text("C = 3\n")
    commit_all(git_repo)
    _, rec = build_path_surface(git_repo, ["a.py", "b.py", "c.py"], max_bytes=20)
    assert rec.truncated is True
    assert rec.omitted == 2


def test_paths_that_matched_no_tracked_file_are_counted(git_repo, commit_all):
    """The third way the surface differs from what was named, and the quietest.

    The panel is told, in the surface text; the operator is told nothing at all.
    """
    (git_repo / "real.py").write_text("R = 1\n")
    commit_all(git_repo)
    _, rec = build_path_surface(
        git_repo, ["real.py", "gone.py", "also_gone.py"], max_bytes=10_000)
    assert rec.unresolved == 2
    assert (rec.omitted, rec.truncated) == (0, False)
    assert rec.degraded is True


def test_a_file_that_could_not_be_read_is_counted_apart_from_the_cap(git_repo,
                                                                    commit_all):
    """Two reasons, two counts. A run that blamed a binary file on the cap would
    be the instrument lying about itself, which is the epoch's whole subject."""
    (git_repo / "a.py").write_text("A = 1\n")
    (git_repo / "img.bin").write_bytes(b"\xff\xfe\x00\x01binary")
    commit_all(git_repo)
    _, rec = build_path_surface(git_repo, ["a.py", "img.bin"], max_bytes=10_000)
    assert rec.unreadable == 1
    assert rec.omitted == 0, "nothing was dropped at the cap"
    assert rec.degraded is True


@pytest.mark.parametrize("cap_delta,expected", [
    (0, (0, False)),     # exactly at the cap: the comparison is strict, so it fits
    (-1, (1, False)),    # one byte over: the LAST file goes, whole
    (-7, (1, False)),    # a byte under the second file: still one whole file lost
])
def test_the_cap_boundary_is_exact(git_repo, commit_all, cap_delta, expected):
    """(input, expected) over the boundary, including a row that must stay clean.

    The row at the cap is the one that keeps this test honest: an assembler that
    reported a reduction unconditionally would satisfy the other two.
    """
    (git_repo / "a.py").write_text("A = 1\n")
    (git_repo / "b.py").write_text("B = 2\n")
    commit_all(git_repo)
    cap = _cap_for(_render_file("a.py", "A = 1"), _render_file("b.py", "B = 2"))
    _, rec = build_path_surface(git_repo, ["a.py", "b.py"], max_bytes=cap + cap_delta)
    assert (rec.omitted, rec.truncated) == expected


def test_a_diff_surface_records_that_no_cap_was_applied(changed_repo):
    """Diff mode is not bounded at all (#27), and the record says so rather than
    claiming a completeness nobody checked."""
    text, rec = gather_diff(changed_repo, "HEAD~1", "HEAD"), DIFF_SURFACE
    assert text
    assert rec.mode == "diff"
    assert rec.bounded is False
    assert rec.degraded is False, "an unbounded surface lost nothing; it risks more"


def test_a_file_that_talks_about_the_cap_is_not_a_cap_breach(git_repo, commit_all):
    """Why the record is returned rather than read back out of the surface text.

    Recovering the breach by searching the string for ``--- OMITTED`` or
    ``surface cap`` reads a fact off wording — the defect ``PersonaStatus``
    forbids one layer up — and it is wrong on the first file that discusses the
    cap. That is not hypothetical: this project's own ``surface.py`` contains
    both markers, and reviewing it in path mode is an ordinary invocation.
    """
    (git_repo / "surface_ish.py").write_text(
        'NOTICE = "--- OMITTED (not shown) ---"\nWHY = "b.py: surface cap"\n')
    commit_all(git_repo)
    text, rec = build_path_surface(git_repo, ["surface_ish.py"], max_bytes=10_000)

    assert "surface cap" in text, "the surface really does contain the marker"
    assert rec.degraded is False, "and nothing was lost"


def test_only_counts_cross_the_boundary(git_repo, commit_all):
    """The record is a fact ABOUT the surface, never the surface.

    It is written into an artefact that gets pasted into a public repo's issues,
    and a review surface is unbounded content — whole files, or a whole diff. The
    same narrowing #67 made for a refused call's arguments.
    """
    (git_repo / "a.py").write_text("SECRET_LOOKING_TOKEN = 1\n")
    commit_all(git_repo)
    _, rec = build_path_surface(git_repo, ["a.py"], max_bytes=10_000)
    assert {f.name for f in fields(rec)} == {
        "mode", "omitted", "truncated", "unreadable", "unresolved", "bounded"}
    assert "SECRET_LOOKING_TOKEN" not in repr(rec)
