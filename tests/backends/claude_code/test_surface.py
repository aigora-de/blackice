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

import subprocess
from dataclasses import fields

import pytest

from kuang.backends.claude_code.surface import (SurfaceError, _expand_paths,
                                                   _render_file,
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
    text, rec = gather_diff(changed_repo, "HEAD~1", "HEAD")
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


def test_only_names_and_counts_cross_the_boundary(git_repo, commit_all):
    """The record is a fact ABOUT the surface, never the surface.

    It is written into an artefact that gets pasted into a public repo's issues,
    and a review surface is unbounded content — whole files, or a whole diff. The
    same narrowing #67 made for a refused call's arguments.

    **Converted, not weakened, by #74** (it was ``test_only_counts_cross_the_
    boundary``). Names now cross too, and the argument for the narrowing is on
    the record: a path is a name, not content; every persona is already handed
    the same list of dropped paths in its prompt; a path is bounded by the
    filesystem where a file is not; and a surface record is our data, not a
    model's. The half that forbids **content** is unchanged, and is strengthened
    below with the content of an OMITTED file — a path the old test never
    exercised, because until now an omitted file had no field to leak into.
    """
    (git_repo / "a.py").write_text("A = 1\n")
    (git_repo / "b.py").write_text("SECRET_LOOKING_TOKEN = 2\n")
    commit_all(git_repo)
    cap = _cap_for(_render_file("a.py", "A = 1"))
    _, rec = build_path_surface(git_repo, ["a.py", "b.py"], max_bytes=cap)

    assert {f.name for f in fields(rec)} == {
        "mode", "bounded", "size", "files", "cap", "refs", "paths",
        "omitted_files", "truncated_file", "unreadable_files", "unresolved_paths"}
    assert rec.omitted_files == ("b.py",), "the NAME of the dropped file crosses"
    assert "SECRET_LOOKING_TOKEN" not in repr(rec), "its CONTENT does not"


# --- what the surface WAS, not only how it fell short (#74) ------------------
#
# #69 recorded the difference between the surface that was asked for and the one
# that was produced, and nothing else: two runs over entirely different files
# were byte-identical in their records. These pin the composition beside the
# loss, so an archived run can be compared with a later one over the "same"
# surface — and so a dropped file is NAMED, which is this issue's criterion and
# was deliberately left out of #69's.

def test_the_record_says_what_the_surface_was_made_of(git_repo, commit_all):
    """The composition, on a run where nothing was lost at all.

    A record that only speaks up when something went wrong cannot answer *what
    was reviewed*, which is #40's fourth exit criterion and not its first.
    """
    (git_repo / "a.py").write_text("A = 1\n")
    (git_repo / "b.py").write_text("B = 2\n")
    commit_all(git_repo)
    surface, rec = build_path_surface(git_repo, ["a.py", "b.py"], max_bytes=10_000)

    assert rec.paths == ("a.py", "b.py"), "the operator's own argv, as given"
    assert rec.files == 2
    assert rec.size == len(surface)
    assert rec.cap == 10_000
    assert rec.degraded is False


def test_two_surfaces_that_lost_nothing_are_told_apart(git_repo, commit_all):
    """The regression, at the source: #69's record could not tell these apart.

    Both runs are healthy, so every loss counter is zero in both. If the record
    is equal here, no artefact downstream can say which files were reviewed.
    """
    (git_repo / "a.py").write_text("A = 1\n")
    (git_repo / "b.py").write_text("B = 2\n" * 20)
    commit_all(git_repo)
    _, first = build_path_surface(git_repo, ["a.py"], max_bytes=10_000)
    _, second = build_path_surface(git_repo, ["b.py"], max_bytes=10_000)

    assert first.degraded is False and second.degraded is False
    assert first != second
    assert (first.paths, first.size) != (second.paths, second.size)


def test_files_dropped_at_the_cap_are_named_in_order(git_repo, commit_all):
    """Named, not just counted — and in the order the assembler dropped them.

    An operator deciding whether to re-run at a higher cap needs to know which
    files never reached a reviewer; a count tells them only how many.
    """
    (git_repo / "a.py").write_text("A = 1\n")
    (git_repo / "b.py").write_text("B = 2\n")
    (git_repo / "c.py").write_text("C = 3\n")
    commit_all(git_repo)
    cap = _cap_for(_render_file("a.py", "A = 1"))
    _, rec = build_path_surface(git_repo, ["a.py", "b.py", "c.py"], max_bytes=cap)

    assert rec.omitted_files == ("b.py", "c.py")
    assert rec.omitted == len(rec.omitted_files), "one source of truth, not two"
    assert rec.files == 1, "and what DID fit is counted"


def test_the_file_cut_mid_way_is_named(tmp_path):
    """The truncation is one named file, so the record names it rather than
    carrying a bare flag a reader cannot act on."""
    (tmp_path / "big.py").write_text("\n".join(f"line{i}" for i in range(500)) + "\n")
    _, rec = build_path_surface(tmp_path, ["big.py"], max_bytes=200)

    assert rec.truncated_file == "big.py"
    assert rec.truncated is True, "and the flag still derives from the name"
    assert rec.omitted_files == ()


def test_unreadable_files_and_unresolved_paths_are_named_apart(git_repo, commit_all):
    """Four losses, four names, and never one blamed on another.

    A run that named a binary file among the cap's casualties would be the
    instrument lying about itself, which is this epoch's whole subject.
    """
    (git_repo / "a.py").write_text("A = 1\n")
    (git_repo / "img.bin").write_bytes(b"\xff\xfe\x00\x01binary")
    commit_all(git_repo)
    _, rec = build_path_surface(
        git_repo, ["a.py", "img.bin", "gone.py", "also_gone.py"], max_bytes=10_000)

    assert rec.unreadable_files == ("img.bin",)
    assert rec.unresolved_paths == ("gone.py", "also_gone.py")
    assert rec.omitted_files == (), "nothing was dropped at the cap"
    assert (rec.unreadable, rec.unresolved) == (1, 2)


def test_a_complete_surface_names_nothing(git_repo, commit_all):
    """The mirror image of every test above: nothing lost, nothing named."""
    (git_repo / "a.py").write_text("A = 1\n")
    commit_all(git_repo)
    _, rec = build_path_surface(git_repo, ["a.py"], max_bytes=10_000)

    assert rec.omitted_files == ()
    assert rec.unreadable_files == ()
    assert rec.unresolved_paths == ()
    assert rec.truncated_file is None
    assert rec.degraded is False


def test_a_name_is_never_recovered_from_the_surface_text(git_repo, commit_all):
    """The rejected design, killed at the source.

    Recovering the dropped files by parsing ``--- OMITTED ---`` back out of the
    string reads a fact off wording, and is wrong on the first file that
    discusses the cap — this project's own ``surface.py`` contains both markers.
    The names come from the assembler, which is the only thing that knows.
    """
    (git_repo / "surface_ish.py").write_text(
        'NOTICE = "--- OMITTED (not shown) ---"\nWHY = "- b.py: surface cap"\n')
    commit_all(git_repo)
    text, rec = build_path_surface(git_repo, ["surface_ish.py"], max_bytes=10_000)

    assert "- b.py: surface cap" in text, "the surface really does say it"
    assert rec.omitted_files == (), "and the record is not fooled by it"
    assert rec.degraded is False


def test_the_size_is_the_surface_as_handed_over_notices_included(git_repo,
                                                                 commit_all):
    """``size`` can exceed ``cap``, and that is the honest number.

    The cap bounds the rendered file content; the assembler's own OMITTED notice
    rides on top of it, and a persona is handed both. Reporting the bounded
    figure instead would report a size nobody was given — and it is the surface,
    not the cap, that a later run is compared against.
    """
    (git_repo / "a.py").write_text("A = 1\n")
    (git_repo / "b.py").write_text("B = 2\n")
    commit_all(git_repo)
    cap = _cap_for(_render_file("a.py", "A = 1"))
    surface, rec = build_path_surface(git_repo, ["a.py", "b.py"], max_bytes=cap)

    assert rec.size == len(surface)
    assert rec.size > rec.cap, "the notice about the loss is part of the surface"
    assert rec.omitted_files == ("b.py",), "and the loss itself is unaffected"


def test_a_diff_record_states_its_refs_size_and_file_count(changed_repo):
    """Diff mode is the default, and "412 bytes" is a weak answer to *of what*.

    The count comes from a second ``git diff --name-only``, which is
    authoritative; parsing it out of the diff text is the design rejected above.
    """
    text, rec = gather_diff(changed_repo, "HEAD~1", "HEAD")

    assert rec.mode == "diff"
    assert rec.refs == ("HEAD~1", "HEAD")
    assert rec.size == len(text)
    assert rec.files == 1
    assert rec.cap is None and rec.bounded is False
    assert rec.degraded is False
    assert (rec.omitted_files, rec.unreadable_files,
            rec.unresolved_paths) == ((), (), ())
    assert rec.truncated_file is None


def test_a_diff_file_count_git_will_not_give_is_said_aloud_not_guessed(
        changed_repo, monkeypatch):
    """A second git call must never kill the run it only describes.

    ``files is None`` is the same shape as ``turn count unreported`` (#70): a
    fact nobody could establish is said aloud, never defaulted to a number that
    reads like a measurement.
    """
    real = subprocess.run

    def _fake(cmd, *a, **kw):  # noqa: ANN001, ANN002, ANN003
        if "--name-only" in cmd:
            return subprocess.CompletedProcess(cmd, 128, "", "fatal: no")
        return real(cmd, *a, **kw)

    monkeypatch.setattr("kuang.backends.claude_code.surface.subprocess.run", _fake)
    text, rec = gather_diff(changed_repo, "HEAD~1", "HEAD")

    assert text, "the run is unharmed"
    assert rec.files is None
    assert rec.size == len(text), "and everything knowable is still known"
