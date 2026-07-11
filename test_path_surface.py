# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""Tests for the path / whole-file review surface (issue #6).

These exercise ``claude_code_backend.build_path_surface`` and its helper
``_expand_paths`` in isolation — no ``claude`` subprocess, no network. A real
git repo is created under ``tmp_path`` so directory expansion and ``.gitignore``
honouring are tested against actual ``git ls-files`` behaviour, not a mock of it.
"""

from __future__ import annotations

import subprocess

import pytest

from claude_code_backend import _expand_paths, _render_file, build_path_surface


def _git(repo, *args) -> None:
    subprocess.run(["git", "-C", str(repo), *args],
                   check=True, capture_output=True, text=True)


@pytest.fixture
def repo(tmp_path):
    """A minimal, committed git repo under ``tmp_path`` with a local identity."""
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "test@example.invalid")
    _git(tmp_path, "config", "user.name", "test")
    return tmp_path


def _commit_all(repo, msg: str = "c") -> None:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", msg)


# --- rendering: full content with citable line numbers ----------------------

def test_single_file_rendered_with_line_numbers(tmp_path):
    """A plain file (no git needed) is rendered whole, with 1-based line numbers."""
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    surface = build_path_surface(tmp_path, ["calc.py"], max_bytes=10_000)
    assert "--- FILE: calc.py (2 lines) ---" in surface
    assert "1| def add(a, b):" in surface
    assert "2|     return a + b" in surface


# --- directory expansion via git ls-files -----------------------------------

def test_directory_expanded_via_git_ls_files(repo):
    (repo / "pkg").mkdir()
    (repo / "pkg" / "a.py").write_text("A = 1\n")
    (repo / "pkg" / "b.py").write_text("B = 2\n")
    _commit_all(repo)
    surface = build_path_surface(repo, ["pkg"], max_bytes=10_000)
    assert "--- FILE: pkg/a.py" in surface
    assert "--- FILE: pkg/b.py" in surface
    assert "A = 1" in surface and "B = 2" in surface


def test_gitignored_file_excluded_from_directory(repo):
    """`.gitignore` is honoured for free: an ignored (untracked) file is absent."""
    (repo / "pkg").mkdir()
    (repo / "pkg" / "keep.py").write_text("KEEP = 1\n")
    (repo / "pkg" / "secret.env").write_text("TOKEN=xyz\n")
    (repo / ".gitignore").write_text("*.env\n")
    _commit_all(repo)
    surface = build_path_surface(repo, ["pkg"], max_bytes=10_000)
    assert "keep.py" in surface
    assert "secret.env" not in surface
    assert "TOKEN=xyz" not in surface


def test_expand_paths_dedups_file_and_dir_overlap(repo):
    (repo / "pkg").mkdir()
    (repo / "pkg" / "a.py").write_text("A = 1\n")
    _commit_all(repo)
    files, missing = _expand_paths(repo, ["pkg", "pkg/a.py"])
    assert len(files) == 1
    assert missing == []


# --- the total-size cap: truncate, never silently drop ----------------------

def test_surface_cap_omits_and_names_dropped_file(repo):
    (repo / "a.py").write_text("A = 1\n")
    (repo / "b.py").write_text("B = 2\n")
    _commit_all(repo)
    cap = len(_render_file("a.py", "A = 1")) + 5  # room for a.py, not b.py
    surface = build_path_surface(repo, ["a.py", "b.py"], max_bytes=cap)
    assert "A = 1" in surface
    assert "B = 2" not in surface
    assert "--- OMITTED" in surface
    assert "b.py: surface cap" in surface


def test_single_file_over_cap_is_truncated_in_place(tmp_path):
    """A lone file bigger than the cap is truncated (with a marker), not dropped."""
    (tmp_path / "big.py").write_text("\n".join(f"line{i}" for i in range(500)) + "\n")
    surface = build_path_surface(tmp_path, ["big.py"], max_bytes=200)
    assert "truncated at surface cap" in surface
    assert "big.py" in surface
    assert len(surface) < 1_000  # the cap is actually honoured


# --- missing / untracked paths are surfaced, not swallowed -------------------

def test_missing_path_is_reported(repo):
    (repo / "real.py").write_text("R = 1\n")
    _commit_all(repo)
    surface = build_path_surface(repo, ["real.py", "does_not_exist.py"], max_bytes=10_000)
    assert "R = 1" in surface
    assert "PATHS WITH NO TRACKED FILES" in surface
    assert "does_not_exist.py" in surface


def test_no_reviewable_files_placeholder(repo):
    surface = build_path_surface(repo, ["nope.py"], max_bytes=10_000)
    assert "(no reviewable files)" in surface
    assert "PATHS WITH NO TRACKED FILES" in surface