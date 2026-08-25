# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""Shared fixtures: the throwaway git repo four test modules used to copy-paste.

Two repos, deliberately not one. ``git_repo`` has no commits: giving it a
committed file would change what ``_expand_paths`` and the surface cap see, which
is the whole subject of ``backends/claude_code/test_surface.py``. ``changed_repo``
adds the two commits diff mode needs, since a review of an empty diff is now
refused outright (see ``backends/claude_code/test_gather.py``).
"""

from __future__ import annotations

import subprocess

import pytest


def _git(repo, *args) -> None:
    subprocess.run(["git", "-C", str(repo), *args],
                   check=True, capture_output=True, text=True)


@pytest.fixture
def commit_all():
    """Stage and commit everything in a repo."""
    def _commit_all(repo, msg: str = "c") -> None:
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", msg)
    return _commit_all


@pytest.fixture
def git_repo(tmp_path):
    """A minimal git repo under ``tmp_path`` with a local identity, no commits."""
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "test@example.invalid")
    _git(tmp_path, "config", "user.name", "test")
    return tmp_path


@pytest.fixture
def changed_repo(git_repo, commit_all):
    """``git_repo`` plus two commits on ``a.py``, so diff mode has real content."""
    (git_repo / "a.py").write_text("def f():\n    return 1\n")
    commit_all(git_repo, "init")
    (git_repo / "a.py").write_text("def f():\n    return 2\n")
    commit_all(git_repo, "change")
    return git_repo
