# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""CLI mode-selection tests (issue #6): diff mode vs path mode.

Exactly one review mode is active per run. These drive ``blackice.main`` with
``--dry-run`` so nothing spawns; the positive cases also exercise the full wiring
(persona sourcing → ``gather`` → prompt assembly) over a throwaway git repo.
"""

from __future__ import annotations

import subprocess

import pytest

from blackice import main


def _git(repo, *args) -> None:
    subprocess.run(["git", "-C", str(repo), *args],
                   check=True, capture_output=True, text=True)


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "test@example.invalid")
    _git(tmp_path, "config", "user.name", "test")
    (tmp_path / "a.py").write_text("def f():\n    return 1\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


# --- exactly-one-mode validation --------------------------------------------

def test_both_modes_rejected(repo):
    with pytest.raises(SystemExit):
        main(["--repo", str(repo), "--base", "HEAD", "--paths", "a.py"])


def test_neither_mode_rejected(repo):
    with pytest.raises(SystemExit):
        main(["--repo", str(repo)])


def test_paths_flag_with_no_values_rejected(repo):
    with pytest.raises(SystemExit):
        main(["--repo", str(repo), "--paths"])


# --- both modes are individually accepted (dry-run: nothing spawns) ----------

def test_path_mode_accepted(repo, capsys):
    rc = main(["--repo", str(repo), "--paths", "a.py", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "dry-run" in out
    # Whole-file review framing, not diff framing.
    assert "WHAT TO REVIEW" in out or "this code" in out


def test_diff_mode_accepted(repo):
    rc = main(["--repo", str(repo), "--base", "HEAD", "--dry-run"])
    assert rc == 0
