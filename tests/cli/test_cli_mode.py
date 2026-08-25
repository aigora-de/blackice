# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""CLI mode-selection tests (issue #6): diff mode vs path mode.

Exactly one review mode is active per run. These drive ``blackice.main`` with
``--dry-run`` so nothing spawns; the positive cases also exercise the full wiring
(persona sourcing → ``gather`` → prompt assembly) over a throwaway git repo.
"""

from __future__ import annotations

import pytest

from blackice.cli import main


# --- exactly-one-mode validation --------------------------------------------

def test_both_modes_rejected(changed_repo):
    with pytest.raises(SystemExit):
        main(["--repo", str(changed_repo), "--base", "HEAD", "--paths", "a.py"])


def test_neither_mode_rejected(changed_repo):
    with pytest.raises(SystemExit):
        main(["--repo", str(changed_repo)])


def test_paths_flag_with_no_values_rejected(changed_repo):
    with pytest.raises(SystemExit):
        main(["--repo", str(changed_repo), "--paths"])


# --- both modes are individually accepted (dry-run: nothing spawns) ----------

def test_path_mode_accepted(changed_repo, capsys):
    rc = main(["--repo", str(changed_repo), "--paths", "a.py", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "dry-run" in out
    # Whole-file review framing, not diff framing.
    assert "WHAT TO REVIEW" in out or "this code" in out


def test_diff_mode_accepted(changed_repo):
    rc = main(["--repo", str(changed_repo), "--base", "HEAD~1", "--dry-run"])
    assert rc == 0
