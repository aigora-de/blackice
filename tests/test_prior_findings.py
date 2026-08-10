# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""Cross-run memory (issue #13): seeding a re-run with a prior run's findings.

blackice has cross-*epoch* memory but starts every invocation cold, so the
hunt -> fix -> re-run workflow cannot tell the panel what the last run already
found. ``--prior-findings`` seeds ``PanelSession`` before epoch 1.

Two things here are easy to get wrong and are pinned deliberately:

* the injection gate is ``epoch > 1`` — a seeded run must reach the panel at
  **epoch 1**, which is the only epoch a confirm-the-fixes re-run may get;
* ``on_epoch`` **overwrites** ``prior_summary`` from the current run's ledger,
  so a seeded summary stored in that field is silently discarded after epoch 1.
  The carried-in findings are therefore held in their own field and reach the
  prompt as a separately labelled block, so a persona can also tell "found
  before the fixes" from "found this run".
"""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from blackice import main
from claude_code_backend import PanelSession, build_prompt, load_prior_findings
from loop import Finding, ReviewSpec, Severity


_LEDGER = {
    "halt_reason": "converged",
    "epochs": 2,
    "findings": [
        {"persona": "P1", "severity": "BLOCKER", "title": "unbounded retry loop",
         "file": "runner.py", "line": 120, "open": True},
        {"persona": "P2", "severity": "NOTE", "title": "stale threshold",
         "file": "calc.py", "line": 7, "open": False},
    ],
}


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


# --- loading ----------------------------------------------------------------

def test_loads_a_saved_findings_json(tmp_path):
    p = tmp_path / "findings.json"
    p.write_text(json.dumps(_LEDGER))

    summary = load_prior_findings(p)

    assert "unbounded retry loop" in summary
    assert "stale threshold" in summary
    assert "runner.py:120" in summary


def test_loads_a_bare_findings_array(tmp_path):
    """Tolerate the array on its own, not only the whole run envelope."""
    p = tmp_path / "findings.json"
    p.write_text(json.dumps(_LEDGER["findings"]))

    assert "unbounded retry loop" in load_prior_findings(p)


def test_extracts_the_json_block_from_a_raw_run_log(tmp_path):
    """The durable archive is ``run.log`` — stdout, prose and all.

    Asking an operator to hand-carve the JSON out of it is how a seeded re-run
    silently gets nothing, so accept the log directly.
    """
    p = tmp_path / "run.log"
    p.write_text(
        "[panel] 6 personas from panel file: P1, P2\n"
        "=== HALT: converged after 2 epoch(s) ===\n"
        "\n--- JSON ---\n" + json.dumps(_LEDGER) + "\n"
    )

    assert "unbounded retry loop" in load_prior_findings(p)


def test_carries_the_open_closed_state(tmp_path):
    """A re-run's job is confirming fixes, so 'was this already closed?' matters."""
    p = tmp_path / "findings.json"
    p.write_text(json.dumps(_LEDGER))

    summary = load_prior_findings(p)

    assert "open" in summary
    assert "resolved" in summary


def test_a_missing_file_fails_loudly(tmp_path):
    with pytest.raises((FileNotFoundError, ValueError)):
        load_prior_findings(tmp_path / "nope.json")


def test_a_file_with_no_findings_fails_loudly(tmp_path):
    """Silently seeding nothing is blackice#11's lesson repeated."""
    p = tmp_path / "junk.json"
    p.write_text("not json at all")

    with pytest.raises(ValueError):
        load_prior_findings(p)


# --- injection --------------------------------------------------------------

def test_seeded_findings_reach_epoch_one(tmp_path):
    """The gate was ``epoch > 1``; a confirm-the-fixes re-run may only get one."""
    spec = ReviewSpec(why="w", what="x")

    prompt = build_prompt(spec, "surface", epoch=1, prior="",
                          surface_kind="files", seeded="- [BLOCKER/open] carried in")

    assert "carried in" in prompt


def test_a_cold_run_is_unchanged(tmp_path):
    spec = ReviewSpec(why="w", what="x")

    prompt = build_prompt(spec, "surface", epoch=1, prior="", surface_kind="files")

    assert "PRIOR" not in prompt


def test_locations_are_framed_as_advisory(tmp_path):
    """Fixes move code.  A seeded ``file:line`` is a hint, not ground truth —
    personas re-read the live surface every epoch and must adjudicate against it.
    """
    spec = ReviewSpec(why="w", what="x")

    prompt = build_prompt(spec, "s", epoch=1, prior="", surface_kind="files",
                          seeded="- [BLOCKER/open] x @ runner.py:120")

    assert "advisory" in prompt.lower() or "may have moved" in prompt.lower()


def test_the_epoch_checkpoint_does_not_clobber_the_seed(repo):
    """``on_epoch`` rewrites ``prior_summary`` from the current ledger.

    Storing the seed in that field would look correct and then vanish after
    epoch 1 — the run would lose exactly the memory it was asked to carry. This
    drives the real checkpoint rather than a helper written to be asserted on.
    """
    session = PanelSession(
        repo_root=repo, spec=ReviewSpec(why="w", what="x"), base="", head="HEAD",
        paths=["a.py"], personas={}, seeded_summary="- [BLOCKER/open] carried in")
    run = SimpleNamespace(ledger={
        "k": Finding("P1", "found this run", Severity.NOTE, "meta",
                     file="a.py", line=1)})

    session.on_epoch(run)

    assert session.seeded_summary == "- [BLOCKER/open] carried in"
    assert "found this run" in session.prior_summary


def test_the_two_memories_stay_distinguishable(repo):
    """A persona must be able to tell "found before the fixes" from "found now".

    Merging them into one block would lose the only thing that makes a seeded
    re-run useful: knowing which items the fixes were supposed to have closed.
    """
    prompt = build_prompt(
        ReviewSpec(why="w", what="x"), "s", epoch=2,
        prior="- [NOTE/open] found this run", surface_kind="files",
        seeded="- [BLOCKER/open] carried in")

    assert "PRIOR RUN'S FINDINGS" in prompt
    assert "PRIOR EPOCHS' FINDINGS" in prompt
    assert prompt.index("carried in") < prompt.index("found this run")


# --- CLI wiring -------------------------------------------------------------

def test_cli_seeds_the_session_and_says_so(repo, tmp_path, capsys):
    """Operator-visible: a seed that silently fails to load is worse than none."""
    p = tmp_path / "prior.json"
    p.write_text(json.dumps(_LEDGER))

    main(["--repo", str(repo), "--paths", "a.py", "--dry-run",
          "--prior-findings", str(p)])

    out = capsys.readouterr().out
    assert "prior findings" in out.lower()
    assert "2" in out


def test_cli_works_in_diff_mode_too(repo, tmp_path, capsys):
    p = tmp_path / "prior.json"
    p.write_text(json.dumps(_LEDGER))

    main(["--repo", str(repo), "--base", "HEAD", "--dry-run",
          "--prior-findings", str(p)])

    assert "prior findings" in capsys.readouterr().out.lower()
