# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""A run asked for ``--out PATH`` leaves a durable artefact behind it, epoch by epoch (#31).

Before #31 the artefact existed only in memory until the run ended, and then only on
stdout: a run interrupted in epoch 2 left epoch 1's titles on the console and nothing a
machine could read, and one killed outright with stdout redirected left nothing at all.

What is pinned here, as pairs, because each half passes trivially without the other:

* a run interrupted after epoch N leaves a parseable artefact with N epochs and
  ``halt_reason: null`` — the partial marker — PAIRED WITH a run that completes,
  whose file is byte-identical to the stdout artefact and carries its halt reason;
* no flag, no file, and stdout unchanged — PAIRED WITH the flag writing one;
* a destination that cannot be used is refused before anything is spawned — PAIRED
  WITH an ordinary destination being accepted;
* a failed write is said once and the run continues — PAIRED WITH a healthy run saying
  nothing about writes at all.

Nothing is spawned; no network. ``_run_claude`` is stubbed at the session, which is
the layer the CLI's other reporting tests stub.

**Which are regressions, measured on main with this file in place** and a shim for the
new module. Recorded in the PR body; any test green on main says so in its own
docstring.
"""

from __future__ import annotations

import json
import os
import stat

import pytest

from kuang.backends.claude_code import session as session_module
from kuang.backends.claude_code.spawn import CallResult
from kuang.cli import main

_CLAUDE_MD = """\
# A repo

# Resident Experts

## Analyst — Correctness

Does the change compute the right thing?

## Critic — Completeness

Find what everyone else missed.
"""

_DELIMITER = "--- JSON ---\n"


def _contract(epoch: int, severity: str) -> CallResult:
    """One finding, distinct per epoch so a second epoch is reached at all."""
    body = json.dumps({"verdict": "NO", "findings": [
        {"title": f"off-by-one in the retry bound (epoch {epoch})",
         "severity": severity, "claim_class": f"correctness-{epoch}",
         "file": "a.py", "line": 1}]})
    return CallResult(f"I reviewed it.\n\n```json\n{body}\n```", 18, num_turns=7)


_CLEAN = CallResult('I reviewed it.\n\n```json\n{"verdict": "YES", "findings": []}\n```',
                    18, num_turns=7)


def _stub(monkeypatch, repo, *, severity="BLOCKER", interrupt_at_epoch=None,
          lose_surface_at_epoch=None, on_gather=None) -> dict:
    """Stub the reviewer, and optionally interrupt or lose the surface at an epoch.

    Returns a mutable record of what happened: how many persona calls were made, and
    how many times ``on_epoch`` ran.
    """
    (repo / "CLAUDE.md").write_text(_CLAUDE_MD)
    state = {"epoch": 0, "calls": 0, "on_epoch": 0}
    real_gather = session_module.PanelSession.gather
    real_on_epoch = session_module.PanelSession.on_epoch

    def _fake_call(self, prompt, mandate, tools, model):  # noqa: ANN001, ARG001
        if "careful synthesiser" in mandate or "formatter" in mandate:
            return CallResult("nothing structured to say.", 0)
        state["calls"] += 1
        if interrupt_at_epoch is not None and state["epoch"] == interrupt_at_epoch:
            raise KeyboardInterrupt
        if severity is None:
            return _CLEAN
        return _contract(state["epoch"], severity)

    def _gather(self, epoch: int) -> str:  # noqa: ANN001
        state["epoch"] = epoch
        if on_gather is not None:
            on_gather(epoch)
        if lose_surface_at_epoch is not None and epoch == lose_surface_at_epoch:
            raise RuntimeError("the surface went away")
        return real_gather(self, epoch)

    def _on_epoch(self, run):  # noqa: ANN001
        state["on_epoch"] += 1
        return real_on_epoch(self, run)

    monkeypatch.setattr(session_module.PanelSession, "_run_claude", _fake_call)
    monkeypatch.setattr(session_module.PanelSession, "gather", _gather)
    monkeypatch.setattr(session_module.PanelSession, "on_epoch", _on_epoch)
    return state


def _argv(repo, *extra, epochs: int = 2) -> list[str]:
    return ["--repo", str(repo), "--base", "HEAD~1", "--max-epochs", str(epochs),
            "--stall-patience", str(epochs), "--no-parallel", *extra]


def _stdout_artefact(out: str) -> str:
    """The bytes after the delimiter line, exactly as the console carried them."""
    head, sep, tail = out.rpartition(_DELIMITER)
    assert sep, "no artefact on stdout"
    return tail


def _files(root) -> set:
    return {p for p in root.rglob("*") if ".git" not in p.parts}


@pytest.fixture
def out_dir(tmp_path_factory):
    """A directory OUTSIDE the repository: ``changed_repo`` is ``tmp_path`` itself."""
    return tmp_path_factory.mktemp("artefacts")


# --- the partial marker, and its pair -------------------------------------------------


def test_a_run_interrupted_in_epoch_2_leaves_epoch_1_marked_incomplete(
        changed_repo, out_dir, capsys, monkeypatch):
    """REGRESSION for #31: on main a Ctrl-C in epoch 2 left no artefact anywhere.

    The partial marker is ``halt_reason: null``. A run that completes always has one;
    a checkpoint write never does, including on the halting epoch (#155), so the two
    states are told apart by a value the artefact already carries.
    """
    _stub(monkeypatch, changed_repo, interrupt_at_epoch=2)
    target = out_dir / "run.json"

    with pytest.raises(KeyboardInterrupt):
        main(_argv(changed_repo, "--out", str(target)))
    capsys.readouterr()

    payload = json.loads(target.read_text())
    assert payload["halt_reason"] is None
    assert payload["epochs"] == 1
    assert [f["title"] for f in payload["findings"]] == [
        "off-by-one in the retry bound (epoch 1)"]
    # Nothing but the artefact: no temporary file left beside it.
    assert sorted(p.name for p in out_dir.iterdir()) == ["run.json"]


@pytest.mark.parametrize("severity, extra, halt", [
    ("BLOCKER", (), "epoch"),
    (None, (), "converged"),
    ("UGLY", (), "escalate_ugly"),
])
def test_a_run_that_completes_leaves_a_file_identical_to_its_stdout_artefact(
        changed_repo, out_dir, capsys, monkeypatch, severity, extra, halt):
    """REGRESSION for #31, the pair of the partial: complete, and byte-identical.

    One builder renders both, so the durable record cannot drift from the console's
    copy (#33, #82). The final write is what carries the halt reason.
    """
    _stub(monkeypatch, changed_repo, severity=severity)
    target = out_dir / "run.json"

    main(_argv(changed_repo, "--out", str(target), *extra))
    out = capsys.readouterr().out

    assert target.read_bytes() == _stdout_artefact(out).encode()
    assert json.loads(target.read_text())["halt_reason"] == halt


def test_a_run_that_loses_its_surface_records_the_loss_in_its_final_file(
        changed_repo, out_dir, capsys, monkeypatch):
    """REGRESSION for #31: an epoch that never began fires no checkpoint (#85).

    So the last per-epoch write is the epoch BEFORE the loss, and only the final
    write can record ``surface_lost``.
    """
    _stub(monkeypatch, changed_repo, lose_surface_at_epoch=2)
    target = out_dir / "run.json"

    rc = main(_argv(changed_repo, "--out", str(target)))
    out = capsys.readouterr().out

    payload = json.loads(target.read_text())
    assert rc == 2
    assert payload["halt_reason"] == "surface_lost"
    assert payload["surface_lost"]["epoch"] == 2
    assert target.read_bytes() == _stdout_artefact(out).encode()


def test_the_halting_epochs_checkpoint_is_not_a_recorded_failure(
        changed_repo, out_dir, capsys, monkeypatch):
    """REGRESSION for #31: the halting epoch's checkpoint sees ``halt_reason is None``.

    A builder that read ``halt_reason.value`` unguarded would raise there on EVERY
    run, recording a ``checkpoint`` failure on a healthy one — a rule firing on a
    healthy run. Pinned at the published record, which is where it would show.
    """
    _stub(monkeypatch, changed_repo)
    target = out_dir / "run.json"

    main(_argv(changed_repo, "--out", str(target)))
    console, _, _ = capsys.readouterr().out.rpartition(_DELIMITER)

    assert json.loads(target.read_text())["seam_failures"] == []
    assert "seam failed" not in console
    assert "[out]" not in console


# --- no flag, no file; and the flag moves nothing on stdout --------------------------


def test_no_file_is_written_without_the_flag(changed_repo, capsys, monkeypatch):
    """GREEN ON MAIN, and kept as the opt-in's guard: kuang writes nothing unasked.

    The pair of the test below. It passes on main because nothing is written at all;
    it is here to go red if a default destination is ever introduced without a ruling
    (#146 owns that question).
    """
    _stub(monkeypatch, changed_repo)
    # The whole pytest base directory: the repository and every sibling of it.
    root = changed_repo.parent.parent
    before = _files(root)

    main(_argv(changed_repo))
    capsys.readouterr()

    assert _files(root) == before


def test_the_flag_writes_a_file_and_moves_nothing_on_stdout(
        changed_repo, out_dir, capsys, monkeypatch):
    """REGRESSION for #31: the flag writes, and a healthy run's stdout is unchanged.

    The second half is acceptance bullet 5 at the unit level; the golden and replay
    captures are its twin.
    """
    _stub(monkeypatch, changed_repo)
    main(_argv(changed_repo))
    without = capsys.readouterr().out

    _stub(monkeypatch, changed_repo)
    target = out_dir / "run.json"
    main(_argv(changed_repo, "--out", str(target)))
    with_flag = capsys.readouterr().out

    assert target.is_file()
    assert with_flag == without


# --- the destination, refused at start ------------------------------------------------


def _refused(capsys, state) -> str:
    captured = capsys.readouterr()
    assert state["calls"] == 0, "a refused destination must be refused before any spend"
    assert "--- JSON ---" not in captured.out
    return captured.err


def test_an_existing_file_is_never_overwritten(changed_repo, out_dir, capsys, monkeypatch):
    """REGRESSION for #31: the destination is refused rather than clobbered.

    Overwriting a previous run's only durable record, unasked, is irreversible. The
    operator can delete it; the tool cannot bring it back.
    """
    state = _stub(monkeypatch, changed_repo)
    target = out_dir / "run.json"
    target.write_text("an earlier run's artefact")

    rc = main(_argv(changed_repo, "--out", str(target)))

    assert rc == 2
    assert "already exists" in _refused(capsys, state)
    assert target.read_text() == "an earlier run's artefact"


def test_a_directory_is_refused(changed_repo, out_dir, capsys, monkeypatch):
    """REGRESSION for #31: a directory is not a file to write the artefact to."""
    state = _stub(monkeypatch, changed_repo)

    rc = main(_argv(changed_repo, "--out", str(out_dir)))

    assert rc == 2
    assert "already exists" in _refused(capsys, state)


def test_a_missing_directory_is_refused(changed_repo, out_dir, capsys, monkeypatch):
    """REGRESSION for #31: the parent is not created on the operator's behalf."""
    state = _stub(monkeypatch, changed_repo)

    rc = main(_argv(changed_repo, "--out", str(out_dir / "absent" / "run.json")))

    assert rc == 2
    assert "does not exist" in _refused(capsys, state)
    assert not (out_dir / "absent").exists()


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0,
                    reason="needs a POSIX permission bit that binds this user")
def test_an_unwritable_directory_is_refused(changed_repo, out_dir, capsys, monkeypatch):
    """REGRESSION for #31: refused at start, not discovered at epoch 1's write."""
    state = _stub(monkeypatch, changed_repo)
    out_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        rc = main(_argv(changed_repo, "--out", str(out_dir / "run.json")))
    finally:
        out_dir.chmod(stat.S_IRWXU)

    assert rc == 2
    assert "cannot write" in _refused(capsys, state)
    assert list(out_dir.iterdir()) == []


@pytest.mark.parametrize("relative", ["run.json", "sub/../run.json"])
def test_a_destination_inside_the_reviewed_repository_is_refused(
        changed_repo, capsys, monkeypatch, relative):
    """REGRESSION for #31: the artefact quotes the source and is not publishable (#121).

    A file inside the reviewed repository is one ``git add -A`` from a public commit,
    which is the defect #149's close-out recorded for ``.claude/worktrees/``.
    """
    state = _stub(monkeypatch, changed_repo)
    (changed_repo / "sub").mkdir()

    rc = main(_argv(changed_repo, "--out", str(changed_repo / relative)))

    assert rc == 2
    assert "inside the reviewed repository" in _refused(capsys, state)
    assert not (changed_repo / "run.json").exists()


def test_a_destination_beside_the_repository_is_accepted(changed_repo, capsys,
                                                         monkeypatch):
    """REGRESSION for #31, the pair of the refusals: an ordinary path is written.

    A sibling of the repository whose name EXTENDS the repository's: the case a
    prefix comparison on strings would get wrong (``repo`` vs ``repo-out.json``).
    """
    _stub(monkeypatch, changed_repo)
    target = changed_repo.parent / f"{changed_repo.name}-out.json"

    rc = main(_argv(changed_repo, "--out", str(target)))
    capsys.readouterr()

    assert rc == 0
    assert target.is_file()


# --- a failed write: said once, and the run continues ---------------------------------


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0,
                    reason="needs a POSIX permission bit that binds this user")
def test_a_failed_write_is_said_once_and_the_run_continues(changed_repo, out_dir, capsys,
                                                           monkeypatch):
    """REGRESSION for #31: the outcome #148 left to this issue, stated.

    The directory becomes unwritable after the start check, so every write fails —
    three epochs and the final write. The run still does what it is for and its
    stdout artefact still arrives, so it CONTINUES; the console says once, above the
    report, that the file is not current. The published detail carries no path: a
    home directory is a person's name (#148, #121).
    """
    def lock(epoch):
        if epoch == 1:
            out_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)

    _stub(monkeypatch, changed_repo, on_gather=lock)
    target = out_dir / "run.json"
    try:
        rc = main(_argv(changed_repo, "--out", str(target), epochs=3))
    finally:
        out_dir.chmod(stat.S_IRWXU)
    out = capsys.readouterr().out
    console, _, tail = out.rpartition(_DELIMITER)
    payload = json.loads(tail)

    assert rc == 0
    assert payload["halt_reason"] == "epoch"
    warnings = [ln for ln in console.splitlines() if ln.startswith("[out] WARNING")]
    assert len(warnings) == 1
    assert "not current" in warnings[0]
    assert [(r["epoch"], r["seam"]) for r in payload["seam_failures"]] == [
        (1, "checkpoint"), (2, "checkpoint"), (3, "checkpoint")]
    for row in payload["seam_failures"]:
        assert "artefact write" in row["detail"]
        assert "[errno" in row["detail"]
        assert str(out_dir) not in row["detail"]
        assert out_dir.name not in row["detail"]
    assert not target.exists()


def test_a_failing_prior_context_refresh_does_not_skip_the_write(
        changed_repo, out_dir, capsys, monkeypatch):
    """REGRESSION for #31: the checkpoint's two jobs fail independently.

    ``on_epoch`` rebuilds the next epoch's prior context; the durable write is the
    second job behind the same seam. A raise in the first must not cost the second.
    """
    _stub(monkeypatch, changed_repo, interrupt_at_epoch=2)

    def _raising(self, run):  # noqa: ANN001, ARG001
        raise RuntimeError("memory went wrong")

    monkeypatch.setattr(session_module.PanelSession, "on_epoch", _raising)
    target = out_dir / "run.json"

    with pytest.raises(KeyboardInterrupt):
        main(_argv(changed_repo, "--out", str(target)))
    capsys.readouterr()

    payload = json.loads(target.read_text())
    assert payload["epochs"] == 1


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0,
                    reason="needs a POSIX permission bit that binds this user")
def test_a_failing_write_does_not_skip_the_prior_context_refresh(
        changed_repo, out_dir, capsys, monkeypatch):
    """REGRESSION for #31: the pair of the test above, the other job first."""
    def lock(epoch):
        if epoch == 1:
            out_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)

    state = _stub(monkeypatch, changed_repo, on_gather=lock)
    try:
        main(_argv(changed_repo, "--out", str(out_dir / "run.json")))
    finally:
        out_dir.chmod(stat.S_IRWXU)
    capsys.readouterr()

    assert state["on_epoch"] == 2


def test_a_prior_context_failure_is_named_as_such(changed_repo, out_dir, capsys,
                                                  monkeypatch):
    """REGRESSION for #31: one ``SeamFailure`` per epoch names the job that failed.

    Two jobs behind one seam publish one row, so the row has to say which of them it
    is about, or a write failure and a memory failure read alike.
    """
    _stub(monkeypatch, changed_repo)

    def _raising(self, run):  # noqa: ANN001, ARG001
        raise RuntimeError("memory went wrong")

    monkeypatch.setattr(session_module.PanelSession, "on_epoch", _raising)
    target = out_dir / "run.json"

    main(_argv(changed_repo, "--out", str(target)))
    _, _, tail = capsys.readouterr().out.rpartition(_DELIMITER)

    details = [r["detail"] for r in json.loads(tail)["seam_failures"]]
    assert len(details) == 2
    for d in details:
        assert "prior context" in d
        assert "memory went wrong" in d
        assert "artefact write" not in d
