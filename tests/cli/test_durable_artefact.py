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
import subprocess
import tempfile

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


def test_a_run_stopped_after_its_halting_epoch_but_before_the_final_write_is_incomplete(
        changed_repo, out_dir, capsys, monkeypatch):
    """REGRESSION for #31: the halting epoch's partial must read as partial too.

    Ctrl-C while the report renders lands between the halting epoch's checkpoint and
    the final write, which leaves exactly that file behind. Its epoch already knows
    the run will halt (``EpochResult.halt``), but the run has not published a halt
    reason, so the file must not claim one: a partial that carried the halt would be
    indistinguishable from the complete file, and it is short of what only the final
    write records.
    """
    from kuang.cli.artefact_file import ArtefactFile

    def interrupted(self, text):  # noqa: ANN001, ARG001
        raise KeyboardInterrupt

    _stub(monkeypatch, changed_repo)
    monkeypatch.setattr(ArtefactFile, "final", interrupted)
    target = out_dir / "run.json"

    with pytest.raises(KeyboardInterrupt):
        main(_argv(changed_repo, "--out", str(target)))
    capsys.readouterr()

    payload = json.loads(target.read_text())
    assert payload["epochs"] == 2
    assert payload["halt_reason"] is None


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


def test_no_file_is_written_without_the_flag(changed_repo, capsys, monkeypatch,
                                             tmp_path_factory):
    """GREEN ON MAIN, and kept as the opt-in's guard: kuang writes nothing unasked.

    The pair of the test below. It passes on main because nothing is written at all;
    it is here to go red if a default destination is ever introduced without a ruling
    (#146 owns that question).

    Two instruments, because one is blind where the other is not. The directory walk
    sees any file that lands near the repository; the spy sees a write ANYWHERE —
    a default under a cache or home directory would pass the walk — through the two
    calls an atomic write cannot avoid.
    """
    _stub(monkeypatch, changed_repo)
    calls = []
    real_mkstemp, real_replace = tempfile.mkstemp, os.replace

    def spy_mkstemp(*a, **k):
        calls.append("mkstemp")
        return real_mkstemp(*a, **k)

    def spy_replace(*a, **k):
        calls.append("replace")
        return real_replace(*a, **k)

    monkeypatch.setattr(tempfile, "mkstemp", spy_mkstemp)
    monkeypatch.setattr(os, "replace", spy_replace)
    # This session's own base directory: the repository and every sibling of it.
    # Never the per-user parent of it, which other pytest processes write into
    # concurrently (a mutation run beside a version run made this red that way).
    root = tmp_path_factory.getbasetemp()
    before = _files(root)

    main(_argv(changed_repo))
    capsys.readouterr()

    assert _files(root) == before
    assert calls == []


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


@pytest.mark.parametrize("relative", ["run.json", "sub/../run.json", "sub/run.json"])
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


def test_a_relative_destination_inside_the_repository_is_refused(
        changed_repo, capsys, monkeypatch):
    """REGRESSION for #31: the check is made on the path as resolved, not as typed.

    Run from inside the repository, ``--out run.json`` names a file there while the
    string is nothing like the repository's path.
    """
    state = _stub(monkeypatch, changed_repo)
    monkeypatch.chdir(changed_repo)

    rc = main(_argv(changed_repo, "--out", "run.json"))

    assert rc == 2
    assert "inside the reviewed repository" in _refused(capsys, state)
    assert not (changed_repo / "run.json").exists()


@pytest.mark.skipif(os.name != "posix", reason="needs a symbolic link")
def test_a_destination_reaching_the_repository_through_a_link_is_refused(
        changed_repo, out_dir, capsys, monkeypatch):
    """REGRESSION for #31: a link outside the repository that points into it."""
    state = _stub(monkeypatch, changed_repo)
    link = out_dir / "into-repo"
    link.symlink_to(changed_repo, target_is_directory=True)

    rc = main(_argv(changed_repo, "--out", str(link / "run.json")))

    assert rc == 2
    assert "inside the reviewed repository" in _refused(capsys, state)
    assert not (changed_repo / "run.json").exists()


def test_a_destination_in_the_working_tree_above_repo_is_refused(
        changed_repo, capsys, monkeypatch):
    """REGRESSION for #31's pre-merge review: the repository is not ``--repo``.

    ``--repo`` may be a subdirectory of the working tree, and its default is ``.``.
    The first shape of the check compared against ``--repo`` alone, so a file at the
    tree's top level was accepted, and ``git status`` then listed it as untracked —
    one ``git add -A`` from a public commit.
    """
    state = _stub(monkeypatch, changed_repo)
    (changed_repo / "sub").mkdir()
    (changed_repo / "sub" / "b.py").write_text("x = 1\n")

    rc = main(_argv(changed_repo / "sub", "--out", str(changed_repo / "leak.json")))

    assert rc == 2
    assert "inside the reviewed repository" in _refused(capsys, state)
    assert not (changed_repo / "leak.json").exists()


def test_a_destination_in_another_working_tree_of_the_repository_is_refused(
        changed_repo, capsys, monkeypatch, commit_all):
    """REGRESSION for #31's pre-merge review: a linked worktree's repository is every tree.

    Reviewing a linked worktree, a destination in the MAIN checkout was accepted. This
    repository keeps its reviewers' worktrees under ``.claude/worktrees/`` in exactly
    that layout, so the case is not hypothetical.
    """
    state = _stub(monkeypatch, changed_repo)
    commit_all(changed_repo, "panel")
    linked = changed_repo / ".claude" / "worktrees" / "w"
    subprocess.run(["git", "-C", str(changed_repo), "worktree", "add", "-q", "--detach",
                    str(linked)], check=True, capture_output=True)

    rc = main(_argv(linked, "--out", str(changed_repo / "leak.json")))

    assert rc == 2
    assert "inside the reviewed repository" in _refused(capsys, state)
    assert not (changed_repo / "leak.json").exists()


def _case_insensitive(path) -> bool:
    flipped = path.parent / path.name.swapcase()
    return flipped != path and flipped.exists()


def test_a_destination_spelled_in_another_case_is_refused(changed_repo, capsys,
                                                         monkeypatch):
    """REGRESSION for #31's pre-merge review: containment is identity, not spelling.

    On a case-insensitive volume (macOS's default) ``REPO`` is ``repo``. A string
    comparison accepted it; ``os.path.samestat`` does not. Skipped where the volume
    is case-sensitive, because there the two spellings ARE different directories.
    """
    if not _case_insensitive(changed_repo):
        pytest.skip("the volume is case-sensitive")
    state = _stub(monkeypatch, changed_repo)
    flipped = changed_repo.parent / changed_repo.name.swapcase()

    rc = main(_argv(changed_repo, "--out", str(flipped / "x.json")))

    assert rc == 2
    assert "inside the reviewed repository" in _refused(capsys, state)


@pytest.mark.skipif(os.name != "posix", reason="needs a symbolic link")
def test_a_dangling_link_at_the_destination_is_refused(changed_repo, out_dir, capsys,
                                                      monkeypatch):
    """REGRESSION for #31: a link kuang did not create is not replaced.

    ``exists()`` follows the link and says False for a dangling one, so the check
    also asks ``is_symlink()``. Without it ``os.replace`` would delete the operator's
    link and put a file in its place.
    """
    state = _stub(monkeypatch, changed_repo)
    target = out_dir / "run.json"
    target.symlink_to(out_dir / "nowhere.json")

    rc = main(_argv(changed_repo, "--out", str(target)))

    assert rc == 2
    assert "already exists" in _refused(capsys, state)
    assert target.is_symlink()


@pytest.mark.parametrize("raw", ["", "somewhere/"])
def test_a_destination_naming_no_file_is_refused(changed_repo, capsys, monkeypatch, raw):
    """REGRESSION for #31: an unset shell variable, or a trailing separator.

    ``Path`` drops a trailing slash, so ``x.json/`` — a directory to POSIX — would be
    written as a file; and ``--out ""`` was refused as "already exists", which tells
    an operator with an unset variable the wrong thing.
    """
    state = _stub(monkeypatch, changed_repo)

    rc = main(_argv(changed_repo, "--out", raw))

    assert rc == 2
    assert "names no file" in _refused(capsys, state)


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
    assert warnings[0].startswith("[out] WARNING: could not write the artefact at epoch 1:")
    assert "not current until a later write succeeds" in warnings[0]
    assert [(r["epoch"], r["seam"]) for r in payload["seam_failures"]] == [
        (1, "checkpoint"), (2, "checkpoint"), (3, "checkpoint")]
    for row in payload["seam_failures"]:
        assert "artefact write" in row["detail"]
        assert "[errno" in row["detail"]
        assert str(out_dir) not in row["detail"]
        assert out_dir.name not in row["detail"]
    assert not target.exists()


def test_a_failed_final_write_is_said_above_the_artefact(changed_repo, out_dir, capsys,
                                                         monkeypatch):
    """REGRESSION for #31: only the final write fails, and the console says so FIRST.

    A line printed after the ``--- JSON ---`` block would make the run log unreadable
    to every reader that takes the last block (#123). The file then holds the last
    epoch's partial, which reads as one, and the warning says exactly that.
    """
    from kuang.cli import artefact_file

    real = artefact_file._replace

    def fails_when_complete(path, text):  # noqa: ANN001
        if json.loads(text)["halt_reason"] is not None:
            raise OSError(28, "No space left on device")
        return real(path, text)

    _stub(monkeypatch, changed_repo)
    monkeypatch.setattr(artefact_file, "_replace", fails_when_complete)
    target = out_dir / "run.json"

    rc = main(_argv(changed_repo, "--out", str(target)))
    out = capsys.readouterr().out
    console, _, tail = out.rpartition(_DELIMITER)

    assert rc == 0
    json.loads(tail)
    warnings = [ln for ln in console.splitlines() if ln.startswith("[out] WARNING")]
    assert len(warnings) == 1
    assert warnings[0].startswith(
        "[out] WARNING: could not write the artefact at the end of the run:")
    assert "not the complete artefact" in warnings[0]
    assert json.loads(target.read_text())["halt_reason"] is None
    assert json.loads(target.read_text())["epochs"] == 2


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


def test_the_help_says_how_to_tell_a_partial_file(capsys):
    """REGRESSION for #31: the partial marker is a convention, so it is stated.

    ``halt_reason: null`` is exact, but a reader who does not know it sees a run with
    no halt reason rather than one that did not finish. The help is where an operator
    choosing ``--out`` meets the convention.
    """
    with pytest.raises(SystemExit):
        main(["--help"])
    text = " ".join(capsys.readouterr().out.split())

    assert "--out PATH" in text
    assert "halt_reason is null is from a run that did not finish" in text
    assert "without this flag nothing is written" in text
