# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""The durable artefact is replaced atomically, and its failures are said once (#31).

The CLI half is ``test_durable_artefact.py``. This is the writer on its own: what a
reader can see if the write is interrupted between its two steps, what is left behind,
and what a failure publishes. All of it is new with #31, so none of it can run on main.
"""

from __future__ import annotations

import errno
import os

import pytest

from kuang.cli import artefact_file
from kuang.cli.artefact_file import (ArtefactFile, ArtefactWriteError,
                                     DestinationError, check_destination,
                                     failure_detail)


def _said():
    lines: list[str] = []
    return lines, lines.append


def test_a_write_replaces_the_file_and_leaves_nothing_else(tmp_path):
    """REGRESSION for #31: the file is the whole text, and no temporary file remains."""
    lines, say = _said()
    out = ArtefactFile(tmp_path / "run.json", say=say)

    out.persist(lambda: '{"epochs": 1}\n', when="at epoch 1")
    out.persist(lambda: '{"epochs": 2}\n', when="at epoch 2")

    assert (tmp_path / "run.json").read_text() == '{"epochs": 2}\n'
    assert [p.name for p in tmp_path.iterdir()] == ["run.json"]
    assert lines == []


def test_a_write_interrupted_before_the_rename_leaves_the_previous_file(tmp_path,
                                                                       monkeypatch):
    """REGRESSION for #31: a reader never sees a half-written artefact.

    Ctrl-C is never contained (``Exception``, never ``BaseException``, #148), so it
    can land between writing the new text and putting it in place. The previous
    epoch's complete file must still be there, and the temporary one must be gone.
    """
    out = ArtefactFile(tmp_path / "run.json", say=lambda _: None)
    out.persist(lambda: '{"epochs": 1}\n', when="at epoch 1")

    def interrupted(src, dst):  # noqa: ARG001
        raise KeyboardInterrupt

    monkeypatch.setattr(artefact_file.os, "replace", interrupted)
    with pytest.raises(KeyboardInterrupt):
        out.persist(lambda: '{"epochs": 2}\n', when="at epoch 2")

    assert (tmp_path / "run.json").read_text() == '{"epochs": 1}\n'
    assert [p.name for p in tmp_path.iterdir()] == ["run.json"]


def test_a_write_interrupted_mid_text_leaves_the_previous_file(tmp_path, monkeypatch):
    """REGRESSION for #31: the same, with the interruption inside the text write.

    This is the row that a write IN PLACE fails: truncating the destination and then
    being stopped leaves an empty or half-written artefact where a whole one was.
    """
    out = ArtefactFile(tmp_path / "run.json", say=lambda _: None)
    out.persist(lambda: '{"epochs": 1}\n', when="at epoch 1")

    def interrupted(fd):  # noqa: ARG001
        raise KeyboardInterrupt

    monkeypatch.setattr(artefact_file.os, "fsync", interrupted)
    with pytest.raises(KeyboardInterrupt):
        out.persist(lambda: '{"epochs": 2}\n', when="at epoch 2")

    assert (tmp_path / "run.json").read_text() == '{"epochs": 1}\n'
    assert [p.name for p in tmp_path.iterdir()] == ["run.json"]


def test_the_text_is_written_byte_for_byte(tmp_path):
    """REGRESSION for #31: no newline translation and no re-encoding.

    The final file is held to byte-identity with the stdout artefact, so the writer
    may not alter a byte of what it is handed.
    """
    out = ArtefactFile(tmp_path / "run.json", say=lambda _: None)
    text = '{\n  "a": "\\u00e9 \\r"\n}\n'

    out.persist(lambda: text, when="at epoch 1")

    assert (tmp_path / "run.json").read_bytes() == text.encode()


def test_a_failure_is_said_once_and_raised_every_time(tmp_path):
    """REGRESSION for #31: one line per spell, one for the final write; raised per write.

    The run keeps trying, because a full disk can clear. The operator is told once per
    spell of failures, because a line per epoch is noise that buries the first; the
    artefact records every failed write, because each is a fact about an epoch. The
    final write is said as well, even inside the spell: its failure reaches no
    artefact, so the console is its only channel.
    """
    lines, say = _said()
    out = ArtefactFile(tmp_path / "absent" / "run.json", say=say)

    for epoch in (1, 2):
        with pytest.raises(ArtefactWriteError):
            out.persist(lambda: "{}\n", when=f"at epoch {epoch}")
    assert out.final("{}\n") is False

    assert len(lines) == 2
    assert lines[0].startswith("[out] WARNING: could not write the artefact at epoch 1")
    assert lines[1].startswith(
        "[out] WARNING: could not write the artefact at the end of the run")


def test_a_builder_failure_is_a_write_failure(tmp_path):
    """REGRESSION for #31: the text not being BUILT persists nothing, and says so."""
    lines, say = _said()
    out = ArtefactFile(tmp_path / "run.json", say=say)

    def broken() -> str:
        raise TypeError("cannot render")

    with pytest.raises(ArtefactWriteError, match="TypeError: cannot render"):
        out.persist(broken, when="at epoch 1")
    assert len(lines) == 1
    assert not (tmp_path / "run.json").exists()


def test_a_healthy_final_write_says_nothing(tmp_path):
    """REGRESSION for #31, the pair of the warning: silence on a healthy run."""
    lines, say = _said()
    out = ArtefactFile(tmp_path / "run.json", say=say)

    assert out.final("{}\n") is True
    assert lines == []


def test_an_os_error_detail_carries_no_path(tmp_path):
    """REGRESSION for #31: an ``OSError``'s message carries the path, and this drops it.

    A home directory is a person's name (#148's hand-off, #121's class). The errno and
    its text say what went wrong; the operator already knows where.
    """
    exc = PermissionError(errno.EACCES, os.strerror(errno.EACCES),
                          str(tmp_path / "someone" / "run.json"))

    detail = failure_detail(exc)

    assert detail == f"PermissionError [errno {errno.EACCES}]: {os.strerror(errno.EACCES)}"
    assert "someone" not in detail


def test_an_os_error_without_an_errno_is_rendered_whole():
    """REGRESSION for #31: the errno form is used only where there is an errno."""
    assert failure_detail(OSError("no errno here")) == "OSError: no errno here"


def test_the_warning_stays_true_when_a_later_write_succeeds(tmp_path, monkeypatch):
    """REGRESSION for #31's pre-merge review: a line never retracted must not go false.

    A full disk can clear. The first shape said "the file is not current" outright,
    and a later write then made it current while the line stood. It now says what
    holds until a later write succeeds.
    """
    lines, say = _said()
    out = ArtefactFile(tmp_path / "run.json", say=say)
    real = artefact_file._replace
    calls = {"n": 0}

    def full_once(path, text):  # noqa: ANN001
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))
        return real(path, text)

    monkeypatch.setattr(artefact_file, "_replace", full_once)
    with pytest.raises(ArtefactWriteError):
        out.persist(lambda: "{}\n", when="at epoch 1")
    out.persist(lambda: '{"epochs": 2}\n', when="at epoch 2")

    assert (tmp_path / "run.json").read_text() == '{"epochs": 2}\n'
    assert len(lines) == 1
    assert "not current until a later write succeeds" in lines[0]


def test_a_failure_after_a_recovery_is_said_again(tmp_path, monkeypatch):
    """REGRESSION for #31's second review pass: a new spell of failures is a new line.

    Said once and never again, "fail, succeed, fail" printed exactly what "fail,
    succeed, succeed" printed, and the one line ("not current until a later write
    succeeds") read as the recovery while the file was stale again. Paired with the
    test above it, which pins that a single spell is said once.
    """
    lines, say = _said()
    out = ArtefactFile(tmp_path / "run.json", say=say)
    real = artefact_file._replace
    outcomes = iter([False, True, False])

    def scripted(path, text):  # noqa: ANN001
        if not next(outcomes):
            raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))
        return real(path, text)

    monkeypatch.setattr(artefact_file, "_replace", scripted)
    with pytest.raises(ArtefactWriteError):
        out.persist(lambda: "{}\n", when="at epoch 1")
    out.persist(lambda: "{}\n", when="at epoch 2")
    # Per-epoch writes only: the final write is always said, so ending on it could
    # not tell a spell that was reset by the success from one that was never reset.
    with pytest.raises(ArtefactWriteError):
        out.persist(lambda: "{}\n", when="at epoch 3")

    assert [ln.split(":")[1].strip() for ln in lines] == [
        "could not write the artefact at epoch 1",
        "could not write the artefact at epoch 3"]


def _tree(path, *, git_file: bool = False):
    path.mkdir(parents=True)
    if git_file:
        (path / ".git").write_text("gitdir: /elsewhere/.git/modules/x\n")
    else:
        (path / ".git").mkdir()
    return path


@pytest.mark.parametrize("where", ["submodule top", "below the submodule",
                                   "superproject", "linked worktree inside"])
def test_any_git_entry_above_the_destination_refuses_it(tmp_path, where):
    """REGRESSION for #31's second review pass: no git is asked, so none can mislead.

    A ``.git`` FILE is what a submodule and a linked worktree carry; inside a
    submodule ``git worktree list`` names the gitdir, which is how the git-listing
    rule was broken. A ``.git`` directory above it is the superproject, or the
    checkout a ``.claude/worktrees/`` tree lives in.
    """
    sup = _tree(tmp_path / "sup")
    sub = _tree(sup / "lib", git_file=True)
    (sub / "d").mkdir()
    reviewed = sub / "d"
    destination = {"submodule top": sub / "x.json",
                   "below the submodule": sub / "d" / "x.json",
                   "superproject": sup / "x.json",
                   "linked worktree inside": _tree(sup / ".claude" / "w", git_file=True)
                   / "x.json"}[where]

    with pytest.raises(DestinationError, match="inside a git working tree"):
        check_destination(str(destination), reviewed)


def test_a_non_git_reviewed_directory_is_refused_by_identity(tmp_path):
    """REGRESSION for #31: a review outside git still keeps its artefact out of it."""
    reviewed = tmp_path / "plain"
    (reviewed / "deep").mkdir(parents=True)

    with pytest.raises(DestinationError, match="the reviewed directory"):
        check_destination(str(reviewed / "deep" / "x.json"), reviewed)


def test_a_destination_outside_every_tree_is_accepted(tmp_path):
    """REGRESSION for #31, the pair of the refusals: a plain directory is usable."""
    reviewed = _tree(tmp_path / "repo")
    (tmp_path / "out").mkdir()

    assert check_destination(str(tmp_path / "out" / "x.json"), reviewed) == (
        tmp_path / "out").resolve() / "x.json"


def test_a_git_file_alone_refuses_the_destination(tmp_path):
    """REGRESSION for #31's final review pass: the ``.git`` FILE is the rule's headline.

    A linked worktree added anywhere (``git worktree add /elsewhere/wt``) and a
    submodule carry a ``.git`` file and no ``.git`` directory. Every other case here
    sat below a ``.git`` directory as well, so "directories only" — the narrower
    design — passed the whole suite.
    """
    wt = _tree(tmp_path / "wt", git_file=True)
    (wt / "d").mkdir()

    with pytest.raises(DestinationError, match="inside a git working tree"):
        check_destination(str(wt / "d" / "x.json"), tmp_path / "elsewhere")


@pytest.mark.skipif(os.name != "posix", reason="needs a symbolic link")
def test_a_non_git_reviewed_directory_is_matched_by_identity_not_spelling(tmp_path):
    """REGRESSION for #31: ``--repo`` named through a link is the same directory.

    The reviewed directory is not in git, so only the identity check can refuse;
    the operator names it through a link, and the destination by its real path.
    """
    real = tmp_path / "plain"
    real.mkdir()
    link = tmp_path / "via-link"
    link.symlink_to(real, target_is_directory=True)

    with pytest.raises(DestinationError, match="the reviewed directory"):
        check_destination(str(real / "x.json"), link)


def test_a_name_too_long_is_refused_not_raised(tmp_path):
    """REGRESSION for #31's final review pass: an operator error, never a traceback.

    Before Python 3.14 ``exists()`` re-raises ENAMETOOLONG, which escaped as an
    unhandled ``OSError``: exit 1 and a traceback. From 3.14 ``exists()`` swallows it
    and the writability probe meets it instead. Either way it must be refused, with
    the errno's own words. The refusal goes to stderr and quotes the path the
    operator gave, as every refusal does; the published artefact never carries it.
    """
    raw = str(tmp_path / ("x" * 300 + ".json"))

    with pytest.raises(DestinationError, match=r"\[errno \d+\]: File name too long"):
        check_destination(raw, tmp_path / "elsewhere")
