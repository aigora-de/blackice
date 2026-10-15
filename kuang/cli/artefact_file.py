# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""The durable run artefact behind ``--out PATH``: refused early, replaced atomically (#31).

Before #31 the artefact existed only in memory until the run ended, and then only on
stdout, so a run that was interrupted or killed left nothing a machine could read. This
module is the file half of the fix, and it is kuang's only file write.

**Opt-in, deliberately.** No flag, no file: an operator who did not ask has nothing
written into their filesystem, and a healthy run's stdout does not move by a byte.
Whether there should be a default destination, and how an operator learns where the
artefact went, is #146's interface question. A run started without ``--out`` is as
unrecoverable as it was before #31.

**How a reader tells a partial file from a complete one.** By ``halt_reason``, which
the artefact already carries. A complete run always has one, because the CLI renders
only after the engine returns. A per-epoch write never does, because ``checkpoint``
runs before the loop publishes the halt, on the halting epoch too (#155). So
``halt_reason: null`` means *partial*, exactly. A partial file is also short of the
newest epoch's ``gate``, ``gate_failure`` and its own ``checkpoint`` failure, which the
loop records after the call (#129, #148). The final write is the stdout artefact byte
for byte.

**Atomic.** The text goes to a temporary file in the same directory, is flushed and
``fsync``-ed, and is then put in place with ``os.replace``. A reader sees the previous
file or the new one, never half of either. The file is ``fsync``-ed and the directory is
not: if the new name is visible its contents are whole, and after a power loss the worst
case is the previous epoch's file, which is still a whole partial. On macOS ``fsync``
does not reach the platter (``F_FULLFSYNC`` would), and that is stated rather than
fixed. What this does not cover: a SIGKILL between creating the temporary file and
replacing the destination leaves a stray ``.<name>.*.tmp`` beside it, and a SIGKILL
during ``os.replace`` is the filesystem's. ``mkstemp`` creates the file readable by its
owner only, which suits an artefact that is not publishable by default (#121).

**A failed write does not stop the run.** Against ``HaltReason``'s bar, a run whose
file cannot be written can still do what it is for, and its stdout artefact still
arrives. The operator is told once, above the report, that the file is not current
until a later write succeeds.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

from kuang.engine.findings import render_exception


class DestinationError(Exception):
    """``--out`` names a destination the run cannot use. An operator error (#18)."""


class ArtefactWriteError(Exception):
    """A durable write failed. Its message is ``failure_detail``'s, so it carries no path."""


def failure_detail(exc: BaseException) -> str:
    """What a failed write publishes: what went wrong, never where.

    An ``OSError``'s message carries the path it failed on, and a home directory is a
    person's name on most installs (#148's hand-off, #121's class). The errno and its
    text say what went wrong. The operator named the path, so they already know where.
    Anything else is rendered whole, as the engine renders a failed seam.
    """
    if isinstance(exc, OSError) and exc.errno is not None:
        return f"{type(exc).__name__} [errno {exc.errno}]: {exc.strerror}"
    return render_exception(exc)


def check_destination(raw: str, repo: Path) -> Path:
    """Refuse an ``--out`` destination before anything is spent, or return it resolved.

    Each refusal is an operator error, reported through the same path as an unusable
    ``--panel``: before the first epoch, so before any spend.

    * Empty, or ending in a separator: no file name was given. An unset shell
      variable produces the first constantly (#106's case).
    * Inside the reviewed repository: the artefact quotes the source and is not
      publishable by default (#121), and a file there is one ``git add -A`` from a
      public commit. See ``_forbidden_roots`` for what "the repository" covers and
      ``_inside`` for how containment is decided.
    * Already present, as a file, a directory or a link: a previous run's only durable
      record is never overwritten unasked, because that cannot be undone.
    * A directory that does not exist, or one this user cannot create a file in. The
      second is probed by creating and removing a real temporary file there, rather
      than asked of ``os.access``, which answers a different question on some
      filesystems.

    What this does not cover: a file created at the destination after this check and
    before the first write is replaced by that write. The window is the run's own.
    """
    if not raw or raw.endswith(("/", os.sep)):
        raise DestinationError(
            f"--out {raw!r} names no file; give the path of the file to write")
    target = Path(raw)
    if not target.parent.is_dir():
        raise DestinationError(
            f"--out {raw}: the directory it would be written in does not exist")
    resolved = target.parent.resolve() / target.name
    if _inside(resolved.parent, _forbidden_roots(repo)):
        raise DestinationError(
            f"--out {raw} is inside the reviewed repository; the artefact quotes the "
            f"source under review and must be written outside it")
    if target.exists() or target.is_symlink():
        raise DestinationError(
            f"--out {raw} already exists; kuang does not overwrite it")
    try:
        fd, probe = tempfile.mkstemp(dir=resolved.parent, prefix=f".{target.name}.",
                                     suffix=".tmp")
        os.close(fd)
        os.unlink(probe)
    except OSError as exc:
        raise DestinationError(
            f"--out {raw}: cannot write in its directory ({failure_detail(exc)})"
        ) from None
    return resolved


def _forbidden_roots(repo: Path) -> list[Path]:
    """Every directory the artefact must not be written under: the reviewed repository.

    One rule, stated once: the repository is ``--repo`` itself AND every working tree
    git lists for the repository ``--repo`` belongs to. ``--repo`` alone is not
    enough, and each way it falls short was measured on #31's diff:

    * ``--repo`` may be a subdirectory of a working tree (its default is ``.``), so a
      destination beside it can still be inside the tree git commits from;
    * a linked worktree's repository includes the main checkout and every other
      linked tree, any of which ``git add -A`` can publish from.

    ``git worktree list`` names all of them, from any of them. If git cannot answer
    (not a repository, or no git), only ``--repo`` is known, and that is all that is
    refused: what this under-reports is a non-git ``--repo`` nested inside a git tree.
    """
    roots = [repo]
    try:
        listed = subprocess.run(
            ["git", "-C", str(repo), "worktree", "list", "--porcelain"],
            capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return roots
    # Line-split rather than ``-z``, which needs git 2.36: on an older git the
    # fallback above would quietly narrow the rule, and a path with a newline in it
    # is not a working tree anyone reviews.
    roots += [Path(line[len("worktree "):]) for line in listed.splitlines()
              if line.startswith("worktree ")]
    return roots


def _inside(directory: Path, roots: list[Path]) -> bool:
    """Whether ``directory`` is one of ``roots`` or below one, decided by IDENTITY.

    Each ancestor of ``directory`` is compared with each root by ``os.path.samestat``:
    the same file on the same device, never the same spelling. A string comparison
    is wrong on a case-insensitive volume (``REPO`` is ``repo``), through any link,
    and wherever two spellings name one directory; identity is wrong in none of
    them. A root that cannot be stat-ed cannot contain anything, and is skipped.
    """
    root_stats = []
    for root in roots:
        try:
            root_stats.append(os.stat(root))
        except OSError:
            continue
    for ancestor in (directory, *directory.parents):
        try:
            st = os.stat(ancestor)
        except OSError:
            continue
        if any(os.path.samestat(st, r) for r in root_stats):
            return True
    return False


_UNTIL_A_LATER_WRITE = ("The --out file is not current until a later write succeeds, "
                        "and a run stopped before then cannot be recovered from it.")
_AT_THE_END = ("The --out file is not the complete artefact: it holds the last "
               "epoch written, if any, and says halt_reason null.")


class ArtefactFile:
    """One run's durable artefact: every write replaces it whole.

    ``persist`` is the per-epoch write, and raises ``ArtefactWriteError`` on failure so
    the checkpoint it runs in is recorded as failed. ``final`` is the end-of-run write,
    which nothing records, so it returns whether it succeeded. Both say the first
    failure once, through ``say``, and never again.
    """

    def __init__(self, path: Path, *, say: Callable[[str], None]) -> None:
        self.path = path
        self._say = say
        self._warned = False

    def persist(self, build: Callable[[], str], *, when: str,
                consequence: str = _UNTIL_A_LATER_WRITE) -> None:
        """Build the text and put it in place, or say so and raise.

        The build is inside the guard: text that could not be rendered persisted
        nothing, exactly as a full disk did, and the operator is owed the same line.

        The line is said once and never retracted, so what it says must stay true
        whatever happens next: a later write can succeed (a full disk clears), and
        then "not current" would be false. So it says what holds UNTIL then.
        ``Exception``, never ``BaseException``: a Ctrl-C mid-write still stops the run,
        and the temporary file is removed on the way out.
        """
        try:
            _replace(self.path, build())
        except Exception as exc:  # noqa: BLE001
            detail = failure_detail(exc)
            if not self._warned:
                self._warned = True
                self._say(f"[out] WARNING: could not write the artefact {when}: "
                          f"{detail}. {consequence}")
            raise ArtefactWriteError(detail) from None

    def final(self, text: str) -> bool:
        """The end-of-run write: the stdout artefact, byte for byte."""
        try:
            self.persist(lambda: text, when="at the end of the run",
                         consequence=_AT_THE_END)
        except ArtefactWriteError:
            return False
        return True


def _replace(path: Path, text: str) -> None:
    """Write ``text`` beside ``path`` and rename it into place."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        # ``newline=""`` and an explicit encoding: the final file is held to
        # byte-identity with the stdout artefact, so no byte may be translated.
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
