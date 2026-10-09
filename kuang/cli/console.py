# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""stdout as the entry point's report sees it, so the report cannot kill the run (#141).

The report is printed whole before the ``--- JSON ---`` block, so a stdout fault
raised anywhere in it took the artefact with it — after every epoch had completed
and been paid for. Measured, two faults reach it, and they are not alike:

* an ENCODING fault (``PYTHONIOENCODING=ascii``, a ``LC_ALL=C`` redirect) is
  recoverable. The characters it trips on are decorative punctuation, so the line is
  written again with ``?`` in their place — and COUNTED, because a substitution
  nobody says is #111's defect one document along;
* a CLOSED PIPE is not. There is nothing to fall back to on stdout, so the stream
  records that it was lost and drops what follows, and the entry point says so on
  stderr and refuses to exit ``0`` for a loss it saw.

ONE wrapper at entry, deliberately, and not 46 call sites or a helper each of them
calls: either of those is a patched table that the next ``print`` nobody remembers
reopens. Installed for the length of ``main`` it covers every print made under it —
the backend's included, which therefore no longer raise there either. That is a
consequence of the site and not a change to the backend: no backend line is touched,
and the backend driven without this entry point behaves exactly as before.

What it under-reports, stated:

* a reader who leaves AFTER the kernel accepted the bytes is invisible to the
  writer. ``| head -1`` on a report that fits the pipe's buffer exits ``0`` with
  nothing said, measured — only a loss some write or flush SEES is reported;
* the gate's own prompt on a terminal is encoded by ``input()`` itself, not written
  through this stream, so it is replaced (``errors`` says ``replace``) but not
  counted;
* the count is of characters, not of the lines they were in;
* a pipe closed while the gate is waiting is noticed, but does not end the run —
  the epochs after it are still spawned and paid for, for a reader who has gone.
  Ending a run early is a halting decision and not this stream's;
* ``writelines`` and ``buffer`` reach the wrapped stream unguarded. Nothing in
  ``kuang`` calls either; ``print`` calls ``write``.
"""

from __future__ import annotations

import os
import threading
from typing import TextIO


class ConsoleStream:
    """A text stream that survives the two stdout faults and records each one.

    Measured on the real ``TextIOWrapper``: a write that cannot be encoded raises
    ``UnicodeEncodeError`` and writes NOTHING, not even its ASCII prefix, so writing
    the replaced text again duplicates nothing.

    Every attribute it does not define is the wrapped stream's, ``encoding`` above
    all: ``input()`` reads ``sys.stdout.encoding`` and ``errors`` to encode a
    terminal prompt ITSELF, bypassing ``write``. ``errors`` is therefore this
    stream's own and says ``replace``, which is what it does — measured on a real
    pty, a strict one made the gate's em-dash prompt raise at an ASCII terminal, and
    the gate recorded that as an answer it could not read.

    The two notices are ORTHOGONAL: one is about the text (what was replaced),
    the other about delivery (whether it arrived). Each is exactly knowable on its
    own; tying the first to the second could not be, because a buffered write is
    accepted long before it is known to have arrived.

    ``None`` is a stream too: with descriptor 1 closed (``>&-``) CPython sets
    ``sys.stdout`` to ``None``, so there was never anywhere to deliver to.
    """

    def __init__(self, inner: TextIO | None) -> None:
        self.inner = inner
        self.substituted = 0
        self.fault_encoding: str | None = None
        self.lost = inner is None
        self._count = threading.Lock()  # parallel personas print from worker threads

    @property
    def errors(self) -> str:
        return "replace"

    def write(self, s: str) -> int:
        if self.inner is None:
            return len(s)
        try:
            return self.inner.write(s)
        except UnicodeEncodeError as exc:
            kept = s.encode(exc.encoding, "replace").decode(exc.encoding)
            with self._count:
                self.fault_encoding = exc.encoding
                self.substituted += sum(1 for a, b in zip(s, kept) if a != b)
            self._deliver(kept)
            return len(s)
        except BrokenPipeError:
            self.lost = True
            return len(s)

    def _deliver(self, s: str) -> None:
        try:
            self.inner.write(s)
        except BrokenPipeError:
            self.lost = True

    def flush(self) -> None:
        if self.inner is None:
            return
        try:
            self.inner.flush()
        except BrokenPipeError:
            self.lost = True

    def __getattr__(self, name: str):
        return getattr(self.inner, name)

    def release(self) -> None:
        """Release the wrapped stdout if it was lost; see ``release``."""
        if self.lost:
            release(self.inner)

    def notices(self) -> list[str]:
        """What the operator must be told about this console, in ASCII.

        ASCII because a notice about a console that cannot encode must not fail the
        way the thing it reports did.
        """
        said = []
        if self.substituted:
            said.append(f"kuang: {self.substituted} character(s) in the console report "
                        f"could not be encoded as {self.fault_encoding} and were "
                        f"replaced by '?'; the JSON artefact is ASCII-escaped and "
                        f"unaffected")
        if self.lost:
            said.append("kuang: stdout was closed before the report was complete; the "
                        "rest of it and the --- JSON --- artefact were not delivered")
        return said


def release(stream: TextIO | None) -> None:
    """Point a lost stream's descriptor at the null device.

    The interpreter flushes ``sys.stdout`` and ``sys.stderr`` once more at exit, and
    the bytes a failed write leaves buffered fail again there: measured, that turns a
    clean return into ``rc=120`` and an ``Exception ignored`` banner — for stderr too,
    when ``2>&1 | head`` closes both, where it turned an UGLY's ``3`` into ``120``. A
    stream with no descriptor (an in-process capture, or ``None``) has no exit flush
    to fail.
    """
    try:
        fd = stream.fileno()
    except (AttributeError, OSError, ValueError):
        return
    null = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null, fd)
    finally:
        os.close(null)
