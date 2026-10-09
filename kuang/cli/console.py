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
  stderr and refuses to exit ``0``.

ONE wrapper at entry, deliberately, and not 46 call sites or a helper each of them
calls: either of those is a patched table that the next ``print`` nobody remembers
reopens. Installed for the length of ``main`` it covers every print made under it —
the backend's included, which therefore no longer raise there either. That is a
consequence of the site and not a change to the backend: no backend line is touched,
and the backend driven without this entry point behaves exactly as before.

What it under-reports, stated: the substitution count is of characters, not of the
lines they were in; and a pipe closed while the gate is waiting is noticed, but
does not end the run — the epochs after it are still spawned and paid for, for a
reader who has gone. Ending a run early is a halting decision and not this stream's.
"""

from __future__ import annotations

import os
from typing import TextIO


class ConsoleStream:
    """A text stream that survives the two stdout faults and records each one.

    Measured on the real ``TextIOWrapper``: a write that cannot be encoded raises
    ``UnicodeEncodeError`` and writes NOTHING, not even its ASCII prefix, so writing
    the replaced text again duplicates nothing.
    """

    def __init__(self, inner: TextIO) -> None:
        self.inner = inner
        self.substituted = 0
        self.encoding: str | None = None
        self.lost = False

    def write(self, s: str) -> int:
        try:
            return self.inner.write(s)
        except UnicodeEncodeError as exc:
            kept = s.encode(exc.encoding, "replace").decode(exc.encoding)
            # Counted only once the replaced text is accepted: the notice says these
            # characters were PRINTED as ``?``, which a retry that met a closed pipe
            # never did. Accepted is not flushed — a buffered stream that is lost
            # later has still counted them, and the lost notice beside it says so.
            if self._deliver(kept):
                self.encoding = exc.encoding
                self.substituted += sum(1 for a, b in zip(s, kept) if a != b)
            return len(s)
        except BrokenPipeError:
            self.lost = True
            return len(s)

    def _deliver(self, s: str) -> bool:
        try:
            self.inner.write(s)
        except BrokenPipeError:
            self.lost = True
            return False
        return True

    def flush(self) -> None:
        try:
            self.inner.flush()
        except BrokenPipeError:
            self.lost = True

    def __getattr__(self, name: str):
        return getattr(self.inner, name)

    def release(self) -> None:
        """Point a lost stdout's descriptor at the null device.

        The interpreter flushes ``sys.stdout`` once more at exit, and the bytes a
        failed flush leaves buffered fail again there: measured, that turns a clean
        return into ``rc=120`` and an ``Exception ignored`` banner. A stream with no
        descriptor (an in-process capture) has no exit flush to fail.
        """
        if not self.lost:
            return
        try:
            fd = self.inner.fileno()
        except (AttributeError, OSError, ValueError):
            return
        null = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(null, fd)
        finally:
            os.close(null)

    def notices(self) -> list[str]:
        """What the operator must be told about this console, in ASCII.

        ASCII because a notice about a console that cannot encode must not fail the
        way the thing it reports did.
        """
        said = []
        if self.substituted:
            said.append(f"kuang: {self.substituted} character(s) in the console report "
                        f"could not be encoded as {self.encoding} and were printed as "
                        f"'?'; the JSON artefact is ASCII-escaped and unaffected")
        if self.lost:
            said.append("kuang: stdout was closed before the report was complete; the "
                        "rest of it and the --- JSON --- artefact were not delivered")
        return said
