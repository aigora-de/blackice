# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""The entry point's own report must not kill the run it reports (#141).

#129 guarded the engine's gate seam so a failing gate no longer discards a paid-for
run. Measured, every trigger the shipped backend can reach there is a STDOUT fault
— and the entry point prints its whole report before the ``--- JSON ---`` block, so
on exactly those runs the report raised after every epoch had completed and the
artefact went with it. ``json.dumps`` escapes to ASCII, so the artefact itself would
have survived; it was the console report, printed first, that took the run down.

Two faults, and they are not alike:

* an ENCODING fault is recoverable. The characters are decorative punctuation, so
  every line is kept with ``?`` where the console cannot represent one — and the
  substitution is SAID, once per run, because a record that does not say what it
  could not keep is #111's defect one document along;
* a CLOSED PIPE is not. There is nothing to fall back to on stdout, so the process
  says so on stderr and exits non-zero: the halt's own code where that is already
  non-zero, ``1`` where it would have been ``0``, so a shell ``&&`` cannot read an
  undelivered artefact as success and an UGLY's ``3`` is never masked.

What must not happen is lines silently dropped, which is the quieter version of the
defect and the shape #11 and #111 exist to refuse.

The faults are reproduced in-process with a stand-in stdout whose ``write`` raises
exactly what the real stream raises — the capture streams pytest installs are UTF-8
whatever the locale, so breaking those reproduces nothing. Two tests then drive the
real ``python -m kuang`` in a subprocess, because one behaviour exists only in a real
process: the interpreter's own flush of stdout at exit, which on a closed pipe turns
a clean return into ``rc=120`` and an ``Exception ignored`` banner.

**Which of these are regression tests, said plainly.** Every test that drives a fault
goes red on ``main``: there the run raises. The healthy-run test and the restoration
test pass before and after, and say so in their own docstrings: they pin what the fix
must NOT move.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path

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

## Sentinel — Ruin

Hunt ruin-class hazards only.
"""

_ROOT = Path(__file__).resolve().parents[2]


class _AsciiConsole(io.StringIO):
    """A stdout whose encoding is ASCII, raising exactly what the real one raises.

    Measured on the real ``TextIOWrapper``: a write that cannot be encoded raises
    ``UnicodeEncodeError`` and writes NOTHING, not even its ASCII prefix, so a
    stand-in that encodes before it stores is faithful on the one point that matters.
    """

    encoding = "ascii"

    def write(self, s: str) -> int:
        s.encode("ascii")
        return super().write(s)


class _ClosedPipe(io.StringIO):
    """A stdout whose reader has gone, after accepting ``after`` writes.

    ``print`` writes its text and its newline separately, so a print is two writes.
    """

    def __init__(self, after: int = 0) -> None:
        super().__init__()
        self.after = after

    def write(self, s: str) -> int:
        if self.after <= 0:
            raise BrokenPipeError(32, "Broken pipe")
        self.after -= 1
        return super().write(s)

    def flush(self) -> None:
        if self.after <= 0:
            raise BrokenPipeError(32, "Broken pipe")


class _AsciiThenClosed(_AsciiConsole):
    """An ASCII stdout whose reader leaves the moment the first encoding fault is raised.

    So the retry of the replaced text meets a closed pipe: both faults, in one write.
    """

    def __init__(self) -> None:
        super().__init__()
        self.gone = False

    def write(self, s: str) -> int:
        if self.gone:
            raise BrokenPipeError(32, "Broken pipe")
        try:
            return super().write(s)
        except UnicodeEncodeError:
            self.gone = True
            raise

    def flush(self) -> None:
        if self.gone:
            raise BrokenPipeError(32, "Broken pipe")


@pytest.fixture
def sourced_repo(changed_repo):
    (changed_repo / "CLAUDE.md").write_text(_CLAUDE_MD)
    return changed_repo


def _ugly_panel(monkeypatch) -> None:
    """Every persona replies; the Sentinel raises one UGLY, so the run halts ``3``."""
    def _reply(findings=()):
        body = json.dumps({"verdict": "NO", "findings": list(findings)})
        return CallResult(f"I reviewed it.\n\n```json\n{body}\n```\n", 0)

    ugly = {"title": "unbounded loss on retry", "severity": "UGLY",
            "claim_class": "logic", "file": "a.py", "line": 1, "evidence": "read it"}

    def _fake(self, prompt, mandate, tools, model):  # noqa: ANN001, ARG001
        return _reply([ugly]) if "ruin" in mandate.lower() else _reply()

    monkeypatch.setattr(session_module.PanelSession, "_run_claude", _fake)


def _dry_run(repo) -> int:
    return main(["--repo", str(repo), "--base", "HEAD~1", "--max-epochs", "1",
                 "--no-parallel", "--dry-run"])


def _healthy(repo, capsys) -> str:
    """The same run's stdout under a console that can encode everything."""
    assert _dry_run(repo) == 0
    return capsys.readouterr().out


def _notices(err: str) -> list[str]:
    return [line for line in err.splitlines() if line.startswith("kuang: ")]


# --- the encoding fault: recoverable, and said --------------------------------

def test_an_ascii_console_keeps_every_line_and_the_exit_code(sourced_repo, capsys,
                                                             monkeypatch):
    """REGRESSION for #141: this raised at the epoch account header, rc=1.

    Line for line the healthy report with ``?`` in place of what the console cannot
    represent — no line dropped, none added, the halt's own exit code returned.
    """
    healthy = _healthy(sourced_repo, capsys)
    console = _AsciiConsole()
    monkeypatch.setattr(sys, "stdout", console)

    assert _dry_run(sourced_repo) == 0
    expected = healthy.encode("ascii", "replace").decode("ascii")
    assert console.getvalue() == expected


def test_an_ascii_console_leaves_the_artefact_byte_identical(sourced_repo, capsys,
                                                             monkeypatch):
    """The notice says the artefact is unaffected; this is what makes that true."""
    healthy = _healthy(sourced_repo, capsys)
    console = _AsciiConsole()
    monkeypatch.setattr(sys, "stdout", console)

    _dry_run(sourced_repo)
    assert (console.getvalue().split("--- JSON ---")[-1]
            == healthy.split("--- JSON ---")[-1])
    json.loads(console.getvalue().split("--- JSON ---")[-1])


def test_a_substitution_is_said_once_with_its_exact_count(sourced_repo, capsys,
                                                          monkeypatch):
    """REGRESSION for #141, and #111's rule one document along.

    One notice per run, never one per line, on stderr — stdout after the JSON block
    is not available, because both parsers read from the last marker to EOF. It
    counts the characters actually replaced, so it is exactly knowable, and it is
    itself ASCII so it cannot fail the way the thing it reports did.
    """
    healthy = _healthy(sourced_repo, capsys)
    replaced = sum(1 for c in healthy if ord(c) > 127)
    assert replaced > 0, "the fixture must exercise the fault"
    monkeypatch.setattr(sys, "stdout", _AsciiConsole())

    _dry_run(sourced_repo)
    notices = _notices(capsys.readouterr().err)
    assert len(notices) == 1
    assert f"{replaced} character(s)" in notices[0]
    assert "ascii" in notices[0]
    assert "artefact is ASCII-escaped and unaffected" in notices[0]
    notices[0].encode("ascii")


def test_a_healthy_console_is_untouched_and_says_nothing(sourced_repo, capsys):
    """GUARD, green on ``main``: a rule that fires on a healthy run is the defect's mirror.

    Under a console that encodes everything, nothing is replaced, nothing is said,
    and the report is the one ``main`` printed.
    """
    _healthy(sourced_repo, capsys)
    assert _dry_run(sourced_repo) == 0
    captured = capsys.readouterr()
    assert "—" in captured.out, "the em dashes reach a console that can print them"
    assert _notices(captured.err) == []


# --- the closed pipe: not recoverable, and never read as success ---------------

def test_a_closed_pipe_is_said_and_never_exits_zero(sourced_repo, capsys, monkeypatch):
    """REGRESSION for #141: this raised ``BrokenPipeError`` out of the halt line.

    The dry run halts ``no_review``, whose code is ``0``. Its artefact was never
    delivered, so returning ``0`` would let a shell ``&&`` read success.
    """
    monkeypatch.setattr(sys, "stdout", _ClosedPipe())

    assert _dry_run(sourced_repo) == 1
    notices = _notices(capsys.readouterr().err)
    assert len(notices) == 1
    assert "not delivered" in notices[0]
    notices[0].encode("ascii")


def test_a_closed_pipe_keeps_what_was_written_before_it_closed(sourced_repo, capsys,
                                                                monkeypatch):
    """REGRESSION for #141: the lines delivered before the reader left are the healthy ones."""
    healthy = _healthy(sourced_repo, capsys)
    pipe = _ClosedPipe(after=6)
    monkeypatch.setattr(sys, "stdout", pipe)

    assert _dry_run(sourced_repo) == 1
    delivered = pipe.getvalue()
    assert delivered and healthy.startswith(delivered)
    assert len(delivered) < len(healthy), "the reader left part-way through"


def test_a_closed_pipe_never_masks_the_circuit_breaker(sourced_repo, capsys,
                                                       monkeypatch):
    """REGRESSION for #141: an UGLY halt keeps its ``3`` when its report is lost.

    The halt's own code is kept wherever it is already non-zero, because each such
    code carries a stronger fact than "not delivered" — and the ruin-class one
    above all.
    """
    _ugly_panel(monkeypatch)
    monkeypatch.setattr(sys, "stdout", _ClosedPipe())

    rc = main(["--repo", str(sourced_repo), "--base", "HEAD~1", "--max-epochs", "1",
               "--no-parallel"])
    assert rc == 3
    assert len(_notices(capsys.readouterr().err)) == 1


def test_a_retry_that_meets_a_closed_pipe_claims_no_substitution(sourced_repo, capsys,
                                                                 monkeypatch):
    """REGRESSION for #141: the replaced text's own write can meet a closed pipe.

    The retry is a second write and a second chance to fail, so it is guarded like
    the first: the run returns and the stream is lost. And the substitution notice
    is NOT said, because it claims characters were printed as ``?`` and these never
    were — the only notice is the one that is true.
    """
    monkeypatch.setattr(sys, "stdout", _AsciiThenClosed())

    assert _dry_run(sourced_repo) == 1
    notices = _notices(capsys.readouterr().err)
    assert len(notices) == 1
    assert "not delivered" in notices[0]


def test_a_closed_stderr_as_well_still_keeps_the_circuit_breaker(sourced_repo,
                                                                  monkeypatch):
    """REGRESSION for #141: ``2>&1 | head`` closes both streams at once.

    Then the notice has nowhere to go either, and the exit code is the only channel
    left — so the notice must not raise, or an UGLY's ``3`` becomes the ``1`` of an
    unhandled exception.
    """
    _ugly_panel(monkeypatch)
    monkeypatch.setattr(sys, "stdout", _ClosedPipe())
    monkeypatch.setattr(sys, "stderr", _ClosedPipe())

    rc = main(["--repo", str(sourced_repo), "--base", "HEAD~1", "--max-epochs", "1",
               "--no-parallel"])
    assert rc == 3


def test_main_restores_the_stdout_it_found(sourced_repo, capsys, monkeypatch):
    """GUARD, green on ``main``: whatever wraps stdout for a run unwraps it after.

    ``main`` is called in-process by every test in this directory, so a wrapper that
    outlived it would leak into the next one.
    """
    console = _AsciiConsole()
    monkeypatch.setattr(sys, "stdout", console)
    try:
        _dry_run(sourced_repo)
    except UnicodeEncodeError:
        pass
    assert sys.stdout is console


# --- the real process ----------------------------------------------------------

def _kuang(repo, **kwargs) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(_ROOT)}
    env.update(kwargs.pop("env", {}))
    return subprocess.run(
        [sys.executable, "-m", "kuang", "--repo", str(repo), "--base", "HEAD~1",
         "--max-epochs", "1", "--no-parallel", "--dry-run"],
        env=env, stderr=subprocess.PIPE, text=True, check=False, **kwargs)


def test_a_real_ascii_console_completes(sourced_repo):
    """REGRESSION for #141, end to end: ``PYTHONIOENCODING=ascii`` ended in a traceback."""
    proc = _kuang(sourced_repo, stdout=subprocess.PIPE,
                  env={"PYTHONIOENCODING": "ascii"})

    assert proc.returncode == 0, proc.stderr
    assert "Traceback" not in proc.stderr
    assert len(_notices(proc.stderr)) == 1
    json.loads(proc.stdout.split("--- JSON ---")[-1])


def test_a_real_closed_pipe_exits_cleanly(sourced_repo):
    """REGRESSION for #141, end to end, and the one fault only a real process has.

    The read end is closed BEFORE the child starts, so every write it makes fails
    and the result does not depend on timing. Beyond the in-process tests this pins
    the interpreter's own flush at exit: unhandled, a closed pipe that nothing
    in-process noticed still exits ``120`` with an ``Exception ignored`` banner.
    """
    read_end, write_end = os.pipe()
    os.close(read_end)
    try:
        proc = _kuang(sourced_repo, stdout=write_end)
    finally:
        os.close(write_end)

    assert proc.returncode == 1, proc.stderr
    assert "Traceback" not in proc.stderr
    assert "Exception ignored" not in proc.stderr
    assert len(_notices(proc.stderr)) == 1