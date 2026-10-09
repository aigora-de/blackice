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
  artefact it KNOWS was undelivered as success and an UGLY's ``3`` is never
  masked. A reader who leaves after the kernel accepted the bytes is not known —
  ``| head`` on a report that fits the pipe's buffer — and the stream says so.

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
from kuang.cli.console import ConsoleStream

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


class _UnbufferedClosedPipe(io.StringIO):
    """A stdout whose reader has gone, with NO buffer: the write fails, the flush cannot.

    What ``python -u`` or ``PYTHONUNBUFFERED`` gives a real pipe — every write is
    pushed at once, so nothing is left for a flush to fail on.
    """

    def write(self, s: str) -> int:
        raise BrokenPipeError(32, "Broken pipe")


class _BufferedClosedPipe(io.StringIO):
    """A stdout whose reader has gone, behind a buffer: writes succeed, the flush fails.

    What a real pipe does when the report fits in the buffer — nothing raises until
    the bytes are pushed, so the loss is only observable at a flush.
    """

    def flush(self) -> None:
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


def test_both_faults_in_one_run_are_both_said(sourced_repo, capsys, monkeypatch):
    """REGRESSION for #141: the two notices are orthogonal, so both faults say both.

    One is about the TEXT (what was replaced), the other about DELIVERY (whether it
    arrived). A first draft tied the first to the second — "counted only once
    accepted" — and the review showed that could not be exact: a buffered write is
    accepted long before it is known to have arrived. That the retry itself survives
    is pinned at the stream, below: end to end the engine's spawn guard masks it.
    """
    monkeypatch.setattr(sys, "stdout", _AsciiThenClosed())

    assert _dry_run(sourced_repo) == 1
    notices = _notices(capsys.readouterr().err)
    assert len(notices) == 2
    assert "replaced by '?'" in notices[0] and "not delivered" in notices[1]


def test_a_loss_only_a_flush_can_see_is_still_seen(sourced_repo, capsys, monkeypatch):
    """REGRESSION for #141: a report that fits in the buffer raises at no ``print``.

    The loss then exists only at the flush, so the flush must happen while the entry
    point can still act on it, not in the interpreter's exit after it has returned.
    """
    monkeypatch.setattr(sys, "stdout", _BufferedClosedPipe())

    assert _dry_run(sourced_repo) == 1
    assert len(_notices(capsys.readouterr().err)) == 1


def test_a_loss_only_a_write_can_see_is_still_seen(sourced_repo, capsys, monkeypatch):
    """REGRESSION for #141: the mirror of the test above, for an unbuffered stdout.

    Measured, a mutation that stopped the WRITE from recording the loss survived every
    other test, because each of their stand-ins also fails the final flush, which
    records it a second time. Unbuffered, nothing is left to flush — the write is the
    only place the loss is ever seen, and without it the run exits ``0``.
    """
    monkeypatch.setattr(sys, "stdout", _UnbufferedClosedPipe())

    assert _dry_run(sourced_repo) == 1
    assert len(_notices(capsys.readouterr().err)) == 1


def test_the_retry_of_replaced_text_survives_a_closed_pipe():
    """REGRESSION for #141, at the stream: the retry is a second write that can fail.

    Pinned directly because end to end it is unreachable as a distinct outcome: the
    first non-ASCII write of a dry run is inside ``spawn``, whose engine guard (#25)
    contains an escaped ``BrokenPipeError`` and lets the run continue to the same
    exit code — measured, a mutation removing this guard survived the CLI tests.
    """
    stream = ConsoleStream(_AsciiThenClosed())
    assert stream.write("a — b") == len("a — b")
    assert stream.lost is True
    assert stream.substituted == 1, "the text had one character replaced"


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
    """``python -m kuang`` with stdout BUFFERED, as an operator's shell has it.

    The inherited stdout settings are dropped, not passed through. Measured: with
    ``PYTHONUNBUFFERED`` set, every write reaches the pipe at once, nothing is left
    for the exit flush, and the closed-pipe test below passes with ``release`` deleted
    — a harness that exports it would have made that test vacuous.
    """
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONUNBUFFERED", "PYTHONIOENCODING")}
    env["PYTHONPATH"] = str(_ROOT)
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

# --- what the pre-merge review found, each measured before it was accepted ------

def test_the_stream_presents_the_encoding_it_wraps_and_the_errors_it_applies():
    """REGRESSION for #141's review: ``input()`` reads these two to encode a prompt.

    A first draft shadowed ``encoding`` with its own fault record, ``None`` until the
    first fault and the fault's codec after it — and ``errors`` was the wrapped
    stream's ``strict``. The pty test below is what that cost.
    """
    stream = ConsoleStream(_AsciiConsole())
    assert stream.encoding == "ascii"
    stream.write("a — b")
    assert stream.encoding == "ascii" and stream.errors == "replace"


_DRIVER = '''\
import json, sys
from kuang.backends.claude_code import session
from kuang.backends.claude_code.spawn import CallResult
from kuang.cli import main

def _fake(self, prompt, mandate, tools, model):
    f = {"title": "weak logic", "severity": sys.argv[2], "claim_class": "logic",
         "file": "a.py", "line": 1, "evidence": "read it"}
    body = json.dumps({"verdict": "NO", "findings": [f]})
    return CallResult(f"I reviewed it.\\n\\n```json\\n{body}\\n```\\n", 0)

session.PanelSession._run_claude = _fake
raise SystemExit(main(["--repo", sys.argv[1], "--base", "HEAD~1", "--max-epochs", "2",
                       "--no-parallel"]))
'''


def _driven(tmp_path, repo, severity: str) -> list[str]:
    """A real ``python`` running ``main`` over a stubbed panel: no reviewer is spawned."""
    driver = tmp_path / "driver.py"
    driver.write_text(_DRIVER)
    return [sys.executable, str(driver), str(repo), severity]


def _env(**extra) -> dict:
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONUNBUFFERED", "PYTHONIOENCODING")}
    env["PYTHONPATH"] = str(_ROOT)
    env.update(extra)
    return env


@pytest.mark.skipif(not hasattr(os, "openpty"), reason="needs a pseudo-terminal")
def test_an_ascii_terminal_still_asks_the_human(sourced_repo, tmp_path):
    """REGRESSION for #141's review, the BLOCKER: the gate skipped the human it asks.

    On a terminal ``input()`` encodes its prompt itself, from ``sys.stdout``'s
    ``encoding`` and ``errors``, bypassing ``write``. With those wrong, the gate's
    em-dash prompt raised at an ASCII terminal and the gate recorded the run as one
    whose answer it could not read — ``asked: null`` for a human who was there.
    Driven on a real pty because no capture stream takes that path.
    """
    master, slave = os.openpty()
    proc = subprocess.Popen(_driven(tmp_path, sourced_repo, "BAD"), stdin=slave,
                            stdout=slave, stderr=subprocess.PIPE,
                            env=_env(PYTHONIOENCODING="ascii"))
    os.close(slave)
    os.write(master, b"c\nc\nc\n")
    out = b""
    while True:
        try:
            chunk = os.read(master, 65536)
        except OSError:
            break
        if not chunk:
            break
        out += chunk
    err = proc.stderr.read().decode("ascii", "replace")
    proc.wait(timeout=60)
    os.close(master)

    text = out.decode("ascii", "replace")
    assert proc.returncode == 0, err
    gate = json.loads(text.split("--- JSON ---")[-1])["gate"]
    assert gate[0]["asked"] is True, "a human at the terminal answered"
    assert "no readable answer" not in text


def test_a_closed_stdout_descriptor_is_a_lost_stream(sourced_repo, capsys, monkeypatch):
    """REGRESSION for #141's review: ``>&-`` leaves ``sys.stdout`` as ``None``.

    The first draft crashed on it before the run began. On ``main`` it returned
    ``0`` with nothing delivered — the "reads as success" shape this issue refuses.
    """
    monkeypatch.setattr(sys, "stdout", None)

    assert _dry_run(sourced_repo) == 1
    notices = _notices(capsys.readouterr().err)
    assert len(notices) == 1 and "not delivered" in notices[0]


def test_a_closed_stderr_descriptor_keeps_the_notice_out_of_the_artefact(
        sourced_repo, monkeypatch):
    """REGRESSION for #141, found while fixing the review: ``2>&-`` makes it ``None``.

    ``print(file=None)`` writes to stdout — after the ``--- JSON ---`` block, where
    both parsers read to EOF. With nowhere to say it, it is not said anywhere else.
    """
    console = _AsciiConsole()
    monkeypatch.setattr(sys, "stdout", console)
    monkeypatch.setattr(sys, "stderr", None)

    assert _dry_run(sourced_repo) == 0
    json.loads(console.getvalue().split("--- JSON ---")[-1])


def test_help_into_a_closed_pipe_exits_cleanly(tmp_path):
    """REGRESSION for #141's review: an ``argparse`` exit skipped the stream's settling.

    Not a regression of the first draft — ``main`` gave the same ``120`` — but the
    stream claims every print made under ``main``, and ``--help`` is one.
    """
    read_end, write_end = os.pipe()
    os.close(read_end)
    try:
        proc = subprocess.run([sys.executable, "-m", "kuang", "--help"], env=_env(),
                              stdout=write_end, stderr=subprocess.PIPE, text=True)
    finally:
        os.close(write_end)

    assert proc.returncode == 0, proc.stderr
    assert "Exception ignored" not in proc.stderr


@pytest.mark.parametrize("severity, code", [("UGLY", 3), ("BAD", 1)])
def test_both_streams_closed_in_a_real_process(sourced_repo, tmp_path, severity, code):
    """REGRESSION for #141's review, the other BLOCKER: ``2>&1 | head``.

    The in-process test above holds against stand-ins with no exit flush. In a real
    process the failed notice left bytes in stderr's buffer, the interpreter's exit
    flush failed on them, and an UGLY's ``3`` became ``120``.
    """
    read_end, write_end = os.pipe()
    os.close(read_end)
    try:
        proc = subprocess.run(_driven(tmp_path, sourced_repo, severity), env=_env(),
                              stdin=subprocess.DEVNULL, stdout=write_end,
                              stderr=write_end)
    finally:
        os.close(write_end)

    assert proc.returncode == code
