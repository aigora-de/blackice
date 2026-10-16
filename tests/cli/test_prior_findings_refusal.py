# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""A ``--prior-findings`` seed that cannot be used is refused, not a traceback (#59).

Before #59 every unusable seed — missing, a directory, undecodable, unreadable, empty,
or carrying no findings — took the run down with an interpreter traceback at rc 1,
after the ``[panel]`` lines had already printed. And ``--prior-findings ""``, which a
shell produces from an unset variable, was read by truthiness as "no flag": the run
went ahead cold and its output was byte-identical to one never asked to be seeded
(#106's defect, one flag along; ruled a member of the class and folded in here).

What is pinned here, as pairs, because each half passes trivially without the other:

* every unusable seed is refused at rc 2, with one ``error: --prior-findings …`` line
  on stderr, nothing on stdout and nothing spawned — PAIRED WITH a valid seed still
  seeding, its echo line where it always was;
* ``""`` is refused — PAIRED WITH no flag at all still running cold and silently;
* a FIFO or ``/dev/fd`` seed with a writer still seeds, so no file-type check has been
  added that would break process substitution;
* a ValueError the loader did NOT declare as an operator fault stays loud, so the
  catch is closed by default: a bug is never reported as "your seed file is bad".

Nothing is spawned; no network. ``_run_claude`` is stubbed at the session, which is
the layer the CLI's other reporting tests stub, and every refusal runs WITHOUT
``--dry-run`` so that "before any spend" is measured, not assumed.

**Which are regressions, measured on main with this file in place** and a shim for
the new exception. Recorded in the PR body; any test green on main says so in its
own docstring.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from kuang.backends.claude_code import PriorFindingsError
from kuang.backends.claude_code import memory as memory_module
from kuang.backends.claude_code import session as session_module
from kuang.backends.claude_code.spawn import CallResult
from kuang.cli import main

_ROOT = Path(__file__).resolve().parents[2]

_SEED = {"findings": [{"persona": "correctness", "severity": "BLOCKER",
                       "title": "an earlier run's finding", "file": "a.py", "line": 1,
                       "open": True}]}

_CLEAN = CallResult('I reviewed it.\n\n```json\n{"verdict": "YES", "findings": []}\n```',
                    18, num_turns=7)


def _stub(monkeypatch) -> dict:
    """Stub the reviewer and count the persona calls, as #31's tests do."""
    state = {"calls": 0, "seeded": None}
    real_init = session_module.PanelSession.__init__

    def _fake_call(self, prompt, mandate, tools, model):  # noqa: ANN001, ARG001
        state["calls"] += 1
        return _CLEAN

    def _init(self, *args, **kwargs):  # noqa: ANN001
        state["seeded"] = kwargs.get("seeded_summary")
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(session_module.PanelSession, "_run_claude", _fake_call)
    monkeypatch.setattr(session_module.PanelSession, "__init__", _init)
    return state


def _argv(repo, *extra) -> list[str]:
    return ["--repo", str(repo), "--base", "HEAD~1", "--max-epochs", "1",
            "--no-parallel", *extra]


@pytest.fixture
def seeds(tmp_path_factory, monkeypatch):
    """A directory of seeds OUTSIDE the repository, and the working directory.

    Every relative path below resolves here and nowhere else: a mutant that accepts a
    path must not read the developer's checkout (#31's lesson).
    """
    d = tmp_path_factory.mktemp("seeds")
    monkeypatch.chdir(d)
    (d / "valid.json").write_text(json.dumps(_SEED))
    (d / "undecodable.json").write_bytes(b"\xff\xfe\x00bad")
    (d / "empty.json").write_text("")
    (d / "nokey.json").write_text(json.dumps({"other": []}))
    (d / "unreadable.json").write_text(json.dumps({"findings": [1, "x", None]}))
    (d / "adir").mkdir()
    return d


def _refused(capsys, state, rc) -> str:
    """The refusal's one stderr line, after checking the refusal's shape."""
    captured = capsys.readouterr()
    assert rc == 2
    assert state["calls"] == 0, "an unusable seed must be refused before any spend"
    assert captured.out == "", "a refusal comes before the first [panel] line"
    lines = captured.err.splitlines()
    assert len(lines) == 1, captured.err
    assert lines[0].startswith("error: --prior-findings")
    return lines[0]


# --- every unusable seed is refused ---------------------------------------------------

@pytest.mark.parametrize(("seed", "expected"), [
    ("missing.json", "--prior-findings missing.json: the file does not exist"),
    ("adir", "--prior-findings adir: IsADirectoryError [errno"),
    ("undecodable.json", "--prior-findings undecodable.json: the file is not"),
    ("empty.json", "--prior-findings empty.json: no findings to seed. Expected"),
    ("nokey.json", "--prior-findings nokey.json: no findings to seed. Expected"),
    ("unreadable.json",
     "--prior-findings unreadable.json: findings present but none were readable"),
])
def test_an_unusable_seed_is_refused_before_anything_is_spent(
        changed_repo, seeds, capsys, monkeypatch, seed, expected):
    """REGRESSION for #59: each of these was a traceback at rc 1 after stdout began."""
    state = _stub(monkeypatch)

    rc = main(_argv(changed_repo, "--prior-findings", seed))

    assert expected in _refused(capsys, state, rc)


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0,
                    reason="needs a POSIX permission bit that binds this user")
def test_an_unreadable_seed_is_refused_before_anything_is_spent(
        changed_repo, seeds, capsys, monkeypatch):
    """REGRESSION for #59: a PermissionError was a traceback at rc 1."""
    state = _stub(monkeypatch)
    (seeds / "noperm.json").write_text(json.dumps(_SEED))
    (seeds / "noperm.json").chmod(0)
    try:
        rc = main(_argv(changed_repo, "--prior-findings", "noperm.json"))
    finally:
        (seeds / "noperm.json").chmod(0o600)

    assert ("--prior-findings noperm.json: PermissionError [errno"
            in _refused(capsys, state, rc))


@pytest.mark.parametrize("raw", ["", "   "])
def test_an_empty_argument_is_refused_not_read_as_no_flag(
        changed_repo, seeds, capsys, monkeypatch, raw):
    """REGRESSION for #59 (folded in by ruling): ``""`` ran cold and silently.

    Measured on main: the output was byte-identical to a run with no flag at all, so a
    seeded re-run asked for through an unset shell variable read as a healthy cold
    one. Whitespace is refused with it, as ``--panel`` refuses it.
    """
    state = _stub(monkeypatch)

    rc = main(_argv(changed_repo, "--prior-findings", raw))

    line = _refused(capsys, state, rc)
    assert line.startswith("error: --prior-findings: no path was given — an empty "
                           "argument is usually an unset shell variable")


@pytest.mark.parametrize("raw", ["missing.json", ""])
def test_the_real_process_prints_no_traceback(changed_repo, seeds, raw):
    """REGRESSION for #59, through the console script's own process.

    In-process, a traceback is only an exception escaping ``main``; this is what the
    operator actually sees. ``PYTHONUNBUFFERED`` is stripped so stdout is measured as
    an operator's shell would buffer it, and ``PYTHONPATH`` names this working tree, as
    ``test_console_faults`` does: the process runs from the seeds directory, where an
    installed copy of kuang — or none — would be imported instead.
    """
    env = {k: v for k, v in os.environ.items() if k != "PYTHONUNBUFFERED"}
    env["PYTHONPATH"] = str(_ROOT)
    proc = subprocess.run(
        [sys.executable, "-m", "kuang", *_argv(changed_repo, "--dry-run",
                                               "--prior-findings", raw)],
        capture_output=True, text=True, env=env, cwd=seeds, timeout=60)

    assert proc.returncode == 2
    assert "Traceback" not in proc.stderr
    assert proc.stdout == ""
    assert proc.stderr.startswith("error: --prior-findings")


# --- the pairs: a healthy seed, and no seed, are untouched ----------------------------

def test_a_valid_seed_still_seeds_and_its_echo_has_not_moved(
        changed_repo, seeds, capsys, monkeypatch):
    """PAIR, green on main: guards against a refusal that fires on a healthy seed.

    The ``[panel]`` lines of a seeded run are the unseeded run's with exactly the echo
    inserted after the ``tools=`` line, so moving the LOAD earlier moved nothing there.
    Only those lines: a dry run prints the prompt's length, which the seed changes.
    """
    state = _stub(monkeypatch)

    def _panel_lines() -> list[str]:
        return [line for line in capsys.readouterr().out.splitlines()
                if line.startswith("[panel]")]

    assert main(_argv(changed_repo, "--dry-run")) == 0
    cold = _panel_lines()
    assert main(_argv(changed_repo, "--dry-run", "--prior-findings", "valid.json")) == 0
    seeded = _panel_lines()

    at = next(i for i, line in enumerate(cold) if line.startswith("[panel] tools="))
    echo = ("[panel] prior findings: 1 carried in from valid.json "
            "(locations advisory)")
    assert seeded == cold[:at + 1] + [echo] + cold[at + 1:]
    assert "an earlier run's finding" in state["seeded"]


def test_no_flag_still_runs_cold_and_silently(changed_repo, seeds, capsys, monkeypatch):
    """PAIR, green on main: the mirror image of the ``""`` refusal.

    A rule that fired on a run never asked to be seeded would be the defect's mirror.
    """
    state = _stub(monkeypatch)

    rc = main(_argv(changed_repo))

    captured = capsys.readouterr()
    assert rc == 0
    assert state["calls"] > 0
    assert state["seeded"] == ""
    assert "prior findings" not in captured.out
    assert "error" not in captured.err


def test_a_fifo_with_a_writer_still_seeds(changed_repo, seeds, capsys, monkeypatch):
    """PAIR, green on main: guards against refusing every non-regular file.

    A FIFO from ``os.mkfifo``, fed by a thread. A FIFO with NO writer still blocks in
    ``read_text()`` — before any spend, ended by Ctrl-C — and is a stated under-report.

    If the CLI never opens the FIFO — as a non-regular-file refusal would not — the
    writer blocks in ``open()`` forever. The finally opens a reader to release it, and
    the thread is a daemon besides, so a failure here fails and cannot hang the suite.
    """
    state = _stub(monkeypatch)
    fifo = seeds / "seed.fifo"
    os.mkfifo(fifo)
    writer = threading.Thread(target=fifo.write_text, args=(json.dumps(_SEED),),
                              daemon=True)
    writer.start()
    try:
        rc = main(_argv(changed_repo, "--dry-run", "--prior-findings", str(fifo)))
    finally:
        if writer.is_alive():
            reader = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
            writer.join(timeout=10)
            os.close(reader)
        writer.join(timeout=10)

    assert rc == 0
    assert "prior findings: 1 carried in" in capsys.readouterr().out
    assert "an earlier run's finding" in state["seeded"]


@pytest.mark.skipif(not os.path.isdir("/dev/fd"), reason="needs /dev/fd")
def test_process_substitution_still_seeds(changed_repo, seeds, capsys, monkeypatch):
    """PAIR, green on main: ``--prior-findings <(jq … run.json)`` is a /dev/fd path.

    An ``os.pipe`` read end named through ``/dev/fd/N``, which is what a shell's
    process substitution hands the CLI. The payload fits the pipe buffer, so the write
    end is filled and closed before the read.
    """
    state = _stub(monkeypatch)
    r, w = os.pipe()
    try:
        os.write(w, json.dumps(_SEED).encode())
        os.close(w)
        rc = main(_argv(changed_repo, "--dry-run", "--prior-findings", f"/dev/fd/{r}"))
    finally:
        os.close(r)

    assert rc == 0
    assert "prior findings: 1 carried in" in capsys.readouterr().out
    assert "an earlier run's finding" in state["seeded"]


# --- closed by default: an undeclared fault stays loud --------------------------------

def test_an_undeclared_value_error_in_the_loader_is_not_reported_as_a_bad_seed(
        changed_repo, seeds, monkeypatch):
    """PAIR, green on main: guards the REJECTED design of catching ValueError in the CLI.

    ``UnicodeDecodeError`` is a ValueError, so catching ``(OSError, ValueError)`` around
    the call would cover every refusal above — and would also report a bug in the
    renderer as an operator's bad file, at rc 2, with the traceback that would have
    found the bug gone. Only the faults the loader names are refused.
    """
    _stub(monkeypatch)

    def _broken(**_kwargs):
        raise ValueError("a bug in the renderer, not in the seed")

    monkeypatch.setattr(memory_module, "ledger_line", _broken)

    with pytest.raises(ValueError, match="a bug in the renderer") as caught:
        main(_argv(changed_repo, "--dry-run", "--prior-findings", "valid.json"))
    assert not isinstance(caught.value, PriorFindingsError)
