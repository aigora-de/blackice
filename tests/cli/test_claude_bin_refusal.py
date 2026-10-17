# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""A live run with no usable ``claude`` is refused before it starts (#51).

Before #51, a live run with no usable binary did not die, because #25 contained the
spawn's ``FileNotFoundError``. It ran a whole epoch that could review nothing, halted
``no_review``, exited 0, and left the cause in one ``NOTE``. And an explicit
``$CLAUDE_BIN`` that could not be used was silently replaced by whatever ``PATH``
found, so the run spent on a binary nobody asked for (ruled a first-criterion member
and folded in).

What is pinned here, as pairs, because each half passes trivially without the other:

* a live run with nothing usable is refused at rc 2, with one ``error:`` line on
  stderr, nothing on stdout and nothing spawned — PAIRED WITH a usable binary still
  running, and printing no warning;
* a live run is refused — PAIRED WITH a dry run in the same environment still
  running, exiting as a healthy one does, and differing from a healthy dry run only
  by one warning line and the binary its ``argv=`` names;
* an explicit unusable ``$CLAUDE_BIN`` is refused with a working ``claude`` on
  ``PATH`` — PAIRED WITH an explicit usable one being the binary actually spawned;
* the binary is resolved once, and the session spawns what the CLI checked;
* a fault the resolver does not name stays loud.

Nothing is spawned; no network. ``_run_claude`` is stubbed at the session, as the
CLI's other reporting tests stub it, and every refusal runs WITHOUT ``--dry-run`` so
that "before any spend" is measured, not assumed. ``HOME``, ``PATH`` and
``CLAUDE_BIN`` are set by each test: ``PATH`` holds only a link to ``git``, so a
mutant that falls through cannot find the developer's real ``claude``.

**Which are regressions, measured on main with this file in place** and a shim for
the new names. Recorded in the PR body; any test green on main says so in its own
docstring.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from kuang.backends.claude_code import session as session_module
from kuang.backends.claude_code import spawn as spawn_module
from kuang.backends.claude_code.spawn import CallResult
from kuang.cli import main

_ROOT = Path(__file__).resolve().parents[2]
_GIT = shutil.which("git")

_CLEAN = CallResult('I reviewed it.\n\n```json\n{"verdict": "YES", "findings": []}\n```',
                    18, num_turns=7)

_WOULD_REFUSE = "[panel] WARNING: a live run would be refused: "


def _exe(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 97\n")
    path.chmod(0o755)
    return path


@pytest.fixture
def env(tmp_path, monkeypatch):
    """No ``claude`` anywhere: ``PATH`` is a directory holding only ``git``."""
    home, bindir = tmp_path / "home", tmp_path / "bin"
    home.mkdir()
    bindir.mkdir()
    (bindir / "git").symlink_to(_GIT)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", str(bindir))
    monkeypatch.delenv("CLAUDE_BIN", raising=False)
    return {"home": home, "bin": bindir, "root": tmp_path}


def _stub(monkeypatch) -> dict:
    """Stub the reviewer; count its calls and record the binary each would spawn."""
    state = {"calls": 0, "bins": []}

    def _fake_call(self, prompt, mandate, tools, model):  # noqa: ANN001, ARG001
        state["calls"] += 1
        state["bins"].append(self._argv(prompt, mandate, tools, model)[0])
        return _CLEAN

    monkeypatch.setattr(session_module.PanelSession, "_run_claude", _fake_call)
    return state


def _argv(repo, *extra) -> list[str]:
    return ["--repo", str(repo), "--base", "HEAD~1", "--max-epochs", "1",
            "--no-parallel", *extra]


def _refused(capsys, state, rc) -> str:
    """The refusal's one stderr line, after checking the refusal's shape."""
    captured = capsys.readouterr()
    assert rc == 2
    assert state["calls"] == 0, "an unusable binary must be refused before any spend"
    assert captured.out == "", "a refusal comes before the first [panel] line"
    lines = captured.err.splitlines()
    assert len(lines) == 1, captured.err
    assert lines[0].startswith("error: ")
    return lines[0]


# --- nothing usable: refused live, previewed dry ------------------------------------

@pytest.mark.parametrize("extra", [(), ("--semantic-dedup",)])
def test_a_live_run_with_no_claude_is_refused_before_anything_is_spent(
        changed_repo, env, capsys, monkeypatch, extra):
    """REGRESSION for #51: this ran an epoch of seven failed spawns, then exited 0.

    With ``--semantic-dedup`` too, whose cluster call spawns the same binary.
    """
    state = _stub(monkeypatch)

    rc = main(_argv(changed_repo, *extra))

    line = _refused(capsys, state, rc)
    assert line.startswith("error: claude: no usable binary was found")
    for place in ("$CLAUDE_BIN is unset", "no `claude` on PATH",
                  "~/.local/bin/claude does not exist"):
        assert place in line
    assert str(env["home"]) not in line


def test_a_usable_binary_still_runs_and_prints_no_warning(
        changed_repo, env, capsys, monkeypatch):
    """PAIR, green on main: guards against a refusal that fires on a healthy run."""
    state = _stub(monkeypatch)
    pinned = _exe(env["root"] / "pinned" / "claude")
    monkeypatch.setenv("CLAUDE_BIN", str(pinned))

    rc = main(_argv(changed_repo))

    captured = capsys.readouterr()
    assert rc == 0, captured.err
    assert state["calls"] > 0
    assert captured.out.startswith("[panel] ")
    assert "WARNING" not in captured.out
    assert captured.err == ""


def test_a_dry_run_with_no_claude_runs_and_says_a_live_run_would_be_refused(
        changed_repo, env, capsys, monkeypatch):
    """REGRESSION for #51: the dry run said nothing, so its preview could not warn.

    It is otherwise the healthy dry run, line for line: the same exit, the one
    warning added, and the bare ``claude`` in each ``argv=`` where the healthy run
    names its binary. So "``--dry-run`` is unaffected" is measured, not asserted.
    """
    state = _stub(monkeypatch)
    rc_faulty = main(_argv(changed_repo, "--dry-run"))
    faulty = capsys.readouterr()

    pinned = _exe(env["root"] / "pinned" / "claude")
    monkeypatch.setenv("CLAUDE_BIN", str(pinned))
    rc_healthy = main(_argv(changed_repo, "--dry-run"))
    healthy = capsys.readouterr()

    assert state["calls"] == 0
    assert rc_faulty == rc_healthy == 0
    assert faulty.err == healthy.err == ""
    warnings = [line for line in faulty.out.splitlines()
                if line.startswith(_WOULD_REFUSE)]
    assert len(warnings) == 1
    assert warnings[0].removeprefix(_WOULD_REFUSE).startswith(
        "claude: no usable binary was found")
    assert _WOULD_REFUSE not in healthy.out
    assert "  argv= claude -p " in faulty.out
    rest = [line for line in faulty.out.splitlines() if line not in warnings]
    assert rest == [line.replace(f"argv= {pinned} ", "argv= claude ")
                    for line in healthy.out.splitlines()]


def test_the_warning_comes_before_any_persona_is_previewed(
        changed_repo, env, capsys, monkeypatch):
    """Said with the other ``[panel]`` warnings, where an operator reads them first."""
    _stub(monkeypatch)
    main(_argv(changed_repo, "--dry-run"))
    out = capsys.readouterr().out.splitlines()
    warning = next(i for i, line in enumerate(out) if line.startswith(_WOULD_REFUSE))
    assert all(line.startswith("[panel]") for line in out[:warning])
    assert not any(line.startswith("[dry-run]") for line in out[:warning])


# --- an explicit $CLAUDE_BIN ----------------------------------------------------------

@pytest.fixture
def claude_on_path(env):
    """A working ``claude`` on PATH, so a silent fallthrough would succeed."""
    return _exe(env["bin"] / "claude")


@pytest.mark.parametrize(("value", "expected"), [
    ("/nonexistent/claude", "error: $CLAUDE_BIN /nonexistent/claude: does not exist"),
    ("", "error: $CLAUDE_BIN: set but empty"),
])
def test_an_explicit_unusable_binary_is_refused_not_replaced(
        changed_repo, claude_on_path, capsys, monkeypatch, value, expected):
    """REGRESSION for #51 (folded in by ruling): each spent on the PATH binary.

    Measured on main with a real ``HOME`` and ``PATH``: both resolved to the
    installed ``~/.local/bin/claude``, and nothing said a different binary ran.
    """
    state = _stub(monkeypatch)
    monkeypatch.setenv("CLAUDE_BIN", value)

    rc = main(_argv(changed_repo))

    assert _refused(capsys, state, rc).startswith(expected)


@pytest.mark.parametrize("shape", ["dir", "noexec"])
def test_an_explicit_directory_or_plain_file_is_refused(
        changed_repo, env, capsys, monkeypatch, shape):
    """REGRESSION for #51: ``os.path.exists`` admitted both; each spawn then failed."""
    state = _stub(monkeypatch)
    target = env["root"] / shape
    if shape == "dir":
        target.mkdir()
    else:
        target.write_text("x\n")
        target.chmod(0o644)
    monkeypatch.setenv("CLAUDE_BIN", str(target))

    rc = main(_argv(changed_repo))

    said = ("is a directory, not an executable file" if shape == "dir"
            else "is not executable")
    assert _refused(capsys, state, rc).startswith(
        f"error: $CLAUDE_BIN {target}: {said} — ")


def test_an_explicit_usable_binary_is_the_one_spawned(
        changed_repo, claude_on_path, capsys, monkeypatch):
    """PAIR, green on main: the binary the operator named is the binary used."""
    state = _stub(monkeypatch)
    pinned = _exe(claude_on_path.parent.parent / "pinned" / "claude")
    monkeypatch.setenv("CLAUDE_BIN", str(pinned))

    assert main(_argv(changed_repo)) == 0
    assert state["bins"] and set(state["bins"]) == {str(pinned)}


def test_unset_the_path_binary_is_the_one_spawned(
        changed_repo, claude_on_path, env, capsys, monkeypatch):
    """PAIR, green on main: with nothing pinned, the search order still holds."""
    state = _stub(monkeypatch)
    _exe(env["home"] / ".local" / "bin" / "claude")

    assert main(_argv(changed_repo)) == 0
    assert set(state["bins"]) == {str(claude_on_path)}


# --- one resolution, and closed by default ---------------------------------------------

def test_the_binary_is_resolved_once_and_the_session_spawns_what_was_checked(
        changed_repo, env, capsys, monkeypatch):
    """The CLI's check and the session's spawn cannot disagree, because there is one.

    The CLI's resolver answers differently on each call, so a second call by the CLI
    shows up as a second answer. A session left to resolve for itself does not reach
    this stub — its ``default_factory`` bound the real resolver when the class was
    made — and the real one, with nothing usable in this environment, refuses.
    """
    state = _stub(monkeypatch)
    first = _exe(env["root"] / "first" / "claude")
    second = _exe(env["root"] / "second" / "claude")
    answers = iter([str(first), str(second)])
    calls = []

    def _resolve():
        calls.append(1)
        return next(answers)

    import kuang.cli as cli_module
    monkeypatch.setattr(cli_module, "resolve_claude_bin", _resolve)

    assert main(_argv(changed_repo)) == 0
    assert len(calls) == 1
    assert set(state["bins"]) == {str(first)}


@pytest.mark.parametrize("fault", [ValueError("a bug"), PermissionError(13, "a bug")])
def test_a_fault_the_resolver_does_not_name_is_not_reported_as_the_operators(
        changed_repo, env, capsys, monkeypatch, fault):
    """Closed by default: a bug escapes ``main`` loudly, never as ``error:`` at rc 2.

    ``PermissionError`` is an ``OSError``: red against the rejected design of catching
    every ``OSError`` around the resolution.
    """
    _stub(monkeypatch)

    def _broken(*_a, **_k):
        raise fault
    monkeypatch.setattr(spawn_module.shutil, "which", _broken)

    with pytest.raises(type(fault)):
        main(_argv(changed_repo))
    assert capsys.readouterr().out == ""


def test_the_real_process_prints_no_traceback(changed_repo, env):
    """REGRESSION for #51, through the console script's own process.

    What the operator actually sees. ``PYTHONUNBUFFERED`` is stripped, as elsewhere,
    and ``PYTHONPATH`` names this working tree. The child's environment is the
    fixture's: no ``claude`` anywhere it can look.
    """
    child = {k: v for k, v in os.environ.items() if k != "PYTHONUNBUFFERED"}
    child["PYTHONPATH"] = str(_ROOT)
    proc = subprocess.run([sys.executable, "-m", "kuang", *_argv(changed_repo)],
                          capture_output=True, text=True, env=child, timeout=60)

    assert proc.returncode == 2
    assert "Traceback" not in proc.stderr
    assert proc.stdout == ""
    assert proc.stderr.startswith("error: claude: no usable binary was found")
