# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""An unreadable persona file, end to end: reported or refused, never a traceback (#104).

The sourcing rules are pinned in ``tests/backends/claude_code/test_persona_file_reads``.
This module pins what an operator sees, and the one axis an in-process test cannot
reach: the LOCALE, whose encoding ``read_text()`` used by default and which is fixed at
interpreter start. Those tests run a child interpreter with an environment built from
scratch, because the dev shell's own locale and ``PYTHONUTF8`` would otherwise leak in.

* ``--panel`` naming an undecodable file is refused at rc 2 with one ``error:`` line,
  nothing on stdout and nothing spawned — run WITHOUT ``--dry-run``, so "before any
  spend" is measured, not assumed;
* an undecodable ``CLAUDE.md`` is reported as a discard on the console and in the
  artefact, and the run goes on — PAIRED WITH a valid one reporting nothing;
* the real process prints no traceback on any of the three paths that had one;
* under an ASCII locale (``LC_ALL=C`` with coercion and UTF-8 mode off, present on every
  POSIX system, so it never skips on CI) a valid UTF-8 panel sources exactly as under a
  UTF-8 locale — C5 and C9, which tracebacked;
* under a Latin-1 locale, where one exists, the artefact's persona names are the
  console's — C3, the silent member, where the console showed ``🎯 The Analyst`` and
  the artefact recorded ``ð¯ The Analyst``.

**Which are regressions, measured on main with this file in place.** Recorded in the PR
body; any test green on main says so in its own docstring.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from kuang.backends.claude_code import session as session_module
from kuang.backends.claude_code.spawn import CallResult
from kuang.cli import main

_ROOT = Path(__file__).resolve().parents[2]

_UNDECODABLE = b"ACM\xff\xfe secret-internal-name\n"
_UNDECODABLE_DETAIL = "UnicodeDecodeError at byte 3: not utf-8 text"

# Hyphen separators are C3's shape: under a Latin-1 locale on main they still split,
# so the panel sourced with mojibake names and nothing warned. The em-dash grounding
# and the accent are what an ASCII locale could not decode at all (C5).
_UTF8_EXPERTS = """\
# A repo

# Resident Experts

## 🎯 The Analyst - Correctness

Does it compute the right thing? Café — arithmetic.

## 🗡️ The Adversary - Threats

Break it.
"""
_NAMES = ["The Analyst", "The Adversary", "completeness-critic", "survivability"]

_CLEAN = CallResult('I reviewed it.\n\n```json\n{"verdict": "YES", "findings": []}\n```',
                    18, num_turns=7)

# An ASCII locale with nothing to rescue it: no PEP 538 coercion, no UTF-8 mode.
_ASCII_LOCALE = {"LC_ALL": "C", "PYTHONCOERCECLOCALE": "0", "PYTHONUTF8": "0"}
_UTF8_LOCALE = {"LC_ALL": "C.UTF-8" if sys.platform != "darwin" else "en_US.UTF-8"}


def _stub(monkeypatch) -> dict:
    """Stub the reviewer and count the persona calls, as #59's tests do."""
    state = {"calls": 0}

    def _fake_call(self, prompt, mandate, tools, model):  # noqa: ANN001, ARG001
        state["calls"] += 1
        return _CLEAN

    monkeypatch.setattr(session_module.PanelSession, "_run_claude", _fake_call)
    return state


def _argv(repo, *extra) -> list[str]:
    return ["--repo", str(repo), "--base", "HEAD~1", "--max-epochs", "1",
            "--no-parallel", *extra]


def _artefact(out: str) -> dict:
    return json.loads(out.split("--- JSON ---")[-1])


# --- in-process: what the operator sees -----------------------------------------------

def test_an_undecodable_explicit_panel_is_refused_before_anything_is_spent(
        changed_repo, tmp_path, capsys, monkeypatch):
    """REGRESSION for #104 (A9): a traceback at rc 1, naming ``personas.py``."""
    state = _stub(monkeypatch)
    panel = tmp_path / "x.md"
    panel.write_bytes(_UNDECODABLE)

    rc = main(_argv(changed_repo, "--panel", str(panel)))

    captured = capsys.readouterr()
    assert rc == 2
    assert state["calls"] == 0, "a refusal comes before any spend"
    assert captured.out == "", "a refusal comes before the first [panel] line"
    assert captured.err.splitlines() == [
        f"error: --panel {panel}: it could not be read or parsed "
        f"({_UNDECODABLE_DETAIL})"]


def test_an_undecodable_claude_md_is_reported_and_the_run_goes_on(
        changed_repo, capsys, monkeypatch):
    """REGRESSION for #104 (A1): a traceback at rc 1, so nothing was reported at all."""
    state = _stub(monkeypatch)
    (changed_repo / "CLAUDE.md").write_bytes(_UNDECODABLE)

    rc = main(_argv(changed_repo))

    out = capsys.readouterr().out
    assert rc == 0
    assert state["calls"] > 0
    assert ("[panel] WARNING: 1 declared panel(s) DISCARDED — the roster above is "
            "not the one they declared") in out.splitlines()
    assert (f"[panel]   CLAUDE.md: it could not be read or parsed — "
            f"{_UNDECODABLE_DETAIL}") in out.splitlines()
    assert _artefact(out)["panel"]["discarded"] == [
        {"origin": "CLAUDE.md", "reason": "malformed", "detail": _UNDECODABLE_DETAIL}]


def test_a_valid_utf8_claude_md_reports_nothing(changed_repo, capsys, monkeypatch):
    """PAIR, green on main: the warning above must not fire on a healthy file."""
    _stub(monkeypatch)
    (changed_repo / "CLAUDE.md").write_text(_UTF8_EXPERTS, encoding="utf-8")

    rc = main(_argv(changed_repo))

    out = capsys.readouterr().out
    assert rc == 0
    assert "WARNING: 1 declared panel(s)" not in out
    assert f"[panel] 4 personas from CLAUDE.md: {', '.join(_NAMES)}" in out.splitlines()
    assert _artefact(out)["panel"]["discarded"] == []


# --- the real process -----------------------------------------------------------------

def _child_env(home: Path, **extra: str) -> dict[str, str]:
    """An environment built from scratch, so nothing of the dev shell leaks in.

    ``PYTHONPATH`` names this working tree, because the child runs from ``home``, where
    an installed copy of kuang — or none — would otherwise be imported. ``CLAUDE_BIN``
    is the conftest's fake, so no real ``claude`` is reachable.
    """
    return {"PATH": "/usr/bin:/bin", "HOME": str(home), "PYTHONPATH": str(_ROOT),
            "CLAUDE_BIN": os.environ["CLAUDE_BIN"], **extra}


def _run(argv: list[str], env: dict[str, str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "kuang", *argv], capture_output=True,
                          env=env, cwd=cwd, timeout=60)


@pytest.mark.parametrize("where", ["CLAUDE.md", "panel.md", "--panel"])
def test_the_real_process_prints_no_traceback(changed_repo, tmp_path, where):
    """REGRESSION for #104 (A1, A5, A9), through the interpreter an operator runs."""
    extra: list[str] = []
    if where == "--panel":
        panel = tmp_path / "x.md"
        panel.write_bytes(_UNDECODABLE)
        extra = ["--panel", str(panel)]
    else:
        (changed_repo / where).write_bytes(_UNDECODABLE)

    proc = _run(_argv(changed_repo, "--dry-run", *extra),
                _child_env(tmp_path, **_UTF8_LOCALE), tmp_path)

    err = proc.stderr.decode("utf-8", "replace")
    assert "Traceback" not in err
    if where == "--panel":
        assert proc.returncode == 2
        assert proc.stdout == b""
        assert err.startswith(f"error: --panel {tmp_path / 'x.md'}: it could not be")
    else:
        assert proc.returncode == 0
        assert f"{where}: it could not be read or parsed — {_UNDECODABLE_DETAIL}" in (
            proc.stdout.decode("utf-8"))


# --- the locale axis ------------------------------------------------------------------

_SOURCE_SCRIPT = """\
import json, sys
from pathlib import Path
from kuang.backends.claude_code.personas import load_personas
roster, source, _ = load_personas(Path(sys.argv[1]))
print(json.dumps({"label": source.label,
                  "discarded": [[d.origin, d.reason, d.detail] for d in source.discarded],
                  "personas": [[p.name, p.grounding] for p in roster]}))
"""


def _sourced_under(repo: Path, home: Path, locale_env: dict[str, str]) -> dict:
    """Source ``repo`` in a child interpreter under ``locale_env``; ASCII-safe JSON out,
    so the result does not depend on the child's console encoding."""
    proc = subprocess.run([sys.executable, "-c", _SOURCE_SCRIPT, str(repo)],
                          capture_output=True, env=_child_env(home, **locale_env),
                          cwd=home, timeout=60)
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    return json.loads(proc.stdout)


@pytest.mark.parametrize("tier", ["CLAUDE.md", "panel.md"])
def test_a_valid_utf8_panel_sources_the_same_under_an_ascii_locale(
        tmp_path, tier):
    """REGRESSION for #104 (C5, C9): a valid UTF-8 file tracebacked under ASCII.

    The pair is in the assertion: the ASCII-locale result must EQUAL the UTF-8-locale
    one, names and groundings byte for byte, and the UTF-8 one must be the healthy
    roster. ``LC_ALL=C`` exists on every POSIX system, so this never skips on CI.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / tier).write_text(_UTF8_EXPERTS, encoding="utf-8")

    under_utf8 = _sourced_under(repo, tmp_path, _UTF8_LOCALE)
    under_ascii = _sourced_under(repo, tmp_path, _ASCII_LOCALE)

    assert under_utf8["discarded"] == []
    assert [name for name, _ in under_utf8["personas"]][:2] == _NAMES[:2]
    assert "Café — arithmetic." in under_utf8["personas"][0][1]
    assert under_ascii == under_utf8


def test_the_whole_cli_runs_a_valid_utf8_panel_under_an_ascii_locale(
        changed_repo, tmp_path):
    """REGRESSION for #104 (C5), through the CLI: rc 1 and a traceback on main.

    The artefact's roster and panel record equal the UTF-8 run's. Only the panel is
    compared: the console encodes what it prints for its own locale (#141).
    """
    (changed_repo / "CLAUDE.md").write_text(_UTF8_EXPERTS, encoding="utf-8")
    argv = _argv(changed_repo, "--dry-run")

    runs = {name: _run(argv, _child_env(tmp_path, **env), tmp_path)
            for name, env in (("utf8", _UTF8_LOCALE), ("ascii", _ASCII_LOCALE))}

    for proc in runs.values():
        assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
        assert b"Traceback" not in proc.stderr
    panels = {name: _artefact(proc.stdout.decode("utf-8", "replace"))["panel"]
              for name, proc in runs.items()}
    assert panels["utf8"]["personas"] == _NAMES
    assert panels["ascii"] == panels["utf8"]


def _latin1_locale() -> str | None:
    """A Latin-1 locale this machine actually has, or ``None``.

    Probed by asking a child interpreter what encoding it gets, not by matching
    ``locale -a`` spellings, which differ between platforms. CI's Linux runners may
    have none, so the test that needs one skips with a reason; the ASCII tests above
    never skip.
    """
    for name in ("en_GB.ISO8859-1", "en_US.ISO8859-1", "en_GB.iso88591",
                 "en_US.iso88591", "de_DE.ISO8859-1"):
        proc = subprocess.run(
            [sys.executable, "-c",
             "import locale; print(locale.getpreferredencoding(False))"],
            capture_output=True, text=True, timeout=30,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": name, "PYTHONUTF8": "0",
                 "PYTHONCOERCECLOCALE": "0"})
        if proc.stdout.strip().replace("-", "").upper() == "ISO88591":
            return name
    return None


_LATIN1 = _latin1_locale()


@pytest.mark.skipif(_LATIN1 is None, reason="no ISO8859-1 locale on this machine")
def test_the_artefact_names_the_panel_the_console_names_under_latin1(
        changed_repo, tmp_path):
    """REGRESSION for #104 (C3), the silent member of #40's first-criterion class.

    On main, under a Latin-1 locale, this file sourced with no warning, the console
    re-encoded the mojibake and SHOWED ``🎯 The Analyst``, and the artefact recorded
    ``ð¯ The Analyst``. Now both say ``The Analyst``, and both equal the roster the
    UTF-8 locale sources.
    """
    (changed_repo / "CLAUDE.md").write_text(_UTF8_EXPERTS, encoding="utf-8")

    proc = _run(_argv(changed_repo, "--dry-run"),
                _child_env(tmp_path, LC_ALL=_LATIN1, PYTHONUTF8="0",
                           PYTHONCOERCECLOCALE="0"), tmp_path)

    assert proc.returncode == 0, proc.stderr.decode("latin-1")
    out = proc.stdout.decode("latin-1")
    assert f"[panel] 4 personas from CLAUDE.md: {', '.join(_NAMES)}" in out.splitlines()
    assert _artefact(out)["panel"]["personas"] == _NAMES
    assert "WARNING: 1 declared panel(s)" not in out
