# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""CHARACTERISATION tests for the subprocess boundary: the argv and the dry-run.

Not regression tests. They pin what ``PanelSession`` hands to ``claude`` today,
and what ``--dry-run`` prints instead, so that #19's consolidation of the two —
one ``build_argv`` feeding both the real call and the dry-run report — is a
visible change rather than a silent one.

Nothing here spawns anything: ``subprocess.run`` is replaced for the duration of
each test and the captured argv is asserted in full. The argv had no coverage at
all before this file, which is the point: a dry-run whose only job is pre-flight
confirmation was describing wiring that nothing checked.
"""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from blackice.backends.claude_code import PanelSession, Persona
from blackice.backends.claude_code import spawn as spawn_module
from blackice.engine import ReviewSpec


@pytest.fixture
def session(tmp_path):
    return PanelSession(
        repo_root=tmp_path, spec=ReviewSpec(why="w", what="x"),
        personas={"p": Persona("p", "be adversarial")},
        base="main", head="HEAD", claude_bin="/fake/claude")


@pytest.fixture
def captured(monkeypatch):
    """Replace ``subprocess.run`` and record every call made through it."""
    calls: list[SimpleNamespace] = []

    def fake_run(argv, **kwargs):
        calls.append(SimpleNamespace(argv=argv, kwargs=kwargs))
        return subprocess.CompletedProcess(
            argv, 0,
            stdout=json.dumps({"result": '```json\n{"verdict": "YES", "findings": []}\n```',
                               "usage": {"output_tokens": 11}}),
            stderr="")

    monkeypatch.setattr(spawn_module.subprocess, "run", fake_run)
    return calls


# --- the argv ---------------------------------------------------------------

def test_the_argv_is_exactly_this(session, captured):
    session._run_claude("PROMPT", "MANDATE", ["Read", "Grep"], None)

    assert captured[0].argv == [
        "/fake/claude", "-p", "PROMPT",
        "--append-system-prompt", "MANDATE",
        "--allowedTools", "Read", "Grep",
        "--disallowedTools", "Edit", "Write", "NotebookEdit", "Bash",
        "--permission-mode", "plan",
        "--output-format", "json",
        "--add-dir", str(session.repo_root),
    ]


def test_a_model_is_appended_only_when_given(session, captured):
    session._run_claude("P", "M", ["Read"], "some-model")
    session._run_claude("P", "M", ["Read"], None)

    assert captured[0].argv[-2:] == ["--model", "some-model"]
    assert "--model" not in captured[1].argv


def test_the_call_runs_in_the_repo_root(session, captured):
    session._run_claude("P", "M", ["Read"], None)

    assert captured[0].kwargs["cwd"] == str(session.repo_root)
    assert captured[0].kwargs["capture_output"] is True
    assert captured[0].kwargs["text"] is True


def test_tools_and_permission_overrides_reach_the_argv(session, captured):
    session.disallowed_tools = ["Edit"]
    session.permission_mode = "acceptEdits"

    session._run_claude("P", "M", ["Bash(pytest:*)"], None)

    argv = captured[0].argv
    assert argv[argv.index("--allowedTools") + 1] == "Bash(pytest:*)"
    assert argv[argv.index("--disallowedTools") + 1] == "Edit"
    assert argv[argv.index("--permission-mode") + 1] == "acceptEdits"


# --- what comes back --------------------------------------------------------

def test_the_result_and_token_count_are_returned(session, captured):
    text, toks, err = session._run_claude("P", "M", ["Read"], None)

    assert '"verdict": "YES"' in text
    assert toks == 11
    assert err is None


def test_a_non_zero_exit_becomes_an_error_string(session, monkeypatch):
    monkeypatch.setattr(spawn_module.subprocess, "run",
                        lambda argv, **kw: subprocess.CompletedProcess(
                            argv, 7, stdout="", stderr="boom"))

    text, toks, err = session._run_claude("P", "M", ["Read"], None)

    assert (text, toks) == ("", 0)
    assert err == "claude exited 7: boom"


def test_raw_non_json_stdout_is_tolerated(session, monkeypatch):
    monkeypatch.setattr(spawn_module.subprocess, "run",
                        lambda argv, **kw: subprocess.CompletedProcess(
                            argv, 0, stdout="just text", stderr=""))

    assert session._run_claude("P", "M", ["Read"], None) == ("just text", 0, None)


# --- the dry-run report -----------------------------------------------------

def test_dry_run_spawns_nothing_and_votes_yes(session, captured, capsys):
    session.dry_run = True

    report = session.spawn("p", "MANDATE", "SURFACE", 1)

    assert captured == [], "a dry run must not spawn a subprocess"
    assert (report.persona, report.verdict, report.findings) == ("p", "YES", [])


def test_the_dry_run_reports_the_argv_it_would_spawn(session, capsys):
    """The one deliberate behaviour change in #19, and its whole justification.

    The report used to be a prose summary written separately from the argv, so
    the two could disagree — and three of the argv's flags (``--output-format``,
    ``--add-dir``, the binary itself) never appeared in it at all. It is now
    rendered from the argv ``build_argv`` returns, which is the same object the
    real call spawns. The prompt and system prompt are elided by length; every
    flag that governs what the reviewer may do is shown.
    """
    session.dry_run = True

    session.spawn("p", "MANDATE", "SURFACE", 1)

    lines = capsys.readouterr().out.splitlines()
    assert lines[1] == "[dry-run] persona=p model=default"
    assert lines[2].startswith("  argv= /fake/claude -p <prompt: ")
    assert lines[2].endswith(
        "--append-system-prompt <system-prompt: 7 chars> "
        "--allowedTools Read Grep Glob "
        "--disallowedTools Edit Write NotebookEdit Bash "
        "--permission-mode plan --output-format json "
        f"--add-dir {session.repo_root}")
    assert lines[3].startswith("  prompt≈ 'Adversarially review this change.")


def test_the_dry_run_report_and_the_real_call_cannot_disagree(session, captured, capsys):
    """The property the consolidation exists for, asserted directly."""
    session.dry_run = True
    session.spawn("p", "MANDATE", "SURFACE", 1)
    reported = capsys.readouterr().out.splitlines()[2].removeprefix("  argv= ")

    session.dry_run = False
    session.spawn("p", "MANDATE", "SURFACE", 1)

    from blackice.report import render_argv
    assert render_argv(captured[0].argv) == reported


def test_the_dry_run_prompt_preview_is_capped_at_280_characters(session, capsys):
    session.dry_run = True

    session.spawn("p", "MANDATE", "x" * 5000, 1)

    preview = capsys.readouterr().out.splitlines()[3]
    assert preview.endswith("…'")
    assert len(preview) < 320
