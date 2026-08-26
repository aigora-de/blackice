# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""The subprocess boundary: the argv, the dry-run, and what comes back.

**Mixed by design, and labelled section by section — do not read the file as one
kind of test.**

*Characterisation*, up to and including the dry-run: these pin what
``PanelSession`` hands to ``claude`` today, and what ``--dry-run`` prints instead,
so that #19's consolidation of the two — one ``build_argv`` feeding both the real
call and the dry-run report — is a visible change rather than a silent one. The
argv had no coverage at all before this file, which was the point: a dry-run whose
only job is pre-flight confirmation was describing wiring that nothing checked.

*Regression*, from `the error envelope (#29)` onward: a failed agent must not be
mistaken for a reviewing one. The file was labelled characterisation-only when the
error path was the one thing here nothing tested, and #29 is that path — so the
label is now half wrong and says which half.

Nothing here spawns anything: ``subprocess.run`` is replaced for the duration of
each test. That hermeticity is also this file's limit, and the #29 section says so
where it matters — a stubbed envelope proves the *handler* correct and proves
nothing whatever about the CLI. The envelopes it stubs are real captures precisely
because that gap cannot be closed from inside the suite.
"""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from kuang.backends.claude_code import PanelSession, Persona
from kuang.backends.claude_code import spawn as spawn_module
from kuang.engine import ReviewSpec


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
    """The fallback, for a failure with no envelope to describe it at all.

    The ``agent error: `` prefix is a deliberate wording change in #29, applied to
    every failure this boundary reports rather than only to the new ones: it is the
    one channel meaning *the process ran and no review came back*, and #30 needs to
    tell it from ``loop.run``'s ``persona failed:``, which means our own code raised.
    Two shapes for one channel would have left #30 matching on both.
    """
    monkeypatch.setattr(spawn_module.subprocess, "run",
                        lambda argv, **kw: subprocess.CompletedProcess(
                            argv, 7, stdout="", stderr="boom"))

    text, toks, err = session._run_claude("P", "M", ["Read"], None)

    assert (text, toks) == ("", 0)
    assert err == "agent error: claude exited 7: boom"


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

    from kuang.report import render_argv
    assert render_argv(captured[0].argv) == reported


def test_the_dry_run_prompt_preview_is_capped_at_280_characters(session, capsys):
    session.dry_run = True

    session.spawn("p", "MANDATE", "x" * 5000, 1)

    preview = capsys.readouterr().out.splitlines()[3]
    assert preview.endswith("…'")
    assert len(preview) < 320


# --- the error envelope (#29) -----------------------------------------------
#
# REGRESSION. The envelopes below are real captures from agent CLI 2.1.246, taken
# by the live probe P5 called for — not invented JSON. Fields irrelevant to the
# boundary are trimmed; every field asserted on is verbatim from the capture.

_MAX_TURNS = {
    "type": "result", "subtype": "error_max_turns", "is_error": True,
    "num_turns": 2, "stop_reason": "tool_use", "terminal_reason": "max_turns",
    "errors": ["Reached maximum number of turns (1)"],
    "total_cost_usd": 0.027003,
    "usage": {"input_tokens": 9, "output_tokens": 187},
}
# Note the shape: `subtype` is "success" and the diagnosis is in `result`, which is
# what the CLI documents for a turn that ended on an API error. There is no
# `errors` key here, and no `result` key in _MAX_TURNS — never both.
_API_ERROR = {
    "type": "result", "subtype": "success", "is_error": True,
    "num_turns": 1, "stop_reason": "stop_sequence", "terminal_reason": "api_error",
    "api_error_status": None,
    "result": "API Error: Connection refused — a firewall or proxy may be blocking it",
    "total_cost_usd": 0, "usage": {"input_tokens": 0, "output_tokens": 0},
}


def _envelope(monkeypatch, env, *, returncode=1, stderr=""):
    """Stub the subprocess with one captured envelope on stdout."""
    monkeypatch.setattr(spawn_module.subprocess, "run",
                        lambda argv, **kw: subprocess.CompletedProcess(
                            argv, returncode, stdout=json.dumps(env), stderr=stderr))


@pytest.mark.parametrize("env, expected", [
    (_MAX_TURNS, "agent error: error_max_turns: Reached maximum number of turns (1)"),
    (_API_ERROR, "agent error: api_error: API Error: Connection refused — "
                 "a firewall or proxy may be blocking it"),
])
def test_the_error_carries_the_agents_own_diagnosis(session, monkeypatch, env, expected):
    """The whole point of #29: a run must be able to say *why* a persona dropped out.

    The envelope is on stdout and was never parsed on the failure path, while
    stderr is empty in both real captures — so the operator's note read literally
    ``claude exited 1: `` and every word of the diagnosis was discarded.
    """
    _envelope(monkeypatch, env)

    _, _, err = session._run_claude("P", "M", ["Read"], None)

    assert err == expected


def test_an_is_error_envelope_is_a_failure_even_on_a_zero_exit(session, monkeypatch):
    """SYNTHETIC, and deliberately so — this case is unreachable on CLI 2.1.246.

    Both real failure envelopes exit 1, so the existing ``returncode != 0`` guard
    happens to catch them. That protection is incidental: it keys on the exit code,
    which merely *correlates* with ``is_error``. This test is the guard against a
    runtime that does not maintain the correlation — a CLI revision, or a second
    backend. It does not reproduce a live defect and must not be read as doing so.
    """
    _envelope(monkeypatch, _MAX_TURNS, returncode=0)

    _, _, err = session._run_claude("P", "M", ["Read"], None)

    assert err == "agent error: error_max_turns: Reached maximum number of turns (1)"


def test_a_failed_call_still_reports_no_tokens(session, monkeypatch):
    """Pins the deferral in #65 so that lifting it is a visible change.

    The envelope reports ``output_tokens: 187`` and ``$0.027003`` for this failure
    and the boundary counts none of it. Counting it decides when a run halts on
    BUDGET, which is control flow with its own acceptance criterion, so it is a
    separate issue rather than a free ride on this one.
    """
    _envelope(monkeypatch, _MAX_TURNS, returncode=0)

    text, toks, err = session._run_claude("P", "M", ["Read"], None)

    assert (text, toks) == ("", 0)
    assert err is not None
    assert _MAX_TURNS["usage"]["output_tokens"] == 187, "the spend the run does not see"


@pytest.mark.parametrize("returncode", [1, 0])
def test_a_failed_agent_never_reaches_the_reformat_retry(session, monkeypatch, returncode):
    """#29's third bullet. The retry launders a failure into a review of record.

    Measured live: handed real error text, the reformatter returns ``verdict: "NO"``
    and a *fabricated* BLOCKER with an invented claim_class — a finding no reviewer
    raised, entering the ledger under a reviewer's name (#63).

    Both exit codes, because only one of them is the fix: at ``returncode=1`` this
    already held on main, by way of a guard that never looked at ``is_error``.
    """
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(
            argv, returncode, stdout=json.dumps(_MAX_TURNS), stderr="")

    monkeypatch.setattr(spawn_module.subprocess, "run", fake_run)

    report = session.spawn("p", "MANDATE", "SURFACE", 1)

    assert len(calls) == 1, "the failure was sent through the reformat retry"
    assert report.verdict is None
    assert [(f.severity, f.claim_class) for f in report.findings] == [(0, "meta")]


@pytest.mark.parametrize("returncode", [1, 0])
def test_a_failed_agent_cannot_produce_converged(session, monkeypatch, returncode):
    """The property, asserted rather than a constant a mutation would drag along.

    Driven through the real ``PanelSession.spawn`` into the real engine, because the
    criterion — *a run in which any persona failed cannot be read as a complete
    panel* — is a claim about the halt reason, not about a string in a report.

    The stub is the laundering path #29 describes, and it takes **two** calls to
    build: the failure alone cannot vote, because error text parses to no verdict at
    all. It is the *reformat retry* that turns a failure into a vote — so the second
    call returns a clean contract, which is the contract-compliant answer for a
    string containing no BLOCKER and no UGLY. Without that second call this test
    would pass on main and prove nothing.

    A single-persona panel is the strongest form: quorum defaults to *all*, so one
    failed persona is the whole panel, and nothing else can be carrying the assertion.
    """
    from kuang.engine import HaltingSet, HaltReason, PanelConfig
    from kuang.engine import run as engine_run

    replies = [
        json.dumps(_MAX_TURNS),
        json.dumps({"result": '```json\n{"verdict": "YES", "findings": []}\n```',
                    "is_error": False, "usage": {"output_tokens": 5}}),
    ]

    def fake_run(argv, **kwargs):
        stdout = replies.pop(0) if replies else replies_exhausted()
        return subprocess.CompletedProcess(argv, returncode if not stdout.startswith(
            '{"result"') else 0, stdout=stdout, stderr="")

    def replies_exhausted():
        raise AssertionError("more calls than the stub models")

    monkeypatch.setattr(spawn_module.subprocess, "run", fake_run)

    review_run = engine_run(
        session.spec, HaltingSet(max_epochs=1),
        PanelConfig(personas=[("p", "mandate")]),
        spawn=session.spawn, gather=lambda e: "surface", parallel=False,
        scope_complete=lambda r: True)

    assert review_run.halt_reason is not HaltReason.CONVERGED


def test_a_backend_failure_reads_differently_from_a_crashed_persona(session, monkeypatch):
    """Two sources, one shape — and #30 has to tell them apart.

    ``loop.run``'s seam guard writes ``persona failed:`` when our own code raised.
    This boundary writes ``agent error:`` when the process succeeded and the agent
    did not. Both are ``verdict=None`` plus a meta NOTE; only the wording separates
    them, and it is set at the source rather than derived later.
    """
    _envelope(monkeypatch, _MAX_TURNS)

    report = session.spawn("p", "MANDATE", "SURFACE", 1)

    assert report.findings[0].title.startswith("agent error: ")
    assert not report.findings[0].title.startswith("persona failed: ")


@pytest.mark.parametrize("stdout", ["[]", "12", '"a string"', "null"])
def test_json_that_is_not_an_envelope_is_tolerated_like_raw_text(session, monkeypatch,
                                                                 stdout):
    """This test exists because mutation 8 survived, not because anyone designed it.

    Removing the ``isinstance(env, dict)`` guard killed no test at all. On main the
    same input raised ``AttributeError`` from ``env.get("result")`` — uncaught, out
    of a worker thread, ending a run for which every persona had already been paid.
    Restructuring the function moved that crash one line earlier, to
    ``env.get("is_error")``, which is how the guard came to be written at all; the
    matrix then showed nothing was holding it in place.

    Tolerating it is the same courtesy the function already extends to stdout that
    is not JSON. A reply the boundary cannot read is not a reason to lose the epoch.
    """
    monkeypatch.setattr(spawn_module.subprocess, "run",
                        lambda argv, **kw: subprocess.CompletedProcess(
                            argv, 0, stdout=stdout, stderr=""))

    assert session._run_claude("P", "M", ["Read"], None) == (stdout, 0, None)
