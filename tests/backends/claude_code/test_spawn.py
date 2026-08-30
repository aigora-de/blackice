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

*Regression*, from `tools the reviewer never had (#67)` onward: a reviewer that
reviewed with fewer tools than the panel granted it must not pass for one that had
them. One test in that section is **characterisation of the runtime, not of our
code**, and says so in its own docstring — it pins the measurement that
``permission_denials`` is empty in exactly the state the issue is about.

*Regression*, from `a review that called no tool (#70)` onward: a reviewer that
answered from the prompt alone must not pass for one that opened the source. This
section also carries a **characterisation of the runtime** —
``test_a_turn_is_not_a_tool_call`` — and it is the argument against the threshold
that issue asked about rather than a guard on our code.

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
from kuang.engine import PersonaStatus, ReviewSpec


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
    result = session._run_claude("P", "M", ["Read"], None)

    assert '"verdict": "YES"' in result.text
    assert result.output_tokens == 11
    assert result.error is None


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

    result = session._run_claude("P", "M", ["Read"], None)

    assert (result.text, result.output_tokens) == ("", 0)
    assert result.error == "agent error: claude exited 7: boom"


def test_raw_non_json_stdout_is_tolerated(session, monkeypatch):
    monkeypatch.setattr(spawn_module.subprocess, "run",
                        lambda argv, **kw: subprocess.CompletedProcess(
                            argv, 0, stdout="just text", stderr=""))

    result = session._run_claude("P", "M", ["Read"], None)

    assert (result.text, result.output_tokens, result.error) == ("just text", 0, None)


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

    assert session._run_claude("P", "M", ["Read"], None).error == expected


def test_an_is_error_envelope_is_a_failure_even_on_a_zero_exit(session, monkeypatch):
    """SYNTHETIC, and deliberately so — this case is unreachable on CLI 2.1.246.

    Both real failure envelopes exit 1, so the existing ``returncode != 0`` guard
    happens to catch them. That protection is incidental: it keys on the exit code,
    which merely *correlates* with ``is_error``. This test is the guard against a
    runtime that does not maintain the correlation — a CLI revision, or a second
    backend. It does not reproduce a live defect and must not be read as doing so.
    """
    _envelope(monkeypatch, _MAX_TURNS, returncode=0)

    assert session._run_claude("P", "M", ["Read"], None).error == (
        "agent error: error_max_turns: Reached maximum number of turns (1)")


def test_a_failed_call_still_reports_no_tokens(session, monkeypatch):
    """Pins the deferral in #65 so that lifting it is a visible change.

    The envelope reports ``output_tokens: 187`` and ``$0.027003`` for this failure
    and the boundary counts none of it. Counting it decides when a run halts on
    BUDGET, which is control flow with its own acceptance criterion, so it is a
    separate issue rather than a free ride on this one.
    """
    _envelope(monkeypatch, _MAX_TURNS, returncode=0)

    result = session._run_claude("P", "M", ["Read"], None)

    assert (result.text, result.output_tokens) == ("", 0)
    assert result.error is not None
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

    result = session._run_claude("P", "M", ["Read"], None)

    assert (result.text, result.output_tokens, result.error) == (stdout, 0, None)


# --- tools the reviewer never had, and tools it was refused (#67) ------------
#
# REGRESSION. The envelopes below are real captures from agent CLI 2.1.246, taken
# by the probe this issue called for — not invented JSON. Trimmed of fields
# irrelevant to the boundary; every field asserted on is verbatim from the
# capture, and the two ``result`` strings are the capture's own opening and its
# own contract block, shortened where marked.
#
# The existing #29 captures above have no ``permission_denials`` key because they
# were trimmed of fields irrelevant to *that* boundary. These were taken for this
# one.

# The issue's own reproduction case: --disallowedTools Read Grep Glob Edit Write
# NotebookEdit Bash, on top of --allowedTools Read Grep Glob. The reviewer could
# open nothing, and the envelope says so NOWHERE — is_error is false, the subtype
# is "success", and permission_denials is EMPTY. A deny-listed tool is absent, not
# refused, so there is no call to refuse and nothing to record.
_DENIED_EVERY_TOOL = {
    "type": "result", "subtype": "success", "is_error": False,
    "num_turns": 3, "stop_reason": "end_turn", "terminal_reason": "completed",
    "permission_denials": [],
    "result": (
        "## Verification note (read first)\n\nThis session exposes no file-read, "
        "grep, or shell tools — the deferred-tool set is "
        "Cron/Monitor/WebFetch/SendMessage only, and `select:Read,Grep,Bash` "
        "returned no match. So I could not open `a.py` or run anything.\n\n"
        # …the review itself, and then its contract. Two of the seven findings the
        # capture returned, verbatim in title/severity/claim_class; evidence elided.
        '```json\n{"verdict": "NO",\n  "findings": [\n'
        '    {"title": "Audit write is a new failure point after the balance '
        'mutation — failed write yields apparent-failure-after-success, retry '
        'double-debits", "severity": "UGLY", "claim_class": '
        '"atomicity/irreversible-state", "file": "a.py", "line": 4, '
        '"evidence": "…"},\n'
        '    {"title": "Unescaped src/dst in f-string allows audit-record forgery '
        'via newline injection", "severity": "BLOCKER", "claim_class": '
        '"log-injection", "file": "a.py", "line": 4, "evidence": "…"}\n  ]}\n```'),
    "total_cost_usd": 0.473272,
    "usage": {"input_tokens": 6, "output_tokens": 8341},
}

# The only shape that DOES populate the field: a tool present in the session but
# not on the allow-list, which the agent actually attempted. Note what came back
# with it — ``tool_input`` carries a whole shell command, verbatim and
# model-controlled. Note also that the agent recovered: it re-ran the commands
# separately and completed the task, so an entry here is not itself a degradation.
_REFUSED_A_TOOL_CALL = {
    "type": "result", "subtype": "success", "is_error": False,
    "num_turns": 5, "stop_reason": "end_turn", "terminal_reason": "completed",
    "permission_denials": [
        {"tool_name": "Bash",
         "tool_use_id": "toolu_019WhRupaf8hM9c6ji421qaD",
         "tool_input": {
             "command": "grep -n 'transfer' a.py; echo \"---exit:$?---\"; ls *.py",
             "description": "Search a.py and list Python files"}}],
    "result": ("Side note: an initial combined Bash command was blocked by the "
               "permission layer (the `echo` part needed approval); I re-ran the "
               "two commands separately and both succeeded."),
    "usage": {"input_tokens": 4, "output_tokens": 212},
}


def test_a_refused_tool_call_reaches_the_caller(session, monkeypatch):
    """#67's own mechanism, on the one shape that populates it."""
    _envelope(monkeypatch, _REFUSED_A_TOOL_CALL, returncode=0)

    assert session._run_claude("P", "M", ["Read"], None).denied_tools == ("Bash",)


def test_only_the_tool_name_crosses_the_boundary(session, monkeypatch):
    """The Auditor's constraint, asserted rather than left to review.

    ``tool_input`` is unbounded model-controlled content — here an entire shell
    command; in general file paths and file contents — and what crosses this
    boundary is written into a run artefact that gets pasted into a **public**
    repository's issues. ``tool_use_id`` is a session-internal handle with no
    operator meaning. The issue asks *which tool*, and ``tool_name`` answers it.

    A deliberate narrowing against #24's and #26's "record it verbatim": that
    doctrine governs a short scalar whose whole meaning *is* the string, and does
    not oblige this boundary to forward a blob.
    """
    _envelope(monkeypatch, _REFUSED_A_TOOL_CALL, returncode=0)

    result = session._run_claude("P", "M", ["Read"], None)

    leaked = repr(result.denied_tools)
    assert "grep -n" not in leaked and "ls *.py" not in leaked
    assert "toolu_019WhRupaf8hM9c6ji421qaD" not in leaked


def test_a_healthy_call_reports_no_refusal(session, captured):
    """The mirror-image defect: a clean run must claim nothing happened."""
    assert session._run_claude("P", "M", ["Read"], None).denied_tools == ()


def test_a_reviewer_denied_every_tool_is_invisible_in_its_own_envelope(session,
                                                                       monkeypatch):
    """The measurement that redirected this fix, pinned so it cannot be forgotten.

    CHARACTERISATION of the runtime, not of our code, and it is the whole argument
    for the deterministic half in ``permissions.unavailable_tools``. The issue
    proposed detecting this state by reading ``permission_denials``. Measured, the
    field is EMPTY in exactly this state — a deny-listed tool is removed from the
    session rather than refused at call time, so no denial is ever recorded — while
    ``is_error`` is false, the error is ``None``, and the reply parses to a vote.

    Everything ``run_claude`` can see here says the call went well. If a future CLI
    revision starts populating the field for a deny-listed tool this goes red, which
    is the honest outcome: the finding would then be reachable from the envelope too.
    """
    _envelope(monkeypatch, _DENIED_EVERY_TOOL, returncode=0)

    result = session._run_claude("P", "M", ["Read", "Grep", "Glob"], None)

    assert result.error is None
    assert result.denied_tools == (), "the envelope does not report this degradation"
    assert '"verdict": "NO"' in result.text, "and the persona voted anyway"


def test_a_refusal_is_reported_even_when_the_call_then_failed(session, monkeypatch):
    """A call can be refused a tool AND then error. Neither fact displaces the other.

    ``run_claude`` returns early on failure, so the refusal is read before that
    branch rather than after it — otherwise the one call most likely to have been
    starved of tools is the one that reports none.
    """
    env = dict(_MAX_TURNS, permission_denials=[{"tool_name": "Grep"}])
    _envelope(monkeypatch, env)

    result = session._run_claude("P", "M", ["Read"], None)

    assert result.error is not None
    assert result.denied_tools == ("Grep",)


@pytest.mark.parametrize("denials, expected", [
    # Nothing readable in it: tolerated, and nothing invented from it.
    (None, ()), ("Bash", ()), (12, ()), ({}, ()),
    ([None], ()), ([12], ()), (["Bash"], ()), ([{}], ()),
    ([{"tool_name": None}], ()), ([{"tool_name": 12}], ()),
    ([{"tool_name": ""}], ()), ([{"tool_use_id": "x"}], ()),
    # …and the half that makes the rest mean something: one readable entry among
    # unreadable ones must still arrive. Without these two rows every row above is
    # satisfied by an extractor that returns nothing at all, which is precisely the
    # defect. (Written this way because the first draft of this test survived the
    # vacuity probe: neutering the extractor entirely left it green.)
    ([12, {"tool_name": "Bash"}, None], ("Bash",)),
    ([{"tool_name": "Bash"}, "Grep", {"tool_name": "Glob"}], ("Bash", "Glob")),
])
def test_a_malformed_denial_list_is_read_for_what_it_has(session, monkeypatch,
                                                         denials, expected):
    """#25's doctrine one field along: the envelope is untrusted input.

    Every level here is model-adjacent or runtime-shaped and none of it is ours, so
    a reply this boundary cannot read is not a reason to lose an epoch every persona
    has already been paid for — the same courtesy the function already extends to
    stdout that is not JSON at all. Tolerating it must not mean discarding it: what
    IS readable still crosses.
    """
    _envelope(monkeypatch, dict(_REFUSED_A_TOOL_CALL, permission_denials=denials),
              returncode=0)

    result = session._run_claude("P", "M", ["Read"], None)

    assert result.error is None
    assert result.denied_tools == expected


def test_duplicate_refusals_of_one_tool_are_reported_once(session, monkeypatch):
    """Three refused ``Bash`` calls are one fact about the reviewer, not three."""
    _envelope(monkeypatch, dict(_REFUSED_A_TOOL_CALL, permission_denials=[
        {"tool_name": "Bash"}, {"tool_name": "Grep"}, {"tool_name": "Bash"}]),
        returncode=0)

    assert session._run_claude("P", "M", ["Read"], None).denied_tools == ("Bash", "Grep")


# --- and onto the persona's report ------------------------------------------

def test_a_refusal_lands_on_the_persona_report(session, monkeypatch):
    """Set at the source (#30's doctrine), and orthogonal to the status (#75's).

    The persona both CONTRIBUTED and was degraded, which is why this is a field
    beside ``unresolved_severities`` and ``unresolved_verdict`` rather than a
    ``PersonaStatus`` member: forcing it into the outcome enum would mean choosing
    one of the two facts and losing the other.
    """
    _envelope(monkeypatch, dict(_REFUSED_A_TOOL_CALL, result=(
        '```json\n{"verdict": "YES", "findings": []}\n```')), returncode=0)

    report = session.spawn("p", "MANDATE", "SURFACE", 1)

    assert report.denied_tools == ["Bash"]
    assert report.status is PersonaStatus.FOUND_NOTHING, "contributed AND degraded"
    assert report.verdict == "YES", "and its vote still counts (#24/#26 doctrine)"


def test_a_refusal_during_the_reformat_retry_is_not_lost(session, monkeypatch):
    """The retry is a second call and can be starved too; the report is of both.

    The first call is refused ``Bash`` and returns prose with no contract; the
    reformat call is refused ``Read`` and returns the contract. A report naming
    only one of them would understate what the reviewer went without.
    """
    replies = [
        dict(_REFUSED_A_TOOL_CALL, result="A prose review with no JSON contract."),
        dict(_REFUSED_A_TOOL_CALL, permission_denials=[{"tool_name": "Read"}],
             result='```json\n{"verdict": "YES", "findings": []}\n```'),
    ]

    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(
            argv, 0, stdout=json.dumps(replies.pop(0)), stderr="")

    monkeypatch.setattr(spawn_module.subprocess, "run", fake_run)

    report = session.spawn("p", "MANDATE", "SURFACE", 1)

    assert report.denied_tools == ["Bash", "Read"]


def test_a_dry_run_claims_no_refusals(session, capsys):
    """Nothing was spawned, so nothing was refused — and nothing may be implied."""
    session.dry_run = True

    assert session.spawn("p", "MANDATE", "SURFACE", 1).denied_tools == []


# --- a review that called no tool (#70) --------------------------------------
#
# REGRESSION. The envelopes below are real captures from agent CLI 2.1.246, taken
# by the probe this issue called for — not invented JSON. Trimmed of fields
# irrelevant to this boundary; every field asserted on is verbatim from the
# capture, and the ``result`` strings are the captures' own openings, shortened
# where marked.
#
# The healthy baseline this issue asked for, because none existed: ten live calls
# under the argv ``build_argv`` builds, over a 5-file surface and a one-line diff,
# with Read/Grep/Glob, with Read alone, and with a scoped Bash grant, ran 6 to 9
# turns — 7, 7, 7, 9, 7, 8 from the probe and 6, 7, 9 from a three-persona panel
# driven end to end. The only ONE-turn call was the reformat retry.

# The healthy case. A read-heavy review of a five-file surface: it opened the
# files, said so, and took SEVEN turns to do it.
_HEALTHY_REVIEW = {
    "type": "result", "subtype": "success", "is_error": False,
    "num_turns": 7, "stop_reason": "end_turn", "terminal_reason": "completed",
    "permission_denials": [],
    "result": (
        "## Review\n\nI read all five files on disk (they match the review "
        "surface byte-for-byte). **I could not execute the tests — this session "
        "has no shell tool** — so every claim below is derived from reading the "
        "source, and I've said in each finding exactly what I checked.\n\n"
        # …the review itself, and then its contract; elided.
        '```json\n{"verdict": "NO", "findings": []}\n```'),
    "total_cost_usd": 0.5662955,
    "usage": {"input_tokens": 6, "output_tokens": 13312},
}

# The defect, reproduced live: ONE turn, ``is_error`` false, no denial, and a
# well-formed contract carrying a vote and an UGLY. Nothing else in the envelope
# distinguishes it from the capture above. It is the reformat retry — the one call
# ``PanelSession.spawn`` makes that reliably needs no tool at all.
_CALLED_NO_TOOL = {
    "type": "result", "subtype": "success", "is_error": False,
    "num_turns": 1, "stop_reason": "end_turn", "terminal_reason": "completed",
    "permission_denials": [],
    "result": (
        '```json\n{"verdict": "NO",\n  "findings": [\n'
        '    {"title": "NaN amount bypasses every guard, permanently poisons '
        'balances and enables unbounded minting", "severity": "UGLY", '
        '"claim_class": "input-validation-ruin", "file": "transfer.py", '
        '"line": 22, "evidence": "…"}\n  ]}\n```'),
    "total_cost_usd": 0.211466,
    "usage": {"input_tokens": 2, "output_tokens": 3925},
}


def test_a_healthy_reviews_turn_count_reaches_the_caller(session, monkeypatch):
    """The baseline nobody had taken. Seven turns, and the boundary now sees it."""
    _envelope(monkeypatch, _HEALTHY_REVIEW, returncode=0)

    assert session._run_claude("P", "M", ["Read", "Grep", "Glob"], None).num_turns == 7


def test_a_review_that_called_no_tool_is_visible_here_and_nowhere_else(session,
                                                                       monkeypatch):
    """The defect. Every other guard in the run reports this call as healthy.

    ``is_error`` is false so #29 cannot see it; ``permission_denials`` is empty and
    the tools were granted so #67 cannot either; it parses to a contract with a
    vote, so #30 records it as a persona that contributed. ``num_turns`` is the
    only field in the envelope that says the reviewer opened nothing.
    """
    _envelope(monkeypatch, _CALLED_NO_TOOL, returncode=0)

    result = session._run_claude("P", "M", ["Read"], None)

    assert result.error is None, "no other guard sees it"
    assert result.denied_tools == (), "and neither does #67's"
    assert '"verdict": "NO"' in result.text, "and it voted"
    assert result.num_turns == 1


@pytest.mark.parametrize("turns, expected", [
    # 0 is "the envelope did not say", never "called no tool" — see below.
    (0, False),
    # The one value the measurement supports.
    (1, True),
    # Everything above it. 3 is _DENIED_EVERY_TOOL, which opened NOTHING; 7, 8 and
    # 9 are the measured healthy range. A rule with a threshold between them would
    # have to call 3 ungrounded and 7 grounded, and the capture below shows that
    # the counter does not carry that meaning.
    (2, False), (3, False), (5, False), (7, False), (8, False), (9, False),
])
def test_only_one_turn_is_claimed_as_no_tool_call(turns, expected):
    """The rule, and the whole of it: ``> 1`` and nothing finer (#70 decision 1).

    Measured on agent CLI 2.1.246, ``num_turns`` counts tool-call round-trips plus
    the final answering turn — so ONE turn is an unambiguous statement that no tool
    was called, and it is the only value that is. The converse does not hold and is
    not claimed: see ``test_a_turn_is_not_a_tool_call``.
    """
    assert spawn_module.called_no_tool(turns) is expected


@pytest.mark.parametrize("value, expected", [
    # Unreadable, or absent: 0, meaning "the envelope did not say".
    (None, 0), ("7", 0), (7.5, 0), (-1, 0), ([7], 0), ({}, 0),
    # ``True`` is an int in Python and would read as one turn — i.e. as the
    # degradation itself — so it is rejected explicitly rather than by accident.
    (True, 0), (False, 0),
    # …and the half that makes the rest mean something: a readable count still
    # arrives. Without these rows every row above is satisfied by an extractor
    # that returns nothing at all, which is the defect this issue is about.
    (1, 1), (7, 7), (0, 0),
])
def test_an_unreadable_turn_count_is_read_for_what_it_has(session, monkeypatch,
                                                          value, expected):
    """#25's doctrine one field along: the envelope is untrusted input.

    A count this boundary cannot read is "we do not know", never "it called no
    tool" — a run must not report a degradation it did not measure, which is the
    same rule that keeps a dry run silent about one.
    """
    _envelope(monkeypatch, dict(_HEALTHY_REVIEW, num_turns=value), returncode=0)

    result = session._run_claude("P", "M", ["Read"], None)

    assert result.num_turns == expected
    assert result.error is None, "an unreadable count is not a failed call"


def test_an_envelope_without_the_field_reports_no_turn_count(session, monkeypatch):
    """A runtime that does not report turns must not read as one that called none."""
    env = {k: v for k, v in _HEALTHY_REVIEW.items() if k != "num_turns"}
    _envelope(monkeypatch, env, returncode=0)

    assert session._run_claude("P", "M", ["Read"], None).num_turns == 0
    assert spawn_module.called_no_tool(0) is False


def test_raw_non_json_stdout_reports_no_turn_count(session, monkeypatch):
    """There is no envelope to read, so there is no count — and no claim."""
    monkeypatch.setattr(spawn_module.subprocess, "run",
                        lambda argv, **kw: subprocess.CompletedProcess(
                            argv, 0, stdout="just text", stderr=""))

    assert session._run_claude("P", "M", ["Read"], None).num_turns == 0


def test_a_failed_call_still_reports_its_turn_count(session, monkeypatch):
    """Read BEFORE the failure branch, for #67's reason at the same boundary.

    A call can call no tool and then error, and the call least likely to have
    opened anything must not be the one that reports nothing about it. Both real
    failure captures carry the field: ``_MAX_TURNS`` at 2, ``_API_ERROR`` at 1.
    """
    _envelope(monkeypatch, _MAX_TURNS)

    result = session._run_claude("P", "M", ["Read"], None)

    assert result.error is not None
    assert result.num_turns == 2


def test_a_turn_is_not_a_tool_call(session, monkeypatch):
    """CHARACTERISATION OF THE RUNTIME, not of our code — and the limit of the rule.

    ``_DENIED_EVERY_TOOL`` is the #67 capture: a reviewer with **no tools at all**
    in its session, which opened nothing and said so in prose, and which reports
    ``num_turns: 3``. So a turn is not a tool call, and no threshold above 1 is
    evidence that source was inspected — the reviewer here spent its turns
    searching for tools that were not there. Measured again for this issue and
    much further out: in a live panel run with every review tool deny-listed, two
    personas reached **12 and 13** turns having opened nothing.

    This is the argument against the threshold this issue asked about, pinned so
    that proposing one later has to argue past a measurement. If a future CLI
    revision changes what the counter counts, this goes red, which is honest.
    """
    _envelope(monkeypatch, _DENIED_EVERY_TOOL, returncode=0)

    result = session._run_claude("P", "M", ["Read", "Grep", "Glob"], None)

    assert result.num_turns == 3
    assert spawn_module.called_no_tool(result.num_turns) is False, \
        "3 turns, and it opened nothing: the count does not measure grounding"


# --- and onto the persona's report ------------------------------------------

def test_a_turn_count_lands_on_the_persona_report(session, monkeypatch):
    """Set at the source (#30's doctrine), and orthogonal to the status (#67's).

    The persona both CONTRIBUTED and reviewed nothing, which is why this is a field
    beside ``unresolved_severities``, ``unresolved_verdict`` and ``denied_tools``
    rather than a ``PersonaStatus`` member: the enum names the OUTCOME of one
    spawn, and forcing a survived degradation into it would mean choosing one of
    the two facts and losing the other.
    """
    _envelope(monkeypatch, _CALLED_NO_TOOL, returncode=0)

    report = session.spawn("p", "MANDATE", "SURFACE", 1)

    assert report.turns == 1
    assert report.status is PersonaStatus.CONTRIBUTED, \
        "it reviewed nothing AND contributed"
    assert report.verdict == "NO", "and its vote still counts (#24/#26/#67 doctrine)"
    assert [f.severity.name for f in report.findings] == ["UGLY"], \
        "including an UGLY, which is why discarding the vote would unlatch the breaker"


def test_the_reformat_retrys_turns_do_not_become_the_reviews(session, monkeypatch):
    """The laundering guard, and it is measured rather than imagined.

    The retry is a formatter: it needs no tool, and the live capture of one is the
    single ONE-turn call in the whole measurement. Two wrong answers are available
    here and the tests below reject both — summing the calls would render 1 + 5 = 6
    and clear the rule, and taking the last call's count would report the retry's 5
    for a review that opened nothing.

    The report is of the REVIEW call, because the review is the thing that was
    grounded or was not; the formatter never had anything to ground.
    """
    replies = [
        dict(_CALLED_NO_TOOL, result="A prose review with no JSON contract."),
        dict(_HEALTHY_REVIEW, num_turns=5,
             result='```json\n{"verdict": "YES", "findings": []}\n```'),
    ]

    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(
            argv, 0, stdout=json.dumps(replies.pop(0)), stderr="")

    monkeypatch.setattr(spawn_module.subprocess, "run", fake_run)

    report = session.spawn("p", "MANDATE", "SURFACE", 1)

    assert report.verdict == "YES", "the retry produced the contract of record"
    assert report.turns == 1, "and the REVIEW still opened nothing"


def test_a_grounded_review_is_not_relabelled_by_its_retry(session, monkeypatch):
    """The mirror image, and the reason the rule is not 'the last call's count'.

    A seven-turn review that missed the contract is grounded; its one-turn
    formatter is not evidence about the review, and must not be reported as if it
    were. A rule that took the retry's count would flag this healthy persona.
    """
    replies = [
        dict(_HEALTHY_REVIEW, result="A prose review with no JSON contract."),
        _CALLED_NO_TOOL,
    ]

    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(
            argv, 0, stdout=json.dumps(replies.pop(0)), stderr="")

    monkeypatch.setattr(spawn_module.subprocess, "run", fake_run)

    assert session.spawn("p", "MANDATE", "SURFACE", 1).turns == 7


def test_a_failed_persona_still_reports_what_the_envelope_said(session, monkeypatch):
    """The fact is set at the source for every persona, whatever became of it."""
    _envelope(monkeypatch, _MAX_TURNS)

    report = session.spawn("p", "MANDATE", "SURFACE", 1)

    assert report.status is PersonaStatus.AGENT_ERROR
    assert report.turns == 2


def test_a_dry_run_reports_no_turn_count(session, capsys):
    """Nothing was spawned, so no turn was taken — and none may be implied.

    Zero is "we never asked", not "it called no tool": #30 separated ``NOT_SPAWNED``
    from ``FOUND_NOTHING`` and ``ReduceState.DRY_RUN`` from ``RAN`` for this reason,
    and a dry run must not report a degradation it did not suffer.
    """
    session.dry_run = True

    report = session.spawn("p", "MANDATE", "SURFACE", 1)

    assert report.turns == 0
    assert report.status is PersonaStatus.NOT_SPAWNED
