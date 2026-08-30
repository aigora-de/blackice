# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""The subprocess boundary: finding the ``claude`` binary and calling it.

Everything that crosses out of this process goes through here, so the permission
flags a reviewer runs under are visible in one place.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class CallResult:
    """What the envelope told us about one ``claude -p`` call.

    The boundary used to return ``(result_text, output_tokens, error)`` and discard
    the envelope, so a diagnostic that was not one of those three could not reach a
    caller at all: #29 read five fields out of it, built one error string, and
    dropped the rest.

    Deliberately *this* channel and not a wider tuple. The envelope carries several
    things nobody reads — ``num_turns`` (#70), ``total_cost_usd`` (#65) — and a
    channel designed as "the one field I need" is redesigned once per issue, with
    every call site re-unpacking each time. A dataclass takes a new field without
    touching a single existing call site.

    Deliberately not the envelope itself, either. It is untrusted, unbounded,
    model-adjacent input (#25), and handing the dict downstream would invite callers
    to reach into arbitrary keys — the drift this boundary exists to prevent. Fields
    are extracted, validated and named here, and nothing else crosses.

    ``num_turns`` is what the runtime says the call cost in turns (#70). Measured
    on agent CLI 2.1.246 it counts tool-call round-trips plus the final answering
    turn, so **1 means the agent answered without calling a single tool** — see
    ``called_no_tool``. **0 means the envelope did not say**, never "called no
    tool": a call that happened takes at least one turn, so zero is a value the
    runtime cannot legitimately report and is free to mean *unknown*.

    ``denied_tools`` holds the NAMES of tools whose use was refused (#67), and only
    the names: ``tool_input`` is unbounded model-controlled content — an entire
    shell command in the capture this was written against — and what crosses here is
    written into a run artefact that gets pasted into public issues. See
    ``permissions.unavailable_tools`` for the other, deterministic half of #67; a
    deny-listed tool is absent rather than refused, so it never appears here.
    """

    text: str
    output_tokens: int
    error: str | None = None
    denied_tools: tuple[str, ...] = field(default=())
    num_turns: int = 0


# The one turn count that is a claim about grounding, and the whole of the rule
# (#70). Measured on agent CLI 2.1.246 under the argv ``build_argv`` builds: ten
# live calls over a five-file surface and a one-line diff, with Read/Grep/Glob,
# with Read alone, and with a scoped Bash grant, ran 6 to 9 turns — including a
# whole three-persona panel driven end to end through the CLI. The only ONE-turn
# call was the reformat retry, a formatter that needs no tool by design.
#
# Deliberately not a threshold. ``num_turns`` counts turns, not tool calls: in a
# live panel with every review tool removed from the session, two of the three
# personas reached 12 and 13 turns having opened nothing, hunting for tools that
# were not there (#67's capture did the same at 3). So a value above 1 is not
# evidence that source was inspected, and any ``>= N`` rule would be a judgement
# about "enough review" on a counter that does not measure review. One turn is
# unambiguous — there was no round-trip for a tool result to come back in — and
# nothing finer is measurable today.
#
# It therefore UNDER-reports, deliberately: in that same run only the third
# persona was caught here. The other two are caught by ``permissions``'
# deterministic half (#67), which knows from the flags alone that their tools were
# cancelled. Neither rule subsumes the other — this one sees a reviewer that did
# not look, that one sees a reviewer that could not — which is why both are
# printed.
UNGROUNDED_TURNS = 1


def called_no_tool(turns: int) -> bool:
    """Whether this call answered without calling a single tool (#70).

    False for 0, which means the envelope reported no count at all — a run must
    not report a degradation it did not measure, and a dry run must not report one
    it did not suffer.
    """
    return turns == UNGROUNDED_TURNS


def _resolve_claude_bin() -> str:
    """Locate the ``claude`` executable robustly (child PATH may differ).

    Order: ``$CLAUDE_BIN`` -> ``PATH`` -> ``~/.local/bin/claude``. Falls back to
    the bare name so the failure, if any, is a clear FileNotFoundError.
    """
    for cand in (os.environ.get("CLAUDE_BIN"), shutil.which("claude"),
                 os.path.expanduser("~/.local/bin/claude")):
        if cand and os.path.exists(cand):
            return cand
    return "claude"


def build_argv(*, claude_bin: str, prompt: str, mandate: str, tools: list[str],
               disallowed_tools: list[str], permission_mode: str,
               repo_root: Path, model: str | None) -> list[str]:
    """Build the argv for one ``claude -p`` reviewer call.

    The single description of the wiring. The real call spawns exactly this, and
    the dry run reports exactly this (via ``report.render_argv``) — previously
    the dry run described the call in prose written separately from the call
    itself, so the two could disagree, and a dry run that misreports the wiring
    is worse than none because pre-flight confirmation is its only job.

    Permissions are deny-by-default and are enforced here, at the boundary: see
    ``permissions``.
    """
    argv = [
        claude_bin, "-p", prompt,
        "--append-system-prompt", mandate,
        "--allowedTools", *tools,
        "--disallowedTools", *disallowed_tools,
        "--permission-mode", permission_mode,
        "--output-format", "json",
        "--add-dir", str(repo_root),
    ]
    if model:
        argv += ["--model", model]
    return argv


def _describe_failure(env: dict | None, *, returncode: int, stderr: str) -> str:
    """Render the agent's own account of why it failed, for the operator's note.

    The envelope is the only account there is: on both observed failure shapes
    ``stderr`` is empty, so a message built from the exit status alone says
    nothing. ``subtype`` names the failure — except in the API-error shape, where
    the subtype is still ``"success"`` and it is ``terminal_reason`` that says
    ``api_error``. The text itself arrives in ``errors`` or in ``result``, never
    both, so both are read and neither is assumed.
    """
    if env is None:
        return f"claude exited {returncode}: {stderr[:300]}"
    subtype = env.get("subtype")
    label = subtype if subtype and subtype != "success" else env.get("terminal_reason")
    errors = env.get("errors")
    if isinstance(errors, list) and errors:
        detail = "; ".join(str(e).strip() for e in errors if str(e).strip())
    else:
        detail = str(env.get("result") or stderr or "").strip()
    label = str(label or f"claude exited {returncode}")
    return f"{label}: {detail[:300]}" if detail else label


def _denied_tools(env: dict | None) -> tuple[str, ...]:
    """Tool names the agent asked for and was refused, deduped, first-seen order.

    #25's doctrine one field along: the envelope is untrusted input. Every level
    here is model-adjacent or runtime-shaped and none of it is ours, so a
    ``permission_denials`` that is not a list, an entry that is not a mapping, and a
    ``tool_name`` that is not a string are all tolerated rather than raised on. A
    reply this boundary cannot read is not a reason to lose an epoch every persona
    has already been paid for.

    Deduped because three refused ``Bash`` calls are one fact about the reviewer,
    not three, and this is read by a human deciding whether the review was grounded.
    """
    if env is None:
        return ()
    denials = env.get("permission_denials")
    if not isinstance(denials, list):
        return ()
    names: list[str] = []
    for entry in denials:
        if not isinstance(entry, dict):
            continue
        name = entry.get("tool_name")
        if isinstance(name, str) and name and name not in names:
            names.append(name)
    return tuple(names)


def _num_turns(env: dict | None) -> int:
    """The turn count the envelope reported, or 0 if it reported none (#70).

    #25's doctrine, and here it decides which way the run errs. The value is
    runtime-shaped and is not ours, so anything unreadable — absent, a string, a
    float, a negative — reads as 0, *we do not know*, rather than as 1, *it called
    no tool*. A boundary that guesses would report a degradation nobody measured,
    which is the same defect as missing one that happened.

    ``bool`` is excluded explicitly: it is an ``int`` in Python, so ``True`` would
    otherwise arrive as one turn — as the degradation itself — by accident.
    """
    if env is None:
        return 0
    turns = env.get("num_turns")
    if isinstance(turns, bool) or not isinstance(turns, int) or turns < 0:
        return 0
    return turns


def run_claude(argv: list[str], *, cwd: Path) -> CallResult:
    """Spawn one ``claude`` call. Returns what the envelope said about it.

    Never raises on the model's behalf: a failure becomes an error string the
    caller records as a meta finding, and stdout that is not the expected JSON
    envelope is tolerated as raw text.

    A failure is ``is_error`` **or** a non-zero exit (#29). Keying on the exit code
    alone made the protection incidental: on the agent CLI the two happen to
    coincide, so an error envelope was caught by a guard that never looked at it —
    and a runtime that does not maintain that coincidence would send the failure on
    to ``parse_findings`` and the reformat retry, which turns it into a review of
    record rather than a failure. The envelope is therefore parsed on the failure
    path too, which is precisely where it carries the diagnosis.

    Tokens are deliberately **not** counted for a failed call, though the envelope
    reports them: what counts against the budget decides when a run halts, which is
    control flow and belongs to #65 with its own evidence.

    Refused tools (#67) and the turn count (#70) are read BEFORE the failure branch,
    not after it: a call can be starved of tools and then error, and the call most
    likely to have been starved must not be the one that reports none. Both real
    failure captures carry a turn count, and one of them is ``1``.
    """
    proc = subprocess.run(argv, capture_output=True, text=True, cwd=str(cwd))
    try:
        env = json.loads(proc.stdout)
    except json.JSONDecodeError:
        env = None
    if not isinstance(env, dict):
        env = None  # valid JSON that is not an envelope is still just text
    denied, turns = _denied_tools(env), _num_turns(env)
    if proc.returncode != 0 or (env is not None and env.get("is_error") is True):
        return CallResult("", 0, "agent error: " + _describe_failure(
            env, returncode=proc.returncode, stderr=proc.stderr), denied, turns)
    if env is None:
        return CallResult(proc.stdout, 0, None, denied, turns)  # tolerate raw text
    return CallResult(env.get("result", ""),
                      int((env.get("usage") or {}).get("output_tokens", 0)),
                      None, denied, turns)
