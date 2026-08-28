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

    Refused tools (#67) are read BEFORE the failure branch, not after it: a call can
    be starved of tools and then error, and the call most likely to have been
    starved must not be the one that reports none.
    """
    proc = subprocess.run(argv, capture_output=True, text=True, cwd=str(cwd))
    try:
        env = json.loads(proc.stdout)
    except json.JSONDecodeError:
        env = None
    if not isinstance(env, dict):
        env = None  # valid JSON that is not an envelope is still just text
    denied = _denied_tools(env)
    if proc.returncode != 0 or (env is not None and env.get("is_error") is True):
        return CallResult("", 0, "agent error: " + _describe_failure(
            env, returncode=proc.returncode, stderr=proc.stderr), denied)
    if env is None:
        return CallResult(proc.stdout, 0, None, denied)  # tolerate raw text
    return CallResult(env.get("result", ""),
                      int((env.get("usage") or {}).get("output_tokens", 0)),
                      None, denied)
