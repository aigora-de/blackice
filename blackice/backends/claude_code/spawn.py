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
from pathlib import Path


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


def run_claude(argv: list[str], *, cwd: Path) -> tuple[str, int, str | None]:
    """Spawn one ``claude`` call. Returns ``(result_text, output_tokens, error)``.

    Never raises on the model's behalf: a non-zero exit becomes an error string
    the caller records as a meta finding, and stdout that is not the expected
    JSON envelope is tolerated as raw text.
    """
    proc = subprocess.run(argv, capture_output=True, text=True, cwd=str(cwd))
    if proc.returncode != 0:
        return "", 0, f"claude exited {proc.returncode}: {proc.stderr[:300]}"
    try:
        env = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return proc.stdout, 0, None  # tolerate raw text
    return env.get("result", ""), int((env.get("usage") or {}).get("output_tokens", 0)), None
