# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""The permission policy: deny-by-default, and never a blanket shell.

One module so there is one place to audit what a reviewer subprocess is allowed
to do. ``personas`` defaults each reviewer's tools from the allow-list;
``session`` and the CLI carry the deny-list to the subprocess boundary.
"""

from __future__ import annotations


# Read-only permission policy (DENY-BY-DEFAULT). Reviewers may READ the diff and
# source (Read/Grep/Glob); they may NOT run shell or mutate anything.
#
# IMPORTANT — how permissions work headless: in `claude -p` there is no
# interactive prompt. An *allowed* tool runs UNSUPERVISED; an unallowed one is
# auto-DENIED (never asked). Putting bare `Bash` here would pre-approve *all*
# shell for an LLM reviewing an untrusted diff (rm, git push, network egress),
# with no human in the per-command loop. In this design HITL is per-EPOCH
# (convene / synthesise / gate), not per-command, so per-command safety must come
# from POLICY, not prompts. Verification tools (pytest/git/ruff) are a deliberate,
# SCOPED add-on for a later version — e.g. --allowedTools "Bash(pytest:*)"
# "Bash(git diff:*)" — ideally shipped via a `--settings` profile and sandboxed
# (no internet). Never bare `Bash`. See panel-review-NOTES.md.
DEFAULT_ALLOWED_TOOLS = ["Read", "Grep", "Glob"]
DEFAULT_DISALLOWED_TOOLS = ["Edit", "Write", "NotebookEdit", "Bash"]


def _base_name(tool: str) -> str:
    """The tool a grant is a grant *of*: ``Bash(pytest:*)`` -> ``Bash``."""
    return tool.split("(", 1)[0].strip()


def unavailable_tools(granted: list[str], denied: list[str]) -> list[str]:
    """Granted tools the deny-list removed, in the order they were granted (#67).

    Deny-by-default is enforced at the boundary above so a reviewer cannot do
    **more** than it should. This is the other direction, which nothing checked: a
    tool named on both lists is not refused when the reviewer calls it — it is
    absent from the session entirely, so the reviewer never calls it and the result
    envelope records nothing. Measured on agent CLI 2.1.246: ``permission_denials``
    is empty, ``is_error`` is false, and a well-formed contract with a vote comes
    back from a reviewer that could not open a file.

    Deterministic, and deliberately so. It needs no subprocess, so it is knowable
    before a panel is paid for and is reportable in a dry run — and it does not
    depend on a runtime continuing to leave that field empty.

    A grant is cancelled if its exact string or its *base name* is deny-listed, and
    the rule runs that way only: ``Bash(pytest:*)`` against a deny-listed ``Bash``
    is cancelled (measured: *"there is no Bash tool in this session"*), while a
    plain ``Bash`` grant against a deny-listed ``Bash(rm:*)`` is not, since it
    remains available for everything else. Claiming otherwise would report a
    degradation that did not happen, which is as bad as missing one that did.

    **Limit, stated rather than half-implemented:** ``--permission-mode`` also
    removes tools — under ``plan`` the agent reports it cannot run ``Bash`` at all,
    attempts no call, and again records no denial — and this check does not model
    it. A mode-to-tool-class table would be runtime knowledge that drifts silently
    as the runtime changes, which is the failure mode this whole issue is about.
    """
    deny = set(denied)
    return [t for t in granted if t in deny or _base_name(t) in deny]
