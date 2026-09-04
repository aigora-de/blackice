# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""The permission policy: deny-by-default, and never a blanket shell.

One module so there is one place to audit what a reviewer subprocess is allowed
to do — and, below, an honest account of how far that audit reaches. ``personas``
defaults each reviewer's tools from the allow-list; ``session`` and the CLI carry
the deny-list to the subprocess boundary.
"""

from __future__ import annotations


# Deny-by-default permission policy. Reviewers may READ the diff and source
# (Read/Grep/Glob); they may NOT edit, write, or run shell — those four tools are
# on the deny-list below and are removed from the reviewer's session.
#
# IMPORTANT — how permissions work headless, and WHAT THIS POLICY DOES NOT BOUND.
# In `claude -p` there is no interactive prompt. An *allowed* tool runs
# UNSUPERVISED. A DENY-LISTED tool is removed from the session entirely, so it is
# never even attempted (#67). A tool on NEITHER list is **not bounded here**:
# `--allowedTools` marks tools as needing no approval, it does not restrict the
# set. Measured on agent CLI 2.1.246 under this exact policy, a reviewer called
# `WebFetch` on an external URL and it SUCCEEDED, with no denial recorded. So this
# policy bounds the tools it NAMES and nothing else, and "read-only" describes
# what a reviewer may do to the REPO, not the whole of its reach. #76 tracks
# bounding the tool set; #4 is the `--settings` profile and sandbox that would do
# it. Do not read the list below as an exhaustive account of a reviewer's powers
# until one of those lands.
#
# Putting bare `Bash` here would additionally pre-approve *all* shell for an LLM
# reviewing an untrusted diff (rm, git push, arbitrary egress), with no human in
# the per-command loop. In this design HITL is per-EPOCH (convene / synthesise /
# gate), not per-command, so per-command safety must come from POLICY, not
# prompts. Verification tools (pytest/git/ruff) are a deliberate, SCOPED add-on
# for a later version — e.g. --allowedTools "Bash(pytest:*)" "Bash(git diff:*)" —
# ideally shipped via a `--settings` profile and sandboxed (no internet). Never
# bare `Bash`. See NOTES.md.
#
# MEASURED, AND THE SCOPE IS NOT ENFORCED (#96). On agent CLI 2.1.259, a
# `Bash(pytest:*)` grant with `Bash` off the deny-list, under the DEFAULT
# `--permission-mode plan`, ran `ls`, `git log -p`, an arbitrary Python heredoc,
# writes outside the repo and two `rm -rf` — with `permission_denials` EMPTY. Under
# `--permission-mode default` the same grant was refused (the agent's `python3 -m
# pytest` does not match `pytest:*`) and every refusal WAS recorded. So the scoped
# add-on above is an INTENTION, not a shipped guarantee: today it is cancelled by
# the deny-list, or refused, or unbounded. Do not describe it as bounding anything
# until #96 lands, and do not add a `Bash(...)` grant to any default.
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
    decides what a grant is worth, and this check does not model it. A
    mode-to-tool-class table would be runtime knowledge that drifts silently as the
    runtime changes, which is the failure mode this whole issue is about — and the
    drift is not hypothetical. This paragraph previously read that under ``plan``
    the agent *"reports it cannot run Bash at all, attempts no call, and again
    records no denial"*. That was measured on agent CLI 2.1.246 and was true then.
    Re-measured on **2.1.259** it is **false in every clause**: the agent attempts,
    the calls SUCCEED, and the scope is not enforced (#96). The envelope's shape did
    not change between those versions, so nothing here could notice. The lesson is
    written into #96's acceptance: pin the CLI version beside any claim about it.
    """
    deny = set(denied)
    return [t for t in granted if t in deny or _base_name(t) in deny]
