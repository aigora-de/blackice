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
