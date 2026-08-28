# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""The permission policy's own blind spot: a tool granted and denied at once (#67).

REGRESSION. Deny-by-default is enforced at the boundary so a reviewer cannot do
*more* than it should; nothing checked that it could do as much as the panel
believed. A tool on both lists is not refused at call time — it is **absent**, so
the reviewer never attempts it and the result envelope records nothing at all.

Measured on agent CLI 2.1.246 before this was written: ``--allowedTools Read Grep
Glob`` against ``--disallowedTools Read Grep Glob …`` returns ``is_error: false``,
``permission_denials: []`` and a well-formed contract, and the reviewer's own prose
says *"This session exposes no file-read, grep, or shell tools"*. That measurement
is why this half exists at all: it is the one that fires on the issue's own
reproduction case, and it needs no subprocess to do it.
"""

from __future__ import annotations

from kuang.backends.claude_code.permissions import (DEFAULT_ALLOWED_TOOLS,
                                                    DEFAULT_DISALLOWED_TOOLS,
                                                    unavailable_tools)


# --- the healthy case: the default policy contradicts itself nowhere ---------

def test_the_shipped_default_policy_is_self_consistent():
    """The mirror-image defect, guarded directly.

    Reporting degradation on a healthy run is as bad as missing it on a degraded
    one, and the default policy is the run every operator gets. If this ever goes
    red, every default run has started claiming it reviewed with no tools.
    """
    assert unavailable_tools(DEFAULT_ALLOWED_TOOLS, DEFAULT_DISALLOWED_TOOLS) == []


def test_a_tool_on_neither_list_is_not_reported():
    assert unavailable_tools(["Read"], ["Edit"]) == []


# --- the defect: granted and denied at once ---------------------------------

def test_a_tool_that_is_granted_and_denied_was_never_available():
    assert unavailable_tools(["Read", "Grep", "Glob"],
                             ["Read", "Grep", "Glob", "Edit"]) == ["Read", "Grep", "Glob"]


def test_only_the_contradicted_tools_are_named():
    """Not "the panel is degraded" — *which* tool, per the issue's acceptance."""
    assert unavailable_tools(["Read", "Grep", "Glob"], ["Grep", "Bash"]) == ["Grep"]


def test_the_granted_order_is_preserved():
    """An operator reads this beside the ``--allow-tools`` they typed."""
    assert unavailable_tools(["Glob", "Read", "Grep"], ["Read", "Glob"]) == ["Glob", "Read"]


# --- the scoped-grant rule, and the measurement behind it -------------------

def test_a_scoped_grant_does_not_survive_a_bare_deny_of_the_same_tool():
    """The rule that makes the ``Bash(pytest:*)`` add-on in ``permissions`` legible.

    MEASURED, not assumed, because guessing wrong here manufactures the mirror-image
    defect. On agent CLI 2.1.246, ``--allowedTools 'Bash(echo:*)'`` against
    ``--disallowedTools … Bash`` (permission mode ``default``, so the mode is not
    what blocks it) returns ``permission_denials: []`` and the agent reports *"there
    is no Bash tool in this session. The call wasn't refused; it was impossible to
    make."* The scoped grant is cancelled, so naming it is correct, not a false
    positive.
    """
    assert unavailable_tools(["Read", "Bash(pytest:*)"],
                             DEFAULT_DISALLOWED_TOOLS) == ["Bash(pytest:*)"]


def test_a_scoped_deny_does_not_cancel_a_broader_grant():
    """The rule runs one way only: the base name of the GRANT is what is matched.

    ``Bash`` granted against ``Bash(rm:*)`` denied leaves Bash available for
    everything else, so claiming it was unavailable would be a false positive.
    """
    assert unavailable_tools(["Bash"], ["Bash(rm:*)"]) == []


def test_an_exact_scoped_match_is_still_a_match():
    assert unavailable_tools(["Bash(pytest:*)"], ["Bash(pytest:*)"]) == ["Bash(pytest:*)"]


# --- edges ------------------------------------------------------------------

def test_an_empty_grant_reports_nothing():
    assert unavailable_tools([], DEFAULT_DISALLOWED_TOOLS) == []


def test_an_empty_deny_list_reports_nothing():
    assert unavailable_tools(DEFAULT_ALLOWED_TOOLS, []) == []
