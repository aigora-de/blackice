# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""Backend tests for the LLM semantic clusterer (issue #1).

These test the DETERMINISTIC glue around the non-deterministic ``claude`` call:
parsing the model's index-groups, enforcing a **partition** (never drop a
panellist's finding), sanitising bad indices, and degrading gracefully to the
identity reduce on any failure. No network, no subprocess — ``_run_claude`` is
monkeypatched to return canned envelopes, so the model's *reasoning* is out of
scope here (that is what the deterministic default + recorded fixtures cover).
"""

from __future__ import annotations

import pytest

from claude_code_backend import PanelSession
from loop import Finding, ReviewSpec, Severity


def _findings():
    return [
        Finding("P1", "fills dropped", Severity.UGLY, "drop", "a.py", 499),
        Finding("P2", "fills dropped, reworded", Severity.BLOCKER, "drop", "b.py", 517),
        Finding("P3", "unrelated tz bug", Severity.NOTE, "tz", "a.py", 216),
    ]


@pytest.fixture
def session(tmp_path):
    return PanelSession(
        repo_root=tmp_path, spec=ReviewSpec(why="w", what="x"),
        personas={}, base="")


def _canned(text, toks=7, err=None):
    """A stand-in for ``_run_claude`` returning a fixed (text, tokens, err)."""
    return lambda prompt, mandate, tools, model: (text, toks, err)


def _members_by_title(clusters):
    return sorted(tuple(sorted(m.title for m in c.members)) for c in clusters)


# --- valid grouping ---------------------------------------------------------

def test_valid_groups_produce_clusters(session):
    session._run_claude = _canned('```json\n{"clusters": [[0, 1], [2]]}\n```')
    clusters = session.reduce(_findings())
    assert len(clusters) == 2
    assert _members_by_title(clusters) == [
        ("fills dropped", "fills dropped, reworded"), ("unrelated tz bug",)]
    # UGLY-preserving: the merged cluster's severity is the max of its members.
    merged = next(c for c in clusters if len(c.members) == 2)
    assert merged.severity is Severity.UGLY


# --- graceful degradation: never raise, always a partition ------------------

def test_malformed_json_falls_back_to_identity(session):
    session._run_claude = _canned("I could not produce JSON, sorry.")
    clusters = session.reduce(_findings())
    # Identity: one cluster per finding, every finding preserved.
    assert len(clusters) == 3
    assert sum(len(c.members) for c in clusters) == 3


def test_call_error_falls_back_to_identity(session):
    session._run_claude = _canned("", toks=0, err="claude exited 1: boom")
    clusters = session.reduce(_findings())
    assert len(clusters) == 3


# --- partition integrity under bad indices ----------------------------------

def test_missing_index_becomes_singleton(session):
    """An unassigned finding is kept as its own cluster — never dropped."""
    session._run_claude = _canned('```json\n{"clusters": [[0, 1]]}\n```')
    clusters = session.reduce(_findings())
    assert sum(len(c.members) for c in clusters) == 3          # nothing dropped
    titles = {m.title for c in clusters for m in c.members}
    assert "unrelated tz bug" in titles                        # index 2 recovered


def test_duplicate_and_out_of_range_indices_sanitised(session):
    """Duplicate indices collapse; out-of-range indices are ignored; still a partition."""
    session._run_claude = _canned('```json\n{"clusters": [[0, 0, 5], [1]]}\n```')
    clusters = session.reduce(_findings())
    members = [m for c in clusters for m in c.members]
    assert len(members) == 3                                   # exactly the input
    assert len({id(m) for m in members}) == 3                  # each exactly once


# --- trivial inputs short-circuit without a call ----------------------------

def test_small_input_does_not_call_claude(session):
    called = False

    def _boom(*a, **k):
        nonlocal called
        called = True
        return ("", 0, None)

    session._run_claude = _boom
    one = [Finding("P1", "solo", Severity.NOTE, "x", "a.py", 1)]
    clusters = session.reduce(one)
    assert len(clusters) == 1
    assert called is False                                     # no spawn for <2 findings
