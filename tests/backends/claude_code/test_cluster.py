# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""Backend tests for the LLM semantic clusterer (issue #1).

These test the DETERMINISTIC glue around the non-deterministic ``claude`` call:
parsing the model's index-groups, enforcing a **partition** (never drop a
panellist's finding), sanitising bad indices, and degrading gracefully to the
identity reduce on any failure. No network, no subprocess — ``_run_claude`` is
monkeypatched to return canned envelopes, so the model's *reasoning* is out of
scope here (that is what the deterministic default + recorded fixtures cover).

The last group is #61, the twin of #26: a clusterer that restates the output
contract instead of answering it used to have the contract's own **example**
partition the findings, merging two that nobody grouped. Those tests go red
without the fix, and one of them exists because a mutation found the gap rather
than because anyone designed for it.
"""

from __future__ import annotations

import pytest

from kuang.backends.claude_code import PanelSession
from kuang.backends.claude_code.cluster import _CLUSTER_CONTRACT, _CLUSTER_TEMPLATE
from kuang.backends.claude_code.spawn import CallResult
from kuang.engine import Finding, ReviewSpec, Severity


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
    """A stand-in for ``_run_claude`` returning a fixed ``CallResult``."""
    return lambda prompt, mandate, tools, model: CallResult(text, toks, err)


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
        return CallResult("", 0, None)

    session._run_claude = _boom
    one = [Finding("P1", "solo", Severity.NOTE, "x", "a.py", 1)]
    clusters = session.reduce(one)
    assert len(clusters) == 1
    assert called is False                                     # no spawn for <2 findings


# --- the clusterer's own tolerance, kept at its own call site (#19) ----------

def test_an_unfenced_reply_is_still_read(session):
    """Unlike the findings contract, the clusterer falls back to the whole reply."""
    session._run_claude = _canned('{"clusters": [[0, 1], [2]]}')

    clusters = session.reduce(_findings())

    assert len(clusters) == 2


def test_an_empty_fenced_block_falls_back_to_identity(session):
    session._run_claude = _canned("```json\n\n```")

    assert len(session.reduce(_findings())) == 3


# --- an echoed cluster contract (#61) ---------------------------------------

def _four_findings():
    """Four findings, so the shipped example ``[[0, 3], [1], [2]]`` is in range."""
    return [Finding(f"P{i}", f"finding {i}", Severity.BLOCKER, f"lens{i}", "a.py", i * 20)
            for i in range(4)]


def test_an_echoed_cluster_example_does_not_partition_the_findings(session):
    """REGRESSION for #61: the shipped example is a *valid* partition.

    Unlike the findings template (#26), which is full of obvious placeholders, the
    cluster example is a plausible real answer — so a clusterer that restates it
    silently merged findings 0 and 3 that nobody grouped, in the view a human reads
    as the canonical issue list. The clusterer's own mandate says merging distinct
    issues is worse than leaving duplicates.

    The accepted cost, stated: a clusterer that genuinely decides exactly this
    grouping for exactly four findings is refused and its merge is lost. That
    failure runs in the safe direction — a duplicate survives — and it is the only
    payload the exact-match rule can refuse.
    """
    session._run_claude = _canned('```json\n{"clusters": [[0, 3], [1], [2]]}\n```')

    clusters = session.reduce(_four_findings())

    assert _members_by_title(clusters) == [("finding 0",), ("finding 1",),
                                           ("finding 2",), ("finding 3",)]


def test_a_real_grouping_survives_a_trailing_echo(session):
    """The review is recovered, not discarded — #26's rule, one module along."""
    session._run_claude = _canned(
        '```json\n{"clusters": [[0, 1], [2], [3]]}\n```\n'
        'The contract asked for:\n'
        '```json\n{"clusters": [[0, 3], [1], [2]]}\n```\n')

    clusters = session.reduce(_four_findings())

    assert _members_by_title(clusters) == [("finding 0", "finding 1"),
                                           ("finding 2",), ("finding 3",)]


def test_a_real_grouping_survives_the_whole_contract_being_restated(session):
    """The contract used to name a fenced ``json`` block inline in its own prose.

    Extraction opened on that mention and closed on the example's fence, capturing
    a fragment of prose — so a restated contract lost the grouping that preceded it
    and degraded to the identity reduce without saying so.
    """
    session._run_claude = _canned(
        '```json\n{"clusters": [[0, 1], [2], [3]]}\n```\n' + _CLUSTER_CONTRACT)

    clusters = session.reduce(_four_findings())

    assert _members_by_title(clusters) == [("finding 0", "finding 1"),
                                           ("finding 2",), ("finding 3",)]


def test_the_shipped_cluster_contract_echoed_whole_is_recognised(session):
    """Anti-drift, end to end: fed the contract we actually ship."""
    session._run_claude = _canned("Understood.\n" + _CLUSTER_CONTRACT)

    assert len(session.reduce(_four_findings())) == 4


def test_the_shipped_cluster_contract_carries_the_example_the_detector_knows():
    assert _CLUSTER_CONTRACT.count("```json") == 1
    assert _CLUSTER_TEMPLATE in _CLUSTER_CONTRACT


def test_a_grouping_that_only_resembles_the_example_is_still_used(session):
    """Exact on purpose: only the payload we shipped is ever refused.

    Rejecting anything that merely looks like the example would discard real
    groupings, which is the defect and not the fix.
    """
    session._run_claude = _canned('```json\n{"clusters": [[0, 2], [1], [3]]}\n```')

    clusters = session.reduce(_four_findings())

    assert _members_by_title(clusters) == [("finding 0", "finding 2"),
                                           ("finding 1",), ("finding 3",)]


def test_an_unfenced_echo_is_filtered_like_a_fenced_one(session):
    """The no-fence fallback is a real path, so the echo filter must cover it.

    Found by mutation, not by design: a filter applied only to fenced blocks
    passes an unfenced restatement straight through to ``json.loads``, and the
    example parses.
    """
    session._run_claude = _canned(_CLUSTER_TEMPLATE)

    assert len(session.reduce(_four_findings())) == 4
