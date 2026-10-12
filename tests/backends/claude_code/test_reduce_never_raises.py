# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""``PanelSession.reduce`` says "Never raises", and it must be true (#149).

Measured on ``main`` at ``0fd8ebb``: the docstring promised that every failure falls
back to the identity reduce, and three inputs raised straight through it instead.

* A clusterer reply nested 2,000 deep raised ``RecursionError`` out of ``json.loads``.
  The reply is model-authored, after a prompt built from findings that reviewers
  wrote while reading the surface under review, so the input is steerable from the
  reviewed code.
* A digit string over 4,300 characters raised ``ValueError``, Python's integer-string
  limit. ``JSONDecodeError`` is a ``ValueError`` subclass, which is why the existing
  ``except (json.JSONDecodeError, TypeError)`` looked complete and was not.
* A clusterer CALL that raised (``FileNotFoundError`` for a ``claude`` binary that is
  not there) propagated too.

Each of those raised **before** ``reduce_states.append``, so the record was left one
entry short. The CLI numbers that list by position (``epoch: i + 1``), so a short list
would report every later epoch's state against the wrong epoch. Today the run dies
first, because the engine calls ``reduce`` bare (#148); once #148 contains it, the
misalignment would be silent. That is why #149 lands first.

**The rule, stated once:** every call records exactly one state and returns a
partition, and no path raises. Two guards, each naming the stage that failed, which
is the vocabulary ``ReduceState`` already has. A call that raised is ``CALL_FAILED``.
A reply whose JSON could not be read is ``NO_GROUPING``, the state a truncated reply
already reached.

**Which are regressions, measured on main with this file in place.** Recorded in the
PR body. The ones green on main say so in their own docstrings: they guard the
``BaseException`` mutation and the healthy path, not the defect.

No network, no subprocess: ``_run_claude`` is replaced, as in ``test_cluster.py``.
"""

from __future__ import annotations

import pytest

from kuang.backends.claude_code import PanelSession
from kuang.backends.claude_code import cluster as cluster_module
from kuang.backends.claude_code.cluster import ReduceState
from kuang.backends.claude_code.spawn import CallResult
from kuang.engine import Finding, ReviewSpec, Severity

NESTED = "[" * 2000 + "]" * 2000
DIGITS = "[[" + "1" * 5000 + "]]"


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
    return lambda prompt, mandate, tools, model: CallResult(text, toks, err)


def _raising(exc):
    def call(prompt, mandate, tools, model):  # noqa: ARG001
        raise exc
    return call


def _is_identity(clusters, findings) -> bool:
    """One cluster per finding, each finding exactly once, in order."""
    return [c.members for c in clusters] == [(f,) for f in findings]


def _reduce(session, findings=None):
    """Call ``reduce``; an ``Exception`` escaping it is a plain test failure.

    ``_contained``'s pattern (``tests/engine/test_unrenderable_exception.py``),
    reduced to what this seam needs: a ``RecursionError`` escaping into pytest's own
    report is noisy enough to hide which test raised it.
    """
    findings = _findings() if findings is None else findings
    try:
        return session.reduce(findings)
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(f"reduce raised {type(exc).__name__}") from None


# --- the defect: hostile reply text --------------------------------------------

@pytest.mark.parametrize("text", [
    NESTED,
    f"```json\n{NESTED}\n```",
    '{"clusters": ' + NESTED + "}",
], ids=["unfenced", "fenced", "inside-the-key"])
def test_a_deeply_nested_reply_falls_back_and_says_so(session, text):
    """REGRESSION for #149: ``RecursionError`` escaped ``reduce`` on main."""
    session._run_claude = _canned(text)

    clusters = _reduce(session)

    assert _is_identity(clusters, _findings())
    assert session.reduce_states == [ReduceState.NO_GROUPING]


def test_an_over_long_integer_falls_back_and_says_so(session):
    """REGRESSION for #149: ``ValueError`` (the integer-string limit) escaped on main.

    Kills the mutation that adds only ``RecursionError`` to the old catch, which is
    the patched table: the catalogue would have been complete for the first input
    measured and wrong for the second.
    """
    session._run_claude = _canned(f"```json\n{DIGITS}\n```")

    clusters = _reduce(session)

    assert _is_identity(clusters, _findings())
    assert session.reduce_states == [ReduceState.NO_GROUPING]


def test_any_failure_to_read_the_reply_is_no_grouping_not_only_the_measured_ones(
        session, monkeypatch):
    """REGRESSION for #149, and the test that pins the RULE rather than a list.

    The reply is untrusted text, so whatever its parse raises means no grouping
    could be read. A class nobody has measured yet (``MemoryError`` on a large
    enough reply is one) must land in the same state. Without this test, a catch
    naming exactly ``ValueError`` and ``RecursionError`` passes every other test
    here and is still the catalogue the issue refuses.
    """
    class _Unforeseen(Exception):
        pass

    def _loads(*a, **k):
        raise _Unforeseen("a parse failure nobody listed")

    monkeypatch.setattr(cluster_module.json, "loads", _loads)
    session._run_claude = _canned('{"clusters": [[0, 1], [2]]}')

    clusters = _reduce(session)

    assert _is_identity(clusters, _findings())
    assert session.reduce_states == [ReduceState.NO_GROUPING]


# --- the defect: a call that raises --------------------------------------------

def test_a_call_that_raises_is_a_failed_call(session):
    """REGRESSION for #149: the raise escaped ``reduce`` on main.

    ``CALL_FAILED`` because that is the stage that failed, and it is what the
    docstring already promised: "a call error ... falls back". No tokens are counted,
    because no call result exists to count them from, which is ``run_claude``'s own
    rule for a failed call.
    """
    session._run_claude = _raising(FileNotFoundError("no claude binary here"))

    clusters = _reduce(session)

    assert _is_identity(clusters, _findings())
    assert session.reduce_states == [ReduceState.CALL_FAILED]
    assert session.tokens == 0


# --- the record: one state per call, on every path -----------------------------

def test_the_record_has_exactly_one_state_per_call_on_every_path(session):
    """REGRESSION for #149: the property the CLI's positional numbering depends on.

    Every path ``reduce`` has, in one session, in the order an epoch sequence would
    meet them. The record must be exactly as long as the number of calls, and each
    entry must be the one for ITS call: a list that is merely the right length could
    still be shifted by one.
    """
    replies = [
        ('{"clusters": [[0, 1], [2]]}', ReduceState.RAN),
        (NESTED, ReduceState.NO_GROUPING),
        (DIGITS, ReduceState.NO_GROUPING),
        (FileNotFoundError("gone"), ReduceState.CALL_FAILED),
        ("", ReduceState.NO_GROUPING),
        ('{"clusters": [[0], [1], [2]]}', ReduceState.RAN),
    ]
    for reply, _ in replies:
        session._run_claude = (_raising(reply) if isinstance(reply, Exception)
                               else _canned(reply))
        _reduce(session)

    assert session.reduce_states == [state for _, state in replies]


# --- what must not change --------------------------------------------------------

def test_a_healthy_grouping_is_still_used(session):
    """GUARD, green on main: the mirror image. A reply that parses is untouched.

    A guard that fired on a healthy reply would be the defect's mirror image: every
    run would silently lose its semantic reduce while reporting a degradation that
    did not happen.
    """
    session._run_claude = _canned('```json\n{"clusters": [[0, 1], [2]]}\n```')

    clusters = _reduce(session)

    assert sorted(len(c.members) for c in clusters) == [1, 2]
    assert session.reduce_states == [ReduceState.RAN]


@pytest.mark.parametrize("where", ["call", "parse"])
def test_ctrl_c_still_escapes_the_reduce(session, monkeypatch, where):
    """GUARD, green on main: ``Exception``, never ``BaseException``.

    Green on main because nothing caught anything. It guards the mutation that
    widens either new catch to ``BaseException``, which would turn an operator's
    Ctrl-C during the clusterer call into a degraded reduce and carry on.
    """
    if where == "call":
        session._run_claude = _raising(KeyboardInterrupt())
    else:
        def _loads(*a, **k):
            raise KeyboardInterrupt
        monkeypatch.setattr(cluster_module.json, "loads", _loads)
        session._run_claude = _canned('{"clusters": [[0, 1], [2]]}')

    with pytest.raises(KeyboardInterrupt):
        session.reduce(_findings())
