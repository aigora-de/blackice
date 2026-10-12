# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""No injected callable can discard a run in which a persona has been spawned (#148).

``kuang.engine.run`` takes nine injected callables. #85, #25 and #129 guarded three of
them (``gather``, ``spawn``, the gate). The other six were called bare: measured on
``0fd8ebb``, a raise from ``adjudicate``, ``reduce``, ``budget_spent``,
``scope_complete`` or ``checkpoint`` escaped ``run`` after the panel had been paid for,
and ``clock`` escaped on every call after the first. A hostile RETURN value did the same
through a second door: ``budget_spent`` returning ``"x"``, ``clock`` returning
``None``, a ``scope_complete`` whose truth test raises, and a ``reduce`` returning
``None`` or a foreign object each raised a moment later, in the engine's own code.

**The rule, stated once:** a seam that fails yields UNKNOWN, never a value. The engine
records it at the source, as a ``SeamFailure`` on the epoch, and never takes an action
whose justification needs the unknown input. Every outcome falls out of it:

* ``adjudicate`` -> the finding stays ``verified=None``: unadjudicated, so still open,
  so still counted by the breaker. ``False`` would need a refutation.
* ``reduce`` -> the identity reduce, the state before the call.
* ``checkpoint`` -> nothing persisted, and the record says so.
* ``scope_complete`` -> it cannot support ``CONVERGED``; inert where not required.
* a DISARMED ceiling's input -> recorded, and the run continues; nothing needed it.
* an ARMED ceiling's input -> neither ``BUDGET`` ("reached") nor continuing ("not
  reached") is justified, so the run halts ``UNMEASURED`` — the maintainer's ruling of
  2026-10-12, recorded on #148. Evaluated last, so any exactly-knowable reason wins,
  and always behind the breaker, which reads no injected input.

The first ``clock()`` is NOT guarded: it precedes any spend, so a raise there loses
nothing (#18's doctrine, as ``gather`` applies it).

**Which are regressions, measured on main with this file in place.** This module
imports ``SeamFailure``, which main does not have, so it was measured there with a
minimal import shim and ``HaltReason.UNMEASURED`` absent. The matrix is in the PR body.
Tests green on main say so in their own docstrings.

Stdlib + stub seams only: no network, no ``claude`` subprocess.
"""

from __future__ import annotations

import pytest

from kuang.engine import (DIAGNOSIS_BOUND, Cluster, Finding, HaltingSet, HaltReason,
                          PanelConfig, PersonaReport, PersonaStatus, ReviewSpec,
                          SeamFailure, Severity, run)
from kuang.engine.halting import _evaluate_halt

PANEL = PanelConfig(personas=[("first", "mandate"), ("second", "mandate")])


class _SeamError(RuntimeError):
    """Stands in for a backend's own exception class, which the engine may not name."""


class _BadBool:
    """A value whose truth test raises: what ``bool(...)`` must meet INSIDE a guard."""

    def __bool__(self):
        raise _SeamError("truth test exploded")


class _NothingRenders(Exception):
    """Neither ``str`` nor ``repr`` renders: the guard must still not be ended by it."""

    def __str__(self):
        raise ValueError("str exploded")

    def __repr__(self):
        raise ValueError("repr exploded")


def _contained(call):
    """Call ``call()``; an ``Exception`` escaping it is a plain test failure.

    ``tests/engine/test_unrenderable_exception.py``'s pattern: an escaping hostile
    exception otherwise crashes pytest's own report with an INTERNALERROR.
    """
    try:
        return call()
    except Exception as exc:  # noqa: BLE001
        name = str.__str__(type.__dict__["__name__"].__get__(type(exc), type))
        raise AssertionError(f"escaped the guard: {name}") from None


def _raises(exc=None):
    def seam(*a, **k):  # noqa: ARG001
        raise exc if exc is not None else _SeamError("the seam failed")
    return seam


def _returns(value):
    return lambda *a, **k: value  # noqa: ARG005


def _spawn_for(severity: Severity, *, calls: list | None = None, verdict: str = "NO"):
    """Each persona raises one finding per epoch, distinct per epoch so nothing stalls.

    ``severity`` is the axis: ``adjudicate`` is only REACHED for a finding at BLOCKER or
    above with ``verified is None``, so a fixture that emits NON_BLOCKING cannot express
    it and a 0-red mutation row would be meaningless.
    """
    def spawn(persona, mandate, surface, epoch):  # noqa: ARG001
        if calls is not None:
            calls.append((persona, epoch))
        findings = [] if severity is None else [
            Finding(persona, f"{persona}: defect at epoch {epoch}", severity,
                    f"{persona}-lens-{epoch}", "a.py", epoch * 100)]
        return PersonaReport(persona=persona, verdict=verdict, tokens=9,
                             status=PersonaStatus.CONTRIBUTED, findings=findings)
    return spawn


def _run(*, severity=Severity.BLOCKER, halting=None, verdict="NO", calls=None, **seams):
    halting = halting if halting is not None else HaltingSet(max_epochs=2, stall_patience=5)
    return _contained(lambda: run(
        ReviewSpec(why="w", what="x"), halting, PANEL,
        spawn=_spawn_for(severity, calls=calls, verdict=verdict),
        gather=lambda epoch: "def f():\n    return 1\n", parallel=False, **seams))


def _seams(review_run) -> list[list[str]]:
    """Which seams failed, per epoch, in order."""
    return [[f.seam for f in e.seam_failures] for e in review_run.epochs]


def _clock_failing_on(n: int, exc=None):
    count = {"n": 0}

    def clock():
        count["n"] += 1
        if count["n"] == n:
            raise exc if exc is not None else _SeamError("clock")
        return 0.0
    return clock


# --- the defect: each callable raises, and the run survives -----------------------

def test_a_raising_adjudicate_leaves_the_findings_unadjudicated_and_open():
    """REGRESSION for #148: escaped ``run`` after two paid spawns on main."""
    review_run = _run(adjudicate=_raises())

    assert review_run.halt_reason is HaltReason.EPOCH
    assert len(review_run.ledger) == 4
    assert all(f.verified is None for f in review_run.ledger.values())
    assert all(f.counts_open for f in review_run.ledger.values())
    assert _seams(review_run) == [["adjudicate", "adjudicate"]] * 2


def test_a_raising_adjudicate_still_trips_the_breaker():
    """REGRESSION for #148, and the ruin row: ``verified=None``, never ``False``.

    ``False`` would DROP the finding, and on an UGLY that silences the breaker. Kills
    the rejected design that substitutes ``False`` for an unknown verdict.
    """
    review_run = _run(severity=Severity.UGLY, adjudicate=_raises())

    assert review_run.halt_reason is HaltReason.ESCALATE_UGLY
    assert len(review_run.open_uglies) == 2


def test_adjudicate_is_contained_per_call_not_per_epoch():
    """REGRESSION for #148: one finding's failure must not leave the rest unadjudicated.

    The first call raises and the second refutes. A guard around the whole loop would
    leave the second finding open; per call, it is refuted and dropped.
    """
    answers = iter([_SeamError("first"), False, True, True])

    def adjudicate(finding, surface):  # noqa: ARG001
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return answer

    review_run = _run(adjudicate=adjudicate)

    epoch_one = review_run.epochs[0].reports
    assert [f.verified for r in epoch_one for f in r.findings] == [None, False]
    assert _seams(review_run) == [["adjudicate"], []]


def test_a_raising_reduce_falls_back_to_the_identity_reduce():
    """REGRESSION for #148: escaped ``run`` on main."""
    review_run = _run(reduce=_raises())

    assert review_run.halt_reason is HaltReason.EPOCH
    assert [c.members for c in review_run.clusters] == [
        (f,) for f in review_run.ledger.values()]
    assert _seams(review_run) == [["reduce"], ["reduce"]]


def test_a_raising_reduce_still_trips_the_breaker():
    """REGRESSION for #148: the identity reduce keeps every UGLY a cluster."""
    review_run = _run(severity=Severity.UGLY, reduce=_raises())

    assert review_run.halt_reason is HaltReason.ESCALATE_UGLY


def test_a_raising_checkpoint_is_recorded_and_the_run_continues():
    """REGRESSION for #148: escaped ``run`` on main. #31 will make this live."""
    review_run = _run(checkpoint=_raises())

    assert review_run.halt_reason is HaltReason.EPOCH
    assert len(review_run.epochs) == 2
    assert _seams(review_run) == [["checkpoint"], ["checkpoint"]]


def test_a_raising_checkpoint_does_not_mask_the_breaker():
    """REGRESSION for #148: an open UGLY was lost with the run on main."""
    review_run = _run(severity=Severity.UGLY, checkpoint=_raises())

    assert review_run.halt_reason is HaltReason.ESCALATE_UGLY
    assert _seams(review_run) == [["checkpoint"]]


# --- the halting inputs: disarmed predicates ----------------------------------------

def test_a_raising_budget_spent_with_no_token_ceiling_is_recorded_and_continues():
    """REGRESSION for #148: escaped on main, though the value could not change the halt."""
    review_run = _run(budget_spent=_raises())

    assert review_run.halt_reason is HaltReason.EPOCH
    assert _seams(review_run) == [["budget_spent"], ["budget_spent"]]


def test_a_raising_scope_complete_that_is_not_required_is_recorded_and_continues():
    """REGRESSION for #148: what the CLI passes (``require_scope_complete=False``)."""
    review_run = _run(halting=HaltingSet(max_epochs=2, stall_patience=5,
                                         require_scope_complete=False),
                      scope_complete=_raises())

    assert review_run.halt_reason is HaltReason.EPOCH
    assert _seams(review_run) == [["scope_complete"], ["scope_complete"]]


def test_an_unknown_scope_cannot_support_convergence():
    """REGRESSION for #148: escaped on main.

    A clean panel that would converge on epoch 1 (no findings, both vote YES) cannot,
    because CONVERGED needs scope KNOWN complete. It runs to the epoch ceiling, which
    claims nothing it did not measure.
    """
    healthy = _run(severity=None, verdict="YES")
    unknown = _run(severity=None, verdict="YES", scope_complete=_raises())

    assert healthy.halt_reason is HaltReason.CONVERGED
    assert unknown.halt_reason is HaltReason.EPOCH
    assert _seams(unknown) == [["scope_complete"], ["scope_complete"]]


def test_a_raising_clock_after_the_start_with_no_time_ceiling_continues():
    """REGRESSION for #148: the second and third calls escaped on main, after 2 and
    4 paid spawns."""
    for n in (2, 3):
        review_run = _run(clock=_clock_failing_on(n))

        assert review_run.halt_reason is HaltReason.EPOCH
        assert _seams(review_run)[n - 2] == ["clock"]


def test_a_disarmed_input_is_still_called_every_epoch():
    """GUARD, green on main: a healthy run makes exactly the calls it made before.

    The alternative — not calling a disarmed input at all — would change what a
    healthy run calls, which is the mirror image of the defect.
    """
    calls = []
    _run(budget_spent=lambda: calls.append("b") or 0)

    assert calls == ["b", "b"]


# --- the halting inputs: armed ceilings (the ruling) ------------------------------

def test_an_unreadable_armed_token_ceiling_halts_unmeasured():
    """REGRESSION for #148, and the ruling: neither BUDGET nor continuing is justified."""
    review_run = _run(halting=HaltingSet(token_budget=100, max_epochs=3, stall_patience=5),
                      budget_spent=_raises())

    assert review_run.halt_reason is HaltReason.UNMEASURED
    assert len(review_run.epochs) == 1
    assert _seams(review_run) == [["budget_spent"]]


def test_an_unreadable_armed_time_ceiling_halts_unmeasured():
    """REGRESSION for #148: the time ceiling is the same rule, not a second one."""
    review_run = _run(halting=HaltingSet(time_budget_s=60, max_epochs=3, stall_patience=5),
                      clock=_clock_failing_on(2))

    assert review_run.halt_reason is HaltReason.UNMEASURED
    assert _seams(review_run) == [["clock"]]


def test_a_knowable_halt_wins_over_an_unknown_ceiling():
    """REGRESSION for #148: the epoch ceiling is exactly knowable, so it is the reason.

    Kills a mutation that evaluates ``UNMEASURED`` ahead of the ceilings.
    """
    review_run = _run(halting=HaltingSet(token_budget=100, max_epochs=1, stall_patience=5),
                      budget_spent=_raises())

    assert review_run.halt_reason is HaltReason.EPOCH


@pytest.mark.parametrize("seam", ["budget_spent", "clock", "scope_complete", "checkpoint"])
def test_the_breaker_fires_whatever_halting_input_fails(seam):
    """REGRESSION for #148, one per halting input: on main the UGLY was lost with the run.

    The breaker reads ``result.open_uglies`` and nothing injected, so no failed input
    can mask it — including an ARMED ceiling, whose ``UNMEASURED`` sits behind it.
    """
    seams = {"clock": _clock_failing_on(2)} if seam == "clock" else {seam: _raises()}
    review_run = _run(severity=Severity.UGLY,
                      halting=HaltingSet(token_budget=100, time_budget_s=60,
                                         max_epochs=3, stall_patience=5),
                      **seams)

    assert review_run.halt_reason is HaltReason.ESCALATE_UGLY
    assert _seams(review_run) == [[seam]]


# --- the predicate itself ---------------------------------------------------------

def _halt(**over):
    from kuang.engine import EpochResult, ReviewRun
    result = EpochResult(index=1, reports=[], new_findings=[],
                         open_blockers=over.pop("open_blockers", 1),
                         open_uglies=over.pop("open_uglies", 0))
    kwargs = dict(epochs_done=1, stall_epochs=0, tokens_spent=0, elapsed_s=0.0,
                  scope_complete=True, quorum_met=False, no_persona_reviewed=False)
    halting = over.pop("halting", HaltingSet(token_budget=100, time_budget_s=60,
                                             max_epochs=3))
    kwargs.update(over)
    return _evaluate_halt(result, ReviewRun(), halting, **kwargs)


def test_the_predicate_reads_an_unknown_input_as_unknown():
    """REGRESSION for #148: the pure predicate states the ruling for both ceilings."""
    assert _halt(tokens_spent=None) is HaltReason.UNMEASURED
    assert _halt(elapsed_s=None) is HaltReason.UNMEASURED
    assert _halt(tokens_spent=None, halting=HaltingSet(max_epochs=3)) is None
    assert _halt(elapsed_s=None, halting=HaltingSet(max_epochs=3)) is None
    assert _halt(tokens_spent=None, elapsed_s=60.0) is HaltReason.BUDGET
    assert _halt(tokens_spent=None, open_uglies=1) is HaltReason.ESCALATE_UGLY
    assert _halt(tokens_spent=None, no_persona_reviewed=True) is HaltReason.NO_REVIEW
    assert _halt(tokens_spent=None, open_blockers=0, quorum_met=True) is HaltReason.CONVERGED
    assert _halt(tokens_spent=None, epochs_done=3) is HaltReason.EPOCH
    assert _halt(tokens_spent=None, stall_epochs=1) is HaltReason.STALL


def test_a_healthy_predicate_never_halts_unmeasured():
    """The pair. Red on main only because the member does not exist (measured).

    ``UNMEASURED`` must be unreachable with every input known, or it is a rule firing
    on a healthy run — the vacuity probe's row.
    """
    for tokens in (0, 99, 100, 10**9):
        for elapsed in (0.0, 59.0, 60.0):
            assert _halt(tokens_spent=tokens, elapsed_s=elapsed) is not HaltReason.UNMEASURED


# --- the second door: hostile return values ---------------------------------------

@pytest.mark.parametrize("value", ["x", None, 1.5, _BadBool()],
                         ids=["str", "none", "float", "bad-bool"])
def test_a_budget_counter_that_returns_no_integer_is_unknown(value):
    """REGRESSION for #148: ``"x"`` against an armed ceiling raised TypeError on main.

    ``operator.index`` inside the guard: an ``int`` (or ``bool``, an ``int``) passes
    exactly, and anything that is not one is a failed read rather than a measurement.
    """
    review_run = _run(halting=HaltingSet(token_budget=100, max_epochs=3, stall_patience=5),
                      budget_spent=_returns(value))

    assert review_run.halt_reason is HaltReason.UNMEASURED
    assert _seams(review_run) == [["budget_spent"]]


def test_a_clock_that_returns_none_is_unknown():
    """REGRESSION for #148: the subtraction raised TypeError on main."""
    values = iter([0.0, None, None, None])
    review_run = _run(halting=HaltingSet(time_budget_s=60, max_epochs=3, stall_patience=5),
                      clock=lambda: next(values))

    assert review_run.halt_reason is HaltReason.UNMEASURED


def test_a_scope_whose_truth_test_raises_is_unknown():
    """REGRESSION for #148: the ``or`` in the predicate raised on main."""
    review_run = _run(scope_complete=_returns(_BadBool()))

    assert review_run.halt_reason is HaltReason.EPOCH
    assert _seams(review_run) == [["scope_complete"], ["scope_complete"]]


@pytest.mark.parametrize("value", [None, [object()], [Cluster(members=(
    Finding("stranger", "not in the ledger", Severity.BLOCKER, "x", "z.py", 1),))]],
    ids=["none", "foreign-object", "member-not-in-ledger"])
def test_a_reduce_that_returns_no_usable_clusters_falls_back(value):
    """REGRESSION for #148: ``None`` and a foreign object raised on main.

    The engine rebuilds every cluster from the ledger's OWN findings by key, inside the
    guard, so a member the ledger does not hold is a failed reduce, not a phantom
    finding. Whether a reduce that DROPS a ledger finding is a failure is #150's to
    rule, not this guard's.
    """
    review_run = _run(reduce=_returns(value))

    assert [c.members for c in review_run.clusters] == [
        (f,) for f in review_run.ledger.values()]
    assert _seams(review_run) == [["reduce"], ["reduce"]]


def test_an_adjudication_whose_truth_test_raises_stays_unadjudicated():
    """REGRESSION for #148: on main the raw object was STORED as ``Finding.verified``.

    It survived only because nothing had read it yet: ``counts_open`` is
    ``verified is not False``, an identity test, so the object counted open by accident.
    """
    review_run = _run(adjudicate=_returns(_BadBool()))

    assert all(f.verified is None for f in review_run.ledger.values())
    assert _seams(review_run) == [["adjudicate", "adjudicate"]] * 2


# --- the record ---------------------------------------------------------------------

def test_the_record_carries_the_rendered_bounded_diagnosis():
    """REGRESSION for #148: collapse, then measure, then bound, as ``SurfaceFailure``.

    Rendered through #142's helper, so the backend's own words survive.
    """
    message = "line one\n  line two " + "x" * (DIAGNOSIS_BOUND * 2)
    review_run = _run(checkpoint=_raises(_SeamError(message)))
    failure = review_run.epochs[0].seam_failures[0]

    collapsed = " ".join(f"_SeamError: {message}".split())
    assert failure == SeamFailure("checkpoint", failure.detail, len(collapsed))
    assert failure.detail.startswith("_SeamError: line one line two x")
    assert len(failure.detail) < len(collapsed)
    assert failure.detail_chars > DIAGNOSIS_BOUND


def test_an_exception_that_cannot_be_rendered_does_not_end_the_guard():
    """REGRESSION for #148: #142's rule at the new guards."""
    review_run = _run(checkpoint=_raises(_NothingRenders()))

    assert review_run.halt_reason is HaltReason.EPOCH
    assert review_run.epochs[0].seam_failures[0].detail


def test_a_healthy_run_records_no_seam_failure_and_halts_as_before():
    """The pair: a rule that fired on a healthy run would be the defect's mirror image.

    Red on main only on the missing attribute: the halt and the ledger it asserts are
    what main already produces.
    """
    for severity in (Severity.BLOCKER, Severity.UGLY, None):
        review_run = _run(severity=severity, verdict="YES" if severity is None else "NO",
                          checkpoint=lambda r: None)

        assert all(e.seam_failures == [] for e in review_run.epochs)
    assert _run().halt_reason is HaltReason.EPOCH
    assert _run(severity=Severity.UGLY).halt_reason is HaltReason.ESCALATE_UGLY
    assert _run(severity=None, verdict="YES").halt_reason is HaltReason.CONVERGED


# --- what must still escape ---------------------------------------------------------

def test_the_first_clock_call_still_propagates():
    """GUARD, green on main: it precedes any spend, so there is nothing to keep.

    Deliberately unguarded (#18's doctrine, as ``gather`` applies it with no completed
    epoch). Guarding it for symmetry would fabricate a run nothing happened in.
    """
    calls = []
    with pytest.raises(_SeamError):
        run(ReviewSpec(why="w", what="x"), HaltingSet(max_epochs=2), PANEL,
            spawn=_spawn_for(Severity.BLOCKER, calls=calls),
            gather=lambda epoch: "s", parallel=False, clock=_clock_failing_on(1))
    assert calls == []


@pytest.mark.parametrize("seam", ["adjudicate", "reduce", "budget_spent", "clock",
                                  "scope_complete", "checkpoint"])
def test_ctrl_c_from_any_guarded_seam_still_stops_the_run(seam):
    """GUARD, green on main: ``Exception``, never ``BaseException``, at every new guard.

    Green on main because nothing caught anything. It kills the mutation that widens
    any one guard to ``BaseException``.
    """
    seams = ({"clock": _clock_failing_on(2, KeyboardInterrupt())} if seam == "clock"
             else {seam: _raises(KeyboardInterrupt())})
    with pytest.raises(KeyboardInterrupt):
        run(ReviewSpec(why="w", what="x"), HaltingSet(max_epochs=2), PANEL,
            spawn=_spawn_for(Severity.BLOCKER), gather=lambda epoch: "s",
            parallel=False, **seams)
