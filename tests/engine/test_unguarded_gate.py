# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""A human gate that fails must not discard the epochs that ran (#129).

``loop.run`` injects three fallible seams and guarded two of them. ``gather`` has
been inside ``except Exception`` since #85 and ``spawn`` since #25, each with a
paragraph saying why; ``human_gate`` was called bare. So a gate that failed
propagated out of ``run`` and took the ``ReviewRun`` with it — every finding,
participation record, surface record, suppression and token count from the epochs
that **did** complete, for a panel already spawned and paid for. The artefact is
stdout-only, so nothing was readable back.

The treatment is the one the other two seams have: the seam is a fallible black
box, the failure is caught as ``Exception`` (never a named backend class, which the
engine may not import, and never ``BaseException``, so a human's Ctrl-C still stops
the loop), and it is **recorded** rather than swallowed.

**What the run then does is #132's ruling, cited and not restated**: it lives in
``GateDecision.asked``'s docstring, which already names "the seam itself failed
(#129)" as one of ``None``'s three cases. The run CONTINUES and records
``GateDecision(stop=False, asked=None)``; ``asked=False`` stays the measured
non-interactive case; there is no new ``HaltReason``, because ``halting.py``'s bar
is a state the loop cannot usefully continue from and the next epoch can still ask.

**"Fails" means five things, all measured against ``c1e0a80`` by execution.** The
gate RAISES; it returns ``None``; it returns a foreign object with no ``.stop``
(which raised identically, in the engine, and never reached the entry point); it
returns a foreign object WITH a duck-typed ``.stop`` (which the engine survived and
then stored in a field typed ``GateDecision | None``); or its ``.stop`` is a
property that raises, which propagated from the attribute read rather than from the
call. The last is why normalisation is by **type** and happens before ``.stop`` is
ever read: a guard wrapped around the call alone does not contain it.

**#85's asymmetry does not arise here, and its absence is the design.** A gather
failure with no completed epoch re-raises, because there is nothing to report. The
gate is only ever called *after* an epoch completed, so that branch has no instance
and asserting one would be a test killed by nothing.

**Which of these are regressions, measured on ``main`` with this file in place
rather than assumed.** Of the eighteen, **one** passes there —
``test_ctrl_c_at_the_gate_still_stops_the_run``, because nothing catches anything at
this seam today, so it guards the ``BaseException`` mutation rather than a
regression. Fifteen are the defect. The remaining two —
``test_a_healthy_run_never_reports_a_gate_failure`` and
``test_a_gate_that_stops_the_run_is_untouched_by_the_guard`` — go red there **only**
on the missing attribute: every other fact they assert passes before and after, so
they are the mirror image and the ordinary path rather than regressions, and they
say so on themselves. Three of these tests asserted something false until the run
was executed; what they asserted and what measurement corrected is recorded on each.
"""

from __future__ import annotations

import pytest

from kuang.engine import (Finding, GateDecision, HaltingSet, HaltReason,
                          PanelConfig, PersonaReport, PersonaStatus, ReviewSpec,
                          Severity, run)

PANEL = PanelConfig(personas=[(n, "mandate") for n in ("correctness", "adversary")])


class _BackendGateError(RuntimeError):
    """Stands in for a backend's own exception class, which the engine may not name.

    Declared here rather than imported from ``kuang.backends`` on purpose: the
    engine's guard must be stated over ``Exception``, so a class it has never heard
    of has to work. ``tests/engine/test_backend_agnostic.py`` enforces the import
    side of the same rule.
    """


def _spawn(persona, mandate, surface, epoch):  # noqa: ANN001, ARG001
    """One persona, one open BLOCKER per epoch: material work worth not discarding.

    The finding is distinct per epoch — a different ``claim_class``, so a different
    ``Finding.key`` — because an identical one deduplicates into the ledger, resets
    nothing, and STALLs the run on epoch 2. That would stop the loop before any of
    these tests reached the state they name.
    """
    return PersonaReport(
        persona=persona, verdict="NO", tokens=9,
        status=PersonaStatus.CONTRIBUTED,
        findings=[Finding(persona, f"{persona}: off-by-one at epoch {epoch}",
                          Severity.BLOCKER, f"{persona}-lens-{epoch}", "a.py", 1)])


def _raises(exc: BaseException | None = None, *, at: int = 1):
    """A ``HumanGate`` that fails from the named epoch on, as a live one would.

    ``at`` defaults to the FIRST gate rather than a later one: the triggers that
    make this issue live are properties of the environment — a closed pipe, a
    stdout that cannot encode an em dash — so a gate that fails at all usually
    fails at every epoch after. The default is the realistic case and the one the
    multiplicity test depends on.
    """
    def gate(result, run):  # noqa: ANN001, ARG001
        if result.index >= at:
            raise exc or _BackendGateError("the gate could not be reached")
        return GateDecision(stop=False, asked=True)
    return gate


def _returns(value):
    """A ``HumanGate`` that hands back exactly what a foreign seam might."""
    def gate(result, run):  # noqa: ANN001, ARG001
        return value
    return gate


def _run(human_gate, *, max_epochs: int = 3):
    return run(ReviewSpec(why="mission-critical", what="the diff"),
               HaltingSet(max_epochs=max_epochs), PANEL,
               spawn=_spawn, gather=lambda epoch: "def f():\n    return 1\n",
               parallel=False, human_gate=human_gate)


# --- the defect ---------------------------------------------------------------

def test_a_raising_gate_returns_the_epochs_that_completed():
    """REGRESSION for #129: this raised out of ``run`` and the run was lost."""
    review_run = _run(_raises(at=2))

    assert review_run.halt_reason is HaltReason.EPOCH
    assert len(review_run.epochs) == 3
    assert len(review_run.ledger) == 6, (
        "two personas across three epochs: the material work the loss discarded")


def test_the_record_names_the_failure_and_the_epoch_it_happened_on():
    """The backend's own diagnosis is kept, not swallowed.

    Both existing guards keep it — ``SurfaceFailure.detail`` at ``gather``, the
    meta finding's ``evidence`` at ``spawn`` — on the ground that it is what an
    operator needs and what the engine could never produce. A gate that fails
    prints nothing at all, so there is no second copy anywhere.
    """
    review_run = _run(_raises(at=2))

    assert review_run.epochs[0].gate_failure is None, "epoch 1's gate answered"
    assert review_run.epochs[1].gate_failure is not None
    assert review_run.epochs[1].gate_failure.detail == (
        "_BackendGateError: the gate could not be reached")


def test_the_record_lives_on_the_epoch_and_does_not_restate_its_index():
    """Per epoch, and deliberately WITHOUT an ``epoch`` field of its own.

    ``SurfaceFailure`` carries one because it hangs off the run: the epoch it names
    never completed, so there is no ``EpochResult`` to hold it. This record hangs
    off the epoch that owns it, and a second copy of the index is a second source of
    truth that can disagree with the first. The artefact's row takes ``epoch`` from
    ``EpochResult.index``, which is the one that cannot drift.
    """
    failure = _run(_raises(at=2)).epochs[1].gate_failure

    assert not hasattr(failure, "epoch"), (
        "the index belongs to the EpochResult that holds this record")
    assert {f for f in vars(failure)} == {"detail", "detail_chars"}


def test_a_gate_that_fails_is_recorded_at_each_epoch_that_reached_it():
    """The reason this is per-epoch and not a run-level singleton.

    A lost surface ENDS the run, so one record says everything there is to say. A
    failed gate does not end it, and the live triggers are environmental, so a gate
    that fails once usually fails at every epoch after. A run-level field would
    report one failure and silently drop the rest — a human-gated run turned into an
    ungated one with nothing in the record saying so. **This is the test that kills
    the rejected design**, which is why it asserts the whole sequence rather than
    that any record exists.

    The third epoch is ``False`` and that is the rule rather than an off-by-one:
    ``loop.run`` breaks on its own halt check BEFORE the gate, so the halting epoch
    never reaches it and has nothing to record. The record's PRESENCE is what says
    the gate ran, exactly as #117 settled for ``EpochResult.gate`` — measured here,
    not assumed, because a sequence of ``True`` was what this test asserted until the
    run was executed.
    """
    review_run = _run(_raises())

    assert len(review_run.epochs) == 3
    assert [e.gate_failure is not None for e in review_run.epochs] == [
        True, True, False]


def test_a_multi_line_message_is_recorded_as_one_line():
    """Collapsed where it is built, as ``gather``'s is, so the channels cannot drift.

    A gate's exception text is environment-fed and has no line contract: a
    ``BrokenPipeError`` carries an errno and a message, and a backend wrapping one
    may carry a traceback's worth. Printed as-is inside the report block the
    continuation lines arrive unprefixed and read as sections of the report.
    """
    review_run = _run(_raises(_BackendGateError(
        "cannot prompt: stdout is closed\nthe run was piped into a reader\n"
        "that exited first"), at=2))

    failure = review_run.epochs[1].gate_failure
    assert "\n" not in failure.detail
    assert failure.detail == (
        "_BackendGateError: cannot prompt: stdout is closed the run was piped "
        "into a reader that exited first")


def test_an_unbounded_message_is_bounded_after_it_is_collapsed():
    """The same 400-character bound, in the same order: collapse, measure, bound.

    A bound applied before the collapse spends itself on whitespace nobody will
    ever be shown. The message carries a long whitespace run so the ORDER can be
    seen at all — the matrix only tests the axis the input can express, which is
    the trap ``test_lost_surface.py`` recorded one seam along.
    """
    review_run = _run(_raises(_BackendGateError("a" + " " * 500 + "b" * 500), at=2))

    detail = review_run.epochs[1].gate_failure.detail
    kept, marker = detail.split("…", 1)
    assert len(kept) == 400
    assert marker.startswith(" [truncated:"), "the cut says so (#111)"
    assert kept.startswith("_BackendGateError: a b"), "one space, not five hundred"
    assert kept.endswith("b"), "the bound spent on diagnosis, not on whitespace"


def test_a_cut_diagnosis_says_it_was_cut_and_records_what_it_was():
    """#111's marker doctrine at a third site, both halves of it.

    The marker is for the person reading it; nothing may read it back.
    ``detail_chars`` is the structural fact — the length of the collapsed diagnosis
    BEFORE bounding — so "was it cut" is ``detail_chars > 400``, exactly knowable
    and immune to any rewording of the marker.
    """
    # Single-spaced and with no trailing space, so the collapse is a no-op here and
    # ``detail_chars`` can be pinned against the message itself. The ORDER of
    # collapse and bound is the test above, which carries whitespace to see it.
    message = "the gate could not be reached: " + " ".join(
        ["unreachable terminal."] * 30)
    review_run = _run(_raises(_BackendGateError(message), at=2))

    failure = review_run.epochs[1].gate_failure
    assert failure.detail_chars == len(f"_BackendGateError: {message}"), (
        "the length of what there was to say, not of what was kept")
    assert failure.detail_chars > 400
    assert "truncated" in failure.detail


def test_a_short_diagnosis_carries_no_marker_and_says_its_own_length():
    """The MIRROR IMAGE, and the half that stops the bound passing vacuously.

    "A cut diagnosis says it was cut" is satisfied by marking every diagnosis,
    which would be this defect inverted — the record claiming a loss that never
    happened. So the pair: an uncut ``detail`` is byte-identical to the collapsed
    message, and ``detail_chars`` agrees with it rather than being a constant.
    """
    failure = _run(_raises(at=2)).epochs[1].gate_failure

    expected = "_BackendGateError: the gate could not be reached"
    assert failure.detail == expected, "unchanged, not merely unmarked"
    assert failure.detail_chars == len(expected) <= 400


# --- the three non-raising failures, which an exception guard does not reach ---

def test_a_gate_that_returns_none_is_not_read_as_a_decision():
    """``AttributeError`` at the ``.stop`` read, AFTER the record was stored.

    Measured: the loop wrote ``result.gate = decision`` and then crashed reading
    ``decision.stop``, so an epoch whose gate RAN was left with no record and the
    whole run was discarded. The assignment moves below the guard for that reason:
    a stored ``None`` renders as "the epoch halted, so the gate was never reached",
    which is false for a gate that ran.
    """
    review_run = _run(_returns(None))

    assert len(review_run.epochs) == 3
    assert [e.gate_failure is not None for e in review_run.epochs] == [
        True, True, False], "the halting epoch never reached the gate"
    assert review_run.epochs[0].gate_failure.detail == (
        "gate returned NoneType, not GateDecision")


def test_a_foreign_object_is_normalised_by_type_and_not_by_attribute():
    """The sharp one: a duck-typed ``.stop`` survives an attribute-shaped guard.

    Measured against ``c1e0a80``, an object with a ``.stop`` the loop can read is
    stored in ``EpochResult.gate`` — a field typed ``GateDecision | None`` — and
    the artefact then publishes ``reached: true`` for a value nothing validated.
    **It is not an artefact-destruction defect**: ``_asked`` resolves by identity
    and ``stopped`` is ``bool(...)``-coerced, so #131 already stopped a foreign
    value from being serialised. The harm is the unchecked claim, and ``isinstance``
    is what refuses it.
    """
    class _Ducked:
        stop = False
        asked = "yes"

    review_run = _run(_returns(_Ducked()))

    stored = review_run.epochs[0].gate
    assert isinstance(stored, GateDecision), "a value of the type the field claims"
    assert review_run.epochs[0].gate_failure.detail == (
        "gate returned _Ducked, not GateDecision")


def test_a_decision_that_raises_when_it_is_read_is_contained():
    """The fifth mode, and the one a guard around the CALL alone does not reach.

    ``.stop`` as a property that raises propagates from the attribute read, not
    from ``human_gate(...)``. Normalising by type before ``.stop`` is ever read is
    what contains it — a second and independent argument for the type check, which
    does not depend on #131's coercion staying where it is.
    """
    class _Raising:
        @property
        def stop(self):
            raise RuntimeError("the decision cannot be read")

    review_run = _run(_returns(_Raising()))

    assert len(review_run.epochs) == 3
    assert review_run.epochs[0].gate_failure.detail == (
        "gate returned _Raising, not GateDecision")


def test_the_diagnosis_for_a_wrong_type_names_the_type_and_nothing_else():
    """``repr`` of a foreign object is unbounded text from outside this codebase.

    The artefact is published, and ``detail`` goes into it. A caller's or a model's
    ``__repr__`` can return anything of any length, and bounding it would keep 400
    characters of it rather than none. The type name is the exactly-knowable fact,
    which is the rule #131 applied to ``asked`` one field along — a value the tool
    cannot read is not a measurement.
    """
    class _Loud:
        stop = False

        def __repr__(self):
            return "SECRET-" + "x" * 5000

    failure = _run(_returns(_Loud())).epochs[0].gate_failure

    assert failure.detail == "gate returned _Loud, not GateDecision"
    assert "SECRET" not in failure.detail
    assert failure.detail_chars == len(failure.detail), (
        "nothing was cut, so nothing was kept back either")


# --- what the run does next: #132's ruling, cited rather than restated ---------

def test_a_failed_gate_does_not_claim_a_human_was_asked():
    """``asked=None``, never ``False``: silence is not a measurement.

    ``False`` means the gate MEASURED that no human could be reached, which is every
    CI run. A seam that failed measured nothing. ``GateDecision.asked``'s docstring
    carries the ruling and already names "the seam itself failed (#129)" as one of
    ``None``'s three cases, so the artefact's vocabulary needs nothing new here.
    """
    decision = _run(_raises(at=2)).epochs[1].gate

    assert decision.stop is False
    assert decision.asked is None, "the gate resolved no answer; it did not measure one"


def test_a_failed_gate_does_not_stop_the_run():
    """No new ``HaltReason``, and ``ABORTED`` would be the lie it already is.

    ``halting.py``'s bar for a new member is a state the loop cannot usefully
    CONTINUE from, and the next epoch can still ask. ``ABORTED`` means the gate
    stopped the loop; recording it here would say a human stopped a run nobody
    stopped — which is the claim #138 is open on independently of this issue.
    """
    review_run = _run(_raises())

    assert review_run.halt_reason is HaltReason.EPOCH
    assert review_run.halt_reason is not HaltReason.ABORTED


def test_the_record_is_present_wherever_the_gate_was_reached():
    """``gate`` and ``halt`` keep the equivalence an ordinary run has.

    ``test_gate_decisions.py`` asserts ``(gate is None) == (halt is not None)`` on
    every epoch of an ordinary run: the loop breaks before the gate when an epoch
    halts, so an absent record means the gate was never reached. Substituting a
    decision keeps that true. Leaving ``gate`` unset on a failure would break it,
    and would publish ``reached: false`` for a gate that was reached and ran.
    """
    review_run = _run(_raises())

    for epoch in review_run.epochs:
        assert (epoch.gate is None) == (epoch.halt is not None)


# --- the boundary: what this change deliberately does NOT touch ----------------

def test_ctrl_c_at_the_gate_still_stops_the_run():
    """GUARD: ``Exception``, never ``BaseException`` — #25's exclusion, restated.

    The gate prompt is where an operator is most likely to press Ctrl-C, and it is
    the one moment it must be reliable: swallowing it into a recorded failure would
    make the tool continue a run a human had stopped.

    **Green on main, so a guard and not a regression.** Nothing catches anything at
    this seam today, so a ``KeyboardInterrupt`` propagates there for the trivial
    reason; it passes here because the guard that now exists excludes it. It guards
    the mutation ``Exception`` → ``BaseException``, which is the design rejected.
    """
    with pytest.raises(KeyboardInterrupt):
        _run(_raises(KeyboardInterrupt(), at=2))


def test_a_healthy_run_never_reports_a_gate_failure():
    """The MIRROR IMAGE: a rule that fires on a healthy run is the defect inverted.

    "A failed gate is recorded" is satisfied by recording one every epoch, which
    would turn every run into a degraded one. This is the partner that forbids it,
    and it is the vacuity probe's target.

    **Red on main only for the missing attribute.** What it pins — that an answering
    gate's run halts on its ceiling with a decision stored on each epoch that reached
    the gate — passes before and after. It is the mirror image rather than a
    regression, and it is honest to say so rather than count it in the matrix as one.
    """
    def _answers(result, run):  # noqa: ANN001, ARG001
        return GateDecision(stop=False, asked=True)

    review_run = _run(_answers)

    assert review_run.halt_reason is HaltReason.EPOCH
    assert len(review_run.epochs) == 3
    assert all(e.gate_failure is None for e in review_run.epochs)
    assert [e.gate is not None for e in review_run.epochs] == [True, True, False]
    assert all(e.gate.asked is True for e in review_run.epochs[:2])


def test_a_gate_that_stops_the_run_is_untouched_by_the_guard():
    """GUARD: the ordinary path still halts ``ABORTED`` on the epoch it stopped.

    The guard sits between the call and the ``stop`` read, so the one behaviour a
    mistake here would break is the gate's actual purpose. It pins what #117
    settled; it is here because the normalisation is new code on that path, not
    because anything about stopping has changed.

    **Red on main only for the missing attribute, measured rather than asserted.**
    Both of the facts it pins — ``ABORTED`` on one epoch — pass before and after;
    only ``gate_failure is None`` fails there, for want of the field. This docstring
    said "green on main" until the run was executed.
    """
    review_run = _run(_returns(GateDecision(stop=True, asked=True)))

    assert review_run.halt_reason is HaltReason.ABORTED
    assert len(review_run.epochs) == 1
    assert review_run.epochs[0].gate_failure is None
