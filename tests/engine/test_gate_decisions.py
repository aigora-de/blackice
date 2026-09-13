# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""A run must record what happened at the human gate (#117).

Channel 9 of #40's table, and the only one reported by nothing at all: a run
could not say whether the gate was reached, what the operator chose, or when.
``HaltReason.ABORTED`` is set at exactly one place (``loop.run``) and had **zero**
test coverage — the only halt reason exercised nowhere.

The seam half lives here. What the loop records is the decision itself, set where
the gate returns rather than derived afterwards: the gate receives the
``EpochResult`` **mutably**, so a reporter deriving "was the gate reached?" from
``EpochResult.halt`` reads a value the seam can rewrite — which
``test_a_gate_that_rewrites_the_halt_cannot_unsay_that_it_ran`` demonstrates is
reachable rather than theoretical. "Set at the source" (#30, #119) over "derive
when derivable" (#103), on the boundary #119 drew: derive only where no seam can
invalidate the derivation.

Stdlib + ``FakeEnsemble`` only — no network, no ``claude`` subprocess.
"""

from __future__ import annotations

import dataclasses

import pytest

from kuang.engine import (Finding, GateDecision, HaltingSet, HaltReason,
                          PanelConfig, PersonaReport, ReviewSpec, Severity, run)
from kuang.engine.fakes import FakeEnsemble


def _f(title, *, claim_class="correctness", persona="A"):
    return Finding(persona=persona, title=title, severity=Severity.BLOCKER,
                   claim_class=claim_class, file="a.py", line=12)


def _report(persona, *findings, verdict="NO") -> PersonaReport:
    return PersonaReport(persona=persona, verdict=verdict, findings=list(findings))


def _two_epochs() -> dict:
    """Epoch 1 and epoch 2 each raise a DISTINCT blocker.

    Distinct so the second epoch is genuinely new material rather than a
    re-sighting, and so a record reporting a constant cannot pass for one
    reporting each epoch.
    """
    return {1: {"A": _report("A", _f("first", claim_class="correctness-1"))},
            2: {"A": _report("A", _f("second", claim_class="correctness-2"))}}


def _panel(script, *, max_epochs=2, patience=9, **kwargs):
    return run(
        ReviewSpec(why="w", what="x"),
        HaltingSet(max_epochs=max_epochs, stall_patience=patience,
                   require_scope_complete=False),
        PanelConfig(personas=[("A", "m")]),
        spawn=FakeEnsemble(script), gather=lambda e: "s", parallel=False,
        **kwargs)


def _stopping_gate(**kwargs):
    """A gate that stops the run.

    Keyword arguments are passed through rather than defaulted, so the bare call
    constructs only what the seam carried before this issue — which is what lets
    the coverage test below be measured honestly against ``main``.
    """
    def gate(result, run):  # noqa: ANN001, ARG001
        return GateDecision(stop=True, **kwargs)
    return gate


# --- 1. the halt reason nothing has ever exercised ---------------------------

def test_a_gate_that_stops_the_run_halts_aborted():
    """COVERAGE, not regression: this passes on ``main`` and always has.

    ``loop.run`` has set ``ABORTED`` since the vocabulary was written and no test
    or capture has ever reached it — it is the one halt reason exercised nowhere,
    which is half of why #117 was filed. Recorded as the guard it is: it pins the
    path rather than a change to it, so it goes green on ``main`` by design.
    """
    review = _panel(_two_epochs(), max_epochs=3, human_gate=_stopping_gate())

    assert review.halt_reason is HaltReason.ABORTED
    assert len(review.epochs) == 1, "the run continued past the gate that stopped it"


# --- 2. what the run records about the gate ----------------------------------

def test_the_run_records_what_the_gate_decided_on_each_epoch():
    """The decision is kept on the epoch it followed, not summarised run-wide."""
    review = _panel(_two_epochs())

    assert review.epochs[0].gate is not None, "the gate ran and the run did not say so"
    assert review.epochs[0].gate.stop is False


def test_an_epoch_that_halted_never_reached_the_gate_and_says_so():
    """``loop.run`` breaks BEFORE the gate when an epoch halts (#117's §3).

    The absence is the fact: the halting epoch has no decision because no decision
    was taken, and a reader must be able to tell that from an epoch whose gate
    continued.
    """
    review = _panel(_two_epochs())

    assert review.epochs[0].gate is not None
    assert review.epochs[-1].gate is None, \
        "an epoch that halted reports a gate decision that never happened"


def test_the_record_agrees_with_the_halt_on_every_epoch_of_an_ordinary_run():
    """``gate is None`` ⟺ ``halt is not None``, for any gate that does not rewrite
    the halt.

    This is the equivalence the artefact's ``reached`` key rests on, and it is
    asserted rather than assumed. It is a property of ORDINARY runs, not an
    invariant of the loop: the one way to break it is a gate that rewrites the
    ``EpochResult`` it was handed, which the test below does deliberately.
    """
    for review in (_panel(_two_epochs()),
                   _panel(_two_epochs(), max_epochs=1),
                   _panel(_two_epochs(), max_epochs=3,
                          human_gate=_stopping_gate())):
        for e in review.epochs:
            assert (e.gate is None) == (e.halt is not None), \
                f"epoch {e.index}: gate={e.gate!r} halt={e.halt!r}"


def test_a_gate_that_rewrites_the_halt_cannot_unsay_that_it_ran():
    """Why the record is STORED rather than derived from ``halt is None``.

    The gate receives the ``EpochResult`` mutably and the loop has already passed
    its own halt check by the time the gate is called, so a gate can write
    ``result.halt`` and the loop will still continue. A reporter deriving "was the
    gate reached?" from that field would then report an epoch the gate DID run on
    as one it never reached — a rule firing on a healthy run, which is the mirror
    image of the defect #117 exists to fix.

    Reachable, not theoretical: the gate is where a human applies fixes, and
    #119 established the same shape one field along for the ledger.
    """
    def meddling_gate(result, run):  # noqa: ANN001, ARG001
        result.halt = HaltReason.STALL
        return GateDecision(stop=False)

    review = _panel(_two_epochs(), human_gate=meddling_gate)

    assert review.epochs[0].gate is not None, \
        "a gate that rewrote the halt erased the record of its own decision"
    assert review.epochs[0].halt is HaltReason.STALL, \
        "the fixture did not actually meddle, so it cannot express the axis"


# --- 3. silence is not a measurement -----------------------------------------

def test_a_gate_that_says_nothing_about_a_human_is_not_read_as_saying_no():
    """``asked`` defaults to None, and the engine's own inert gate sets nothing.

    ``PersonaStatus.UNREPORTED`` exists for exactly this reason — "a default
    naming one would let a backend's silence pass for a fact" — so a gate that
    does not say whether a human was consulted must not be recorded as one that
    consulted nobody. Those are different facts: the first is an unwired seam, the
    second is a measured non-interactive run.
    """
    review = _panel(_two_epochs())

    assert review.epochs[0].gate.asked is None, \
        "the engine's inert gate is recorded as having measured something"


def test_a_gate_can_stop_a_run_without_a_human_and_the_record_says_so():
    """The fifth reachable row, and the one that matters most.

    The engine takes ANY ``HumanGate``, so ``stopped`` with no human asked is
    reachable — a run the TOOL stopped. ``CLAUDE.md``'s first core principle is
    that the tool never decides, so if it ever does, the record must not describe
    it as a human's decision. ``HaltReason.ABORTED``'s own comment claims "the
    human stopped it"; this is the state that falsifies the claim.
    """
    review = _panel(_two_epochs(), max_epochs=3,
                    human_gate=_stopping_gate(asked=False))

    assert review.halt_reason is HaltReason.ABORTED
    assert review.epochs[0].gate.asked is False
    assert review.epochs[0].gate.stop is True


# --- 4. the record is a record -----------------------------------------------

def test_the_decision_cannot_be_rewritten_once_the_run_holds_it():
    """Frozen, like every other record the run stores.

    ``Finding``, ``Cluster``, ``Suppression`` and ``SurfaceFailure`` are all
    frozen; an unfrozen decision stored per epoch would also let a backend
    returning one shared instance alias every epoch's record to one object.
    """
    decision = GateDecision(stop=False)

    with pytest.raises(dataclasses.FrozenInstanceError):
        decision.stop = True
