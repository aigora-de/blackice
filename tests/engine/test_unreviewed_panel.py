# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""A panel that did not review must not produce a verdict (#72).

``CONVERGED``'s three conjuncts are all satisfied by **absence**: no open blocker,
no open ugly, and a quorum of votes. So a panel that never ran at all — every
persona reporting a verdict and no review behind it — met every one of them and
halted on the good verdict, which is the instrument reporting a review that never
happened.

Two rules, and each covers a case the other cannot:

* **a persona the run knows produced no review does not vote**, which is what stops
  the lie. It is the only half that reaches a *mixed* panel, where some personas ran
  and some did not and the ones that did not pad the quorum;
* **an epoch in which the run knows no persona reviewed halts**, which is what stops
  the loop. Without it the first rule makes ``CONVERGED`` unreachable and the loop
  runs its whole epoch budget re-printing wiring, stopping at the human gate each
  time — a worse outcome than the defect.

Both are stated over ``PersonaStatus``, which is the engine's own vocabulary, and
neither can be stated with the words "dry run". The backend that motivated this is
one caller of a general rule.

**The boundary this file also defends.** A vote is a claim *about a review*. A
persona that produced no review made no claim, so nothing is discarded by not
counting it. A persona that reviewed *badly* made a claim, and dropping that would
be the tool judging — which #67 refused for a reviewer denied its tools, and which
#77 and #82 both sit on the far side of. The tests named for those two issues are
the reproductions of states this change deliberately leaves exactly as they are.

Five of the ten tests here go red on ``main`` and are regressions; the five under
the last two headings pass before this change and after it, and pin what must not
move. Said plainly so neither group is read as the other.
"""

from __future__ import annotations

from kuang.engine import (Finding, HaltingSet, HaltReason, PanelConfig,
                          PersonaReport, PersonaStatus, ReviewSpec, Severity, run)

PANEL = PanelConfig(personas=[(n, "mandate") for n in ("correctness", "adversary")])


def _run(spawn, *, panel: PanelConfig = PANEL, max_epochs: int = 1):
    return run(ReviewSpec(why="mission-critical", what="the diff"),
               HaltingSet(max_epochs=max_epochs), panel,
               spawn=spawn, gather=lambda epoch: "diff", parallel=False)


def _every_persona(**kwargs):
    def spawn(persona, mandate, surface, epoch):  # noqa: ANN001, ARG001
        return PersonaReport(persona=persona, **kwargs)
    return spawn


# --- the rule that stops the lie: a vote is a claim about a review -----------

def test_a_persona_that_did_not_run_does_not_vote():
    """REGRESSION for #72, at the seam the backend cannot reach.

    The report carries a well-formed ``"YES"`` — the exact shape #26 hardened the
    tally to accept — and a status saying no call was made. The verdict is not
    removed from the report, because nothing is discarded; it simply is not counted,
    and quorum is therefore short by one.
    """
    def spawn(persona, mandate, surface, epoch):  # noqa: ANN001, ARG001
        if persona == "adversary":
            return PersonaReport(persona=persona, verdict="YES",
                                 status=PersonaStatus.NOT_SPAWNED)
        return PersonaReport(persona=persona, verdict="YES",
                             status=PersonaStatus.CONTRIBUTED)

    review_run = _run(spawn)

    assert review_run.halt_reason is not HaltReason.CONVERGED
    assert review_run.epochs[0].reports[1].verdict == "YES", (
        "the fabricated verdict is not counted, and is not discarded either")


def test_a_backend_that_did_not_say_still_votes():
    """The guard on the guard, and the reason the rule is not ``not reviewed``.

    ``UNREPORTED`` is the default and means the backend said nothing about the
    outcome. A run must not report a degradation it did not measure, so silence
    must not be read as "did not run" — and every report in ``test_quorum.py`` is
    built this way, so reading it that way would make convergence unreachable for
    a whole file of tests without any of them naming this rule.
    """
    review_run = _run(_every_persona(verdict="YES"))

    assert review_run.halt_reason is HaltReason.CONVERGED


# --- the rule that stops the loop -------------------------------------------

def test_a_panel_nobody_reviewed_halts_no_review():
    """REGRESSION for #72: this halted ``CONVERGED`` on nothing at all."""
    review_run = _run(_every_persona(verdict="YES",
                                     status=PersonaStatus.NOT_SPAWNED))

    assert review_run.halt_reason is HaltReason.NO_REVIEW


def test_a_panel_that_wholly_errored_halts_no_review():
    """The rule stated without the words "dry run", and its second caller.

    A panel whose every call returned no review has nothing to converge on and
    nothing to iterate towards; today such a run spends its whole epoch budget
    discovering that. Nothing here knows what a dry run is.
    """
    review_run = _run(_every_persona(status=PersonaStatus.AGENT_ERROR),
                      max_epochs=3)

    assert review_run.halt_reason is HaltReason.NO_REVIEW


def test_no_review_pre_empts_the_epoch_ceiling():
    """What stops epoch 2, and why it is a rule rather than a special case.

    Evaluated on every epoch of every run: an epoch the run knows no persona
    reviewed cannot converge, and repeating it cannot change that. So the halt is
    the reason the loop stops, not the epoch ceiling arriving later.
    """
    review_run = _run(_every_persona(verdict="YES",
                                     status=PersonaStatus.NOT_SPAWNED),
                      max_epochs=3)

    assert review_run.halt_reason is HaltReason.NO_REVIEW
    assert len(review_run.epochs) == 1


def test_an_empty_panel_reviewed_nothing():
    """Quorum over an empty panel is ``0 >= 0``: the same lie, reached by arithmetic."""
    review_run = _run(_every_persona(verdict="YES"),
                      panel=PanelConfig(personas=[]))

    assert review_run.halt_reason is HaltReason.NO_REVIEW


def test_the_breaker_still_outranks_no_review():
    """Ruin is checked first and is never a permitted halt-with-open state.

    An UGLY raised in epoch 1 stays open in the ledger, so an epoch 2 that reviewed
    nothing must still escalate rather than report the milder reason.
    """
    def spawn(persona, mandate, surface, epoch):  # noqa: ANN001, ARG001
        if epoch == 1:
            return PersonaReport(
                persona=persona, verdict="NO", status=PersonaStatus.CONTRIBUTED,
                findings=[Finding(persona, f"{persona}: unbounded loss",
                                  Severity.UGLY, f"{persona}-lens", "a.py", 1)])
        return PersonaReport(persona=persona, status=PersonaStatus.NOT_SPAWNED)

    review_run = _run(spawn, max_epochs=2)

    assert review_run.halt_reason is HaltReason.ESCALATE_UGLY


# --- the boundary: what this change deliberately does NOT touch --------------

def test_a_panel_of_one_that_opened_nothing_still_converges():
    """#82's state, reproduced here so it stays open and stays visible.

    One persona, which reviewed and found nothing, having called no tool at all
    (``turns == 1``, #70). Its agreement is worth very little and the run says so
    on its participation line — but it *reviewed*, so it votes, and the run still
    converges exactly as it does today. #82 is a good verdict for a review that
    happened and was worth nothing; this issue is a good verdict for one that never
    happened, and a fix for the second must not quietly take the first.
    """
    review_run = _run(_every_persona(verdict="YES", turns=1,
                                     status=PersonaStatus.FOUND_NOTHING),
                      panel=PanelConfig(personas=[("sole reviewer", "mandate")]))

    assert review_run.halt_reason is HaltReason.CONVERGED


def test_a_persona_denied_its_tools_still_votes():
    """#67's settlement, which #77 inherits, pinned at the engine this time.

    A reviewer whose tools were removed reviewed under a policy that made its
    findings speculation — and #24 and #26 settled that such a vote is recorded and
    reported, never discarded, because dropping it moves quorum for a reason nobody
    can see and can silently *unlatch* the breaker. That is grounding, not
    participation, and this change gates only participation.
    """
    review_run = _run(_every_persona(verdict="YES", turns=3,
                                     denied_tools=["Read", "Grep"],
                                     status=PersonaStatus.CONTRIBUTED))

    assert review_run.halt_reason is HaltReason.CONVERGED


def test_a_healthy_unanimous_panel_still_converges():
    """The mirror image of the defect, written before the fix.

    A rule that fires on a healthy run is the same instrument failure the other way
    round. Every persona ran, reviewed, and voted YES: the halt reason, the epoch
    count and the good verdict are all unchanged.
    """
    review_run = _run(_every_persona(verdict="YES",
                                     status=PersonaStatus.CONTRIBUTED),
                      max_epochs=3)

    assert review_run.halt_reason is HaltReason.CONVERGED
    assert len(review_run.epochs) == 1
