# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""Why the loop stops: the halt reasons, the halting set, and the predicate.

The halting set is an OR of predicates evaluated in a fixed order, and the order
is the doctrine: the UGLY circuit-breaker is checked first and short-circuits even
an unexhausted budget. ``_evaluate_halt`` is a pure function of its arguments —
no clock, no randomness — so the engine's control flow is reproducible.

Imports from ``findings`` are annotations only, which is what keeps the
dependency one-way: ``findings`` imports ``HaltReason`` from here at run time.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:                    # annotations only — see the module docstring
    from .findings import ReviewRun


class HaltReason(Enum):
    """Why the loop stopped. Every reason returns the accumulated report."""

    CONVERGED = "converged"            # good: no open blocker/ugly, scope complete
    ESCALATE_UGLY = "escalate_ugly"    # ruin-class finding -> circuit-break
    BUDGET = "budget"                  # token or time ceiling reached
    EPOCH = "epoch"                    # max epochs reached
    STALL = "stall"                    # no new material findings, blockers still open
    ABORTED = "aborted"                # the human gate stopped the loop
    NO_REVIEW = "no_review"            # no persona reviewed: there is no verdict
    SURFACE_LOST = "surface_lost"      # the surface could not be re-assembled
    UNMEASURED = "unmeasured"          # an armed ceiling's input could not be read

    # NO_REVIEW is for a state the loop cannot usefully CONTINUE from, which is the
    # rule that keeps this vocabulary from growing once per issue. Nothing was
    # reviewed, so another epoch changes nothing and the run must stop and say so.
    # A review that HAPPENED and was degenerate — one voice, or a voice that opened
    # nothing (#82) — can continue, and #82 is explicit that the tool must not gate
    # it: the tool informs, the human decides. That case is QUALIFIED beside the
    # verdict, never halted, so this member is not a precedent for one naming it.
    #
    # SURFACE_LOST passes that rule on its own merits rather than by citing it (#85).
    # The surface is the INPUT to every epoch: without one there is nothing to hand
    # any persona, and no action at the human gate can produce one, because the loop
    # is already past the gate when the next epoch's gather fails. The operator's
    # next move is a new run against corrected scope, which is a different run.
    #
    # It is a seventh member rather than a reuse because every existing one would be
    # a lie: ABORTED means the GATE stopped it and nothing did, BUDGET/EPOCH/STALL
    # name ceilings that were not reached, and NO_REVIEW is a claim about PERSONAS —
    # here they reviewed, and their findings print two lines below the halt line.
    # Which channel failed is the distinction #30 drew between SPAWN_FAILED and
    # AGENT_ERROR, and folding the two together would lose it.
    #
    # UNMEASURED passes the rule on its merits too, by the maintainer's ruling on #148
    # (2026-10-12). An operator who ARMS a token or time ceiling has set the run's spend
    # bound. If its input cannot be read, ``BUDGET`` would claim a ceiling was REACHED
    # that nobody measured, and continuing would claim one was NOT, so the run spends
    # past a bound it can no longer see. The loop cannot continue within the operator's
    # bound, which is "a state it cannot usefully continue from". #129's ruling to
    # continue past a failed gate does not carry here: part of its argument was that the
    # token budget still bounds the run, and here that is what is down. A DISARMED
    # ceiling's input cannot change the halt, so its failure is recorded and the run
    # continues. #32 was told, because the vocabulary is its subject too.


@dataclass
class HaltingSet:
    """The bounded-loop halting conditions (evaluated as an OR of predicates).

    UGLY is always a circuit-breaker regardless of these limits, and never a
    permitted halt-with-open state.
    """

    token_budget: int | None = None   # None -> unbounded (rely on epoch/stall)
    time_budget_s: float | None = None
    max_epochs: int = 3
    stall_patience: int = 1           # K epochs with no new material -> STALL
    require_scope_complete: bool = True  # convergence needs an explicit "scope covered"


def _evaluate_halt(
    run: ReviewRun,
    halting: HaltingSet,
    *,
    epochs_done: int,
    stall_epochs: int,
    tokens_spent: int | None,
    elapsed_s: float | None,
    scope_complete: bool,
    quorum_met: bool,
    no_persona_reviewed: bool,
) -> HaltReason | None:
    """Pure predicate: return a HaltReason if any halting condition is met.

    Order matters: the UGLY circuit-breaker is checked first and short-circuits
    even an unexhausted budget. CONVERGED requires no open ugly AND no open
    blocker AND (optionally) scope complete AND quorum — and every one of those
    three conjuncts is satisfied by ABSENCE, which is why NO_REVIEW is evaluated
    ahead of it (#72).

    ``tokens_spent`` or ``elapsed_s`` is ``None`` where its injected input could not be
    read (#148). ``None`` is UNKNOWN, never zero: it fires no ceiling, and where its
    ceiling is armed it halts ``UNMEASURED``, last, so that every exactly-knowable
    reason is preferred to it.

    Every zero-test on open severity reads the LEDGER, not the clusters (#150): "is any
    UGLY open" for the breaker, and "is any BLOCKER open" for ``CONVERGED`` and
    ``STALL``. The ledger holds what the panel raised and adjudication kept; a cluster is
    whatever an injected ``reduce`` returned. So no value a reduce returns, and none of
    the three halting inputs, can switch the breaker off. (The ledger is still made of
    ``spawn``'s reports and ``adjudicate``'s verdicts, and a refutation withdrawing an
    UGLY is the design, not a masking.) Measured on ``f17c256`` reading clusters:
    ``reduce -> []`` with open UGLYs in the ledger halted ``epoch``, or ``converged`` on
    a panel voting YES.

    The epoch's ``EpochResult`` is deliberately NOT a parameter. Its ``open_uglies`` and
    ``open_blockers`` are what the run PUBLISHES as these decisions' inputs (#117), and
    they agree with the ledger in zero-ness wherever the loop built them, because it
    admits only clusters that partition the ledger (``_as_partition``) and
    ``Cluster.open_severity`` is the max over open members. The BLOCKER count agrees only
    while no UGLY is open, since a cluster merging the two counts as UGLY, and while one
    is open the breaker halts before either BLOCKER test is reached. But the record is
    MUTABLE, and ``scope_complete`` is handed it before this runs: the pre-merge review
    measured a scope check that zeroed the counts converging an UGLY run on ``f17c256``.
    A predicate that cannot see the record cannot be steered through it.
    """
    # 1. Circuit-breaker: a ruin-class finding halts and escalates immediately.
    if run.open_uglies:
        return HaltReason.ESCALATE_UGLY

    blocker_open = bool(run.open_blockers)

    # 2. Nothing was reviewed (#72). An epoch in which the run KNOWS no persona
    #    produced a review cannot converge, and repeating it cannot change that:
    #    there is no new material, and no action at the human gate makes an
    #    unspawned or failed call run. Ahead of CONVERGED because zero findings
    #    from nobody is not a clean bill of health, and ahead of the ceilings so
    #    the run reports the reason it actually stopped rather than the ceiling
    #    it would have reached later. Behind the breaker, which is unconditional.
    if no_persona_reviewed:
        return HaltReason.NO_REVIEW

    # 3. Convergence (good): nothing dangerous or blocking is open.
    scope_ok = scope_complete or not halting.require_scope_complete
    if not blocker_open and scope_ok and quorum_met:
        return HaltReason.CONVERGED

    # 4. Resource ceilings (partial halts — BADs may remain, but must be tracked).
    if (halting.token_budget is not None and tokens_spent is not None
            and tokens_spent >= halting.token_budget):
        return HaltReason.BUDGET
    if (halting.time_budget_s is not None and elapsed_s is not None
            and elapsed_s >= halting.time_budget_s):
        return HaltReason.BUDGET
    if epochs_done >= halting.max_epochs:
        return HaltReason.EPOCH

    # 5. Stall: no new material findings for K epochs while blockers persist.
    if stall_epochs >= halting.stall_patience and blocker_open:
        return HaltReason.STALL

    # 6. An armed ceiling that could not be read (#148). LAST, because continuing is the
    #    one action it forbids: anything above that halts the run for a reason the run
    #    actually measured is the truer reason. A disarmed ceiling's unknown is no
    #    obstacle to continuing, so it is not consulted.
    if ((halting.token_budget is not None and tokens_spent is None)
            or (halting.time_budget_s is not None and elapsed_s is None)):
        return HaltReason.UNMEASURED

    return None
