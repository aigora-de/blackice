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
    from .findings import EpochResult, ReviewRun


class HaltReason(Enum):
    """Why the loop stopped. Every reason returns the accumulated report."""

    CONVERGED = "converged"            # good: no open blocker/ugly, scope complete
    ESCALATE_UGLY = "escalate_ugly"    # ruin-class finding -> circuit-break
    BUDGET = "budget"                  # token or time ceiling reached
    EPOCH = "epoch"                    # max epochs reached
    STALL = "stall"                    # no new material findings, blockers still open
    ABORTED = "aborted"                # human gate stopped the loop
    NO_REVIEW = "no_review"            # no persona reviewed: there is no verdict

    # NO_REVIEW is for a state the loop cannot usefully CONTINUE from, which is the
    # rule that keeps this vocabulary from growing once per issue. Nothing was
    # reviewed, so another epoch changes nothing and the run must stop and say so.
    # A review that HAPPENED and was degenerate — one voice, or a voice that opened
    # nothing (#82) — can continue, and #82 is explicit that the tool must not gate
    # it: the tool informs, the human decides. That case is QUALIFIED beside the
    # verdict, never halted, so this member is not a precedent for one naming it.


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
    result: EpochResult,
    run: ReviewRun,
    halting: HaltingSet,
    *,
    epochs_done: int,
    stall_epochs: int,
    tokens_spent: int,
    elapsed_s: float,
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
    """
    # 1. Circuit-breaker: a ruin-class finding halts and escalates immediately.
    if result.open_uglies > 0:
        return HaltReason.ESCALATE_UGLY

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
    if result.open_blockers == 0 and scope_ok and quorum_met:
        return HaltReason.CONVERGED

    # 4. Resource ceilings (partial halts — BADs may remain, but must be tracked).
    if halting.token_budget is not None and tokens_spent >= halting.token_budget:
        return HaltReason.BUDGET
    if halting.time_budget_s is not None and elapsed_s >= halting.time_budget_s:
        return HaltReason.BUDGET
    if epochs_done >= halting.max_epochs:
        return HaltReason.EPOCH

    # 5. Stall: no new material findings for K epochs while blockers persist.
    if stall_epochs >= halting.stall_patience and result.open_blockers > 0:
        return HaltReason.STALL

    return None
