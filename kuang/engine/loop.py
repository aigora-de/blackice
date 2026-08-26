# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""HITL-convened adversarial review loop.

A generalisation of the "two-pass adversarial panel" pattern (see
``two-pass-adversarial-review-pattern.md``) into a **bounded, human-convened
iteration loop** over an ensemble of adversarial reviewer personas.

The insight this module encodes: an agent runtime (e.g. Claude Code) can already
fan out one epoch — one subagent per persona, in parallel, over a review
surface. What is missing is the *control loop*: convening with a spec and a
halting set, accounting for a token/time budget, detecting convergence and
stalls, and — most importantly for mission-critical code — a circuit-breaker
that halts immediately on a ruin-class ("ugly") finding. That control layer is
what lives here.

Design seams (dependency-injected so the module is backend-agnostic and testable
offline — it does not itself depend on any particular LLM SDK):

* ``SpawnPersona``   — run one reviewer persona over the surface -> ``PersonaReport``.
                       Wire this to the Claude Agent SDK, the Messages API, or an
                       in-session orchestrator. A ``FakeEnsemble`` is provided for
                       tests/demos.
* ``Adjudicate``     — verify a finding's claim against source -> bool. Refuted
                       findings are dropped (the "author withdraws on the
                       evidence" step). Optional; defaults to "trust".
* ``Reduce``         — fold an epoch's signature-deduped findings into canonical
                       *clusters* (the semantic dedup / synthesis step), feeding
                       BOTH stall/convergence detection AND the human view.
                       Deterministic default (identity = one cluster per
                       signature); an LLM clusterer is a backend concern.
* ``GatherSurface``  — produce the review surface for an epoch (e.g. ``git diff``).
* ``HumanGate``      — the HITL touchpoint between epochs: apply fixes / adjust
                       scope / file issues / stop. "Human-on-the-loop", not in
                       every step.
* ``budget_spent`` / ``clock`` — injected token counter and monotonic clock, so
                       halting is deterministic and testable (no wall-clock
                       coupling).

Severity ladder maps onto the good/bad/ugly framing:

* GOOD  — the *absence* of open blockers/uglies with scope complete (a halt
          target, not a finding).
* BAD   — NOTE / NON_BLOCKING / BLOCKER: bugs, weak logic, incomplete scope,
          scope-creep. Drive iteration or become tracked residuals.
* UGLY  — ruin-class: dangerous errors/omissions with non-linear, multiplicative
          or cascading consequences that threaten survivability in the local
          context. A **circuit-breaker**: it halts the loop and escalates, and it
          is a non-negotiable convergence gate (you may halt on budget with BADs
          outstanding-and-tracked; you must never halt with an open UGLY).

This module is the loop itself. The vocabulary it works on lives in ``findings``,
the halting predicate in ``halting``, the seams listed above in ``protocols``, the
default reduce in ``reduce``, and the offline ensemble in ``fakes``. The package
``kuang.engine`` re-exports all of it, so importers need not track which
sibling holds what.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Sequence

from .findings import EpochResult, Finding, PersonaReport, ReviewRun, Severity
from .halting import HaltingSet, HaltReason, _evaluate_halt
from .protocols import (Adjudicate, GateDecision, GatherSurface, HumanGate,
                        Reduce, ReviewSurface, SpawnPersona)
from .reduce import _identity_reduce


# =============================================================================
# Inputs: the spec and the ensemble (the halting set lives in ``halting``)
# =============================================================================

@dataclass
class ReviewSpec:
    """The *why / what / scope* — never the *how* (that is the skill/ensemble)."""

    why: str                          # the risk being guarded against
    what: str                         # the change / module / execution path
    in_scope: Sequence[str] = ()      # what this review must cover
    out_of_scope: Sequence[str] = ()  # explicitly deferred (still tracked)


@dataclass
class PanelConfig:
    """The reviewer ensemble (the 'how' — supplied by the skill).

    ``personas`` is an ordered list of (name, mandate) pairs; the mandate is the
    adversarial instruction handed to each subagent.
    """

    personas: list[tuple[str, str]]
    quorum: int | None = None         # min YES verdicts for convergence (default: all)


def _trust_all(finding: Finding, surface: ReviewSurface) -> bool:  # noqa: ARG001
    return True


def _auto_continue(result: EpochResult, run: ReviewRun) -> GateDecision:  # noqa: ARG001
    return GateDecision(stop=False)


# =============================================================================
# The loop
# =============================================================================

def run(
    spec: ReviewSpec,
    halting: HaltingSet,
    panel: PanelConfig,
    *,
    spawn: SpawnPersona,
    gather: GatherSurface,
    adjudicate: Adjudicate = _trust_all,
    reduce: Reduce = _identity_reduce,
    human_gate: HumanGate = _auto_continue,
    budget_spent: Callable[[], int] = lambda: 0,
    clock: Callable[[], float] = lambda: 0.0,
    checkpoint: Callable[[ReviewRun], None] | None = None,
    scope_complete: Callable[[EpochResult], bool] = lambda r: True,
    parallel: bool = True,
) -> ReviewRun:
    """Run the human-convened adversarial review loop until a halting condition.

    Args:
        spec: The why/what/scope of the review (not the how).
        halting: The halting set (budgets, max epochs, stall patience).
        panel: The reviewer ensemble (personas + mandates).
        spawn: Runs one persona subagent over the surface. May raise; an
            exception is contained as a meta finding on that persona's report.
        gather: Produces the review surface for each epoch.
        adjudicate: Verifies a finding against source; False refutes/drops it.
        reduce: Folds the deduped ledger into canonical clusters (semantic dedup);
            defaults to identity (one cluster per signature). Feeds stall/
            convergence AND the human-facing grouping.
        human_gate: Called after each epoch; may stop the loop.
        budget_spent: Returns cumulative output tokens spent (for the budget gate).
        clock: Returns a monotonic time in seconds (for the time gate).
        checkpoint: Optional persistence hook, called each epoch for resumability.
        scope_complete: Predicate: has the in-scope surface been fully covered?

    Returns:
        A ``ReviewRun`` with per-epoch results, the deduped findings ledger, and
        the halt reason.
    """
    review_run = ReviewRun()
    quorum = panel.quorum if panel.quorum is not None else len(panel.personas)
    start = clock()
    stall_epochs = 0

    epoch = 0
    while True:
        epoch += 1
        surface = gather(epoch)

        # Fan out: one persona per subagent. Real backends spawn a subprocess
        # per persona, so run them concurrently (subprocess calls release the
        # GIL, so threads parallelise fine).
        def _run(pm: tuple[str, str]) -> PersonaReport:
            try:
                return spawn(pm[0], pm[1], surface, epoch)
            except Exception as exc:  # noqa: BLE001
                # A persona is a fallible black box, and one that raises must not
                # take the panel's other reviews with it (#25): the exception used
                # to propagate out of pool.map and end a run that had already paid
                # for everyone. Recorded as a finding rather than swallowed, and
                # deliberately not BaseException — a human's Ctrl-C still stops it.
                return PersonaReport(persona=pm[0], verdict=None, findings=[
                    Finding(pm[0], f"persona failed: {type(exc).__name__}: {exc}",
                            Severity.NOTE, "meta", evidence=repr(exc)[:400])])

        if parallel and len(panel.personas) > 1:
            with ThreadPoolExecutor(max_workers=len(panel.personas)) as pool:
                reports = list(pool.map(_run, panel.personas))
        else:
            reports = [_run(pm) for pm in panel.personas]

        # Adjudicate BLOCKER/UGLY claims against source; refuted -> dropped.
        for report in reports:
            checked: list[Finding] = []
            for f in report.findings:
                if f.severity >= Severity.BLOCKER and f.verified is None:
                    ok = adjudicate(f, surface)
                    f = Finding(**{**f.__dict__, "verified": ok})
                checked.append(f)
            report.findings = checked

        # Layer 1 (deterministic, always on): signature dedup into the ledger.
        # Snapshot the keys seen through *previous* epochs before this epoch's
        # findings land, so we can tell which clusters are genuinely new below.
        prior_keys = set(review_run.ledger)
        new_findings: list[Finding] = []
        for report in reports:
            for f in report.findings:
                if not f.counts_open:
                    continue
                if f.key not in review_run.ledger:
                    review_run.ledger[f.key] = f
                    new_findings.append(f)

        # Layer 2 (reduce/view): fold the whole deduped ledger into canonical
        # clusters. The default is identity (one cluster per signature); a semantic
        # reducer collapses re-worded / re-located dups of a single concept.
        review_run.clusters = reduce(list(review_run.ledger.values()))

        # A cluster is *new this epoch* iff every member first appeared this epoch
        # (no member key was seen before). A cluster that merges a new finding into
        # a previously-seen concept therefore reads as NOT new — so a re-worded dup
        # no longer inflates material or resets the stall counter.
        new_clusters = [c for c in review_run.clusters
                        if all(m.key not in prior_keys for m in c.members)]

        result = EpochResult(
            index=epoch,
            reports=reports,
            new_findings=new_findings,
            new_clusters=new_clusters,
            open_blockers=len(review_run.open_blocker_clusters),
            open_uglies=len(review_run.open_ugly_clusters),
        )

        # Stall accounting: only *material* (blocker/ugly) new clusters reset it.
        material_new = [c for c in new_clusters if c.severity >= Severity.BLOCKER]
        stall_epochs = 0 if material_new else stall_epochs + 1

        # str(): a backend may hand back a non-string verdict, and the quorum count
        # is outside the spawn seam's guard — a crash here would end the run (#25).
        # A report with no verdict is not a YES, which is what keeps a crashed or
        # unreadable persona from helping produce a CONVERGED verdict.
        yes_votes = sum(1 for r in reports
                        if str(r.verdict or "").upper().startswith("YES"))
        quorum_met = yes_votes >= quorum

        result.halt = _evaluate_halt(
            result, review_run, halting,
            epochs_done=epoch,
            stall_epochs=stall_epochs,
            tokens_spent=budget_spent(),
            elapsed_s=clock() - start,
            scope_complete=scope_complete(result),
            quorum_met=quorum_met,
        )
        review_run.epochs.append(result)
        if checkpoint is not None:
            checkpoint(review_run)

        if result.halt is not None:
            review_run.halt_reason = result.halt
            break

        # Between-epoch HITL gate: apply fixes / adjust scope / file issues / stop.
        decision = human_gate(result, review_run)
        if decision.stop:
            review_run.halt_reason = HaltReason.ABORTED
            break

    return review_run
