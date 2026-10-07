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

from .findings import (EpochResult, Finding, GateFailure, PersonaReport,
                       PersonaStatus, ReviewRun, Severity, Suppression,
                       SurfaceFailure, bounded_diagnosis)
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

    @property
    def effective_quorum(self) -> int:
        """The YES count convergence actually requires (default: unanimity).

        The loop applied this rule in a local variable nothing outside could see,
        so a run printed ``converged`` without ever stating the agreement behind it
        (#82). It is a property, not a second copy in a reporter, for the reason
        ``AFFIRMATIVE_VERDICT`` is a constant: the number a run *states* and the
        number its gate *applies* must not be able to drift apart.

        Zero over an empty panel, deliberately, rather than a softened 1 — that
        ``0 >= 0`` is the arithmetic behind #72's empty-panel case, and reporting
        anything else here would make the printed number disagree with the applied
        one. What stops that run is ``NO_REVIEW``, which is a different rule.
        """
        return self.quorum if self.quorum is not None else len(self.personas)


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
        gather: Produces the review surface for each epoch. May raise; a failure
            after at least one epoch has completed halts the run as
            ``SURFACE_LOST`` and returns what those epochs produced, and a failure
            with no completed epoch propagates (there is nothing to report, and a
            surface that cannot be built is an operator error, never a review).
        adjudicate: Verifies a finding against source; False refutes/drops it.
        reduce: Folds the deduped ledger into canonical clusters (semantic dedup);
            defaults to identity (one cluster per signature). Feeds stall/
            convergence AND the human-facing grouping.
        human_gate: Called after each epoch; may stop the loop. May raise, and may
            return something that is not a ``GateDecision``: either is contained as a
            ``GateFailure`` on that epoch, a ``GateDecision(stop=False, asked=None)``
            is substituted, and the run CONTINUES (#129). What is stored is always a
            decision the engine constructed, so a subclass's extra state is dropped.
        budget_spent: Returns cumulative output tokens spent (for the budget gate).
        clock: Returns a monotonic time in seconds (for the time gate).
        checkpoint: Optional persistence hook, called each epoch for resumability.
        scope_complete: Predicate: has the in-scope surface been fully covered?

    Returns:
        A ``ReviewRun`` with per-epoch results, the deduped findings ledger, and
        the halt reason.
    """
    review_run = ReviewRun()
    quorum = panel.effective_quorum
    start = clock()
    stall_epochs = 0

    epoch = 0
    while True:
        epoch += 1
        try:
            surface = gather(epoch)
        except Exception as exc:  # noqa: BLE001
            # The other fallible seam, guarded the way ``spawn`` is below (#85, #25).
            # ``gather`` is called once per EPOCH, so a fix applied at the human gate
            # that removes a named file — or a base ref that stops resolving — used to
            # propagate out of ``run`` and take the whole ``ReviewRun`` with it: every
            # finding, participation record and token count from the epochs that DID
            # complete, for a panel already spawned and paid for.
            #
            # ``Exception`` because the engine may not name a backend's exception class
            # (``tests/engine/test_backend_agnostic.py``), and deliberately not
            # ``BaseException`` — a human's Ctrl-C still stops the loop.
            #
            # The predicate is that NO EPOCH HAS COMPLETED, not that the counter is at
            # one. They coincide, but only this one states the rule: with nothing to
            # report there is no run to hand back, so the operator error propagates and
            # #18's doctrine is untouched — a surface that cannot be built is an
            # operator error, never a review with no findings. It is load-bearing as
            # well as honest: a reporter reaching for ``epochs[-1]`` would fail on a
            # run halted with none.
            if not review_run.epochs:
                raise
            # Collapsed to one line, then bounded, and both in the ENGINE so the
            # console and the artefact cannot disagree about the same string. Not
            # cosmetic: a real diff-mode failure carries git's stderr, measured at
            # three lines, and the continuation lines would print unprefixed between
            # the halt line and the findings, reading as sections of the report.
            #
            # Collapse, then measure, then bound and mark (#111). The length is
            # taken from the COLLAPSED string because that is the diagnosis an
            # operator was ever going to be shown; measuring the raw exception
            # would report whitespace as loss. ``bounded_diagnosis`` carries the
            # marker doctrine and its citation.
            collapsed = " ".join(f"{type(exc).__name__}: {exc}".split())
            review_run.surface_failure = SurfaceFailure(
                epoch=epoch, detail=bounded_diagnosis(collapsed),
                detail_chars=len(collapsed))
            review_run.halt_reason = HaltReason.SURFACE_LOST
            break

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
                # The status is set HERE, at the source, and is the engine's own:
                # ``SPAWN_FAILED`` means our code raised, against a backend's
                # ``AGENT_ERROR`` for a runtime that ran and returned no review.
                # A run must be able to tell those apart (#30).
                return PersonaReport(
                    persona=pm[0], verdict=None,
                    status=PersonaStatus.SPAWN_FAILED,
                    findings=[
                        Finding(pm[0], f"persona failed: {type(exc).__name__}: {exc}",
                                Severity.NOTE, "meta",
                                evidence=bounded_diagnosis(repr(exc)),
                                about_run=True)])

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
        # Every open finding this epoch emitted ends in exactly one of three
        # states, and a run that reports only the first cannot be asked what
        # happened to the others (#119).
        suppressed: list[Suppression] = []
        resighted = 0
        for report in reports:
            for f in report.findings:
                if not f.counts_open:
                    continue
                held = review_run.ledger.get(f.key)
                if held is None:
                    review_run.ledger[f.key] = f
                    new_findings.append(f)
                elif held.key_components == f.key_components:
                    # The coarse dedup doing its job: one signature is one
                    # finding, however many personas raise it or re-word it.
                    resighted += 1
                else:
                    # A COLLISION. The signature joins four values — two of them
                    # written by the persona — through an encoding that is
                    # ambiguous (#125), so two findings that are NOT the same
                    # finding can hash alike, and this one loses its place to a
                    # claim it has nothing to do with. It reached no ledger, no
                    # artefact and no count, and nothing said so.
                    #
                    # Diagnosed on the pre-hash components, never on the digest:
                    # the digest is exactly what cannot tell this case from the
                    # re-sighting above.
                    #
                    # Recorded HERE, where the ledger decides, rather than
                    # derived by a reporter afterwards. ``HumanGate`` receives
                    # the run mutably, so a later walk reads the ledger as it
                    # ENDS — and a gate that removes a key would make it report a
                    # suppression that never happened, which is a rule firing on
                    # a healthy run. "Set at the source" (#30) governs: a fact
                    # this exact must not rest on a precondition no seam enforces.
                    #
                    # Reported, not kept. Giving the claim a ledger place of its
                    # own means giving it a distinct key, which is #125 wearing a
                    # different hat — so what this loses is stated rather than
                    # quietly fixed: the claim counts toward no total and resets
                    # no stall counter.
                    suppressed.append(Suppression(finding=f, holder=held))

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
            suppressed=suppressed,
            resighted=resighted,
        )

        # Stall accounting: only *material* (blocker/ugly) new clusters reset it.
        # The predicate lives on the result the reporters also read (#33), so the
        # number a run states and the number this gate applies cannot drift apart
        # — ``EpochResult.material_new_clusters`` carries the rule and the reasons.
        stall_epochs = 0 if result.material_new_clusters else stall_epochs + 1

        # What counts as a vote is one predicate, and it lives on the report it is
        # a fact about (#26, #72) — see ``PersonaReport.counted_vote``, which
        # carries the whole rule and the reasons for it. The engine gates on it
        # here; a reporter reads the same property rather than restating it (#82).
        yes_votes = sum(1 for r in reports if r.counted_vote)
        quorum_met = yes_votes >= quorum

        # And if the run knows NOBODY reviewed, the rule above makes CONVERGED
        # unreachable, so something must stop the loop or it spends its whole epoch
        # budget re-printing wiring and stopping at the gate each time. An empty
        # panel falls out of this as well: quorum over nobody is 0 >= 0, the same
        # verdict from the same absence, reached by arithmetic instead.
        no_persona_reviewed = all(r.status.did_not_review for r in reports)

        result.halt = _evaluate_halt(
            result, review_run, halting,
            epochs_done=epoch,
            stall_epochs=stall_epochs,
            tokens_spent=budget_spent(),
            elapsed_s=clock() - start,
            scope_complete=scope_complete(result),
            quorum_met=quorum_met,
            no_persona_reviewed=no_persona_reviewed,
        )
        review_run.epochs.append(result)
        if checkpoint is not None:
            checkpoint(review_run)

        if result.halt is not None:
            review_run.halt_reason = result.halt
            break

        # Between-epoch HITL gate: apply fixes / adjust scope / file issues / stop.
        #
        # ONE RULE, stated once and applied to the whole seam: the engine never reads
        # an unvalidated attribute off a seam's return value outside the guard. The
        # call, the type check and every read are inside it, and what is stored is a
        # ``GateDecision`` the ENGINE constructed. A type check with the read left
        # bare below it was the first shape of this fix, and a pre-merge review pass
        # broke it twice over: ``isinstance`` admits a SUBCLASS whose ``stop`` is a
        # property that raises, and ``stop`` is coerced nowhere, so even on the
        # engine's own frozen record a ``__bool__`` can raise. Both reproduced the
        # whole defect #129 was opened on — a paid-for run discarded — through a door
        # the check left open while the comment beside it claimed otherwise. Patching
        # the two cases would have been the patched table this project refuses; the
        # rule above is the one principle both fall out of.
        try:
            returned = human_gate(result, review_run)
            if isinstance(returned, GateDecision):
                # Re-made rather than kept as handed back, which is what puts the
                # two attribute reads inside the guard. A subclass's extra state is
                # dropped deliberately: nothing in the engine or the artefact reads
                # it, and #130 removed the one field that carried free text here.
                # ``stop`` is coerced because the loop halts on TRUTHINESS below, so
                # the record must say what the loop acted on (the rule the CLI
                # applies to the same field); ``asked`` is NOT coerced, because a
                # value the tool cannot read is not a measurement and the publisher
                # resolves it by identity (#131).
                decision = GateDecision(stop=bool(returned.stop), asked=returned.asked)
                diagnosis = None
            else:
                # A seam can fail by returning the wrong thing rather than by
                # raising: ``None``, or a foreign object. ``None`` and anything
                # without a ``.stop`` raised at the read and discarded the run
                # exactly as an exception did; one WITH a duck-typed ``.stop``
                # survived and was stored in a field typed ``GateDecision | None``,
                # so the artefact published ``reached: true`` off a value nothing had
                # checked.
                #
                # The diagnosis names the TYPE and stops there: ``repr`` of what came
                # back is unbounded text authored outside this codebase, on its way
                # to a published artefact (``GateFailure`` states that in full).
                diagnosis = f"gate returned {type(returned).__name__}, not GateDecision"
        except Exception as exc:  # noqa: BLE001
            # The third of three injected seams, and the last to be guarded (#129).
            # ``gather`` has had this treatment since #85 and ``spawn`` since #25,
            # and both paragraphs apply here unchanged: a fallible black box must
            # not take the whole ``ReviewRun`` with it — every finding,
            # participation record and token count from the epochs that DID
            # complete, for a panel already spawned and paid for, with the artefact
            # stdout-only so none of it is readable back.
            #
            # ``Exception``, never ``BaseException``: the gate prompt is where an
            # operator presses Ctrl-C to stop a run, and that must stay reliable at
            # the one moment it is most needed. ``EOFError`` is an ``Exception`` and
            # ``KeyboardInterrupt`` is not — though the shipped backend traps EOF at
            # its own ``input()`` call, so what reaches here in practice is a stdout
            # fault, a foreign gate, or a bug in a gate's branch logic.
            diagnosis = f"{type(exc).__name__}: {exc}"
        # What the run does next is RULED, not decided here: see ``GateDecision.asked``,
        # which names "the seam itself failed (#129)" as one of ``None``'s three cases.
        # The run CONTINUES and records that no answer was resolved.
        #
        # Why that is right for THIS case and not merely cited from the one it was
        # ruled on, because the facts differ on the axis the ruling turns on. #132
        # ruled on an operator who was present and typed something unreadable; a seam
        # can fail before the prompt is ever printed, so there may be no human to have
        # seen anything. What carries over is not "they can override it" but the bar
        # in ``halting.py``: a halt reason is for a state the loop cannot usefully
        # continue from, and a run whose gate is down can still do the thing it is
        # for — the panel reviews, the ledger fills, and the human adjudicates the
        # artefact afterwards, which is the human-ON-the-loop contract rather than a
        # human in every step. Halting instead would be the TOOL deciding to stop a
        # review nobody stopped, and ``ABORTED`` would record exactly that.
        #
        # The cost is real and is not hidden: a gate that is down stays down (its
        # triggers are environmental — see ``GateFailure``), so the epochs after it
        # run without the between-epoch touchpoint, spending budget the operator
        # cannot interrupt through the gate. It is bounded by ``max_epochs``, the
        # stall patience and the token budget, and the artefact says at which epochs
        # the channel was gone. Ctrl-C remains the operator's unconditional stop.
        if diagnosis is None:
            # Set on BOTH paths, at the source, rather than left to the field's
            # default: the gate holds this ``EpochResult`` mutably, so a gate that
            # wrote a failure record itself and then answered normally would publish
            # a decision AND a failure for the same epoch — a rule firing on a
            # healthy run. The loop's suppression record is set at the source for
            # the same reason (#30).
            result.gate_failure = None
        else:
            # Collapsed, then measured, then bounded — in that order, once, for both
            # branches. A bound applied before the collapse spends itself on
            # whitespace nobody will be shown, and ``detail_chars`` is the length of
            # the COLLAPSED diagnosis so that "was this cut" stays the exactly
            # knowable ``detail_chars > DIAGNOSIS_BOUND`` (#111). One site because
            # the first shape of this fix collapsed only the exception branch while
            # ``GateFailure``'s docstring promised both, and a class name can carry a
            # newline, which would have reached the report as a section of its own.
            collapsed = " ".join(diagnosis.split())
            result.gate_failure = GateFailure(bounded_diagnosis(collapsed),
                                              len(collapsed))
            decision = GateDecision(stop=False, asked=None)
        # Recorded HERE, below the guard, rather than derived by a reporter
        # afterwards (#117). The channel a human acts through was reported by
        # nothing at all: a run could not say whether the gate was reached, what
        # was chosen, or at which epoch. The record's PRESENCE is what says the
        # gate was REACHED — see ``EpochResult.gate`` for why that is not derived
        # from ``halt``, which this same seam can rewrite.
        #
        # Below the guard rather than above it, which is where #129 found it: a
        # gate returning ``None`` was stored and only then read, so an epoch whose
        # gate RAN was left with a record that renders as "the epoch halted, so the
        # gate was never reached". Whatever is stored here is a ``GateDecision`` by
        # the time it is stored, and ``gate_failure`` beside it is what distinguishes
        # a substituted decision from an answered one.
        result.gate = decision
        if decision.stop:
            review_run.halt_reason = HaltReason.ABORTED
            break

    return review_run
