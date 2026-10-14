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
                       BOTH the stall counter AND the human view. No halt's
                       zero-test reads them (#150): those are a ledger snapshot.
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
default reduce in ``reduce``, the copies a seam is handed in ``_handed`` (#155), and
the offline ensemble in ``fakes``. The package
``kuang.engine`` re-exports all of it, so importers need not track which
sibling holds what.
"""

from __future__ import annotations

import math
import operator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Sequence, TypeVar

from .findings import (Cluster, EpochResult, Finding, GateFailure, PersonaReport,
                       PersonaStatus, ReviewRun, SeamFailure, Severity, Suppression,
                       SurfaceFailure, bounded_diagnosis, render_exception,
                       repr_exception, type_name)
from ._handed import (Claim, UncopiableRecord, claimed_severity, handed,
                      is_immutable_value)
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


_T = TypeVar("_T")


def _call_seam(seam: str, failures: list[SeamFailure],
               call: Callable[[], _T]) -> tuple[bool, _T | None]:
    """Call one injected callable; a failure is UNKNOWN, recorded, never a value (#148).

    ONE guard for the six callables #85, #25 and #129 did not name, rather than six local
    ``try``s each with its own outcome: the outcome is the CALLER's, and every caller
    states it beside the call. What is uniform is everything else — the catch, the
    record, and #129's rule that the engine never reads an unvalidated value off a seam's
    return outside the guard. So ``call`` is the seam's call AND the conversion of what
    it returned, and both run in here. Measured on ``0fd8ebb``: a raise from any of the
    six, and a hostile RETURN from five, escaped ``run`` after the panel was paid for.

    Returns ``(True, value)``, or ``(False, None)`` where the call failed, so that no
    value a seam can return is mistaken for the failure marker. Sites that discard the
    flag and read ``None`` as UNKNOWN may do so only because their conversion never
    returns ``None``: each of the ``_as_*`` conversions below returns its type or raises.

    ``Exception``, never ``BaseException``: Ctrl-C still stops the run. The diagnosis is
    rendered by #142's helper — never by interpolating ``exc``, which reopened #142 at
    every guard written the old way — then collapsed, measured and bounded, in the order
    ``SurfaceFailure`` and ``GateFailure`` use (#111).
    """
    try:
        return True, call()
    except Exception as exc:  # noqa: BLE001
        collapsed = " ".join(render_exception(exc).split())
        failures.append(SeamFailure(seam, bounded_diagnosis(collapsed), len(collapsed)))
        return False, None


def _as_verdict(value: object) -> bool:
    """An adjudication is ``True`` or ``False`` EXACTLY; anything else is no answer.

    Categorical, never truthiness. ``bool(...)`` was this fix's first shape, and the
    pre-merge review measured what it did: an adjudicator returning ``None`` — which is
    ``Finding.verified``'s own word for "not checked" — or ``0``, ``""`` or ``[]`` was
    coerced to ``False``, a REFUTATION, so an UGLY was dropped, the run halted
    ``converged``, and nothing was recorded. Main had kept the raw value, which counted
    open by accident. ``bool`` cannot be subclassed, so ``type(...) is bool`` admits
    exactly the two values a verdict can be.
    """
    if type(value) is not bool:
        raise TypeError(f"adjudicate returned {type_name(value)}, not bool")
    return value


def _as_scope(value: object) -> bool:
    """A scope answer is ``True`` or ``False`` exactly, for ``_as_verdict``'s reason."""
    if type(value) is not bool:
        raise TypeError(f"scope_complete returned {type_name(value)}, not bool")
    return value


def _as_token_count(value: object) -> int:
    """A token count is a non-negative integer, returned as an exact ``int``.

    Excluded: a ``bool`` (``True`` read as one token), anything that is not an ``int``
    (a float, a string, ``None``), and a negative, which can never reach the ceiling and
    so silently disarms the operator's spend bound for the rest of the run. An ``int``
    SUBCLASS is admitted and normalised by ``operator.index``: it is a real count, and
    the second pre-merge review measured an exact-type check halting a healthy run
    ``unmeasured`` on one. No healthy counter produces an excluded value.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"budget_spent returned {type_name(value)}, not a token count")
    count = operator.index(value)
    if count < 0:
        raise ValueError("budget_spent returned a negative token count")
    return count


def _as_clock_reading(value: object) -> float:
    """A clock reading is a FINITE ``int`` or ``float`` (not a ``bool``), as a ``float``.

    The same rule as the token count, so one kind of input meets one rule: the second
    pre-merge review measured ``float(...)`` admitting ``"5"`` and ``True`` as readings.
    NaN compares false with every ceiling, so it read as a measurement that never
    reaches one (the same silent disarm), and ``inf`` as one that reached it at once.

    A reading BELOW the start is still a measurement and is kept: an injected wall
    clock can step backwards, and refusing it would fire on a healthy run. It can delay
    the time ceiling by the size of the step; it cannot disarm it.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"clock returned {type_name(value)}, not a reading")
    reading = float(value)
    if not math.isfinite(reading):
        raise ValueError("clock returned a non-finite reading")
    return reading


def _elapsed(value: object, start: float) -> float:
    """Seconds since ``start``, which must itself be finite: two finite readings far
    enough apart overflow to ``inf`` (measured at ``1e308`` then ``-1e308``)."""
    elapsed = _as_clock_reading(value) - start
    if not math.isfinite(elapsed):
        raise ValueError("clock readings differ by a non-finite interval")
    return elapsed


def _claim_note(claim: Claim) -> str:
    """What a refused report's failure carries, said in fixed words (#155, #30).

    Empty where it claimed nothing above NOTE, so a spawn that RAISED is titled exactly
    as it always was. An UGLY that was claimed and one carried because nothing could be
    read are different facts, and a run must be able to tell them apart.
    """
    if not claim.read:
        return " — what it claimed could not be read, so it is carried as UGLY (fail closed)"
    if claim.severity > Severity.NOTE:
        return f" — it claimed {claim.severity.name}"
    return ""


def _rebuilt_clusters(returned: object, ledger: dict[str, Finding]) -> list[Cluster]:
    """What ``reduce`` returned, rebuilt by the engine from the ledger's OWN findings.

    Called inside ``_call_seam``, so anything this cannot read is a failed reduce. The
    clusters the engine stores are the engine's own, every member is the ledger's own
    ``Finding`` (looked up by key). So nothing the loop reads afterwards — ``members``,
    ``key``, ``severity`` — can raise from a foreign object. The title is carried as
    given: the loop never reads it, and requiring an exact ``str`` here recorded the
    engine's OWN identity reduce as a failed seam whenever a finding's title was not
    one (the pre-merge review). What a title may be is the findings contract's
    question, so ``Cluster.title``'s annotation is a contract on reducers, not a
    guarantee this function adds. A member the ledger does not hold is a ``KeyError`` here, never a phantom
    finding. A reduce that leaves a ledger finding OUT is ``_as_partition``'s to refuse.

    **Except where it is a value the copy cannot share (#155).** The title is stored in
    the run's record (``EpochResult.new_clusters``), so a ``str`` subclass kept there was
    in every later copy of the run, and each was refused: the fifth pre-merge review
    measured the gate never reached again and nothing checkpointed. Such a title is a
    failed reduce, recorded, and the identity follows. The rule is the COPY's
    (``is_immutable_value``: an exact ``str``, ``None`` or other exact scalar), not "a
    ``str``": the identity reduce's titles are a ledger finding's, which the copy admitted
    by that same rule, so the engine's own reduce can never fail it (the objection
    above, which an exact-``str`` rule reopened, measured on this repair's first shape).
    """
    clusters = [Cluster(members=tuple(ledger[m.key] for m in c.members), title=c.title)
                for c in returned]  # type: ignore[attr-defined]
    if not all(is_immutable_value(c.title) for c in clusters):
        raise TypeError("reduce returned a cluster title the engine cannot keep")
    return clusters


def _as_partition(clusters: list[Cluster], ledger: dict[str, Finding]) -> list[Cluster]:
    """The clusters, if they partition the ledger; otherwise the reduce failed (#150).

    #61's rule, "a reduce must partition the findings", was doctrine in ``Reduce`` and
    enforced only in one backend's clusterer. This makes it the ENGINE's rule: every
    ledger finding in exactly one cluster, and no cluster empty. Called inside
    ``_call_seam`` after ``_rebuilt_clusters``, so a non-partition is a failed reduce
    and the identity follows. That is #148's rule unchanged, because the clusters the
    engine counts are a conversion of what ``reduce`` returned.

    Measured on ``f17c256``: ``reduce -> []`` with open UGLYs in the ledger halted
    ``epoch`` on a panel voting NO and ``converged`` on one voting YES, recording
    nothing. The breaker no longer depends on this (its zero-test is a snapshot of
    the ledger, taken before ``reduce`` runs), but the COUNTS do. A cluster list that
    omits, repeats or invents an issue publishes a number of canonical issues that is
    not one, and with no failure recorded the run would halt correctly and say nothing
    about the reducer that could not count.

    An EMPTY cluster is a non-partition, by the maintainer's ruling of 2026-10-13. It
    omits nothing and moves no halt (``severity`` defaults to NOTE, ``open_severity``
    is ``None``), but ``all`` over no members is ``True``, so it was a NEW cluster in
    every epoch, and as an issue in the change it is a claim about nothing.

    Falling back rather than repairing, which is what ``_groups_to_clusters`` does one
    layer down: the identity is the state before the call, and a reducer that has shown
    it cannot count is trusted with nothing, including the merges it got right. On a
    failed epoch, ``new_clusters`` and the stall counter are therefore exactly what a run
    without a clusterer computes.

    The detail is built from COUNTS. A title is model-authored text on its way to a
    published artefact (#118, #121), and which findings went missing is not named, as
    ``SeamFailure`` already states for ``adjudicate``.
    """
    placed: dict[str, int] = {}
    for c in clusters:
        for m in c.members:
            placed[m.key] = placed.get(m.key, 0) + 1
    missing = sum(1 for key in ledger if key not in placed)
    repeated = sum(1 for n in placed.values() if n > 1)
    empty = sum(1 for c in clusters if not c.members)
    if missing or repeated or empty:
        raise ValueError(
            f"reduce did not partition the ledger: findings in no cluster: {missing} of "
            f"{len(ledger)}; findings placed more than once: {repeated}; "
            f"empty clusters: {empty}")
    return clusters


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
            exception is contained as a meta finding on that persona's report. What it
            returns is copied as it enters the run (#155); a return the engine cannot
            copy is contained as a raise is, at the highest severity it claimed, or
            UGLY where that cannot be read (fail closed).
        gather: Produces the review surface for each epoch. May raise; a failure
            after at least one epoch has completed halts the run as
            ``SURFACE_LOST`` and returns what those epochs produced, and a failure
            with no completed epoch propagates (there is nothing to report, and a
            surface that cannot be built is an operator error, never a review).
        adjudicate: Verifies a finding against source; False refutes/drops it.
        reduce: Folds the deduped ledger into canonical clusters (semantic dedup);
            defaults to identity (one cluster per signature). Feeds the stall
            counter AND the human-facing grouping. Its clusters must partition the
            ledger, or the reduce has failed (#150). No zero-test on open severity
            reads them: those are a snapshot of the ledger taken before it runs.
        human_gate: Called after each epoch; may stop the loop. May raise, and may
            return something that is not a ``GateDecision``: either is contained as a
            ``GateFailure`` on that epoch, a ``GateDecision(stop=False, asked=None)``
            is substituted, and the run CONTINUES (#129). What is stored is always a
            decision the engine constructed, so a subclass's extra state is dropped.
        budget_spent: Returns cumulative output tokens spent (for the budget gate).
        clock: Returns a monotonic time in seconds (for the time gate).
        checkpoint: Optional persistence hook, called each epoch for resumability.
        scope_complete: Predicate: has the in-scope surface been fully covered?

    ``adjudicate``, ``reduce``, ``budget_spent``, ``clock`` (after its first call),
    ``scope_complete`` and ``checkpoint`` may each raise, or return something unusable,
    and neither discards the run (#148). The failure is recorded as a ``SeamFailure`` on
    the epoch and its value is UNKNOWN, never substituted by a measurement: a finding
    stays unadjudicated (``verified=None``, still open), the reduce is the identity, an
    unknown scope cannot support ``CONVERGED``, nothing is persisted, and an unknown
    token or time reading halts the run ``UNMEASURED`` where that ceiling is armed and is
    only recorded where it is not. The FIRST ``clock()`` precedes any spend and is not
    guarded, so a failure there propagates.

    Every record a seam is handed is a copy the engine built (#155): ``adjudicate``'s
    finding, ``reduce``'s findings, ``scope_complete``'s epoch, ``checkpoint``'s run and
    the gate's run and epoch. A seam's only channel into the run is its return value. A
    record that cannot be copied is never handed over live: the seam is not called, and
    that is its recorded failure, with a detail naming the copy.

    Returns:
        A ``ReviewRun`` with per-epoch results, the deduped findings ledger, and
        the halt reason.
    """
    review_run = ReviewRun()
    quorum = panel.effective_quorum
    # A persona's NAME is the one value the engine itself writes into the record from its
    # caller (a contained spawn failure's ``persona``), so it must be one the copy can keep
    # (#155): the sixth pre-merge review measured a ``StrEnum`` name refusing every later
    # copy of the run, and every report a healthy spawn returned under it. Checked here,
    # before anything is spent, so it is an operator error that propagates (#18), as the
    # first ``clock()`` below is. The shipped CLI's names are always exact ``str``.
    if not all(type(name) is str for name, _ in panel.personas):
        raise TypeError("every persona name in the panel must be an exact str")
    # NOT guarded, deliberately (#148): this precedes any spend, so a raise loses nothing,
    # which is #18's doctrine as ``gather`` applies it with no completed epoch. Guarding it
    # for symmetry would fabricate a run in which nothing happened. ``float`` here so a
    # hostile return fails HERE, before the panel is paid for, rather than at every
    # epoch's subtraction after it.
    start = _as_clock_reading(clock())
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
            #
            # Rendered by ``render_exception``, never by interpolating ``exc``
            # here: an exception that cannot be rendered raised from inside this
            # handler and discarded the run it exists to keep (#142).
            collapsed = " ".join(render_exception(exc).split())
            review_run.surface_failure = SurfaceFailure(
                epoch=epoch, detail=bounded_diagnosis(collapsed),
                detail_chars=len(collapsed))
            review_run.halt_reason = HaltReason.SURFACE_LOST
            break

        # Fan out: one persona per subagent. Real backends spawn a subprocess
        # per persona, so run them concurrently (subprocess calls release the
        # GIL, so threads parallelise fine).
        def _run(pm: tuple[str, str]) -> PersonaReport:
            returned: object = None
            try:
                # Copied ONCE, here, as it enters the run (#155, by the maintainer's
                # ruling of 2026-10-14), so the run's record holds only values the
                # engine built (a reduce's cluster titles are checked where they are
                # rebuilt, and a gate's foreign ``asked`` is copied as ``None``). A
                # value the engine cannot copy, kept in the record, was measured
                # refusing EVERY later copy of the run, so the gate was
                # never reached again; refused here, it costs this persona's report
                # and nothing after it. And a backend that kept a reference to what it
                # returned and wrote it later now writes nothing the run holds. What
                # comes back must be exactly a ``PersonaReport`` (the copy refuses a
                # subclass, and copies any other exact record it is given), or the
                # spawn has failed like any other.
                returned = spawn(pm[0], pm[1], surface, epoch)
                report = handed(returned)
                if type(report) is not PersonaReport:
                    raise UncopiableRecord("spawn returned a value that is not a report")
                return report
            except Exception as exc:  # noqa: BLE001
                # A persona is a fallible black box, and one that raises must not
                # take the panel's other reviews with it (#25): the exception used
                # to propagate out of pool.map and end a run that had already paid
                # for everyone. Recorded as a finding rather than swallowed, and
                # deliberately not BaseException — a human's Ctrl-C still stops it.
                # The status is set HERE, at the source, and is the engine's own:
                # ``SPAWN_FAILED`` means our code raised, or returned what the engine
                # refused (#155), against a backend's
                # ``AGENT_ERROR`` for a runtime that ran and returned no review.
                # A run must be able to tell those apart (#30). Both strings come
                # from the guarded helpers, never from ``exc`` directly (#142).
                #
                # NOTE where the seam raised. Where it RETURNED a report the engine
                # refused (#155), the highest severity that report claimed, or UGLY
                # where that cannot be read, so an UGLY in it still trips the breaker:
                # see ``claimed_severity``. The title says which, in fixed words; where
                # the seam raised it is exactly what it always was.
                claim = claimed_severity(returned)
                return PersonaReport(
                    persona=pm[0], verdict=None,
                    status=PersonaStatus.SPAWN_FAILED,
                    findings=[
                        Finding(pm[0], f"persona failed: {render_exception(exc)}"
                                       f"{_claim_note(claim)}",
                                claim.severity, "meta",
                                evidence=bounded_diagnosis(repr_exception(exc)),
                                about_run=True)])

        if parallel and len(panel.personas) > 1:
            with ThreadPoolExecutor(max_workers=len(panel.personas)) as pool:
                reports = list(pool.map(_run, panel.personas))
        else:
            reports = [_run(pm) for pm in panel.personas]

        # Every injected callable that fails from here on is recorded on this epoch (#148).
        failures: list[SeamFailure] = []

        # Adjudicate BLOCKER/UGLY claims against source; refuted -> dropped.
        #
        # Contained PER CALL, so one finding's failure does not leave the rest of the
        # epoch unadjudicated. A failed call leaves ``verified=None``: unadjudicated,
        # which is exactly what it is, so it still counts open and still trips the
        # breaker. ``False`` would need a refutation, and substituting it for an UNKNOWN
        # would drop the finding, which on an UGLY silences the circuit-breaker. So the
        # verdict is CATEGORICAL (``_as_verdict``), never truthiness: a ``None`` coerced to
        # ``False`` was exactly that ruin, measured on this fix's first shape.
        for report in reports:
            checked: list[Finding] = []
            for f in report.findings:
                # Never an instrument finding (#155), from any backend: it is a fact
                # about the RUN, not a claim about the source, so there is nothing to
                # verify. The engine's own that can be BLOCKER or above is a refused
                # report's failure, carrying the severity that report claimed. Handed to an adjudicator, a claim
                # naming no file was refuted and an UGLY run converged (the third
                # pre-merge review). On main no instrument finding is above NOTE.
                if (f.severity >= Severity.BLOCKER and f.verified is None
                        and not f.about_run):
                    adjudicated, verdict = _call_seam(
                        "adjudicate", failures,
                        lambda f=f: _as_verdict(adjudicate(handed(f), surface)))
                    if adjudicated:
                        # Built from ``f``, which the seam never held (#155): on main
                        # this read the handed object, so a severity it rewrote was
                        # what the ledger kept, and an UGLY run converged.
                        f = Finding(**{**f.__dict__, "verified": verdict})
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
                    # derived by a reporter afterwards. ``HumanGate`` received
                    # the run mutably until #155, so a later walk read the ledger
                    # as it ENDED — and a gate that removed a key made it report a
                    # suppression that never happened, which is a rule firing on
                    # a healthy run. "Set at the source" (#30) governs: a fact
                    # this exact must not rest on a precondition no seam enforces,
                    # including the one #155 now enforces.
                    #
                    # Reported, not kept. Giving the claim a ledger place of its
                    # own means giving it a distinct key, which is #125 wearing a
                    # different hat — so what this loses is stated rather than
                    # quietly fixed: the claim counts toward no total and resets
                    # no stall counter.
                    suppressed.append(Suppression(finding=f, holder=held))

        # The halt's open-severity zero-tests, taken HERE (#150): from the LEDGER, after
        # the insert, and before ``reduce`` or any halting input is called. The ledger is
        # what the panel raised and adjudication kept; the clusters below are whatever an
        # injected ``reduce`` returned, and reading them let ``reduce -> []`` switch the
        # breaker off. Taken NOW rather than when the predicate runs, because every seam
        # called between here and there is handed objects that alias the ledger's own
        # findings (``reduce`` the findings, ``scope_complete`` the epoch's record), and a
        # frozen dataclass without slots can still be written through ``__dict__``: the
        # second pre-merge review measured a ``scope_complete`` doing exactly that
        # converging an UGLY run when the ledger was read live. Main's cluster counts were
        # taken before the halting inputs too. Neither read raises on any value a seam
        # wrote into a finding's severity or verdict (see ``ReviewRun.open_uglies``).
        # Since #155 no seam is handed the ledger's own findings at all, so no seam's
        # write through what it is handed reaches this snapshot, this epoch's or the next
        # (a seam reaching into this frame is outside any in-process guarantee): ``reduce``,
        # ``scope_complete``, ``checkpoint`` and the gate are handed copies, and the
        # ledger entry ``adjudicate`` verified is built from the finding it never held.
        # The snapshot is kept: it is what makes the breaker independent of ``reduce``'s
        # RETURN value, which a copy does nothing about.
        ugly_open = bool(review_run.open_uglies)
        blocker_open = bool(review_run.open_blockers)

        # Layer 2 (reduce/view): fold the whole deduped ledger into canonical
        # clusters. The default is identity (one cluster per signature); a semantic
        # reducer collapses re-worded / re-located dups of a single concept.
        #
        # A failed reduce is the identity reduce, the state before the call (#148), and
        # one that does not partition the ledger has failed (#150).
        ledger_findings = list(review_run.ledger.values())
        reduced, clusters = _call_seam(
            "reduce", failures,
            lambda: _as_partition(_rebuilt_clusters(reduce(handed(ledger_findings)),
                                                    review_run.ledger),
                                  review_run.ledger))
        review_run.clusters = clusters if reduced else _identity_reduce(ledger_findings)

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
            seam_failures=failures,
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

        # The three halting INPUTS (#148). A failed read is ``None``, which the predicate
        # reads as UNKNOWN, never as a measurement: it fires no ceiling, a disarmed
        # ceiling's unknown lets the run continue, and an armed one's halts it
        # ``UNMEASURED`` (the maintainer's ruling; see ``HaltReason``). An unknown scope
        # cannot support ``CONVERGED``, which needs it KNOWN complete, so it reads as
        # ``False`` — inert where convergence does not require it. The breaker reads none
        # of the three. Called in the order they always were, and every one is still
        # called where its predicate is disarmed, so a healthy run's calls are unchanged.
        _, tokens_spent = _call_seam(
            "budget_spent", result.seam_failures, lambda: _as_token_count(budget_spent()))
        _, elapsed_s = _call_seam(
            "clock", result.seam_failures, lambda: _elapsed(clock(), start))
        _, scope_known = _call_seam(
            "scope_complete", result.seam_failures,
            lambda: _as_scope(scope_complete(handed(result))))
        # The decision is a LOCAL, and the loop acts on it, never on ``result.halt`` read
        # back (#155). Every seam below is handed a copy, so nothing can write the field
        # this publishes; the local is what states that the loop does not depend on it.
        halt = _evaluate_halt(
            halting,
            ugly_open=ugly_open,
            blocker_open=blocker_open,
            epochs_done=epoch,
            stall_epochs=stall_epochs,
            tokens_spent=tokens_spent,
            elapsed_s=elapsed_s,
            scope_complete=scope_known is True,
            quorum_met=quorum_met,
            no_persona_reviewed=no_persona_reviewed,
        )
        result.halt = halt
        review_run.epochs.append(result)
        if checkpoint is not None:
            # Nothing persisted, and the record says so (#148). #31 makes this a durable
            # write and states its own outcome; the guard is what it goes behind. The
            # failure is recorded AFTER the call, so a checkpoint never sees its own.
            _call_seam("checkpoint", result.seam_failures,
                       lambda: checkpoint(handed(review_run)))

        if halt is not None:
            review_run.halt_reason = halt
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
            # One copy for both arguments, so ``run.epochs[-1] is result`` holds for
            # the gate as it always has (#155). It is built INSIDE the guard, so even a
            # copy that failed would be a contained gate failure, never a lost run.
            gate_run = handed(review_run)
            returned = human_gate(gate_run.epochs[-1], gate_run)
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
                diagnosis = f"gate returned {type_name(returned)}, not GateDecision"
        except Exception as exc:  # noqa: BLE001
            # The third of three injected seams guarded before #148, which guarded the six
            # #129 did not name (``_call_seam``).
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
            # fault (where the entry point does not absorb it, as ``kuang``'s has
            # since #141), a foreign gate, or a bug in a gate's branch logic.
            #
            # Rendered by the guarded helper, as at the other two seams: this line
            # is OUTSIDE the ``try`` it follows, so nothing contains it (#142).
            diagnosis = render_exception(exc)
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
            # default: until #155 the gate held this ``EpochResult`` mutably, so a
            # gate that wrote a failure record itself and then answered normally
            # published a decision AND a failure for the same epoch — a rule firing
            # on a healthy run. The gate now writes a copy; this stays set at the
            # source, as the loop's suppression record does (#30).
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
        # from ``halt``, which this same seam could rewrite until #155.
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
