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
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Callable, Protocol, Sequence


# =============================================================================
# Severity & halting vocabulary
# =============================================================================

class Severity(IntEnum):
    """Finding severity, ordered so comparisons are meaningful."""

    NOTE = 0          # observation, no action required
    NON_BLOCKING = 1  # should be addressed or tracked, but not a gate
    BLOCKER = 2       # must be resolved-or-tracked before convergence
    UGLY = 3          # ruin-class: circuit-breaker + non-negotiable gate

    @property
    def is_bad(self) -> bool:
        """BAD = a correctness/quality/scope finding below the ruin line."""
        return self in (Severity.NOTE, Severity.NON_BLOCKING, Severity.BLOCKER)

    @property
    def is_ugly(self) -> bool:
        return self is Severity.UGLY


class HaltReason(Enum):
    """Why the loop stopped. Every reason returns the accumulated report."""

    CONVERGED = "converged"            # good: no open blocker/ugly, scope complete
    ESCALATE_UGLY = "escalate_ugly"    # ruin-class finding -> circuit-break
    BUDGET = "budget"                  # token or time ceiling reached
    EPOCH = "epoch"                    # max epochs reached
    STALL = "stall"                    # no new material findings, blockers still open
    ABORTED = "aborted"                # human gate stopped the loop


# =============================================================================
# Findings & reports
# =============================================================================

@dataclass(frozen=True)
class Finding:
    """One issue raised by one persona in one epoch.

    ``verified`` records adjudication against source: ``None`` = not yet checked,
    ``True`` = confirmed, ``False`` = refuted (dropped; the panellist "withdrew on
    the evidence"). Only confirmed/unadjudicable findings count toward halting.
    """

    persona: str
    title: str
    severity: Severity
    claim_class: str                 # coarse category, used for dedup
    file: str | None = None
    line: int | None = None
    evidence: str = ""
    verified: bool | None = None

    @property
    def key(self) -> str:
        """Stable signature for semantic dedup across epochs.

        Deliberately coarse (file + line-bucket + claim-class + severity) so a
        persona re-wording the *same* finding does not read as *new material*.
        """
        bucket = "" if self.line is None else str(self.line // 10)
        raw = f"{self.file}|{bucket}|{self.claim_class}|{self.severity.name}"
        return hashlib.sha1(raw.encode()).hexdigest()[:12]

    @property
    def counts_open(self) -> bool:
        """A finding is 'open' unless it was refuted by adjudication."""
        return self.verified is not False


@dataclass
class PersonaReport:
    """One persona's output for one epoch."""

    persona: str
    findings: list[Finding] = field(default_factory=list)
    verdict: str | None = None       # e.g. "SOUND-WITH-CONCERNS" / YES / NO
    tokens: int = 0


@dataclass
class EpochResult:
    """Everything produced by one iteration of the loop."""

    index: int
    reports: list[PersonaReport]
    new_findings: list[Finding]      # material findings not seen in prior epochs
    open_blockers: int
    open_uglies: int
    halt: HaltReason | None = None


@dataclass
class ReviewRun:
    """The full record of a review loop."""

    epochs: list[EpochResult] = field(default_factory=list)
    ledger: dict[str, Finding] = field(default_factory=dict)  # key -> first sighting
    halt_reason: HaltReason | None = None

    @property
    def converged(self) -> bool:
        return self.halt_reason is HaltReason.CONVERGED

    @property
    def open_uglies(self) -> list[Finding]:
        return [f for f in self.ledger.values() if f.severity.is_ugly and f.counts_open]

    @property
    def open_blockers(self) -> list[Finding]:
        return [
            f for f in self.ledger.values()
            if f.severity is Severity.BLOCKER and f.counts_open
        ]


# =============================================================================
# Inputs: the spec, the halting set, the ensemble, the surface
# =============================================================================

@dataclass
class ReviewSpec:
    """The *why / what / scope* — never the *how* (that is the skill/ensemble)."""

    why: str                          # the risk being guarded against
    what: str                         # the change / module / execution path
    in_scope: Sequence[str] = ()      # what this review must cover
    out_of_scope: Sequence[str] = ()  # explicitly deferred (still tracked)


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


@dataclass
class PanelConfig:
    """The reviewer ensemble (the 'how' — supplied by the skill).

    ``personas`` is an ordered list of (name, mandate) pairs; the mandate is the
    adversarial instruction handed to each subagent.
    """

    personas: list[tuple[str, str]]
    quorum: int | None = None         # min YES verdicts for convergence (default: all)


# A surface is opaque to the loop; the injected callables know how to read it
# (e.g. a git ref/diff, a set of paths). Kept as a free-form payload.
ReviewSurface = object


# =============================================================================
# Injected seams (Protocols)
# =============================================================================

class SpawnPersona(Protocol):
    """Run one reviewer persona over the surface and return its report."""

    def __call__(
        self, persona: str, mandate: str, surface: ReviewSurface, epoch: int
    ) -> PersonaReport: ...


class GatherSurface(Protocol):
    """Produce the review surface for an epoch (e.g. current diff)."""

    def __call__(self, epoch: int) -> ReviewSurface: ...


class Adjudicate(Protocol):
    """Verify a finding's claim against source. Return False to refute (drop)."""

    def __call__(self, finding: Finding, surface: ReviewSurface) -> bool: ...


@dataclass
class GateDecision:
    """The human's decision at a between-epoch gate."""

    stop: bool = False
    note: str = ""


class HumanGate(Protocol):
    """The HITL touchpoint between epochs (fixes/scope/file-issues/stop)."""

    def __call__(self, result: EpochResult, run: ReviewRun) -> GateDecision: ...


def _trust_all(finding: Finding, surface: ReviewSurface) -> bool:  # noqa: ARG001
    return True


def _auto_continue(result: EpochResult, run: ReviewRun) -> GateDecision:  # noqa: ARG001
    return GateDecision(stop=False)


# =============================================================================
# The loop
# =============================================================================

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
) -> HaltReason | None:
    """Pure predicate: return a HaltReason if any halting condition is met.

    Order matters: the UGLY circuit-breaker is checked first and short-circuits
    even an unexhausted budget. CONVERGED requires no open ugly AND no open
    blocker AND (optionally) scope complete AND quorum.
    """
    # 1. Circuit-breaker: a ruin-class finding halts and escalates immediately.
    if result.open_uglies > 0:
        return HaltReason.ESCALATE_UGLY

    # 2. Convergence (good): nothing dangerous or blocking is open.
    scope_ok = scope_complete or not halting.require_scope_complete
    if result.open_blockers == 0 and scope_ok and quorum_met:
        return HaltReason.CONVERGED

    # 3. Resource ceilings (partial halts — BADs may remain, but must be tracked).
    if halting.token_budget is not None and tokens_spent >= halting.token_budget:
        return HaltReason.BUDGET
    if halting.time_budget_s is not None and elapsed_s >= halting.time_budget_s:
        return HaltReason.BUDGET
    if epochs_done >= halting.max_epochs:
        return HaltReason.EPOCH

    # 4. Stall: no new material findings for K epochs while blockers persist.
    if stall_epochs >= halting.stall_patience and result.open_blockers > 0:
        return HaltReason.STALL

    return None


def run(
    spec: ReviewSpec,
    halting: HaltingSet,
    panel: PanelConfig,
    *,
    spawn: SpawnPersona,
    gather: GatherSurface,
    adjudicate: Adjudicate = _trust_all,
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
        spawn: Runs one persona subagent over the surface.
        gather: Produces the review surface for each epoch.
        adjudicate: Verifies a finding against source; False refutes/drops it.
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
            return spawn(pm[0], pm[1], surface, epoch)

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

        # Dedup vs the running ledger to find *new material* findings.
        new_findings: list[Finding] = []
        for report in reports:
            for f in report.findings:
                if not f.counts_open:
                    continue
                if f.key not in review_run.ledger:
                    review_run.ledger[f.key] = f
                    new_findings.append(f)

        open_blockers = len(review_run.open_blockers)
        open_uglies = len(review_run.open_uglies)
        result = EpochResult(
            index=epoch,
            reports=reports,
            new_findings=new_findings,
            open_blockers=open_blockers,
            open_uglies=open_uglies,
        )

        # Stall accounting: only *material* (blocker/ugly) new findings reset it.
        material_new = [f for f in new_findings if f.severity >= Severity.BLOCKER]
        stall_epochs = 0 if material_new else stall_epochs + 1

        yes_votes = sum(1 for r in reports if (r.verdict or "").upper().startswith("YES"))
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


# =============================================================================
# Reference fake ensemble (for tests/demos — no network, deterministic)
# =============================================================================

class FakeEnsemble:
    """A scripted ``SpawnPersona`` that replays canned per-epoch reports.

    ``script[epoch][persona]`` -> ``PersonaReport``. Missing entries yield an
    empty, YES-verdict report (a persona with nothing left to say).
    """

    def __init__(self, script: dict[int, dict[str, PersonaReport]]) -> None:
        self._script = script

    def __call__(
        self, persona: str, mandate: str, surface: ReviewSurface, epoch: int  # noqa: ARG002
    ) -> PersonaReport:
        return self._script.get(epoch, {}).get(
            persona, PersonaReport(persona=persona, verdict="YES")
        )


def _demo() -> ReviewRun:
    """Tiny end-to-end demo: epoch 1 finds a BLOCKER, it is fixed, epoch 2 converges."""
    panel = PanelConfig(personas=[("quant", "find wrong maths"), ("engineer", "find bugs")])
    spec = ReviewSpec(why="records-critical path", what="the guard")
    halting = HaltingSet(max_epochs=5, stall_patience=1, require_scope_complete=False)

    script = {
        1: {
            "quant": PersonaReport(
                persona="quant",
                verdict="NO",
                findings=[Finding("quant", "off-by-one in allocation",
                                  Severity.BLOCKER, "alloc", "x.py", 42)],
            ),
            "engineer": PersonaReport(persona="engineer", verdict="YES"),
        },
        # epoch 2: the blocker was fixed between epochs, both personas clear.
        2: {
            "quant": PersonaReport(persona="quant", verdict="YES"),
            "engineer": PersonaReport(persona="engineer", verdict="YES"),
        },
    }

    # Human gate marks the blocker resolved before epoch 2 by refuting it via
    # adjudication on re-look (here simulated: after epoch 1 we "fix" it, so on
    # epoch 2 nobody re-raises it and the ledger blocker is retired by the gate).
    fixed: dict[str, bool] = {}

    def gate(result: EpochResult, run: ReviewRun) -> GateDecision:
        for f in run.open_blockers:
            fixed[f.key] = True
            run.ledger[f.key] = Finding(**{**f.__dict__, "verified": False})  # resolved
        return GateDecision(stop=False, note="applied fix")

    review_run = run(
        spec, halting, panel,
        spawn=FakeEnsemble(script),
        gather=lambda epoch: f"surface@{epoch}",
        human_gate=gate,
    )
    return review_run


if __name__ == "__main__":
    result = _demo()
    print(f"halt_reason = {result.halt_reason.value}")
    print(f"epochs      = {len(result.epochs)}")
    print(f"converged   = {result.converged}")
    assert result.halt_reason is HaltReason.CONVERGED, result.halt_reason
    assert len(result.epochs) == 2, len(result.epochs)
    print("demo OK")
