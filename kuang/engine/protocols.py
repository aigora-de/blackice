# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""The dependency-injected seams, and the values they exchange.

Every one of these is a ``Protocol``: the engine is backend-agnostic because it
names the shape it needs and never the implementation. A backend supplies them
(see ``kuang.backends.claude_code.session``); the defaults the loop falls back
to are deliberately inert — trust every claim, continue at every gate, reduce to
identity.

**Two rules govern every seam, one for each direction a value can travel.**

* What a seam RETURNS (#129): the engine never reads an unvalidated value off it
  outside the guard. It converts the return categorically inside ``_call_seam``, or
  re-makes it (the ``GateDecision`` the gate hands back).
* What a seam DOES to what it is handed (#155): a seam's only channel into the run is
  its return value. Every record a seam is handed is a copy the engine built
  (``kuang.engine._handed``), so nothing it does to that object reaches the engine's
  decisions or its record. That includes the gate: a human applying fixes, adjusting
  scope or filing issues acts on the world, and retiring a fixed finding is a channel
  into the run, #157's, on a return value.

Neither rule says anything about what a seam does to the WORLD (the repository, the
filesystem, the network). That is the permission policy (#76, #79, #96).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from .findings import Cluster, EpochResult, Finding, PersonaReport, ReviewRun


# A surface is opaque to the loop; the injected callables know how to read it
# (e.g. a git ref/diff, a set of paths). Kept as a free-form payload.
ReviewSurface = object


class SpawnPersona(Protocol):
    """Run one reviewer persona over the surface and return its report.

    An implementation may raise: the loop contains an exception from this seam as
    a ``meta`` finding on that persona's report, so one reviewer cannot end a run
    the panel has already been paid for (#25). It is contained, not hidden — the
    exception reaches the run artefact as a finding.

    What it returns is copied as it enters the run (#155), so a backend that keeps the
    report writes nothing the run holds afterwards. A return the engine cannot copy, or
    that is not a ``PersonaReport``, is contained exactly as a raise is, and the
    failure's finding carries the highest severity the report claimed, or UGLY where
    what it claimed cannot be read (fail closed); its title says which.
    """

    def __call__(
        self, persona: str, mandate: str, surface: ReviewSurface, epoch: int
    ) -> PersonaReport: ...


class GatherSurface(Protocol):
    """Produce the review surface for an epoch (e.g. current diff)."""

    def __call__(self, epoch: int) -> ReviewSurface: ...


class Adjudicate(Protocol):
    """Verify a finding's claim against source. Return False to refute (drop)."""

    def __call__(self, finding: Finding, surface: ReviewSurface) -> bool: ...


class Reduce(Protocol):
    """Fold the signature-deduped ledger into canonical clusters (the reduce step).

    Must be a *pure function of its input* so the engine's control logic stays
    reproducible given the seam's output (CLAUDE.md — "the engine is deterministic;
    only the agents are not"). It must be a **partition**: every input finding
    appears in exactly one output cluster (never drop a panellist's claim), and no
    cluster is empty. The ENGINE enforces this (#150): a return that is not a
    partition is a failed reduce, recorded, and the identity is used instead. The
    default is deterministic identity; an LLM implementation lives in a backend.
    """

    def __call__(self, findings: Sequence[Finding]) -> list[Cluster]: ...


@dataclass(frozen=True)
class GateDecision:
    """What a gate decided between epochs, and whether a human was consulted.

    Not "the human's decision": the engine takes any ``HumanGate``, so a gate may
    decide without a human at all. Saying otherwise here would be a guarantee this
    module cannot provide — which is why ``asked`` exists beside ``stop`` rather
    than being inferred from it.

    ``asked`` is **tri-state and defaults to None**, and the default is the point.
    ``None`` means the gate did not resolve a human's answer; ``False`` means it
    measured that none could be reached (a run with no terminal, which is every
    run in CI); ``True`` means one was asked **and gave an answer the gate
    resolved**. A boolean default would let a gate's silence pass for a
    measurement, which is the defect ``PersonaStatus.UNREPORTED`` exists to refuse
    one seam along — a run must not report a fact it never took.

    ``True`` is deliberately narrower than "a human was asked" (#132). A gate that
    prompts a human and cannot read what comes back has obtained no decision, and
    recording ``True`` beside ``stop=False`` would publish *"a human continued the
    run"* off an answer nobody could read. So ``None`` carries three cases that
    are one fact — the gate said nothing, the answer never resolved, or the seam
    itself failed (#129) — and the cost of that is stated rather than hidden: a
    run where the operator typed past the gate's bound is archived identically to
    one whose gate was never wired. The operator saw the re-prompts, the archive
    did not. The exactly-knowable rule is preferred to a richer one nothing could
    check.

    **#129 narrowed that cost by exactly one case, and widening the claim to cover
    the rest would be the defect this paragraph exists to avoid.**
    ``EpochResult.gate_failure`` is a sibling field, and where it is present the
    third case — the seam failed — is recoverable after all, with the backend's own
    diagnosis beside it. The first two remain indistinguishable from each other: a
    gate that said nothing and a gate that could not read what came back both leave
    ``asked=None`` and no failure record, because neither is a failure of the seam.
    That is the residue, and nothing here recovers it.

    Frozen, like every other record a run stores (``Finding``, ``Cluster``,
    ``Suppression``, ``SurfaceFailure``): the loop keeps this on the epoch it
    followed, and a mutable one would also let a backend returning a single shared
    instance alias every epoch's record to one object.
    """

    stop: bool = False
    asked: bool | None = None


class HumanGate(Protocol):
    """The HITL touchpoint between epochs (fixes/scope/file-issues/stop).

    An implementation may raise, and may return something that is not a
    ``GateDecision``: the loop contains either as a ``GateFailure`` on that epoch,
    substitutes ``GateDecision(stop=False, asked=None)`` and CONTINUES, because a
    gate is not a state the loop cannot proceed from (#129). Contained, not hidden —
    the diagnosis reaches the run artefact.

    **What is stored is a decision the ENGINE constructed**, not the object handed
    back: ``stop`` is coerced with ``bool`` because the loop halts on truthiness, and
    a subclass carrying its own state has that state dropped. An implementation
    needing to record something of its own has no channel here; #130 removed the one
    that existed. What it is HANDED is a copy (#155): one copy of the run, with
    ``result`` as that copy's newest epoch, so writing either reaches nothing.
    """

    def __call__(self, result: EpochResult, run: ReviewRun) -> GateDecision: ...
