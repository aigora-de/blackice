# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""The dependency-injected seams, and the values they exchange.

Every one of these is a ``Protocol``: the engine is backend-agnostic because it
names the shape it needs and never the implementation. A backend supplies them
(see ``blackice.backends.claude_code.session``); the defaults the loop falls back
to are deliberately inert — trust every claim, continue at every gate, reduce to
identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from .findings import Cluster, EpochResult, Finding, PersonaReport, ReviewRun


# A surface is opaque to the loop; the injected callables know how to read it
# (e.g. a git ref/diff, a set of paths). Kept as a free-form payload.
ReviewSurface = object


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


class Reduce(Protocol):
    """Fold the signature-deduped ledger into canonical clusters (the reduce step).

    Must be a *pure function of its input* so the engine's control logic stays
    reproducible given the seam's output (CLAUDE.md — "the engine is deterministic;
    only the agents are not"). It must be a **partition**: every input finding
    appears in exactly one output cluster (never drop a panellist's claim). The
    default is deterministic identity; an LLM implementation lives in a backend.
    """

    def __call__(self, findings: Sequence[Finding]) -> list[Cluster]: ...


@dataclass
class GateDecision:
    """The human's decision at a between-epoch gate."""

    stop: bool = False
    note: str = ""


class HumanGate(Protocol):
    """The HITL touchpoint between epochs (fixes/scope/file-issues/stop)."""

    def __call__(self, result: EpochResult, run: ReviewRun) -> GateDecision: ...
