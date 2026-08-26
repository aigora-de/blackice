# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""The review vocabulary: severities, findings, clusters, and the run record.

The nouns the engine and every backend share. ``Severity`` is the good/bad/ugly
ladder; ``Finding`` is one persona's claim in one epoch; ``Cluster`` is a *view*
over findings a reduce step judged the same concept; ``ReviewRun`` accumulates
the whole loop. Nothing here decides anything — the halting predicate lives in
``halting.py`` and the loop in ``loop.py``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import IntEnum

# Imported at run time, not only for typing: ``ReviewRun.converged`` compares
# against ``HaltReason.CONVERGED`` in its body. The dependency stays one-way —
# ``halting`` imports from here for annotations only.
from .halting import HaltReason


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


@dataclass(frozen=True)
class Cluster:
    """A canonical issue: raw findings a reduce step judged 'the same concept'.

    A cluster is a *view over* the deduped ledger, never a replacement for it —
    every member finding (persona + evidence) stays visible, grouped. ``severity``
    is the ``max`` of its members (UGLY-preserving: a merge can never hide a
    ruin-class finding); ``open_severity`` — the max over members still *open* — is
    what the halt gate reads, so a withdrawn UGLY does not keep the breaker latched.
    """

    members: tuple[Finding, ...]
    title: str = ""

    @property
    def severity(self) -> Severity:
        """Highest severity among all members (UGLY-preserving)."""
        return max((m.severity for m in self.members), default=Severity.NOTE)

    @property
    def open_severity(self) -> Severity | None:
        """Highest severity among *open* members, or ``None`` if all resolved."""
        open_sevs = [m.severity for m in self.members if m.counts_open]
        return max(open_sevs) if open_sevs else None

    @property
    def counts_open(self) -> bool:
        return any(m.counts_open for m in self.members)

    @property
    def key(self) -> str:
        """Deterministic canonical id: the smallest member signature.

        Order-independent; for a singleton cluster it equals that finding's
        ``key`` — so the identity reduce reproduces today's ledger keys exactly.
        """
        return min((m.key for m in self.members), default="")


@dataclass
class PersonaReport:
    """One persona's output for one epoch.

    ``unresolved_severities`` holds the raw severity strings this persona emitted
    that did not resolve to a level — an unreadable value and a declared tie
    alike. It is how a run says that it escalated a severity rather than reading
    it (#24): the value stays visible to the operator, verbatim, instead of the
    finding quietly appearing at a level nobody claimed.
    """

    persona: str
    findings: list[Finding] = field(default_factory=list)
    verdict: str | None = None       # e.g. "SOUND-WITH-CONCERNS" / YES / NO
    tokens: int = 0
    unresolved_severities: list[str] = field(default_factory=list)


@dataclass
class EpochResult:
    """Everything produced by one iteration of the loop."""

    index: int
    reports: list[PersonaReport]
    new_findings: list[Finding]      # material findings not seen in prior epochs
    open_blockers: int               # cluster-level count (canonical issues)
    open_uglies: int                 # cluster-level count (canonical issues)
    new_clusters: list[Cluster] = field(default_factory=list)  # new canonical issues
    halt: HaltReason | None = None


@dataclass
class ReviewRun:
    """The full record of a review loop."""

    epochs: list[EpochResult] = field(default_factory=list)
    ledger: dict[str, Finding] = field(default_factory=dict)  # key -> first sighting
    clusters: list[Cluster] = field(default_factory=list)     # latest epoch's reduce
    halt_reason: HaltReason | None = None

    @property
    def converged(self) -> bool:
        return self.halt_reason is HaltReason.CONVERGED

    # --- raw, finding-level view (every panellist's claim stays visible) ---
    @property
    def open_uglies(self) -> list[Finding]:
        return [f for f in self.ledger.values() if f.severity.is_ugly and f.counts_open]

    @property
    def open_blockers(self) -> list[Finding]:
        return [
            f for f in self.ledger.values()
            if f.severity is Severity.BLOCKER and f.counts_open
        ]

    # --- canonical, cluster-level view (what the halt gate counts) ---
    # ``open_severity`` is the max over a cluster's *open* members, so these are
    # zero exactly when the finding-level lists above are (a withdrawn UGLY does
    # not count) — the breaker/convergence *timing* is unchanged, only the reported
    # magnitude collapses to canonical issues.
    @property
    def open_ugly_clusters(self) -> list[Cluster]:
        return [c for c in self.clusters if c.open_severity is Severity.UGLY]

    @property
    def open_blocker_clusters(self) -> list[Cluster]:
        return [c for c in self.clusters if c.open_severity is Severity.BLOCKER]
