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
from enum import Enum, IntEnum

# Imported at run time, not only for typing: ``ReviewRun.converged`` compares
# against ``HaltReason.CONVERGED`` in its body. The dependency stays one-way —
# ``halting`` imports from here for annotations only.
from .halting import HaltReason


# The one string that is a vote. Quorum is a conjunct of CONVERGED, so the set of
# verdicts that can produce a good verdict is the set that can make a run lie —
# and a prefix match let the contract's own placeholder "YES | NO" into it (#26).
# Defined here, in the vocabulary, so a backend normalising a persona's reply and
# the loop counting the result cannot drift apart.
AFFIRMATIVE_VERDICT = "YES"


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


class PersonaStatus(Enum):
    """What became of one persona's turn on the panel.

    A closed set naming the OUTCOME of one spawn, and the answer to the question
    a run could not previously answer at all: a persona that reviewed and found
    nothing, one that was never spawned, and one that does not exist produced
    identical output (#30).

    **Set at the source, never derived.** "Zero findings" is precisely the
    ambiguous signal, so a status computed from the ledger reproduces the defect,
    and recovering "this one failed" by matching a finding's title would be the
    same defect one layer up — and would break the moment the wording changed.
    ``AGENT_ERROR`` and ``SPAWN_FAILED`` are separate members for that reason:
    the backend's ``agent error: …`` (#29) and the engine's ``persona failed: …``
    (#25) are worded differently at the source so they can be told apart here.

    Deliberately an outcome and *only* an outcome. A degradation a persona
    survives — an unreadable severity (#24), a verdict that was not a vote (#26),
    tools it was denied but still voted without (#67) — is orthogonal: such a
    persona both contributed and degraded, so forcing it into this enum would
    mean choosing one and losing the other. Those live as their own fields on
    ``PersonaReport``, and this vocabulary did not have to reopen to take them —
    #67 landed as ``denied_tools`` and this enum is unchanged, as predicted.
    """

    UNREPORTED = "unreported"        # the backend did not say (the default)
    CONTRIBUTED = "contributed"      # a readable review with at least one finding
    FOUND_NOTHING = "found_nothing"  # a readable review with no findings
    UNREADABLE = "unreadable"        # replied; the reply was not a readable review
    AGENT_ERROR = "agent_error"      # the process ran, no review came back (#29)
    SPAWN_FAILED = "spawn_failed"    # our own code raised inside the seam (#25)
    NOT_SPAWNED = "not_spawned"      # no call was made (dry run)


@dataclass
class PersonaReport:
    """One persona's output for one epoch.

    ``unresolved_severities`` holds the raw severity strings this persona emitted
    that did not resolve to a level — an unreadable value and a declared tie
    alike. It is how a run says that it escalated a severity rather than reading
    it (#24): the value stays visible to the operator, verbatim, instead of the
    finding quietly appearing at a level nobody claimed.

    ``unresolved_verdict`` is the same record for the other model-controlled value
    that drives halting (#26). Severity gates the circuit-breaker; the verdict
    gates CONVERGED. A verdict that is not a vote must not silently count and must
    not silently *not*-count either, so the raw string is kept verbatim for the
    human who decides what the persona meant. ``None`` means the persona either
    cast a readable vote or claimed no verdict at all — neither is something we
    misread.

    ``status`` is what the run says about whether this persona took part at all
    (#30). It defaults to ``UNREPORTED`` rather than to any outcome, because a
    default naming one would let a backend's silence pass for a fact; a backend
    that does not set it says so, loudly, in the operator's output.

    ``denied_tools`` names the tools this persona asked for and was refused (#67) —
    the third of the orthogonal degradations, and the one a persona most visibly
    survives: the process succeeded, a well-formed contract came back, and it voted.
    It is a field rather than a ``PersonaStatus`` member for exactly that reason;
    such a persona both contributed *and* degraded, and the enum names an outcome,
    so forcing it in would mean choosing one fact and losing the other. Names only,
    never the refused call's arguments: those are unbounded model-controlled content
    and this record is written into an artefact meant to be shared.

    Empty is not a claim that the persona had every tool. A tool the panel granted
    *and* deny-listed is absent from the reviewer's session rather than refused, so
    no runtime ever reports it here; that half of #67 is deterministic and lives in
    the backend's ``permissions.unavailable_tools``.
    """

    persona: str
    findings: list[Finding] = field(default_factory=list)
    verdict: str | None = None       # AFFIRMATIVE_VERDICT / "NO" / None
    tokens: int = 0
    unresolved_severities: list[str] = field(default_factory=list)
    unresolved_verdict: str | None = None
    status: PersonaStatus = PersonaStatus.UNREPORTED
    denied_tools: list[str] = field(default_factory=list)


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
