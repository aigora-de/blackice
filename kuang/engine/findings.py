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

    ``about_run`` says this finding reports the **instrument**, not the change
    under review (#73) — ``agent error: …`` (#29), ``persona failed: …`` (#25),
    and the contract parser's diagnoses. Without it an operator's headline count
    adds one failure of the panel to one defect in the code, which are not the
    same kind of thing.

    **Set by the code that produces the finding, and never matched out of
    ``claim_class``.** That is a measurement, not a preference: ``claim_class`` is
    a model-controlled string taken from the persona's own JSON, and ``meta`` is
    an ordinary word in review — a finding about a metaclass or about metadata is
    naturally labelled that way. Excluding ``claim_class == "meta"`` from the
    ledger was measured to turn a run that halts ``escalate_ugly`` with one open
    UGLY into one that halts ``converged`` with none: a consumer-side match on
    model data **unlatches the circuit-breaker**, which is #24's and #26's
    doctrine one field along. A finding built from a persona's reply therefore
    never carries this flag, because it is not something a persona can say.

    Deliberately outside ``key``: the flag changes what a run *reports*, never
    which findings are the same finding, so ledger identity is unchanged and
    prior runs still match.
    """

    persona: str
    title: str
    severity: Severity
    claim_class: str                 # coarse category, used for dedup
    file: str | None = None
    line: int | None = None
    evidence: str = ""
    verified: bool | None = None
    about_run: bool = False          # reports the instrument, not the change (#73)

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
    def about_run(self) -> bool:
        """True if ANY member reports the instrument (#73).

        ``max``-preserving, exactly as ``severity`` is, and for the same reason: a
        reduce step must not be able to launder an instrument failure into the
        change's issue count by merging it with a real finding.
        """
        return any(m.about_run for m in self.members)

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
    tools it was denied but still voted without (#67), a review that called no
    tool at all (#70) — is orthogonal: such a persona both contributed and
    degraded, so forcing it into this enum would mean choosing one and losing the
    other. Those live as their own fields on ``PersonaReport``, and this
    vocabulary did not have to reopen to take them — #67 landed as
    ``denied_tools`` and #70 as ``turns``, and this enum is unchanged.

    The line that draws it: **did a review happen** is an outcome and belongs
    here; **how well was it grounded** is a fact about a review that did happen
    and belongs beside it.

    That line was written expecting a persona with **nothing to review at all**
    (#69) to fall on THIS side of it and land as a member — an asymmetry argued
    rather than assumed, and offered to be broken on the record. It was. Measured,
    the state it named is **unreachable**: an empty surface raises ``SurfaceError``
    in both of the backend's modes and the run stops before a persona exists (#18),
    so the persona that IS reachable reviewed a *partial* surface — it contributed,
    and it was under-grounded. A member would have named a state the code refuses
    to enter. #69 therefore landed beside this enum as well, and not even on
    ``PersonaReport``: the surface is gathered once per EPOCH and handed to every
    persona unchanged, so it is not a fact about one spawn at all.

    What survives is the line itself. What did not survive is a prediction about
    which side a case falls on, made before anyone checked whether it can happen.
    """

    UNREPORTED = "unreported"        # the backend did not say (the default)
    CONTRIBUTED = "contributed"      # a readable review with at least one finding
    FOUND_NOTHING = "found_nothing"  # a readable review with no findings
    UNREADABLE = "unreadable"        # replied; the reply was not a readable review
    AGENT_ERROR = "agent_error"      # the process ran, no review came back (#29)
    SPAWN_FAILED = "spawn_failed"    # our own code raised inside the seam (#25)
    NOT_SPAWNED = "not_spawned"      # no call was made (dry run)

    @property
    def reviewed(self) -> bool:
        """Whether a review of record came back — material that enters the run.

        The two statuses whose persona produced findings and a vote. Used to
        decide whether a degradation *of a review* may be reported against this
        persona at all: a call that errored, crashed, or never happened has no
        review to be ungrounded (#70), and reporting one against it would say the
        same failure twice under two headings.
        """
        return self in (PersonaStatus.CONTRIBUTED, PersonaStatus.FOUND_NOTHING)

    @property
    def did_not_review(self) -> bool:
        """Whether the run KNOWS this persona produced no review at all (#72).

        Deliberately **not** the negation of ``reviewed``. ``UNREPORTED`` is the
        default and means the backend did not say, and a run must not report a
        degradation it did not measure (#70's rule, one field along) — so silence
        is excluded from both sides. What is left is the three ways a call can end
        with no review of record plus the one where no call was made, and a member
        added later joins this side unless someone puts it on ``reviewed``, which
        is the safe default for an enum whose new members are failure outcomes.

        This is the **participation** axis, and the whole of what a vote may be
        gated on. Three axes meet on a ``PersonaReport`` and #82's central warning
        is that they get conflated, so they are named here once:

        * **participation** — *did a review happen?* This enum (#30, #72).
        * **grounding** — *was the review that happened worth anything?*
          ``turns`` (#70), ``denied_tools`` (#67), a cancelled grant (#77).
          Fields beside this status, never members of it.
        * **quorum and coverage** — *did enough of the right lenses agree?* (#82),
          and derivable from neither of the others.

        A vote is a claim **about a review**. A persona that produced none made no
        claim, so not counting it discards nothing. A persona that reviewed badly
        made a claim, and dropping that would be the tool judging — which #24, #26
        and #67 all refused. That is why this property is on the first axis alone.
        """
        return self is not PersonaStatus.UNREPORTED and not self.reviewed


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

    ``turns`` is what the backend's runtime said this persona's REVIEW cost in
    turns (#70) — the fourth orthogonal degradation, and the one that is invisible
    everywhere else: a reviewer that called no tool answered from the prompt alone,
    and its process succeeded, its tools were granted, and it voted. The raw count
    is kept rather than a verdict on it, for #24's reason — the operator decides
    what a count means, and a backend that reports a number nobody can interpret
    at least reports it. **0 means the backend did not say**; a persona that was
    never spawned is at 0 and has suffered nothing. What a given count implies is
    the backend's knowledge, not this vocabulary's: see
    ``backends.claude_code.spawn.called_no_tool``.
    """

    persona: str
    findings: list[Finding] = field(default_factory=list)
    verdict: str | None = None       # AFFIRMATIVE_VERDICT / "NO" / None
    tokens: int = 0
    unresolved_severities: list[str] = field(default_factory=list)
    unresolved_verdict: str | None = None
    status: PersonaStatus = PersonaStatus.UNREPORTED
    denied_tools: list[str] = field(default_factory=list)
    turns: int = 0

    @property
    def counted_vote(self) -> bool:
        """Whether this persona's verdict counts toward quorum.

        **A vote is the word, or it is not a vote** (#26). The count used to be a
        prefix match, so the contract's own placeholder ``"YES | NO"`` voted for
        convergence — and so would any hedge a persona chose to open with. This is
        the last line, not the only one: a backend normalises its own replies, but
        the engine takes any ``SpawnPersona`` and must not be talked into a good
        verdict by one. ``str()`` because a backend may hand back a non-string
        verdict and the tally sits outside the spawn seam's guard, where a crash
        would end the run (#25). A report with no verdict is not a YES, which is
        what keeps a crashed or unreadable persona from helping produce a
        ``CONVERGED`` verdict.

        **A vote is also a claim ABOUT A REVIEW**, so a persona the run knows
        produced no review does not cast one (#72): a dry run's whole panel
        returned a well-formed ``"YES"`` from a call nobody made, met unanimity,
        and halted on the good verdict. Keyed on the status — a categorical fact
        set at the source — and never on the verdict text. Nothing is discarded:
        the verdict stays on the report and in the run's output, it is not counted.

        Deliberately the **participation** axis only. A persona that reviewed
        *badly* made a claim, and dropping it would move quorum for a reason nobody
        can see and could silently unlatch the breaker (#24, #26, #67). How well
        that review was grounded is a separate axis, and #82 reports it *beside*
        the verdict rather than gating it here.

        A property rather than a line in the loop because #82 must report which
        votes a verdict rests on and how many of those were grounded — which needs
        the voters, not just the count. A second copy of this predicate in a
        reporter would be somewhere for the printed number and the gated number to
        drift apart, which is what #72 found the denominator had already done.
        """
        return (not self.status.did_not_review
                and str(self.verdict or "").strip().upper() == AFFIRMATIVE_VERDICT)


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

    # --- what the findings are ABOUT: the change, or the run itself (#73) ---
    # A partition of the ledger, never a filter of it. Both halves stay in
    # ``ledger`` and in the artefact — a failed persona's diagnosis has to remain
    # recoverable — and only the reported COUNT is split.
    @property
    def change_findings(self) -> list[Finding]:
        """Findings about the code under review: what the panel was convened for."""
        return [f for f in self.ledger.values() if not f.about_run]

    @property
    def run_findings(self) -> list[Finding]:
        """Findings about the instrument: what went wrong while reviewing."""
        return [f for f in self.ledger.values() if f.about_run]

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
