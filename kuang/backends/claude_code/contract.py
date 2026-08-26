# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""The findings contract: the prompt a persona is handed, and the reply it owes.

An output contract appended to every prompt asks for exactly one fenced ``json``
block; this module assembles the prompt and parses that block back into
``Finding`` objects. A reply that carries no parseable block becomes a sentinel
report, which is what drives the reformat retry in ``session``.
"""

from __future__ import annotations

import json
import re

from kuang.engine import (AFFIRMATIVE_VERDICT, Finding, PersonaReport,
                          ReviewSpec, Severity)


_SEV = "NOTE | NON_BLOCKING | BLOCKER | UGLY"

# The block the contract asks for, hoisted out of the prompt so that the text we
# ship and the text ``_is_contract_echo`` recognises are one string and cannot
# drift apart (#26). A persona that restates the contract instead of filling it in
# emits exactly this, and it must not be mistaken for a review.
_TEMPLATE_BLOCK = f"""{{"verdict": "YES | NO",
  "findings": [
    {{"title": "...", "severity": "{_SEV}", "claim_class": "short-category",
      "file": "path/or/null", "line": 0, "evidence": "what you checked and found"}}
  ]}}"""

# Compared whitespace-normalised, so a persona that re-indents the template when
# restating it is still restating it.
_ECHO_SIGNATURE = " ".join(_TEMPLATE_BLOCK.split())

# Note the deliberate absence of a fenced ``json`` marker in the prose below: the
# contract used to name one inline, so a persona restating the contract handed the
# extraction regex an opening fence in the middle of a sentence and it captured a
# fragment of prose instead of a payload (#26). We wrote that trip hazard; the
# only literal fence in this string is the template's own.
FINDINGS_CONTRACT = f"""
---
OUTPUT CONTRACT (mandatory). Verify every claim against the source before making
it — read the files, run the tests. Do NOT speculate. End your reply with EXACTLY
one fenced `json` block and nothing after it:

```json
{_TEMPLATE_BLOCK}
```

Severity — write exactly one of these four, uppercase, alone in the field:
  NOTE          an observation; no action required.
  NON_BLOCKING  should be addressed or tracked, but does not gate approval.
  BLOCKER       must be fixed or tracked before approval.
  UGLY          a ruin-class hazard (non-linear/multiplicative/cascading/
                irreversible) — use it only for that.
Put no qualifier, parenthetical or reasoning in that field; it belongs in
"evidence". If you genuinely cannot decide between two levels, say so rather than
inventing a certainty you do not have: name both (e.g. "BLOCKER/UGLY") and give
your reasons in "evidence". Anything that does not resolve to one of the four
levels — declared ties included — is recorded verbatim and escalated to a human as
BLOCKER; we never pick a level on your behalf.

Verdict — write exactly one word, uppercase, alone in the field: YES or NO.
Write YES only if nothing you found gates approval — that is, if you raised
no BLOCKER and no UGLY. Otherwise write NO. Put no qualifier, condition or
reasoning in that field: conditions and reservations belong in the findings,
where their severity already says whether they gate. A field holding anything
but one of those two words is not a vote, and reaches a human verbatim.
"""

# The level an unresolvable severity takes. Deliberately not NOTE (#24: a silent
# downgrade lost the circuit-breaker) and deliberately not UGLY: defaulting *up*
# is still guessing at a severity, and a breaker that trips on a typo is one an
# operator learns to route around. BLOCKER is the lowest level that cannot be
# swallowed — ``_evaluate_halt`` needs zero open blockers to reach CONVERGED — so
# a value nobody could read cannot produce a good verdict, and a human resolves it.
UNRESOLVED_SEVERITY = Severity.BLOCKER

# Words that turn a level named later in the string into prose about it rather
# than a competing claim: "BLOCKER (not UGLY)" is a BLOCKER.
_NEGATORS = frozenset({"NOT", "NO", "NEVER"})

# A model-controlled string reaches the run artefact; bound what we keep of it.
# One cap for both values we record verbatim — a severity we could not resolve
# (#24) and a verdict that was not a vote (#26).
_RAW_VALUE_CAP = 120

# The two words that are a verdict. Anything else is not one: no prefix match, no
# leading-label rule, no synonyms (#26). Unlike a severity, a verdict is a vote,
# and it is not a value a decorated string can express more precisely — so nothing
# decorated is read. The affirmative comes from the engine, which is what counts
# the votes, so the two cannot disagree about what a YES is.
_VERDICTS = (AFFIRMATIVE_VERDICT, "NO")


def _tokens(raw: str) -> list[str]:
    """The vocabulary words in a field, case- and punctuation-insensitive.

    Shared by the two normalisers so that ``"  blocker "`` and ``"YES."`` are read
    the same way; what each does with the resulting list is where they differ.
    """
    return [t for t in re.split(r"[^A-Z]+", raw.upper()) if t]


def normalise_verdict(raw: str) -> str | None:
    """Resolve a persona's verdict to a vote, or ``None`` if it is not one.

    The rule is categorical: **the field holds exactly one word, and that word is
    YES or NO**. No prose is read. ``"YES"``, ``"yes"`` and ``"YES."`` are votes;
    ``"YES | NO"`` (the contract's own placeholder, echoed back), ``"YES, no
    issues found"``, ``"SOUND-WITH-CONCERNS"`` and ``"MAYBE"`` are not.

    Deliberately *not* ``normalise_severity``'s leading-label rule, and the
    difference is the point. That rule exists to read a decorated value, because a
    severity legitimately carries prose about itself and #24 sanctions a persona
    declaring a tie. A verdict is a vote: quorum is a conjunct of CONVERGED, so
    the strings that resolve here are exactly the strings that can make a run
    report a good verdict, and a vote must not be decorated at all. Two literals
    is the smallest that surface can be — there is no phrase left for a reply to
    invent its way into.

    The asymmetry is deliberate too: a verdict wrongly counted manufactures
    CONVERGED and the run lies, while one wrongly *un*counted costs an epoch and
    halts on EPOCH/BUDGET with every finding visible and a human deciding. So the
    caller records what did not resolve rather than guessing at it — an
    unrecognised verdict must not silently count, and must not silently
    not-count either.

    Returns:
        ``AFFIRMATIVE_VERDICT``, ``"NO"``, or ``None`` if the field is not a vote.
    """
    tokens = _tokens(raw)
    if len(tokens) == 1 and tokens[0] in _VERDICTS:
        return tokens[0]
    return None


def normalise_severity(raw: str) -> Severity | None:
    """Resolve a persona's severity string to a level, or ``None`` if we cannot.

    The rule is positional: **the severity is the leading label, and whatever
    follows it is prose**. So ``"UGLY (ruin-class)"`` is UGLY and
    ``"BLOCKER (not UGLY)"`` is BLOCKER, while ``"not a blocker"`` and
    ``"non blocker"`` resolve to nothing — neither *begins* with a word of the
    vocabulary, and choosing a level for them would be guessing, which is how #24
    happened. ``"non-blocking"`` does resolve, because it *is* ``NON_BLOCKING``
    modulo case and separators; the leading pair is checked before the leading
    token so the two-word name survives being split.

    One guard: a second, un-negated level word later in the string means the
    persona named two levels and chose neither (``"Blocker/Ugly"``, or the output
    contract echoed back verbatim). That is a legitimate thing for an undecided
    reviewer to say and the contract invites it — so we do not resolve it either,
    and it goes to a human. ``max()`` over the levels named is deliberately not
    used: a cluster's max is UGLY-preserving because merging two *findings* must
    not hide a ruin-class one, but a compound string is one persona declining to
    pick, and picking for it is the guess this parser exists to refuse.

    Returns:
        The resolved ``Severity``, or ``None`` if the value names no level, or
        names more than one.
    """
    tokens = _tokens(raw)
    pair = "_".join(tokens[:2])
    if pair in Severity.__members__:
        label, rest = Severity[pair], tokens[2:]
    elif tokens and tokens[0] in Severity.__members__:
        label, rest = Severity[tokens[0]], tokens[1:]
    else:
        return None
    for i, token in enumerate(rest):
        other = Severity.__members__.get(token)
        if other is not None and other is not label:
            if i and rest[i - 1] in _NEGATORS:
                continue                     # "(not UGLY)" is prose about the label
            return None                      # two levels named, neither chosen
    return label


def _coerce_line(raw: object) -> int | None:
    """Read a location best-effort: the first integer in the value, or ``None``.

    Model-shaped locations are ordinary — ``"~120"``, ``"42-58"``, ``"L42"`` — and
    every one of them used to raise ``ValueError`` out of a worker thread and kill
    the run (#25). So: the first run of digits wins (a range yields its start,
    which is where a human looks), a container or a value with no digits yields no
    line, and ``bool`` is rejected explicitly because it is an ``int`` subclass and
    ``true`` is not line 1 — the same trap ``cluster._extract_cluster_groups``
    guards against.

    Best-effort here and refuse-to-guess for severity (``normalise_severity``) are
    one doctrine, not two: severity feeds the circuit-breaker and the convergence
    gate, so guessing it can lose ruin or manufacture a halt, while a location
    drives nothing — ``Finding.key`` buckets it by ``// 10`` precisely because it
    is approximate, and a carried-in location is labelled advisory (``memory``).

    A line is a positive integer or it is absent: ``0`` and ``"0"`` now agree (both
    mean "no line"), and a negative number is no line rather than a nonsense one.
    """
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        n = int(raw)
    elif isinstance(raw, str):
        match = re.search(r"\d+", raw)
        if match is None:
            return None
        n = int(match.group())
    else:
        return None                      # a list or object names no single line
    return n if n > 0 else None


def _contract_violation(persona: str, what: str, evidence: str) -> PersonaReport:
    """The contract-miss sentinel, for a payload whose findings are unreachable.

    Deliberately the shape ``_is_parse_failure`` keys on, so ``session.spawn``
    fires its existing reformat retry and the review is recovered rather than
    discarded. ``verdict`` is dropped with it: a reply we could not read must not
    vote (quorum counts YES votes), which is what stops an unparseable persona
    contributing to a good verdict.
    """
    return PersonaReport(persona=persona, verdict=None, findings=[
        Finding(persona, f"findings contract violated: {what}",
                Severity.NOTE, "meta", evidence=evidence[:400])])


def build_prompt(spec: ReviewSpec, surface: str, epoch: int, prior: str,
                 surface_kind: str = "diff", seeded: str = "") -> str:
    """Assemble the per-epoch review task handed to every persona.

    ``surface_kind`` selects the framing: ``"diff"`` reviews a change,
    ``"files"`` reviews existing code presented as whole files (issue #6).

    ``prior`` is this run's cross-*epoch* memory and only means anything from
    epoch 2.  ``seeded`` is cross-*run* memory loaded by ``--prior-findings``
    (issue #13) and is injected from **epoch 1** — a re-run whose whole purpose is
    confirming fixes may only ever get one epoch, so gating it behind ``epoch > 1``
    would withhold it from the run that needs it most.
    """
    is_diff = surface_kind == "diff"
    subject = "this change" if is_diff else "this code"
    what_label = "WHAT CHANGED" if is_diff else "WHAT TO REVIEW"
    surface_label = "diff" if is_diff else "files"
    scope = ""
    if spec.in_scope:
        scope += "\nIN SCOPE: " + "; ".join(spec.in_scope)
    if spec.out_of_scope:
        scope += "\nOUT OF SCOPE (deferred, do not fault): " + "; ".join(spec.out_of_scope)
    memory = ""
    if seeded:
        # Locations are advisory: the fixes that prompted the re-run move code, so
        # a carried-in file:line is a hint about where to look, never a claim about
        # what is there now.  Personas re-read the live surface every epoch and
        # adjudicate against it.
        memory += ("\n\nPRIOR RUN'S FINDINGS (carried in from an earlier review; "
                   "fixes may have landed since, so the file:line of each is "
                   "ADVISORY and may have moved — adjudicate against the current "
                   "source. Say explicitly which are now resolved, which are still "
                   "open, and then look for what that run missed):\n"
                   f"{seeded}\n")
    if epoch > 1 and prior:
        memory += ("\n\nPRIOR EPOCHS' FINDINGS (this run — build on these; say if "
                   f"they are resolved, and look for what they missed):\n{prior}\n")
    return (
        f"Adversarially review {subject}. Be critical: find where it is wrong, "
        f"incomplete, or dangerous — approve only what you cannot break.\n\n"
        f"WHY THIS MATTERS: {spec.why}\n{what_label}: {spec.what}{scope}\n"
        f"{memory}\n--- REVIEW SURFACE ({surface_label}) ---\n{surface}\n{FINDINGS_CONTRACT}"
    )


def extract_json_blocks(text: str) -> list[str]:
    """Return every fenced ``json`` block in a reply, in the order they appear.

    One implementation of the contract's extraction, shared by the findings
    parser and the clusterer (``cluster``) — which previously wrote the same
    regex twice, so a fix to one silently left the other.

    Extraction only, and the plural exists to keep it that way: choosing *which*
    block a reply meant is a policy that differs between the callers (#26 makes
    the findings parser skip a block that is verbatim our own template), so it
    belongs at the call site rather than being averaged into one rule here.
    """
    return re.findall(r"```json\s*(.*?)```", text, re.DOTALL)


def extract_json_block(text: str) -> str | None:
    """Return the LAST fenced ``json`` block in a reply, or ``None`` if there is none.

    The contract says the JSON block ends the reply, so the last one is what a
    caller with no further policy should read. This is the clusterer's entry
    point; ``parse_findings`` takes the list and applies its own selection.

    Tolerance stays at the call site: what to *do* about a missing or unparseable
    block differs legitimately between the two callers (a persona that emitted no
    block gets a sentinel finding and a reformat retry; the clusterer falls back
    to the whole reply and then to the identity reduce).
    """
    blocks = extract_json_blocks(text)
    return blocks[-1] if blocks else None


def _is_contract_echo(block: str) -> bool:
    """True if a block is the shipped template restated rather than filled in.

    Exact on purpose — whitespace-normalised equality with ``_TEMPLATE_BLOCK`` and
    nothing else. This predicate causes a block to be **skipped**, and a block
    skipped in error is a review silently discarded, which is the defect (#26),
    not the fix. So the only text it can ever match is text we shipped: a persona
    cannot use it to promote a block of its choosing, because it would have to
    make its own final block byte-equal to our template to displace it.

    A paraphrased template is therefore *not* detected, deliberately. It wins its
    block and is caught by the other guard instead: its verdict is not a vote and
    its severity does not resolve, so it blocks convergence loudly rather than
    being guessed at.
    """
    return " ".join(block.split()) == _ECHO_SIGNATURE


def parse_findings(persona: str, result_text: str) -> PersonaReport:
    """Parse a persona's reply into a report. Never raises (#25).

    The whole payload is model-produced, so it is treated as untrusted input:
    every value is coerced or reported, and no shape a persona can emit reaches an
    exception. One reviewer citing ``"line": "~120"`` used to end a run that had
    already paid for every other reviewer.

    Block selection is positional with exactly one exception (#26): the last
    fenced block wins, unless it is verbatim our own template, in which case it is
    skipped and recorded. A persona that helpfully restated the contract after its
    review used to lose the review *and* have the literal placeholder
    ``"YES | NO"`` counted as a YES — the whole reply discarded and a vote for
    convergence cast in its place.
    """
    blocks = extract_json_blocks(result_text)
    if not blocks:
        return PersonaReport(persona=persona, verdict=None, findings=[
            Finding(persona, "no structured output (parse failure)",
                    Severity.NOTE, "meta", evidence=result_text[:400])])
    # Scan back from the end, past our own template. Only echoes actually skipped
    # are counted, so a reply whose last block is real reports nothing.
    echoes = 0
    block = None
    for candidate in reversed(blocks):
        if _is_contract_echo(candidate):
            echoes += 1
            continue
        block = candidate
        break
    if block is None:
        # Every block was the contract restated: there is no review to recover, so
        # this takes the contract-miss path and its reformat retry rather than
        # being read as a finding titled "..." that votes YES.
        return _contract_violation(
            persona, "the reply echoed the output contract without filling it in",
            blocks[-1])
    try:
        data = json.loads(block)
    except json.JSONDecodeError as exc:
        return PersonaReport(persona=persona, verdict=None, findings=[
            Finding(persona, f"unparseable JSON findings: {exc}",
                    Severity.NOTE, "meta", evidence=block[:400])])
    if not isinstance(data, dict):
        return _contract_violation(
            persona, f"the payload was a {type(data).__name__}, not an object", block)
    # Absent or null means "I found nothing" — a clean review, and the commonest
    # reply we hope to see. Only a PRESENT, non-list value is a violation: reading
    # an empty review as malformed would cost a second call, discard the persona's
    # YES, and put CONVERGED out of reach for a healthy panel.
    raw_findings = data.get("findings")
    if raw_findings is None:
        raw_findings = []
    if not isinstance(raw_findings, list):
        return _contract_violation(
            persona, f"'findings' was a {type(raw_findings).__name__}, not a list",
            block)

    findings = []
    unresolved: list[str] = []
    dropped = 0
    for f in raw_findings:
        if not isinstance(f, dict):
            dropped += 1        # one unusable entry loses itself, not the reply
            continue
        raw = f.get("severity")
        if raw is None:
            sev = Severity.NOTE     # nothing was claimed, so nothing was misread
        elif isinstance(raw, (str, int, float)) and not isinstance(raw, bool):
            sev = normalise_severity(str(raw))
            if sev is None:
                # Never a silent downgrade (#24): escalate it AND say we did, so
                # the operator can see the value we refused to interpret.
                unresolved.append(str(raw)[:_RAW_VALUE_CAP])
                sev = UNRESOLVED_SEVERITY
        else:
            # A container is a payload shape, not a level. Reading ``["UGLY"]``
            # through ``str()`` resolved it by accident of punctuation, while
            # ``["UGLY", "BLOCKER"]`` did not — inconsistent for no reason a
            # caller could predict, so neither is read.
            unresolved.append(repr(raw)[:_RAW_VALUE_CAP])
            sev = UNRESOLVED_SEVERITY
        raw_file = f.get("file")
        findings.append(Finding(
            persona=persona, title=str(f.get("title", "")), severity=sev,
            claim_class=str(f.get("claim_class", "uncategorised")),
            file=str(raw_file) if raw_file else None, line=_coerce_line(f.get("line")),
            evidence=str(f.get("evidence", ""))))
    if dropped:
        entries = "entry" if dropped == 1 else "entries"
        findings.append(Finding(
            persona, f"{dropped} malformed finding {entries} discarded (not objects)",
            Severity.NOTE, "meta", evidence=str(raw_findings)[:400]))
    if echoes:
        # Recorded, not retried: the review itself was recovered, so a second paid
        # call would buy nothing. NOTE because it is a fact about the reply, not a
        # claim about the code — at BLOCKER it would gate convergence, and even at
        # NON_BLOCKING it could reset the stall counter as if it were material.
        blocks_word = "block" if echoes == 1 else "blocks"
        findings.append(Finding(
            persona, f"output contract echoed: {echoes} template {blocks_word} ignored",
            Severity.NOTE, "meta", evidence=_TEMPLATE_BLOCK[:400]))
    # Absent or null is the default below and records nothing: a persona that
    # claimed no verdict is not one whose verdict we misread (#26), exactly as an
    # absent severity is not a level we got wrong.
    raw_verdict = data.get("verdict")
    verdict = unresolved_verdict = None
    if isinstance(raw_verdict, (str, int, float)) and not isinstance(raw_verdict, bool):
        verdict = normalise_verdict(str(raw_verdict))
        if verdict is None:
            unresolved_verdict = str(raw_verdict)[:_RAW_VALUE_CAP]
    elif raw_verdict is not None:
        # A container is a payload shape, not a vote, and is never normalised:
        # ``["YES"]`` would resolve through ``str()`` by accident of punctuation,
        # exactly as ``["UGLY"]`` did for severity (#25).
        unresolved_verdict = repr(raw_verdict)[:_RAW_VALUE_CAP]
    return PersonaReport(persona=persona, verdict=verdict, findings=findings,
                         unresolved_severities=unresolved,
                         unresolved_verdict=unresolved_verdict)


def _is_parse_failure(report: PersonaReport) -> bool:
    """True if a report is the contract-miss sentinel (no parseable JSON block)."""
    return (report.verdict is None and len(report.findings) == 1
            and report.findings[0].claim_class == "meta")
