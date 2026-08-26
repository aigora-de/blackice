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

from kuang.engine import Finding, PersonaReport, ReviewSpec, Severity


_SEV = "NOTE | NON_BLOCKING | BLOCKER | UGLY"

FINDINGS_CONTRACT = f"""
---
OUTPUT CONTRACT (mandatory). Verify every claim against the source before making
it — read the files, run the tests. Do NOT speculate. End your reply with EXACTLY
one fenced ```json block and nothing after it:

```json
{{"verdict": "YES | NO",
  "findings": [
    {{"title": "...", "severity": "{_SEV}", "claim_class": "short-category",
      "file": "path/or/null", "line": 0, "evidence": "what you checked and found"}}
  ]}}
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

Set verdict "YES" only if you found nothing you cannot approve.
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
_RAW_SEVERITY_CAP = 120


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
    tokens = [t for t in re.split(r"[^A-Z]+", raw.upper()) if t]
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


def extract_json_block(text: str) -> str | None:
    """Return the LAST fenced ``json`` block in a reply, or ``None`` if there is none.

    One implementation of the contract's extraction, shared by the findings
    parser and the clusterer (``cluster``) — which previously wrote the same
    regex twice, so a fix to one silently left the other.

    Extraction only: what to *do* about a missing or unparseable block differs
    legitimately between the two callers (a persona that emitted no block gets a
    sentinel finding and a reformat retry; the clusterer falls back to the whole
    reply and then to the identity reduce), so that tolerance stays at the call
    site rather than being averaged into one policy here.
    """
    blocks = re.findall(r"```json\s*(.*?)```", text, re.DOTALL)
    return blocks[-1] if blocks else None


def parse_findings(persona: str, result_text: str) -> PersonaReport:
    """Extract the fenced JSON contract from a persona's reply (defensively)."""
    block = extract_json_block(result_text)
    if block is None:
        return PersonaReport(persona=persona, verdict=None, findings=[
            Finding(persona, "no structured output (parse failure)",
                    Severity.NOTE, "meta", evidence=result_text[:400])])
    try:
        data = json.loads(block)
    except json.JSONDecodeError as exc:
        return PersonaReport(persona=persona, verdict=None, findings=[
            Finding(persona, f"unparseable JSON findings: {exc}",
                    Severity.NOTE, "meta", evidence=block[:400])])
    findings = []
    unresolved: list[str] = []
    for f in data.get("findings", []):
        raw = f.get("severity")
        if raw is None:
            sev = Severity.NOTE     # nothing was claimed, so nothing was misread
        else:
            sev = normalise_severity(str(raw))
            if sev is None:
                # Never a silent downgrade (#24): escalate it AND say we did, so
                # the operator can see the value we refused to interpret.
                unresolved.append(str(raw)[:_RAW_SEVERITY_CAP])
                sev = UNRESOLVED_SEVERITY
        line = f.get("line") or None
        findings.append(Finding(
            persona=persona, title=str(f.get("title", "")), severity=sev,
            claim_class=str(f.get("claim_class", "uncategorised")),
            file=f.get("file") or None, line=int(line) if line else None,
            evidence=str(f.get("evidence", ""))))
    return PersonaReport(persona=persona, verdict=data.get("verdict"),
                         findings=findings, unresolved_severities=unresolved)


def _is_parse_failure(report: PersonaReport) -> bool:
    """True if a report is the contract-miss sentinel (no parseable JSON block)."""
    return (report.verdict is None and len(report.findings) == 1
            and report.findings[0].claim_class == "meta")
