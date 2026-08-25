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

from blackice.engine import Finding, PersonaReport, ReviewSpec, Severity


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

Severity: UGLY = a ruin-class hazard (non-linear/multiplicative/cascading/
irreversible) — use it only for that. BLOCKER = must be fixed or tracked before
approval. Set verdict "YES" only if you found nothing you cannot approve.
"""


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


def parse_findings(persona: str, result_text: str) -> PersonaReport:
    """Extract the fenced JSON contract from a persona's reply (defensively)."""
    blocks = re.findall(r"```json\s*(.*?)```", result_text, re.DOTALL)
    if not blocks:
        return PersonaReport(persona=persona, verdict=None, findings=[
            Finding(persona, "no structured output (parse failure)",
                    Severity.NOTE, "meta", evidence=result_text[:400])])
    try:
        data = json.loads(blocks[-1])
    except json.JSONDecodeError as exc:
        return PersonaReport(persona=persona, verdict=None, findings=[
            Finding(persona, f"unparseable JSON findings: {exc}",
                    Severity.NOTE, "meta", evidence=blocks[-1][:400])])
    findings = []
    for f in data.get("findings", []):
        try:
            sev = Severity[str(f.get("severity", "NOTE")).strip().upper()]
        except KeyError:
            sev = Severity.NOTE
        line = f.get("line") or None
        findings.append(Finding(
            persona=persona, title=str(f.get("title", "")), severity=sev,
            claim_class=str(f.get("claim_class", "uncategorised")),
            file=f.get("file") or None, line=int(line) if line else None,
            evidence=str(f.get("evidence", ""))))
    return PersonaReport(persona=persona, verdict=data.get("verdict"), findings=findings)


def _is_parse_failure(report: PersonaReport) -> bool:
    """True if a report is the contract-miss sentinel (no parseable JSON block)."""
    return (report.verdict is None and len(report.findings) == 1
            and report.findings[0].claim_class == "meta")
