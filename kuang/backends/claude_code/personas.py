# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""Persona sourcing: who sits on the panel, and where they came from.

Personas are a parameter, not a hard-code. Precedence: a target repo's
``CLAUDE.md`` "Resident Experts" -> ``panel.yaml`` / ``panel.md`` -> a distilled
default set. Whatever the source, a completeness-critic and a ruin (survivability)
lens are guaranteed present.

A persona's *identity* is its lens; we do not impose a prescriptive checklist that
would lead the witness.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .permissions import DEFAULT_ALLOWED_TOOLS


@dataclass
class Persona:
    """A reviewer. ``grounding`` is an open-ended lens, not a checklist."""

    name: str
    grounding: str
    tools: list[str] = field(default_factory=lambda: list(DEFAULT_ALLOWED_TOOLS))
    model: str | None = None


@dataclass(frozen=True)
class LensCoverage:
    """Which persona covers one required lens, and on what evidence (#82).

    A record returned *beside* the roster, following ``surface``'s pattern (#69,
    #74): sourcing already knew this and threw it away, so no artefact stated what
    the panel covered and a ``converged`` verdict never said which lenses were
    behind it.

    ``injected`` is the whole of the exact/heuristic boundary, and it is why this
    record says **how** a lens is covered rather than **whether**:

    * ``True`` — nothing matched, so the default persona was appended. Exact: this
      persona **is** that lens, by construction.
    * ``False`` — a sourced persona's name or grounding contained a capability
      keyword and suppressed the default. Heuristic: a claim on the strength of a
      substring, which nothing checked and nothing can until a lens is
      **declarable** (#2). A grounding that mentions cascading failures in passing
      suppresses the ruin lens exactly as a real ruin lens does.

    **Whether** is deliberately not modelled, because it was measured and is not a
    question: ``_ensure_specialists`` appends the missing default on every branch
    ``load_personas`` can return through, so a required lens is never absent from a
    roster. A present/absent column could only ever say yes, and a rule that cannot
    fire is decoration — #69's lesson, where an enum member was nearly added for a
    state the code refuses to enter.
    """

    lens: str
    persona: str
    injected: bool


# Distilled generic default panel — used only when a repo defines no experts.
# The lenses are deliberately broad (adversarial, not box-ticking). Prior art
# that informed these roles is credited in two-pass-adversarial-review-pattern.md
# (agent-review-panel, Deep Review, CodeProbe); no text is copied from them.
DEFAULT_PERSONAS: list[Persona] = [
    Persona("correctness", "Does the change compute the right thing? Hunt logic "
            "errors, wrong assumptions, and boundary/edge cases."),
    Persona("adversary", "Try to break it. Worst-case and malformed inputs, "
            "race conditions, resource exhaustion, pathological states."),
    Persona("constraints", "What external rules must this not violate? "
            "Security, privacy, regulatory, licensing, API contracts."),
    Persona("engineer", "Code quality, hidden state, error handling, "
            "maintainability, and change discipline (scope creep / drift)."),
    Persona("empiricist", "Test rigour: would each test fail without the change? "
            "Run the tests, mutation-check load-bearing ones, find coverage gaps."),
]

# Always-present specialists, regardless of source.
COMPLETENESS_CRITIC = Persona(
    "completeness-critic",
    "Your only job is to find what everyone else MISSED: an unexamined modality, "
    "an unverified claim, an execution path or failure mode nobody reviewed. "
    "Assume the other reviewers suffered from shared blind spots.",
)
SURVIVABILITY = Persona(
    "survivability",
    "Hunt ONLY ruin-class hazards: non-linear, multiplicative, cascading or "
    "irreversible failures that threaten survivability in this system's context "
    "(e.g. data/records corruption, unbounded loss, cascading feedback). "
    "Tag any such finding UGLY — it is a circuit-breaker.",
)


def parse_claude_md_experts(text: str) -> list[Persona]:
    """Extract personas from a ``CLAUDE.md`` "Resident Experts" section.

    Recognises subsections of the form ``## <emoji?> Name — Role`` (em-dash or
    hyphen) and uses the whole subsection body as the persona's open-ended
    grounding. Returns ``[]`` if no experts section/subsections are found.
    """
    # Isolate the Resident Experts region (from its heading to EOF or next H1).
    m = re.search(r"(?im)^#+\s*Resident Experts\b.*?$", text)
    if not m:
        return []
    region = text[m.end():]
    next_h1 = re.search(r"(?m)^#\s+\S", region)
    if next_h1:
        region = region[: next_h1.start()]

    personas: list[Persona] = []
    # Split on level-2 headings; capture "Name — Role" and the body.
    parts = re.split(r"(?m)^##\s+", region)
    for part in parts[1:]:
        header, _, body = part.partition("\n")
        name_role = re.split(r"\s+[—–-]\s+", header.strip(), maxsplit=1)
        if len(name_role) < 2:
            # A subsection with no "Name — Role" separator is a process/meta
            # heading (e.g. "Invoking the Experts"), not a persona. Skip it.
            continue
        raw_name, role = name_role[0], name_role[1].strip()
        # Strip a leading emoji/symbol token if present.
        name = re.sub(r"^[^\w]+", "", raw_name).strip() or raw_name.strip()
        grounding = f"You are {name} — {role}.\n\n{body}".strip()
        personas.append(Persona(name=name, grounding=grounding))
    return personas


def _load_panel_file(repo_root: Path) -> list[Persona]:
    """Load personas from ``panel.yaml`` or ``panel.md`` if present (best effort)."""
    yml = repo_root / "panel.yaml"
    if yml.exists():
        try:
            import yaml  # optional dependency
            data = yaml.safe_load(yml.read_text()) or {}
            return [
                Persona(name=p["name"], grounding=p.get("grounding", ""),
                        tools=p.get("tools", list(DEFAULT_ALLOWED_TOOLS)),
                        model=p.get("model"))
                for p in data.get("personas", [])
            ]
        except Exception as exc:  # noqa: BLE001
            print(f"[panel] failed to parse panel.yaml: {exc}", file=sys.stderr)
    md = repo_root / "panel.md"
    if md.exists():
        return parse_claude_md_experts(md.read_text())
    return []


# The lenses a panel must have, whatever it was sourced from: the lens name a run
# reports, the capability keywords that let a sourced persona suppress the default,
# and the default itself. One table rather than two near-identical blocks, so a
# third lens is a row and not a fourth copy of the same three lines. The keyword
# sets are unchanged — replacing them with a declarable ``role:`` tag is #2's, and
# #82 must report what is true today rather than wait for it.
REQUIRED_LENSES: tuple[tuple[str, tuple[str, ...], Persona], ...] = (
    ("completeness",
     ("completeness", "blind spot", "what everyone else"), COMPLETENESS_CRITIC),
    ("ruin",
     ("survivab", "ruin", "antifragil", "tail risk", "tail-risk", "cascading",
      "fat-tail"), SURVIVABILITY),
)


def load_personas(repo_root: Path) -> tuple[list[Persona], str, list[LensCoverage]]:
    """Resolve the persona set by precedence.

    Returns ``(personas, source_label, coverage)`` — who sits on the panel, where
    they came from, and which required lens each of them covers (#82). The third
    element is a fact sourcing has always known and never told anyone, which is why
    no run could state what its panel covered.
    """
    claude_md = repo_root / "CLAUDE.md"
    if claude_md.exists():
        experts = parse_claude_md_experts(claude_md.read_text())
        if experts:
            roster, coverage = _ensure_specialists(experts)
            return roster, "CLAUDE.md", coverage
    panel = _load_panel_file(repo_root)
    if panel:
        roster, coverage = _ensure_specialists(panel)
        return roster, "panel file", coverage
    roster, coverage = _ensure_specialists(list(DEFAULT_PERSONAS))
    return roster, "default", coverage


def _ensure_specialists(
        personas: list[Persona]) -> tuple[list[Persona], list[LensCoverage]]:
    """Guarantee a completeness-critic and a survivability (ruin) lens are present.

    A sourced persona already covering one of these roles suppresses the default,
    detected by capability keywords over each persona's **name + grounding** — not
    by any project's persona names (a per-persona capability tag would be more
    robust; see #2).

    Returns the roster **and** a ``LensCoverage`` per required lens, naming the
    persona that covers it and whether the default was injected. The suppression
    decision was always being made here; before #82 only its effect on the roster
    survived, so a run could not say that one persona had suppressed both defaults
    and left a panel of one — which is the state #82 was filed for.
    """
    texts = [(p.name + " " + p.grounding).lower() for p in personas]
    out = list(personas)
    coverage: list[LensCoverage] = []
    for lens, keys, default in REQUIRED_LENSES:
        matched = next((p for p, text in zip(personas, texts)
                        if any(k in text for k in keys)), None)
        if matched is None:
            out.append(default)
        coverage.append(LensCoverage(
            lens=lens, persona=(default if matched is None else matched).name,
            injected=matched is None))
    return out, coverage
