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


def load_personas(repo_root: Path) -> tuple[list[Persona], str]:
    """Resolve the persona set by precedence. Returns (personas, source_label)."""
    claude_md = repo_root / "CLAUDE.md"
    if claude_md.exists():
        experts = parse_claude_md_experts(claude_md.read_text())
        if experts:
            return _ensure_specialists(experts), "CLAUDE.md"
    panel = _load_panel_file(repo_root)
    if panel:
        return _ensure_specialists(panel), "panel file"
    return _ensure_specialists(list(DEFAULT_PERSONAS)), "default"


def _ensure_specialists(personas: list[Persona]) -> list[Persona]:
    """Guarantee a completeness-critic and a survivability (ruin) lens are present.

    A sourced persona already covering one of these roles suppresses the default,
    detected by capability keywords over each persona's **name + grounding** — not
    by any project's persona names (a per-persona capability tag would be more
    robust; see NOTES.md).
    """
    texts = [(p.name + " " + p.grounding).lower() for p in personas]
    out = list(personas)
    if not any(k in t for t in texts
               for k in ("completeness", "blind spot", "what everyone else")):
        out.append(COMPLETENESS_CRITIC)
    _RUIN_KEYS = ("survivab", "ruin", "antifragil", "tail risk", "tail-risk",
                  "cascading", "fat-tail")
    if not any(k in t for t in texts for k in _RUIN_KEYS):
        out.append(SURVIVABILITY)
    return out
