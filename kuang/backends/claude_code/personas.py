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


# The origins a discarded declaration can have, and the reasons it can be
# discarded. Closed sets, set where the failure happens and never derived from
# prose: there is no model anywhere in this path, which makes it the easiest
# place in the codebase to be exact and the least excusable place to be vague.
DISCARD_ORIGINS: tuple[str, ...] = ("CLAUDE.md", "panel.yaml", "panel.md")
DISCARD_REASONS: tuple[str, ...] = ("empty", "malformed", "unsupported")
# Where a roster came from. ``--panel`` is the operator's own instruction;
# ``panel file`` is deliberately unchanged from before #16 so an archived
# artefact's vocabulary still reads.
PANEL_LABELS: tuple[str, ...] = ("--panel", "CLAUDE.md", "panel file", "default")

# What each reason means to a reader, and what it asks them to do. One sentence
# per categorical value: rendering, not a rule. Shared by the console warning and
# by ``PanelError`` so a run cannot describe one failure two ways.
_REASON_NOTE: dict[str, str] = {
    "empty": "it named no persona",
    "malformed": "it could not be read or parsed",
    "unsupported": ("reading a YAML panel needs the optional extra: "
                    "pip install 'kuang[yaml]'"),
}


def discard_note(reason: str) -> str:
    """The one sentence a reader gets for a categorical discard reason."""
    return _REASON_NOTE[reason]


class PanelError(Exception):
    """An explicitly named panel could not be used.

    Raised only for ``--panel``. The implicit tiers record a ``Discarded`` and
    carry on; an explicit instruction is not silently overridden, because doing
    so would be #16's own defect one tier up with the operator having been
    maximally explicit. What the resulting exit code MEANS is #32's.
    """


@dataclass(frozen=True)
class Discarded:
    """A persona declaration that was reached and yielded nobody (#16).

    ``origin`` is which declaration, ``reason`` is why it could not be used, and
    ``detail`` is what we can say about the failure — recorded rather than
    discarded, and reported rather than printed to a stream the artefact never
    sees. Before #16 the ``panel.yaml`` case went to stderr and the other three
    went nowhere at all.

    ``detail`` is **built by** ``_diagnosis``, never taken from the exception's
    own rendering: that rendering can quote the file, and this record is
    published (#105). It names the exception type and a line/column where there
    is one, which is a name and a coordinate rather than content.
    """

    origin: str          # one of DISCARD_ORIGINS
    reason: str          # one of DISCARD_REASONS
    detail: str = ""     # from _diagnosis: a type and a coordinate, never content


@dataclass(frozen=True)
class PanelSource:
    """Where the panel came from, and what was thrown away getting there (#16).

    A record beside the roster, following ``SurfaceRecord`` (#69, #74) and
    ``LensCoverage`` (#82). It replaces the bare source *string* rather than
    joining it as a fourth tuple element: the label alone answered six distinct
    causes with one word, and #30's own artefact comment named the gap when it
    added that label — *recording the label a sourcing step returned is not the
    same as noticing that the step fell back.*

    The rule ``discarded`` is built from is one sentence: **a degradation is a
    persona declaration that was reached and yielded no personas.** A
    *declaration* is the thing whose sole purpose is to declare a panel —
    ``--panel`` by construction, ``panel.yaml``/``panel.md`` by existing at all,
    and inside ``CLAUDE.md`` (a file that exists for other reasons) the
    ``Resident Experts`` heading.

    Two consequences are deliberate, and both are boundaries rather than gaps:

    * a repo with no ``CLAUDE.md``, and a ``CLAUDE.md`` with no experts heading,
      declared nothing and read **clean** — a rule that fires on a healthy run is
      the mirror image of the defect it reports;
    * only tiers the precedence chain actually **reached** can be discarded, so a
      good ``CLAUDE.md`` beside a broken ``panel.yaml`` is clean. Recording it
      would be a claim about a file this run never opened.
    """

    label: str                                # one of PANEL_LABELS
    path: str | None = None                   # --panel's argument, as given
    discarded: tuple[Discarded, ...] = ()

    @property
    def degraded(self) -> bool:
        """Whether anything this run asked for was thrown away."""
        return bool(self.discarded)


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


# The declaration marker inside CLAUDE.md. Named because #16 needs to ask
# whether one was present, not only whether personas came out: a file with no
# such heading declared nothing to us and must read clean, while a file carrying
# one and yielding nobody is a declaration that was thrown away.
_EXPERTS_HEADING = re.compile(r"(?im)^#+\s*Resident Experts\b.*?$")


def _experts_section_present(text: str) -> bool:
    """Whether ``text`` declares a Resident Experts section at all."""
    return _EXPERTS_HEADING.search(text) is not None


def parse_claude_md_experts(text: str) -> list[Persona]:
    """Extract personas from a ``CLAUDE.md`` "Resident Experts" section.

    Recognises subsections of the form ``## <emoji?> Name — Role`` (em-dash or
    hyphen) and uses the whole subsection body as the persona's open-ended
    grounding. Returns ``[]`` if no experts section/subsections are found.
    """
    # Isolate the Resident Experts region (from its heading to EOF or next H1).
    m = _EXPERTS_HEADING.search(text)
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


def _diagnosis(exc: Exception) -> str:
    """What we can say about a failure without quoting the file it came from (#105).

    The exception's own **rendering** is deliberately not recorded. ``pyyaml``'s
    ``MarkedYAMLError`` embeds source lines — ``Mark.__str__`` calls
    ``get_snippet()`` — and whatever goes in a ``Discarded`` reaches the run
    artefact *and* any archived ``run.log``, both of which get pasted into a
    public repo's issues and the latter of which ``load_prior_findings`` reads
    back. #16 moved this text out of stderr for good reasons; carrying the raw
    rendering with it was the defect.

    The boundary is #69's, stated there for the surface and applying unchanged
    here: a record carries **names and coordinates, never the content they point
    at**. The type is ours to name and a line/column is a coordinate, so both
    stay; the parser's rendering is the file's own text and goes.

    One rule for every exception, not a table per library — including the
    ``OSError`` and ``ImportError`` cases, whose renderings are in fact harmless.
    Narrowing only where a leak is known would be a claim about every parser we
    do not use yet, and the same rule applied everywhere costs those two cases an
    errno string that the recorded path and the categorical reason already say.

    What it under-reports, plainly: the parser's own description of the grammar
    failure, which is often the useful half of it. The sibling mechanism is the
    coordinate — an operator opens their own file at the right line, which they
    can do and a reader of the artefact cannot.
    """
    mark = getattr(exc, "problem_mark", None) or getattr(exc, "context_mark", None)
    line, column = getattr(mark, "line", None), getattr(mark, "column", None)
    where = (f" at line {line + 1}, column {column + 1}"
             if line is not None and column is not None else "")
    return f"{type(exc).__name__}{where}"


def _load_declared(path: Path) -> tuple[list[Persona], str, str]:
    """Load one declared panel file. Returns ``(personas, reason, detail)``.

    ``reason`` is ``""`` where personas came out, and otherwise one of
    ``DISCARD_REASONS``. Dispatch is by suffix over the two loaders that already
    existed, so a file is read the same way whether the precedence chain found it
    or ``--panel`` named it.

    The ``import`` sits outside the parse's ``try`` on purpose. Before #16 both
    landed in one ``except Exception``, so a missing optional extra and a broken
    file were the same event — and they are not: one says *install the extra*, the
    other says *fix the file*. Collapsing them is this issue's own conflation, one
    field along.
    """
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # optional dependency: the `yaml` extra
        except ImportError as exc:
            return [], "unsupported", _diagnosis(exc)
        try:
            data = yaml.safe_load(path.read_text()) or {}
            personas = [
                Persona(name=p["name"], grounding=p.get("grounding", ""),
                        tools=p.get("tools", list(DEFAULT_ALLOWED_TOOLS)),
                        model=p.get("model"))
                for p in data.get("personas", [])
            ]
        except Exception as exc:  # noqa: BLE001
            return [], "malformed", _diagnosis(exc)
    else:
        try:
            personas = parse_claude_md_experts(path.read_text())
        except OSError as exc:
            return [], "malformed", _diagnosis(exc)
    return (personas, "", "") if personas else ([], "empty", "")


def _load_panel_file(repo_root: Path) -> tuple[list[Persona], tuple[Discarded, ...]]:
    """Load personas from ``panel.yaml`` or ``panel.md``, recording what failed.

    The precedence is **unchanged** from before #16, deliberately: a ``panel.yaml``
    that parses answers this tier even when it names nobody, and only one that
    could not be parsed falls through to ``panel.md``. #16 is a reporting issue,
    so what a run *does* is untouched and only what it *says* is new.

    Nothing is printed here any more. The stderr line the YAML branch used to emit
    was the one signal any of these causes produced, and it went to a stream the
    artefact never sees — so it moves into the record rather than being copied
    into it. One degradation, marked once (#74's rule).
    """
    discarded: list[Discarded] = []
    yml = repo_root / "panel.yaml"
    if yml.exists():
        personas, reason, detail = _load_declared(yml)
        if personas:
            return personas, ()
        discarded.append(Discarded("panel.yaml", reason, detail))
        if reason == "empty":
            return [], tuple(discarded)
    md = repo_root / "panel.md"
    if md.exists():
        personas, reason, detail = _load_declared(md)
        if personas:
            return personas, tuple(discarded)
        discarded.append(Discarded("panel.md", reason, detail))
    return [], tuple(discarded)


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


def _load_explicit_panel(
        path: Path) -> tuple[list[Persona], PanelSource, list[LensCoverage]]:
    """Source the panel from ``--panel``, or refuse.

    The asymmetry with the implicit tiers is the point. Those record a discard and
    carry on, because the operator asked for nothing in particular and the tool
    informs rather than decides. Here the operator named a file, so falling back
    to a panel nobody asked for would silently override an explicit instruction —
    which is precisely the defect #16 exists to remove, one tier up.

    It refuses through the same operator-error path ``SurfaceError`` already uses,
    before the first epoch and so before anything is spent, rather than inventing
    a third behaviour for the class (#59). What the exit code MEANS stays #32's.
    """
    if not path.exists():
        raise PanelError(f"--panel {path}: the file does not exist")
    personas, reason, detail = _load_declared(path)
    if not personas:
        said = f" ({detail})" if detail else ""
        raise PanelError(f"--panel {path}: {discard_note(reason)}{said}")
    roster, coverage = _ensure_specialists(personas)
    return roster, PanelSource("--panel", path=str(path)), coverage


def load_personas(
        repo_root: Path, panel_path: Path | None = None,
) -> tuple[list[Persona], PanelSource, list[LensCoverage]]:
    """Resolve the persona set by precedence.

    Returns ``(personas, source, coverage)`` — who sits on the panel, a record of
    where they came from and what was thrown away getting there (#16), and which
    required lens each of them covers (#82).

    ``panel_path`` is ``--panel``: an explicit override ahead of every other tier,
    which when given is the only thing consulted. It raises ``PanelError`` rather
    than falling back; see ``_load_explicit_panel``.

    The middle element became a **record** rather than gaining a fourth tuple
    element, so ``load_personas(repo)[0]`` keeps working and an archived
    artefact's ``source`` vocabulary is unchanged. Before #16 it was the bare
    label, which answered six distinct causes — three of them byte-identical in
    stdout and artefact — with the single word ``"default"``.
    """
    if panel_path is not None:
        return _load_explicit_panel(Path(panel_path))

    discarded: list[Discarded] = []
    claude_md = repo_root / "CLAUDE.md"
    if claude_md.exists():
        text = claude_md.read_text()
        experts = parse_claude_md_experts(text)
        if experts:
            roster, coverage = _ensure_specialists(experts)
            return roster, PanelSource("CLAUDE.md"), coverage
        # A CLAUDE.md is a project-instructions file that exists for other
        # reasons, and we read one section out of it. So the DECLARATION is the
        # heading, not the file: with no heading nothing was declared to us and
        # the run is clean, and with a heading yielding nobody a declaration was
        # thrown away. Two plausible slips reach the second state — `###` instead
        # of `##`, and a colon instead of ` — `.
        if _experts_section_present(text):
            discarded.append(Discarded("CLAUDE.md", "empty"))

    panel, panel_discards = _load_panel_file(repo_root)
    discarded.extend(panel_discards)
    if panel:
        roster, coverage = _ensure_specialists(panel)
        return roster, PanelSource("panel file", discarded=tuple(discarded)), coverage
    roster, coverage = _ensure_specialists(list(DEFAULT_PERSONAS))
    return roster, PanelSource("default", discarded=tuple(discarded)), coverage


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
        matched = [p for p, text in zip(personas, texts)
                   if any(k in text for k in keys)]
        if not matched:
            out.append(default)
            coverage.append(LensCoverage(lens, default.name, injected=True))
            continue
        # EVERY matcher, not the first. Which persona "is" the ruin lens is not a
        # question the keyword rule can answer, so picking one would be the report
        # judging: a panel whose Analyst says "cascading" in passing and whose
        # Sentinel is a real ruin lens would have the lens reported against the
        # Analyst, and a run in which the Analyst errored would say the ruin lens
        # failed while the Sentinel reviewed it perfectly well. Over-reporting a
        # degradation on a healthy panel is the mirror image of the defect #82 is
        # about, so each matcher gets its own record and the reader sees the set.
        coverage.extend(LensCoverage(lens, p.name, injected=False) for p in matched)
    return out, coverage
