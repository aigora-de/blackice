# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""Claude Code CLI backend for the adversarial review loop.

Binds ``loop.SpawnPersona`` to the ``claude`` binary in headless print
mode: **one ``claude -p`` subprocess per persona per epoch**, driven from the
terminal. No SDK dependency — it shells out to the same ``claude`` the user runs,
so it "works in Claude Code" by *being* Claude Code.

Design choices realised here (see ``two-pass-adversarial-review-pattern.md`` and
the design discussion):

* **Persona sourcing (layered):** parse ``CLAUDE.md`` "Resident Experts" when
  present -> else ``panel.yaml`` / ``panel.md`` -> else a distilled default set.
* **Open-ended mandates.** A persona's *identity* (its role/responsibilities from
  ``CLAUDE.md``) is its lens; we do not impose a prescriptive checklist that would
  "lead the witness".
* **Tools as behavioural grounding.** Every reviewer gets read-only source
  inspection + test/lint *execution* (``Read``/``Grep``/``git``/``pytest``/
  ``ruff``) and **no edit tools** — biasing them to verify against source and run
  the tests rather than speculate.
* **Independent within an epoch;** epoch > 1 is handed *all* prior epochs'
  findings (cross-epoch memory), so the panel builds on itself without
  intra-epoch debate (which would reintroduce groupthink).
* **Structured findings** enforced by an output contract appended to each prompt.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


def _resolve_claude_bin() -> str:
    """Locate the ``claude`` executable robustly (child PATH may differ).

    Order: ``$CLAUDE_BIN`` -> ``PATH`` -> ``~/.local/bin/claude``. Falls back to
    the bare name so the failure, if any, is a clear FileNotFoundError.
    """
    for cand in (os.environ.get("CLAUDE_BIN"), shutil.which("claude"),
                 os.path.expanduser("~/.local/bin/claude")):
        if cand and os.path.exists(cand):
            return cand
    return "claude"

from typing import Sequence

from loop import (
    Cluster,
    EpochResult,
    Finding,
    GateDecision,
    PersonaReport,
    ReviewRun,
    ReviewSpec,
    Severity,
    _identity_reduce,
)

# Read-only permission policy (DENY-BY-DEFAULT). Reviewers may READ the diff and
# source (Read/Grep/Glob); they may NOT run shell or mutate anything.
#
# IMPORTANT — how permissions work headless: in `claude -p` there is no
# interactive prompt. An *allowed* tool runs UNSUPERVISED; an unallowed one is
# auto-DENIED (never asked). Putting bare `Bash` here would pre-approve *all*
# shell for an LLM reviewing an untrusted diff (rm, git push, network egress),
# with no human in the per-command loop. In this design HITL is per-EPOCH
# (convene / synthesise / gate), not per-command, so per-command safety must come
# from POLICY, not prompts. Verification tools (pytest/git/ruff) are a deliberate,
# SCOPED add-on for a later version — e.g. --allowedTools "Bash(pytest:*)"
# "Bash(git diff:*)" — ideally shipped via a `--settings` profile and sandboxed
# (no internet). Never bare `Bash`. See panel-review-NOTES.md.
DEFAULT_ALLOWED_TOOLS = ["Read", "Grep", "Glob"]
DEFAULT_DISALLOWED_TOOLS = ["Edit", "Write", "NotebookEdit", "Bash"]


# =============================================================================
# Persona model + sourcing
# =============================================================================

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


# =============================================================================
# Prompt assembly + the structured-findings contract
# =============================================================================

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
                 surface_kind: str = "diff") -> str:
    """Assemble the per-epoch review task handed to every persona.

    ``surface_kind`` selects the framing: ``"diff"`` reviews a change,
    ``"files"`` reviews existing code presented as whole files (issue #6).
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
    if epoch > 1 and prior:
        memory = ("\n\nPRIOR EPOCHS' FINDINGS (build on these; say if they are "
                  f"resolved, and look for what they missed):\n{prior}\n")
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


# =============================================================================
# Semantic reduce: the LLM clusterer (issue #1)
# =============================================================================
#
# This is the non-deterministic ``loop.Reduce`` implementation. The engine owns
# the deterministic identity default; here a cheap model folds re-worded /
# re-located dups of a single concept into one canonical cluster. Everything the
# engine's control logic depends on is DETERMINISTIC glue around the model call:
# the output is forced into a **partition** (every finding in exactly one cluster,
# never dropped), bad indices are sanitised, and any failure degrades to the
# identity reduce. Severity is UGLY-preserving by construction (``Cluster.severity``
# is the max of its members), so a merge can never hide a ruin-class finding.

_CLUSTER_MANDATE = (
    "You are a careful synthesiser. Group same-issue code-review findings "
    "conservatively. Do NOT review, add, drop, or re-severitise findings — only "
    "cluster the ones you are given.")

_CLUSTER_CONTRACT = """
---
OUTPUT CONTRACT (mandatory). End your reply with EXACTLY one fenced ```json block
and nothing after it, assigning finding indices to groups. Every index from 0 to
N-1 must appear in exactly one group; singletons are expected and encouraged:

```json
{"clusters": [[0, 3], [1], [2]]}
```
"""


def build_cluster_prompt(findings: Sequence[Finding]) -> str:
    """Assemble the clusterer task: a numbered finding list + the group contract."""
    lines = []
    for i, f in enumerate(findings):
        loc = f"{f.file}:{f.line}" if f.file else "-"
        lines.append(f"[{i}] ({f.severity.name}) [{f.claim_class}] {f.title} @ {loc}")
    return (
        "You are given a numbered list of code-review findings raised by several "
        "independent reviewers. Group ONLY those that describe the SAME underlying "
        "issue (same root cause / same defect) — even if worded differently, at "
        "different line numbers, or in different files.\n\n"
        "Be CONSERVATIVE: precision over recall. If in doubt, DO NOT merge — leave "
        "the finding in its own group. Merging distinct issues is worse than "
        "leaving duplicates.\n\n"
        f"FINDINGS:\n" + "\n".join(lines) + f"\n{_CLUSTER_CONTRACT}")


def _extract_cluster_groups(text: str) -> list[list[int]] | None:
    """Pull ``[[0,3],[1],...]`` index-groups from a clusterer reply, or None.

    Tolerant: takes the last fenced ```json block (or the whole text), accepts a
    ``clusters``/``groups`` key or a bare list, and normalises a stray bare int to
    a singleton group. Returns None only when nothing list-shaped can be found —
    the caller then falls back to the identity reduce.
    """
    blocks = re.findall(r"```json\s*(.*?)```", text, re.DOTALL)
    raw = blocks[-1] if blocks else text
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(data, dict):
        groups = data.get("clusters", data.get("groups"))
    elif isinstance(data, list):
        groups = data
    else:
        groups = None
    if not isinstance(groups, list):
        return None
    out: list[list[int]] = []
    for g in groups:
        if isinstance(g, bool):        # bool is an int subclass — reject explicitly
            continue
        if isinstance(g, int):
            out.append([g])
        elif isinstance(g, list):
            out.append([i for i in g if isinstance(i, int) and not isinstance(i, bool)])
    return out


def _groups_to_clusters(findings: Sequence[Finding],
                        groups: list[list[int]]) -> list[Cluster]:
    """Turn model index-groups into a strict partition of ``findings``.

    Guarantees: each finding appears in exactly one cluster. Out-of-range and
    duplicate indices are ignored (first placement wins); any finding the model
    left unplaced becomes its own singleton — a panellist's claim is never dropped.
    """
    n = len(findings)
    assigned: set[int] = set()
    clusters: list[Cluster] = []
    for group in groups:
        members: list[Finding] = []
        for idx in group:
            if 0 <= idx < n and idx not in assigned:
                assigned.add(idx)
                members.append(findings[idx])
        if members:
            clusters.append(Cluster(members=tuple(members), title=members[0].title))
    for idx in range(n):               # unplaced findings -> singletons (never dropped)
        if idx not in assigned:
            f = findings[idx]
            clusters.append(Cluster(members=(f,), title=f.title))
    return clusters


# =============================================================================
# Path / whole-file review surface (the non-diff gather mode)
# =============================================================================

def _expand_paths(repo_root: Path, paths: list[str]) -> tuple[list[Path], list[str]]:
    """Expand file/dir arguments into an ordered, de-duplicated file list.

    Directories are expanded via ``git ls-files`` so ``.gitignore`` is honoured
    and only tracked files are surfaced — reusing git rather than reimplementing
    ignore semantics. Plain files are taken as given. Returns the files plus the
    arguments that resolved to nothing (missing, or an untracked directory) so the
    caller can surface them rather than silently drop them.
    """
    repo_root = repo_root.resolve()
    files: list[Path] = []
    seen: set[Path] = set()
    missing: list[str] = []
    for raw in paths:
        cand = Path(raw)
        p = cand.resolve() if cand.is_absolute() else (repo_root / cand).resolve()
        if p.is_dir():
            out = subprocess.run(
                ["git", "-C", str(repo_root), "ls-files", "-z", "--", str(p)],
                capture_output=True, text=True)
            found = False
            for rel in out.stdout.split("\0"):
                if not rel:
                    continue
                fp = (repo_root / rel).resolve()
                if fp not in seen and fp.is_file():
                    seen.add(fp)
                    files.append(fp)
                    found = True
            if not found:
                missing.append(raw)
        elif p.is_file():
            if p not in seen:
                seen.add(p)
                files.append(p)
        else:
            missing.append(raw)
    return files, missing


def _render_file(rel: str, text: str) -> str:
    """Render one file with a path header and 1-based line numbers.

    Line numbers let personas cite real ``file:line`` (as diff mode already does),
    so adjudication and the human can trust cited locations.
    """
    lines = text.splitlines()
    width = max(len(str(len(lines))), 1)
    body = "\n".join(f"{i:>{width}}| {ln}" for i, ln in enumerate(lines, 1))
    return f"\n--- FILE: {rel} ({len(lines)} lines) ---\n{body}\n"


def build_path_surface(repo_root: Path, paths: list[str], max_bytes: int) -> str:
    """Assemble a whole-file review surface from the named files/directories.

    Renders each file's full content with ``file:line`` fidelity, honouring a
    total-size cap: once the cap is reached, remaining files are dropped and named
    in an explicit OMITTED notice — never a silent truncation. A lone file that by
    itself exceeds the cap is truncated in place with a marker, so there is always
    something to review while the cap still bounds the surface.
    """
    repo_root = repo_root.resolve()
    files, missing = _expand_paths(repo_root, paths)
    chunks: list[str] = []
    omitted: list[tuple[str, str]] = []
    used = 0
    for fp in files:
        rel = os.path.relpath(fp, repo_root)
        try:
            text = fp.read_text()
        except (OSError, UnicodeDecodeError):
            omitted.append((rel, "unreadable/binary"))
            continue
        rendered = _render_file(rel, text)
        if used + len(rendered) > max_bytes:
            if not chunks:
                # Nothing has fit yet: keep the cap by truncating this one file in
                # place (a marker signals it), rather than omitting it wholesale.
                budget = max(max_bytes - used, 0)
                chunks.append(rendered[:budget]
                              + f"\n… [truncated at surface cap: {rel}]\n")
                used = max_bytes
            else:
                omitted.append((rel, "surface cap"))
            continue
        chunks.append(rendered)
        used += len(rendered)

    surface = "".join(chunks) if chunks else "(no reviewable files)"
    if missing:
        surface += ("\n--- PATHS WITH NO TRACKED FILES (missing/untracked) ---\n"
                    + "\n".join(f"- {m}" for m in missing) + "\n")
    if omitted:
        surface += ("\n--- OMITTED (not shown) ---\n"
                    + "\n".join(f"- {rel}: {why}" for rel, why in omitted) + "\n")
    return surface


# =============================================================================
# The session: gather / spawn / checkpoint / gate, sharing cross-epoch state
# =============================================================================

@dataclass
class PanelSession:
    """Wires the loop seams to the ``claude`` CLI and carries cross-epoch memory."""

    repo_root: Path
    spec: ReviewSpec
    personas: dict[str, Persona]
    base: str
    head: str = "HEAD"
    paths: list[str] | None = None      # path mode (issue #6): review these files/dirs
    max_surface_bytes: int = 200_000    # total-size cap for the path-mode surface
    default_model: str | None = None
    cluster_model: str | None = None    # model for the semantic reduce (cheap; None -> default)
    dry_run: bool = False
    claude_bin: str = field(default_factory=_resolve_claude_bin)
    disallowed_tools: list[str] = field(
        default_factory=lambda: list(DEFAULT_DISALLOWED_TOOLS))
    permission_mode: str = "plan"
    tokens: int = 0
    prior_summary: str = ""

    # --- gather: the review surface (re-read each epoch so fixes are visible) ---
    def gather(self, epoch: int) -> str:  # noqa: ARG002
        if self.paths:  # path mode: full content of the named files/dirs (issue #6)
            return build_path_surface(self.repo_root, self.paths, self.max_surface_bytes)
        out = subprocess.run(
            ["git", "-C", str(self.repo_root), "diff", f"{self.base}...{self.head}"],
            capture_output=True, text=True)
        return out.stdout or "(empty diff)"

    # --- one `claude -p` call: returns (result_text, output_tokens, error) ---
    def _run_claude(self, prompt: str, mandate: str, tools: list[str],
                    model: str | None) -> tuple[str, int, str | None]:
        argv = [
            self.claude_bin, "-p", prompt,
            "--append-system-prompt", mandate,
            "--allowedTools", *tools,
            "--disallowedTools", *self.disallowed_tools,
            "--permission-mode", self.permission_mode,
            "--output-format", "json",
            "--add-dir", str(self.repo_root),
        ]
        if model:
            argv += ["--model", model]
        proc = subprocess.run(argv, capture_output=True, text=True, cwd=str(self.repo_root))
        if proc.returncode != 0:
            return "", 0, f"claude exited {proc.returncode}: {proc.stderr[:300]}"
        try:
            env = json.loads(proc.stdout)
        except json.JSONDecodeError:
            return proc.stdout, 0, None  # tolerate raw text
        return env.get("result", ""), int((env.get("usage") or {}).get("output_tokens", 0)), None

    # --- spawn: one persona review, with retry-on-contract-miss ---
    def spawn(self, persona: str, mandate: str, surface: str, epoch: int) -> PersonaReport:
        p = self.personas[persona]
        model = p.model or self.default_model
        surface_kind = "files" if self.paths else "diff"
        prompt = build_prompt(self.spec, surface, epoch, self.prior_summary, surface_kind)

        if self.dry_run:
            preview = (prompt[:280] + "…") if len(prompt) > 280 else prompt
            print(f"\n[dry-run] persona={persona} model={model or 'default'}"
                  f"\n  tools={p.tools} disallow={self.disallowed_tools} mode={self.permission_mode}"
                  f"\n  prompt≈ {preview!r}")
            return PersonaReport(persona=persona, verdict="YES")

        text, toks, err = self._run_claude(prompt, mandate, p.tools, model)
        self.tokens += toks
        if err:
            return PersonaReport(persona=persona, verdict=None, findings=[
                Finding(persona, err, Severity.NOTE, "meta")])
        report = parse_findings(persona, text)
        report.tokens = toks

        # Retry-on-contract-miss: the persona reviewed but did not emit the JSON
        # contract (so its findings were lost). Reformat its raw review into the
        # contract via one cheap follow-up call rather than discarding it.
        if _is_parse_failure(report):
            reformat = (
                "Extract the findings from the review below into the EXACT JSON "
                "contract. Output ONLY the fenced ```json block, nothing else.\n\n"
                f"REVIEW:\n{text}\n\n{FINDINGS_CONTRACT}")
            text2, toks2, err2 = self._run_claude(
                reformat, "You are a formatter: reformat, do not review.", ["Read"], model)
            self.tokens += toks2
            if not err2:
                report2 = parse_findings(persona, text2)
                if not _is_parse_failure(report2):
                    report2.tokens = toks + toks2
                    return report2
        return report

    # --- reduce: the semantic clusterer (loop.Reduce), degrades to identity ---
    def reduce(self, findings: Sequence[Finding]) -> list[Cluster]:
        """Fold the deduped ledger into canonical clusters via one cheap model call.

        A ``loop.Reduce`` implementation. Never raises: a call error, a missing
        contract, or fewer than two findings all fall back to the deterministic
        identity reduce, so the engine's control loop keeps working even when the
        clusterer is unavailable.
        """
        findings = list(findings)
        if len(findings) < 2 or self.dry_run:
            return _identity_reduce(findings)
        model = self.cluster_model or self.default_model
        text, toks, err = self._run_claude(
            build_cluster_prompt(findings), _CLUSTER_MANDATE, ["Read"], model)
        self.tokens += toks
        if err:
            return _identity_reduce(findings)
        groups = _extract_cluster_groups(text)
        if groups is None:
            return _identity_reduce(findings)
        return _groups_to_clusters(findings, groups)

    # --- checkpoint: refresh cross-epoch memory from the ledger ---
    def on_epoch(self, run: ReviewRun) -> None:
        lines = []
        for f in run.ledger.values():
            state = "resolved" if not f.counts_open else "open"
            loc = f"{f.file}:{f.line}" if f.file else "-"
            lines.append(f"- [{f.severity.name}/{state}] ({f.persona}) {f.title} @ {loc}")
        self.prior_summary = "\n".join(lines)

    def budget_spent(self) -> int:
        return self.tokens

    # --- gate: the HITL touchpoint between epochs ---
    def interactive_gate(self, result: EpochResult, run: ReviewRun) -> GateDecision:
        print(f"\n=== epoch {result.index} synthesis ===")
        print(f"new findings: {len(result.new_findings)} | open blockers: "
              f"{result.open_blockers} | open uglies: {result.open_uglies} | "
              f"tokens: {self.tokens}")
        for f in result.new_findings:
            print(f"  [{f.severity.name}] ({f.persona}) {f.title}")
        if not sys.stdin.isatty():
            return GateDecision(stop=False)
        ans = input("gate — [c]ontinue / [s]top? ").strip().lower()
        return GateDecision(stop=ans.startswith("s"))
