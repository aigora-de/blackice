# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""Surface assembly: what the panel is actually given to review.

Two modes. Diff mode reviews a change (``git diff base...head``); path mode
(issue #6) reviews whole files, rendered with citable line numbers and bounded by
a size cap. Both refuse to produce a surface they could not assemble, because a
panel handed an empty string finds nothing, votes YES, and the loop halts
CONVERGED — the tool reporting GOOD on a review it never performed (#18).

Because both refuse the empty case, the surface failure that IS reachable is a
surface that is **present and is not the one that was asked for**: files dropped
at the cap, one file cut mid-way, a file that could not be read, or a named path
that matched nothing tracked. Path mode has always computed every one of those
and thrown it away, returning only the string — so a panel handed a third of what
was named reported ``found_nothing`` like any other clean review (#69). Assembly
now returns a ``SurfaceRecord`` beside the text.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


class SurfaceError(RuntimeError):
    """The review surface could not be assembled, so the run must not proceed.

    Raised rather than returned as a placeholder: a panel handed an empty surface
    finds nothing, votes YES, and the loop halts ``CONVERGED`` — the tool would
    report GOOD on a review it never performed. A surface that cannot be built is
    an operator error to be corrected, never a review with no findings.
    """


@dataclass(frozen=True)
class SurfaceRecord:
    """What the assembled surface WAS, and how it differs from the one that was
    asked for (#69, then #74).

    Set where the surface is built and never derived. The difference is known
    exactly — what the assembler dropped, not an estimate of what a reviewer
    could read — which is why there is no threshold here and none is honest:
    "the surface looks implausibly small" is a rule on a value nobody has
    baselined, and it fires on a healthy one-line diff.

    **Composition, not only loss (#74).** Recording only how a surface fell short
    leaves two runs over entirely different files byte-identical in their
    artefacts, so an archived run cannot be compared with a later one over the
    "same" surface. ``size``, ``files``, ``cap``, ``refs`` and ``paths`` say what
    it was made of. ``size`` is deliberately NOT an input to ``degraded``: a small
    surface is not a lost one, and a diff surface SHRINKS as fixes land at the
    gate, which is the tool working rather than failing.

    ``size`` is the length of the string a persona is actually handed, so it can
    exceed ``cap``: the cap bounds the rendered file content, and the assembler's
    own OMITTED and missing-path notices ride on top of it. Reporting the bounded
    figure instead would report a number nobody was given.

    **Names of the exceptions, counts of the whole.** The files the operator
    named and did not get are named; the files they got are counted. This record
    is written into an artefact that gets pasted into a public repo's issues, so
    naming everything included would approach shipping the surface itself. A
    path is a name and not content, it is bounded by the filesystem where a file
    is not, every persona is already handed the same list of dropped paths in its
    prompt, and — the reason the narrowing is safe — a surface record is our data,
    never a model's. ``files`` is ``None`` where the count could not be
    established, said aloud rather than defaulted to a number that reads like a
    measurement.

    ``bounded`` is what the rule does NOT cover, stated in the run's own output
    rather than only in a docstring: diff mode applies no cap at all (#27), so it
    can lose nothing to one and this record can detect nothing there. The mirror
    image — a surface larger than a reviewer can actually read — needs #27, and
    ``bounded=False`` is where an operator reads that it went unchecked.

    The four loss counts are **properties over the names**, so "the count and the
    names disagree" is impossible by construction rather than forbidden by a test.
    """

    mode: str                              # "paths" | "diff"
    bounded: bool = True                   # whether a cap was applied to this mode
    # What the surface WAS (#74).
    size: int = 0                          # bytes of the surface as handed over
    files: int | None = None               # files in it; None = it could not be known
    cap: int | None = None                 # the cap applied; None = none was
    refs: tuple[str, str] | None = None    # (base, head), diff mode
    paths: tuple[str, ...] = ()            # --paths as the operator gave them
    # What was LOST (#69), now named rather than only counted (#74).
    omitted_files: tuple[str, ...] = ()    # whole files dropped at the cap
    truncated_file: str | None = None      # the file cut because it alone over-ran
    unreadable_files: tuple[str, ...] = ()  # named files that could not be decoded
    unresolved_paths: tuple[str, ...] = ()  # named paths that matched no tracked file

    @property
    def omitted(self) -> int:
        return len(self.omitted_files)

    @property
    def truncated(self) -> bool:
        return self.truncated_file is not None

    @property
    def unreadable(self) -> int:
        return len(self.unreadable_files)

    @property
    def unresolved(self) -> int:
        return len(self.unresolved_paths)

    @property
    def degraded(self) -> bool:
        """Whether the panel was given less than the operator named.

        One rule, four shapes, and every shape falls out of it. An unbounded
        surface is NOT one of them: nothing was lost, and a run must not report a
        degradation it did not suffer. Neither is a small one.
        """
        return bool(self.omitted or self.truncated or self.unreadable
                    or self.unresolved)


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


def build_path_surface(repo_root: Path, paths: list[str],
                       max_bytes: int) -> tuple[str, SurfaceRecord]:
    """Assemble a whole-file review surface from the named files/directories.

    Renders each file's full content with ``file:line`` fidelity, honouring a
    total-size cap: once the cap is reached, remaining files are dropped and named
    in an explicit OMITTED notice — never a silent truncation. A lone file that by
    itself exceeds the cap is truncated in place with a marker, so there is always
    something to review while the cap still bounds the surface.

    Returns the surface **and** a ``SurfaceRecord`` of how it differs from what
    was asked for (#69). The record is returned rather than recovered later by
    matching ``--- OMITTED ---`` out of the string: reading a fact back off
    wording is the defect ``PersonaStatus`` forbids one layer up, and it breaks
    the moment a notice is reworded.

    Raises:
        SurfaceError: None of the requested paths yielded reviewable content.
    """
    repo_root = repo_root.resolve()
    files, missing = _expand_paths(repo_root, paths)
    chunks: list[str] = []
    omitted: list[tuple[str, str]] = []
    dropped_at_cap: list[str] = []
    unreadable: list[str] = []
    truncated_file: str | None = None
    used = 0
    for fp in files:
        rel = os.path.relpath(fp, repo_root)
        try:
            text = fp.read_text()
        except (OSError, UnicodeDecodeError):
            omitted.append((rel, "unreadable/binary"))
            unreadable.append(rel)
            continue
        rendered = _render_file(rel, text)
        if used + len(rendered) > max_bytes:
            if not chunks:
                # Nothing has fit yet: keep the cap by truncating this one file in
                # place (a marker signals it), rather than omitting it wholesale.
                budget = max(max_bytes - used, 0)
                chunks.append(rendered[:budget]
                              + f"\n… [truncated at surface cap: {rel}]\n")
                truncated_file = rel
                used = max_bytes
            else:
                omitted.append((rel, "surface cap"))
                dropped_at_cap.append(rel)
            continue
        chunks.append(rendered)
        used += len(rendered)

    if not chunks:
        detail = f": {', '.join(missing)}" if missing else ""
        raise SurfaceError(
            f"no reviewable files in the requested paths{detail}. Check they exist, "
            "are tracked by git, and are not gitignored.")
    surface = "".join(chunks)
    if missing:
        surface += ("\n--- PATHS WITH NO TRACKED FILES (missing/untracked) ---\n"
                    + "\n".join(f"- {m}" for m in missing) + "\n")
    if omitted:
        surface += ("\n--- OMITTED (not shown) ---\n"
                    + "\n".join(f"- {rel}: {why}" for rel, why in omitted) + "\n")
    # Named apart, because they are lost for different reasons and a run that
    # blamed a binary file on the cap would be the instrument lying about itself.
    # ``files`` counts what actually reached a reviewer, a truncated file
    # included: it was reviewed, in part, and the truncation is recorded beside.
    return surface, SurfaceRecord(
        mode="paths", size=len(surface), files=len(chunks), cap=max_bytes,
        paths=tuple(paths), omitted_files=tuple(dropped_at_cap),
        truncated_file=truncated_file, unreadable_files=tuple(unreadable),
        unresolved_paths=tuple(missing))


def _diff_file_count(repo_root: Path, base: str, head: str) -> int | None:
    """How many files the diff touches, from git rather than from the diff text.

    A second, authoritative call: recovering the count by counting ``diff --git``
    markers in the surface would read a fact off wording, which is the design
    ``build_path_surface`` rejects one function up and a test pins.

    **It must never kill the run it only describes.** A failure here returns
    ``None`` — the count could not be established, which the report says aloud
    rather than defaulting to a 0 that reads like a measurement (#70's shape).
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "diff", "--name-only", "-z",
             f"{base}...{head}"],
            capture_output=True, text=True)
    except OSError:
        return None
    if out.returncode != 0:
        return None
    return len([name for name in out.stdout.split("\0") if name])


def gather_diff(repo_root: Path, base: str,
                head: str) -> tuple[str, SurfaceRecord]:
    """Assemble a diff-mode review surface, or refuse to run.

    git's return code and stderr are both checked, so a mistyped ref stops the
    run instead of handing the panel an empty string to unanimously approve.

    Returns the surface **and** a ``SurfaceRecord`` of what it was made of, the
    same shape path mode returns (#74). The module constant this replaced could
    carry neither the refs nor the size, so two runs over different changes were
    indistinguishable in the run artefact.

    Raises:
        SurfaceError: git failed, or the diff is empty.
    """
    out = subprocess.run(
        ["git", "-C", str(repo_root), "diff", f"{base}...{head}"],
        capture_output=True, text=True)
    if out.returncode != 0:
        raise SurfaceError(
            f"git diff {base}...{head} failed ({out.returncode}): "
            f"{out.stderr.strip() or 'no stderr'}")
    if not out.stdout.strip():
        raise SurfaceError(
            f"git diff {base}...{head} is empty — there is nothing to "
            "review. Check the base and head refs.")
    return out.stdout, SurfaceRecord(
        mode="diff", bounded=False, size=len(out.stdout),
        files=_diff_file_count(repo_root, base, head), refs=(base, head))
