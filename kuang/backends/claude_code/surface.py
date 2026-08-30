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
    """How the assembled surface differs from the one that was asked for (#69).

    Set where the surface is built and never derived. The difference is known
    exactly — a count of what the assembler dropped, not an estimate of what a
    reviewer could read — which is why there is no threshold here and none is
    honest: "the surface looks implausibly small" is a rule on a value nobody has
    baselined, and it fires on a healthy one-line diff.

    **Counts, never names, and never the surface itself.** This record is written
    into an artefact that gets pasted into a public repo's issues, and a review
    surface is unbounded content — whole files, or a whole diff. Naming the files
    that were lost is #11's and #74's acceptance criterion, deliberately left to
    them so this cannot be mistaken for having met it.

    ``bounded`` is what the rule does NOT cover, stated in the run's own output
    rather than only in a docstring: diff mode applies no cap at all (#27), so it
    can lose nothing to one and this record can detect nothing there. The mirror
    image — a surface larger than a reviewer can actually read — needs #27, and
    ``bounded=False`` is where an operator reads that it went unchecked.
    """

    mode: str                 # "paths" | "diff"
    omitted: int = 0          # whole files dropped once the cap was reached
    truncated: bool = False   # the first file, cut in place because it alone over-ran
    unreadable: int = 0       # named files that could not be decoded (binary, IO)
    unresolved: int = 0       # named paths that matched no tracked file
    bounded: bool = True      # whether a cap was applied to this mode at all

    @property
    def degraded(self) -> bool:
        """Whether the panel was given less than the operator named.

        One rule, four shapes, and every shape falls out of it. An unbounded
        surface is NOT one of them: nothing was lost, and a run must not report a
        degradation it did not suffer.
        """
        return bool(self.omitted or self.truncated or self.unreadable
                    or self.unresolved)


# Diff mode assembles the whole of ``git diff`` or refuses (see ``gather_diff``),
# so its record is a constant rather than a computation. It is emitted anyway,
# every epoch, because an absent record makes no claim and "we did not check" is
# itself the thing an operator needs told (#27).
DIFF_SURFACE = SurfaceRecord(mode="diff", bounded=False)


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
    unreadable = 0
    truncated = False
    used = 0
    for fp in files:
        rel = os.path.relpath(fp, repo_root)
        try:
            text = fp.read_text()
        except (OSError, UnicodeDecodeError):
            omitted.append((rel, "unreadable/binary"))
            unreadable += 1
            continue
        rendered = _render_file(rel, text)
        if used + len(rendered) > max_bytes:
            if not chunks:
                # Nothing has fit yet: keep the cap by truncating this one file in
                # place (a marker signals it), rather than omitting it wholesale.
                budget = max(max_bytes - used, 0)
                chunks.append(rendered[:budget]
                              + f"\n… [truncated at surface cap: {rel}]\n")
                truncated = True
                used = max_bytes
            else:
                omitted.append((rel, "surface cap"))
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
    # Counted apart, because they are lost for different reasons and a run that
    # blamed a binary file on the cap would be the instrument lying about itself.
    return surface, SurfaceRecord(
        mode="paths", omitted=len(omitted) - unreadable, truncated=truncated,
        unreadable=unreadable, unresolved=len(missing))


def gather_diff(repo_root: Path, base: str, head: str) -> str:
    """Assemble a diff-mode review surface, or refuse to run.

    git's return code and stderr are both checked, so a mistyped ref stops the
    run instead of handing the panel an empty string to unanimously approve.

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
    return out.stdout
