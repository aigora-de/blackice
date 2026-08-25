# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""Surface assembly: what the panel is actually given to review.

Two modes. Diff mode reviews a change (``git diff base...head``); path mode
(issue #6) reviews whole files, rendered with citable line numbers and bounded by
a size cap. Both refuse to produce a surface they could not assemble, because a
panel handed an empty string finds nothing, votes YES, and the loop halts
CONVERGED — the tool reporting GOOD on a review it never performed (#18).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


class SurfaceError(RuntimeError):
    """The review surface could not be assembled, so the run must not proceed.

    Raised rather than returned as a placeholder: a panel handed an empty surface
    finds nothing, votes YES, and the loop halts ``CONVERGED`` — the tool would
    report GOOD on a review it never performed. A surface that cannot be built is
    an operator error to be corrected, never a review with no findings.
    """


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

    Raises:
        SurfaceError: None of the requested paths yielded reviewable content.
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
    return surface


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
