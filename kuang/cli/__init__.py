# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""kuang — entry point.

Run this. It wires the two pieces together:

* ``kuang.engine``   — the generic engine: the bounded loop, halting
                          predicates, dedup/stall, token budget, and the UGLY
                          circuit-breaker. Knows nothing about Claude.
* ``kuang.backends.claude_code`` — the Claude Code binding: sources personas,
                          and spawns one ``claude -p`` per persona per epoch.

This module parses the CLI, loads the panel, wires the backend's seams into the
engine's ``run``, and prints the result. A different agent runtime would be a
different backend swapped in here — the engine is unchanged. It is the only
module that imports both.

The verb interface (``review`` / ``panel`` / ``surface`` / ``inspect``) is issue
#20; this is the flag interface, unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from kuang.backends.claude_code import (DEFAULT_DISALLOWED_TOOLS,
                                          UNRESOLVED_SEVERITY, PanelSession,
                                          SurfaceError, load_personas,
                                          load_prior_findings)
from kuang.engine import HaltingSet, PanelConfig, ReviewSpec, run


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Adversarial review panel loop over a git diff.")
    ap.add_argument("--repo", default=".", help="repository root")
    # Exactly one review mode is active per run: diff mode (--base/--head) OR
    # path mode (--paths). Neither is argparse-required; main() enforces the XOR.
    ap.add_argument("--base", default=None, help="diff mode: base ref (e.g. main)")
    ap.add_argument("--head", default="HEAD", help="diff mode: head ref")
    ap.add_argument("--paths", nargs="*", default=None, metavar="PATH",
                    help="path mode: adversarially review the full content of these "
                         "files/directories (directories expand via git ls-files, "
                         "honouring .gitignore) instead of a diff")
    ap.add_argument("--max-surface-bytes", type=int, default=200_000,
                    help="path mode: cap on total review-surface size (default 200000)")
    ap.add_argument("--why", default="mission-critical change", help="why the review matters")
    ap.add_argument("--what", default=None, help="what changed / what to review")
    ap.add_argument("--max-epochs", type=int, default=3)
    ap.add_argument("--token-budget", type=int, default=None)
    ap.add_argument("--stall-patience", type=int, default=1)
    ap.add_argument("--model", default=None, help="default model for personas")
    ap.add_argument("--allow-tools", nargs="*", default=None, metavar="TOOL",
                    help="override allowed tools for ALL personas, e.g. --allow-tools "
                         "Read Grep Glob 'Bash(git:*)' 'Bash(pytest:*)' 'Bash(poetry:*)'")
    ap.add_argument("--disallow-tools", nargs="*", default=None, metavar="TOOL",
                    help="override the disallowed-tools list (default: Edit Write NotebookEdit Bash)")
    ap.add_argument("--permission-mode", default="plan",
                    help="claude --permission-mode: plan|default|acceptEdits|bypassPermissions")
    ap.add_argument("--semantic-dedup", action="store_true",
                    help="fold same-concept findings into canonical clusters via a "
                         "cheap model call each epoch (default: deterministic "
                         "signature dedup only, no extra call)")
    ap.add_argument("--cluster-model", default=None,
                    help="model for the --semantic-dedup clusterer (default: --model)")
    ap.add_argument("--prior-findings", default=None, metavar="PATH",
                    help="seed the panel with an earlier run's findings (issue #13): a "
                         "saved findings.json, a run's JSON output, or a raw run.log "
                         "containing the '--- JSON ---' block. Injected from epoch 1, "
                         "with locations flagged advisory since fixes move code")
    ap.add_argument("--dry-run", action="store_true", help="print the wiring, spawn nothing")
    ap.add_argument("--no-parallel", action="store_true")
    args = ap.parse_args(argv)

    # Exactly-one-mode selection (deny ambiguity rather than silently prefer one).
    path_mode = args.paths is not None
    if path_mode and args.base is not None:
        ap.error("choose one mode: --base/--head (diff) OR --paths (whole-file), not both")
    if not path_mode and args.base is None:
        ap.error("specify a review mode: --base <ref> (diff) or --paths <path...> (whole-file)")
    if path_mode and not args.paths:
        ap.error("--paths needs at least one file or directory")
    what = args.what or ("existing code (full-file review)" if path_mode
                         else "the pending diff")

    repo = Path(args.repo).resolve()
    personas, source = load_personas(repo)
    if args.allow_tools:  # override the read-only default for ALL personas
        for p in personas:
            p.tools = list(args.allow_tools)
    print(f"[panel] {len(personas)} personas from {source}: "
          f"{', '.join(p.name for p in personas)}")
    print(f"[panel] tools={personas[0].tools} mode={args.permission_mode}")

    # Cross-run memory (issue #13). Loaded BEFORE the session so a bad path fails
    # here, not three epochs into a paid run, and echoed so a seed that loaded
    # nothing cannot be mistaken for one that worked.
    seeded = ""
    if args.prior_findings:
        seeded = load_prior_findings(args.prior_findings)
        print(f"[panel] prior findings: {len(seeded.splitlines())} carried in from "
              f"{args.prior_findings} (locations advisory)")

    spec = ReviewSpec(why=args.why, what=what)
    session = PanelSession(
        repo_root=repo, spec=spec, base=args.base or "", head=args.head,
        paths=args.paths, max_surface_bytes=args.max_surface_bytes,
        personas={p.name: p for p in personas},
        default_model=args.model, cluster_model=args.cluster_model,
        dry_run=args.dry_run, seeded_summary=seeded,
        disallowed_tools=(list(args.disallow_tools) if args.disallow_tools is not None
                          else list(DEFAULT_DISALLOWED_TOOLS)),
        permission_mode=args.permission_mode)
    panel = PanelConfig(personas=[(p.name, p.grounding) for p in personas])
    halting = HaltingSet(token_budget=args.token_budget, max_epochs=args.max_epochs,
                         stall_patience=args.stall_patience, require_scope_complete=False)

    # Opt-in semantic reduce; otherwise the engine's deterministic identity default.
    reduce_kwargs = {"reduce": session.reduce} if args.semantic_dedup else {}
    try:
        review_run = run(
            spec, halting, panel,
            spawn=session.spawn, gather=session.gather,
            human_gate=session.interactive_gate, checkpoint=session.on_epoch,
            budget_spent=session.budget_spent, parallel=not args.no_parallel,
            **reduce_kwargs)
    except SurfaceError as exc:
        # No surface means no review: report the operator error and print no
        # verdict, rather than a halt line for a panel that reviewed nothing.
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"\n=== HALT: {review_run.halt_reason.value} after {len(review_run.epochs)} epoch(s) ===")
    print(f"open uglies: {len(review_run.open_uglies)} | open blockers: {len(review_run.open_blockers)}"
          f" | tokens: {session.tokens}")
    for f in review_run.open_uglies + review_run.open_blockers:
        print(f"  [{f.severity.name}] ({f.persona}) {f.title} @ {f.file}:{f.line}")
    # Severities the panel emitted that did not resolve to a level (#24). Reported
    # rather than absorbed: the finding was escalated, not read, and the operator
    # is the one who decides what the persona meant. Silent when there are none.
    unresolved = [
        {"epoch": e.index, "persona": r.persona, "raw": raw}
        for e in review_run.epochs for r in e.reports
        for raw in r.unresolved_severities]
    if unresolved:
        print(f"\nunresolved severities: {len(unresolved)} — escalated to "
              f"{UNRESOLVED_SEVERITY.name}, never downgraded")
        for u in unresolved:
            print(f"  (epoch {u['epoch']}) ({u['persona']}) {u['raw']!r}")
    # Canonical issues: the semantic reduce/view over the raw ledger. Every raw
    # finding stays visible above; this groups them (panel is raw material).
    if args.semantic_dedup and review_run.clusters:
        print(f"\ncanonical issues: {len(review_run.clusters)} "
              f"(reduced from {len(review_run.ledger)} raw findings)")
        for c in sorted(review_run.clusters, key=lambda c: c.severity, reverse=True):
            print(f"  [{c.severity.name}] ({len(c.members)}x) {c.title}")
    # Machine-readable output for the convening (synthesis) session to consume.
    print("\n--- JSON ---")
    print(json.dumps({
        "halt_reason": review_run.halt_reason.value,
        "epochs": len(review_run.epochs),
        "open_uglies": len(review_run.open_uglies),
        "open_blockers": len(review_run.open_blockers),
        "tokens": session.tokens,
        # Named per persona and verbatim, so a run's own artefact says which
        # severities it escalated rather than read (#24).
        "unresolved_severities": unresolved,
        "findings": [
            {"persona": f.persona, "severity": f.severity.name, "title": f.title,
             "file": f.file, "line": f.line, "open": f.counts_open}
            for f in review_run.ledger.values()],
        # Canonical clusters (the reduce/view). With the default identity reduce
        # this is one cluster per finding; with --semantic-dedup it collapses
        # same-concept findings while every raw finding stays under "findings".
        "clusters": [
            {"title": c.title, "severity": c.severity.name, "open": c.counts_open,
             "size": len(c.members),
             "members": [{"persona": m.persona, "severity": m.severity.name,
                          "title": m.title, "file": m.file, "line": m.line}
                         for m in c.members]}
            for c in review_run.clusters],
    }, indent=2))
    return 3 if review_run.halt_reason.value == "escalate_ugly" else 0
