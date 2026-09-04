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
import textwrap
from collections import Counter
from pathlib import Path

from kuang.backends.claude_code import (DEFAULT_DISALLOWED_TOOLS,
                                          UNRESOLVED_SEVERITY, PanelSession,
                                          ReduceState, SurfaceError,
                                          SurfaceRecord, called_no_tool,
                                          load_personas, load_prior_findings,
                                          unavailable_tools, ungrounded_keys)
from kuang.engine import (HaltingSet, PanelConfig, PersonaReport, ReviewSpec,
                          run)


def _turn_note(report: PersonaReport) -> str:
    """What one persona's participation line says about its turn count (#70).

    Three states, kept apart because collapsing any two of them is the defect this
    reports. A count the runtime gave; a count it did not give for a persona that
    nonetheless reviewed — said aloud, because a backend's silence must not pass
    for a fact; and nothing at all for a persona that was never called, whose
    status already says so and which suffered no degradation to report.

    The CALLED NO TOOL note is attached only to a persona that produced a review
    of record. A call that errored or crashed has no review to be ungrounded, and
    #29 already reports it under its own heading — one failure, one heading.
    """
    if report.turns > 0:
        note = f", {report.turns} turn(s)"
        if report.status.reviewed and called_no_tool(report.turns):
            note += " — CALLED NO TOOL"
        return note
    return ", turn count unreported" if report.status.reviewed else ""


def _surface_note(record: SurfaceRecord | None) -> str:
    """What one epoch's line says about the surface the panel was handed (#69).

    Silent unless the panel was given less than the operator named — a rule that
    fires on a small diff or a surface that fits is the mirror image of the defect
    it reports. What the surface WAS is said unconditionally, one line up
    (``_surface_epoch_line``); this is the clause that says something went wrong,
    and absence is a claim it can afford to make.

    Each loss is named for what it was, never totalled: files dropped whole at the
    cap, the one file cut mid-way because it alone over-ran, a file that could not
    be decoded, and a named path that matched nothing tracked are four different
    things to have gone wrong with a review, and an operator deciding whether to
    re-run at a higher cap needs to know which. The counts here; the names below
    (#74).
    """
    if record is None or not record.degraded:
        return ""
    lost = []
    if record.omitted:
        lost.append(f"{record.omitted} file(s) omitted at the cap")
    if record.truncated:
        lost.append("1 file cut mid-way at the cap")
    if record.unreadable:
        lost.append(f"{record.unreadable} file(s) could not be read")
    if record.unresolved:
        lost.append(f"{record.unresolved} named path(s) matched no tracked file")
    return f" — SURFACE REDUCED ({'; '.join(lost)})"


def _surface_complete(record: SurfaceRecord) -> str:
    """What one epoch's line says when nothing was lost — and the two modes do
    not say the same thing (#74).

    Path mode applies a cap and can report against it, so it can say every named
    path was included. Diff mode applies no cap at all (#27), so all it can
    honestly say is that nothing was dropped and nothing was bounded: claiming a
    completeness nobody checked is the failure this whole epoch is about. Two
    sentences, two source lines, each mutable apart.
    """
    if record.mode == "diff":
        return " — nothing was dropped, and no cap was applied"
    return " — every named path was included"


def _surface_header(record: SurfaceRecord) -> str:
    """The run-constant inputs: the mode, what was named, and the cap.

    Taken from the first epoch's record because these are the operator's own
    argv and do not vary between epochs; what does vary is on the epoch lines.
    """
    if record.mode == "diff":
        refs = f"{record.refs[0]}...{record.refs[1]}" if record.refs else "unreported"
        return (f"review surface: diff — {refs} | "
                "no cap was applied (diff mode is unbounded)")
    named = " ".join(record.paths) if record.paths else "unreported"
    cap = f"cap {record.cap} bytes" if record.cap is not None else "no cap was applied"
    return f"review surface: paths — {named} | {cap}"


def _surface_epoch_line(index: int, record: SurfaceRecord) -> str:
    """What this epoch's surface was made of, and how it fell short if it did.

    The surface is re-gathered every epoch and legitimately CHANGES between them:
    a diff shrinks as fixes land at the gate, which is the tool working. So the
    composition is per-epoch, and a smaller surface is never reported as a loss.

    A file count the runtime could not give is said aloud rather than printed as
    a 0 that reads like a measurement — ``_turn_note``'s rule, applied to a fact
    of ours instead of a backend's.
    """
    counted = (f"{record.files} file(s)" if record.files is not None
               else "file count unreported")
    line = f"  epoch {index}: {counted}, {record.size} bytes"
    return line + (_surface_note(record) or _surface_complete(record))


def _name_line(label: str, names: tuple[str, ...]) -> str:
    """One loss, named. Wrapped with a hanging indent, count first.

    The list is not capped: a cap on it would be a threshold on a value nobody
    has baselined, and the artefact already carries unbounded ``findings``. The
    count leads, so a reader who does not want the names has already read the
    number. Wrapping is presentation and nothing else — no name is elided.
    """
    return textwrap.fill(f"{label} ({len(names)}): {', '.join(names)}", width=88,
                         initial_indent="    ", subsequent_indent="      ")


def _surface_loss_lines(record: SurfaceRecord) -> list[str]:
    """The files the operator named and did not get (#74's third criterion).

    Named because a count tells an operator how many files never reached a
    reviewer but not which, and the decision they have to make — re-run at a
    higher cap, or fix the path — needs the which. The files that WERE included
    are counted and never named: that boundary is #69's, and it holds.
    """
    lines = []
    if record.omitted_files:
        lines.append(_name_line("omitted at the cap", record.omitted_files))
    if record.truncated_file:
        lines.append(_name_line("cut mid-way at the cap", (record.truncated_file,)))
    if record.unreadable_files:
        lines.append(_name_line("could not be read", record.unreadable_files))
    if record.unresolved_paths:
        lines.append(_name_line("matched no tracked file", record.unresolved_paths))
    return lines


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
                         "Read Grep Glob. A Bash(...) grant does NOT bound what is run "
                         "(issue #96): it is cancelled by the default deny-list, or, "
                         "with Bash removed from it, becomes an unrestricted shell.")
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
    disallowed = (list(args.disallow_tools) if args.disallow_tools is not None
                  else list(DEFAULT_DISALLOWED_TOOLS))
    # Tools the panel granted AND deny-listed (#67). Such a tool is not refused when
    # a reviewer calls it — it is absent from the reviewer's session, so the reviewer
    # never calls it and the result envelope records nothing at all. Named HERE, not
    # only in the summary, because it follows from the flags alone: an operator who
    # first learns of it after the run has already paid for the whole panel.
    cancelled = {p.name: unavailable_tools(p.tools, disallowed) for p in personas}
    if any(cancelled.values()):
        starved = [n for n, t in cancelled.items() if t]
        print(f"[panel] WARNING: {len(starved)} of {len(personas)} persona(s) were "
              f"granted tools the deny-list removed, so those tools were NOT "
              f"available to them: "
              f"{', '.join(sorted({t for v in cancelled.values() for t in v}))}")

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
        disallowed_tools=disallowed,
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
    # Verdicts that were not a vote (#26). The same doctrine one field along: a
    # value that decides whether the run may report a good verdict is reported,
    # never absorbed, so quorum falling short is legible rather than mysterious.
    unread_verdicts = [
        {"epoch": e.index, "persona": r.persona, "raw": r.unresolved_verdict}
        for e in review_run.epochs for r in e.reports
        if r.unresolved_verdict is not None]
    if unread_verdicts:
        print(f"\nunresolved verdicts: {len(unread_verdicts)} — not counted "
              f"toward quorum")
        for u in unread_verdicts:
            print(f"  (epoch {u['epoch']}) ({u['persona']}) {u['raw']!r}")
    # Whether the panel actually RAN (#30). Printed on every run, and this is the
    # one place the pattern above is deliberately not copied: #24's and #26's
    # sections report an exception, so their absence is itself a complete claim.
    # "The panel ran in full" is the claim an operator most needs to trust and no
    # absent section can make it — and absent data is precisely this defect. Every
    # status is SET at the source; none is inferred from the ledger, because "zero
    # findings" is the ambiguous signal that made this issue necessary.
    # ``turns`` rides on the same records rather than opening a fourth always-on
    # section (#70). A turn count is a fact about one spawn, which is exactly what
    # a participation record is, and this section is already always-on — so the
    # claim "the panel actually opened the source" is made where the claim "the
    # panel ran" is, without a further block on every run. Measured on agent CLI
    # 2.1.246, ONE turn means the reviewer answered from the prompt alone: see
    # ``spawn.called_no_tool`` for the rule and why nothing finer is claimed.
    participation = [
        {"epoch": e.index, "persona": r.persona, "status": r.status.value,
         "findings": len(r.findings), "turns": r.turns}
        for e in review_run.epochs for r in e.reports]
    # What the panel was GIVEN (#69), one record per epoch, in the order they were
    # gathered. Known before any subprocess exists, so it is reported in a dry run
    # too — the opposite of ``turns``, where 0 means the runtime did not say, and
    # the same as #67's deterministic half. Pre-flight confirmation is the dry
    # run's only job, and "you named three files and one fitted" is the single
    # most useful thing it can say about a path-mode review.
    # What it WAS is appended after how it fell short (#74), so every key an
    # earlier artefact carried is unchanged and in the same place: a run archived
    # before this landed diffs against one taken after it, purely additively.
    surface_epochs = [
        {"epoch": i + 1, "mode": rec.mode, "omitted": rec.omitted,
         "truncated": rec.truncated, "unreadable": rec.unreadable,
         "unresolved": rec.unresolved, "bounded": rec.bounded,
         "degraded": rec.degraded,
         "size": rec.size, "files": rec.files, "cap": rec.cap,
         "refs": list(rec.refs) if rec.refs else None, "paths": list(rec.paths),
         "omitted_files": list(rec.omitted_files),
         "truncated_file": rec.truncated_file,
         "unreadable_files": list(rec.unreadable_files),
         "unresolved_paths": list(rec.unresolved_paths)}
        for i, rec in enumerate(session.surface_states)]
    # Always on, for the reason participation is (#30): "this is what the panel
    # looked at" is a claim no absent section can make, exactly as "the panel ran
    # in full" is. What the panel was GIVEN, then what it DID with it — so #69's
    # mark rides here, on the line carrying the composition it qualifies, rather
    # than on the participation line as well. Two sections about one thing is the
    # failure this section exists to avoid.
    if session.surface_states:
        print(f"\n{_surface_header(session.surface_states[0])}")
        for i, rec in enumerate(session.surface_states, 1):
            print(_surface_epoch_line(i, rec))
            for line in _surface_loss_lines(rec):
                print(line)
    print(f"\npanel participation: {len(personas)} persona(s) x "
          f"{len(review_run.epochs)} epoch(s)")
    for e in review_run.epochs:
        tally = Counter(r.status.value for r in e.reports)
        line = f"  epoch {e.index}: " + ", ".join(
            f"{n} {name.replace('_', ' ')}" for name, n in sorted(tally.items()))
        # Said once per epoch, in words, so a reader scanning a seven-persona
        # three-epoch run does not have to reconstruct it from the marks below.
        ungrounded = [r for r in e.reports
                      if r.status.reviewed and called_no_tool(r.turns)]
        if ungrounded:
            line += (f" — {len(ungrounded)} CALLED NO TOOL "
                     f"(answered from the prompt alone)")
        # What they were GIVEN is one section up (#74), not repeated here. Still
        # two marks and two failures: a surface the operator did not get and a
        # reviewer that did not look have different causes, either can happen
        # without the other, and neither may absorb the other — they are simply
        # reported where each belongs.
        print(line)
        for r in e.reports:
            # The count is findings EMITTED, meta findings included, so it agrees
            # with what "canonical issues" counts rather than quietly disagreeing.
            print(f"    [{r.status.value.replace('_', ' ')}] ({r.persona}) "
                  f"— {len(r.findings)} finding(s){_turn_note(r)}")
    # What the panel could actually DO (#67). Two facts, and they are not one fact
    # reported twice. A tool the panel granted AND deny-listed was never in the
    # reviewer's session — measured on agent CLI 2.1.246, the runtime removes it
    # rather than refusing it, so ``permission_denials`` is empty and no envelope
    # will ever report it. ``refusals`` is the other half: a tool that WAS in the
    # session, whose use the agent attempted and was refused.
    #
    # Always on, for the reason participation is (#30): "the panel had the tools it
    # says it had" is the claim an operator most needs to trust, and no absent
    # section can make it. The two halves are reported differently on purpose — a
    # deny-listed grant is OUR configuration contradicting itself and is a
    # degradation; a refused call may be the deny-by-default policy working exactly
    # as intended, so it is stated and not classified.
    permission_personas = [
        {"persona": p.name, "granted": list(p.tools),
         "unavailable": cancelled[p.name]}
        for p in personas]
    refusals = [
        {"epoch": e.index, "persona": r.persona, "tools": list(r.denied_tools)}
        for e in review_run.epochs for r in e.reports if r.denied_tools]
    print(f"\npanel permissions: mode={args.permission_mode} | "
          f"deny-list: {', '.join(disallowed) or 'none'}")
    starved_records = [r for r in permission_personas if r["unavailable"]]
    if starved_records:
        print(f"  DEGRADED — {len(starved_records)} of {len(personas)} persona(s) "
              f"were granted tools the deny-list removed:")
        for r in starved_records:
            print(f"    ({r['persona']}) granted {', '.join(r['granted'])} — "
                  f"{', '.join(r['unavailable'])} not available")
    else:
        print("  every granted tool was available to every persona")
    for e in review_run.epochs:
        refused = [r for r in refusals if r["epoch"] == e.index]
        if args.dry_run:
            # "no tool call was refused" would be true and would mean the opposite
            # of what an operator reads into it, so the dry run says which it is.
            print(f"  epoch {e.index}: nothing was spawned")
        elif refused:
            for r in refused:
                print(f"  epoch {e.index}: ({r['persona']}) refused: "
                      f"{', '.join(r['tools'])}")
        else:
            print(f"  epoch {e.index}: no tool call was refused")
    # Whether the reduce step ran (#30). Stated, never implied by a ratio: a dead
    # clusterer and a live one that merged nothing used to differ only by the nine
    # tokens the call itself cost. Two of the states are not degradations, and are
    # not reported as such.
    reduce_epochs = [
        {"epoch": i + 1, "state": st.value, "degraded": st.degraded}
        for i, st in enumerate(session.reduce_states)]
    if not args.semantic_dedup:
        state = ReduceState.NOT_REQUESTED.value.replace("_", " ")
        print(f"\nsemantic reduce: {state} (--semantic-dedup off)")
    else:
        print("\nsemantic reduce:")
        for entry in reduce_epochs:
            state = entry["state"].replace("_", " ")
            line = (f"DEGRADED — {state} (identity reduce used)"
                    if entry["degraded"] else state)
            print(f"  epoch {entry['epoch']}: {line}")
    # Canonical issues: the semantic reduce/view over the raw ledger. Every raw
    # finding stays visible above; this groups them (panel is raw material).
    #
    # Counted in two halves (#73). A tooling failure — ``agent error:`` (#29),
    # ``persona failed:`` (#25), the contract parser's diagnoses — is a finding
    # about the INSTRUMENT, and adding it to the count of defects in the change
    # is the run reporting itself as part of its own results. Nothing is excluded
    # to achieve it: both halves stay in the ledger and in the artefact, because a
    # failed persona's diagnosis has to remain recoverable, and only the count is
    # split. The flag is set where each finding is produced and is never matched
    # out of ``claim_class``, which is model-controlled — see ``Finding``.
    #
    # Both numbers are printed even when one is zero: "0 about the run" is the
    # claim an operator needs, and an absent clause cannot make it.
    issues_in_change = sum(1 for c in review_run.clusters if not c.about_run)
    issues_about_run = len(review_run.clusters) - issues_in_change
    if args.semantic_dedup and review_run.clusters:
        print(f"\ncanonical issues: {issues_in_change} in the change | "
              f"{issues_about_run} about the run "
              f"(reduced from {len(review_run.ledger)} raw findings)")
        for c in sorted(review_run.clusters, key=lambda c: c.severity, reverse=True):
            mark = " — about the run" if c.about_run else ""
            print(f"  [{c.severity.name}] ({len(c.members)}x) {c.title}{mark}")
    # Machine-readable output for the convening (synthesis) session to consume.
    ungrounded = ungrounded_keys(review_run)
    print("\n--- JSON ---")
    print(json.dumps({
        "halt_reason": review_run.halt_reason.value,
        "epochs": len(review_run.epochs),
        "open_uglies": len(review_run.open_uglies),
        "open_blockers": len(review_run.open_blockers),
        "tokens": session.tokens,
        # Named per persona and verbatim, so a run's own artefact says which
        # severities it escalated rather than read (#24), and which verdicts it
        # declined to count as votes (#26).
        "unresolved_severities": unresolved,
        "unresolved_verdicts": unread_verdicts,
        # Who was ASKED, and from where. Participation below says who took part;
        # without the roster an artefact read back cold cannot tell a seven-persona
        # panel with five silent from a two-persona panel that ran in full. It does
        # NOT close #16: recording the label a sourcing step returned is not the
        # same as noticing that the step fell back.
        "panel": {"source": source, "personas": [p.name for p in personas]},
        # Per persona, per epoch: whether it contributed, found nothing, or never
        # reviewed — and, if it did not, which channel failed (#30) — plus what the
        # review cost in turns, which is how a run says whether the reviewer opened
        # anything at all (#70). ``turns: 0`` means the runtime did not say; a
        # persona that was never spawned is at 0 and suffered nothing.
        "participation": participation,
        # What the panel could DO: the policy the run actually used, the granted
        # tools the deny-list cancelled, and the calls the agent was refused (#67).
        # The policy is recorded beside the verdict because "unavailable: []" is a
        # claim a reader coming to the artefact cold has no other way to audit.
        "permissions": {"mode": args.permission_mode, "denied": disallowed,
                        "personas": permission_personas, "refusals": refusals},
        # What the panel was GIVEN, per epoch: whether the assembled surface was
        # the one that was asked for, and — where it was not — how it fell short
        # (#69). Always present, because an absent key makes no claim and a run
        # read back cold cannot otherwise tell a review of everything from a
        # review of a third of it. ``bounded: false`` says a cap was never applied
        # to this mode at all (#27), which is what this record does NOT check.
        # What the surface WAS, rather than how it fell short, is #74.
        "surface": surface_epochs,
        # Whether the reduce step ran, was not asked for, had nothing to fold, or
        # degraded — and on which epoch (#30).
        "reduce": {"requested": bool(args.semantic_dedup), "epochs": reduce_epochs},
        # How many canonical issues are about the CHANGE and how many are about
        # the RUN (#73), so a headline read cold cannot mix a defect in the code
        # with a failure of the panel that read it. Every finding and every
        # cluster below carries the same flag, so the split is auditable rather
        # than asserted, and nothing is dropped from either array to produce it.
        "issues_in_change": issues_in_change,
        "issues_about_run": issues_about_run,
        # ``ungrounded`` is the other half of a finding's provenance (#71): the
        # call that first raised it called no tool, so the claim was answered from
        # the prompt alone. It rides here rather than only in cross-epoch memory
        # because this artefact is the INPUT to a later run's ``--prior-findings``
        # — without it a contaminated run seeds the next one clean. Computed by the
        # same join ``on_epoch`` uses, so the two cannot disagree.
        "findings": [
            {"persona": f.persona, "severity": f.severity.name, "title": f.title,
             "file": f.file, "line": f.line, "open": f.counts_open,
             "about_run": f.about_run, "ungrounded": key in ungrounded}
            for key, f in review_run.ledger.items()],
        # Canonical clusters (the reduce/view). With the default identity reduce
        # this is one cluster per finding; with --semantic-dedup it collapses
        # same-concept findings while every raw finding stays under "findings".
        "clusters": [
            {"title": c.title, "severity": c.severity.name, "open": c.counts_open,
             "size": len(c.members), "about_run": c.about_run,
             "members": [{"persona": m.persona, "severity": m.severity.name,
                          "title": m.title, "file": m.file, "line": m.line}
                         for m in c.members]}
            for c in review_run.clusters],
    }, indent=2))
    return 3 if review_run.halt_reason.value == "escalate_ugly" else 0
