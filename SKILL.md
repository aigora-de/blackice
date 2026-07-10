---
name: blackice
description: >-
  Run a human-in-the-loop adversarial review PANEL over mission-critical /
  records-critical / irreversible code (money movement, tax or regulatory
  records, data integrity, safety, migrations, auth). It drives one `claude -p`
  subagent PER reviewer persona PER epoch, iterating until a halting condition
  (converged / budget / epochs / stall) or a ruin-class "UGLY" circuit-breaker.
  Use when a change "looks obviously right" but being wrong is expensive, or when
  the user asks for a deep/rigorous/adversarial/multi-expert review, a "panel",
  or to "find edge cases, traps, pathologies". NOT for routine, reversible, or
  well-covered changes — it is deliberately heavy and spends tokens.
---

# blackice — HITL adversarial review panel

## What it is
A generalisation of the two-pass adversarial panel (see
`two-pass-adversarial-review-pattern.md`) into a **bounded, human-convened
iteration loop**. A deterministic Python engine (`loop.py`) owns the
control logic — halting, semantic dedup, stall detection, token/time budget,
and the **UGLY circuit-breaker** — and binds to Claude Code via
`claude_code_backend.py`, which spawns **one `claude -p` subprocess per persona
per epoch** (read-only: no edit tools). You (the convening `main` session) supply
the scope, run the loop, **synthesise** its output, and gate decisions with the
human. You are the *synthesiser, not the judge* — the human decides.

## How to run it

1. **Scope it.** Decide `--why` (the risk being guarded against) and `--what`
   (the change), and the review surface (`--base`/`--head`, or the working diff).
2. **(Optional pre-flight)** `--dry-run` to confirm *which* personas were sourced
   (e.g. from `CLAUDE.md`) and eyeball the assembled prompt. This spawns **no**
   `claude` process — it only prints the planned wiring — so it costs nothing.
   Worth doing the first time on a repo; skip it thereafter.
   ```
   python blackice.py \
     --repo <root> --base <base> --head <head> --dry-run
   ```
3. **Run live** (each persona is a real `claude -p` subprocess — costs tokens):
   ```
   python blackice.py \
     --repo <root> --base <base> --head <head> \
     --why "<why it matters>" --what "<what changed>" \
     --max-epochs 3 --token-budget 400000 [--model <alias>]
   ```
   The script prints a per-epoch synthesis, pauses at an interactive HITL gate
   (continue/stop) between epochs, and finally emits a `--- JSON ---` block for
   you to consume. Exit code `3` means an UGLY circuit-break.

## Your responsibilities as the convening session
- **Between epochs / after halt: synthesise.** Present each persona's view
  *distinctly* (don't blend them); surface disagreement.
- **Adjudicate BLOCKER/UGLY findings against source before relaying them.** A
  blocker is a hypothesis until verified — reviewers over-claim off misreads.
  Report the adjudication (confirmed / refuted).
- **Mutation-verify** load-bearing test claims where practical (neuter the fix →
  confirm exactly-red).
- **Apply fixes / adjust scope between epochs** if the human directs it, then let
  the loop re-gather the (now-updated) diff on the next epoch.
- **Never auto-approve a merge.** The human owns the verdict.
- **On `escalate_ugly`: stop and escalate immediately** — do not continue or
  "optimise" past a ruin-class finding.
- **File deferred concerns as tracked, dependency-ordered issues** — "complete to
  scope" is not "complete".

## Halting set (OR of predicates; UGLY checked first)
- **ESCALATE_UGLY** — any ruin-class finding → circuit-break + escalate.
- **CONVERGED** — no open UGLY, no open BLOCKER, (scope complete), quorum agrees.
- **BUDGET** — token or time ceiling reached (partial halt; BADs may remain **if
  tracked**).
- **EPOCH** — max epochs reached.
- **STALL** — K epochs with no new *material* findings while blockers remain open.

## Good / Bad / Ugly (severity → behaviour)
- **GOOD** — the absence of open blockers/uglies with scope covered (a halt target).
- **BAD** — NOTE / NON_BLOCKING / BLOCKER: bugs, weak logic, incomplete scope,
  scope-creep. Drive iteration or become tracked residuals.
- **UGLY** — ruin-class: non-linear, multiplicative, cascading, irreversible.
  A circuit-breaker **and** a non-negotiable gate: you may halt on budget with
  BADs outstanding-and-tracked; **never** with an open UGLY.

## Personas (the "how")
Resolved by precedence, all layers supported:
1. **`CLAUDE.md` "Resident Experts"** — parsed into personas (their defined role
   *is* their lens; mandates stay open-ended so we don't lead the witness).
2. **`panel.yaml` / `panel.md`** — an explicit panel definition.
3. **Distilled default set** — correctness, adversary, constraints, engineer,
   empiricist.
A **completeness-critic** and a **survivability (ruin) lens** are always ensured
(the ruin lens is skipped only when the sourced set already has one, e.g. a
tail-risk persona). Reviewers are **independent within an epoch**; epoch > 1
receives *all* prior epochs' findings (cross-epoch memory, no intra-epoch debate).
**Tools ground behaviour + permission policy:** deny-by-default. Personas get
**read-only** source inspection (`Read`/`Grep`/`Glob`) and **no shell or edit
tools** — because headless `claude -p` runs any *allowed* tool unsupervised (no
prompt), and HITL here is per-epoch, not per-command. Scoped verification tools
(`Bash(pytest:*)`, `Bash(git diff:*)`) are **opt-in** via `--allow-tools` — the
permissioned mode that lets the empiricist actually run the suite. A shipped
`--settings` profile + sandbox are further hardening. Never bare `Bash`. See `NOTES.md`.

## Status
Experimental, but exercised end-to-end (dogfooded over a real diff — it found
bugs a human-convened two-pass run missed). Implemented: read-only default +
`--allow-tools` for scoped verification; retry-on-contract-miss; the UGLY
circuit-breaker. Open work is in `NOTES.md` (notably semantic dedup and a richer
default persona set). The structured-findings contract is enforced via prompt.
