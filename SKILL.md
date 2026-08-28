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
iteration loop**. A deterministic Python engine (`kuang/engine/`) owns the
control logic — halting, dedup (a coarse signature always-on, plus an opt-in
semantic **reduce** into canonical clusters), stall detection, token/time budget,
and the **UGLY circuit-breaker** — and binds to Claude Code via
`kuang/backends/claude_code/`, which spawns **one `claude -p` subprocess per persona
per epoch** (read-only: no edit tools). You (the convening `main` session) supply
the scope, run the loop, **synthesise** its output, and gate decisions with the
human. You are the *synthesiser, not the judge* — the human decides.

## How to run it

1. **Scope it.** Decide `--why` (the risk being guarded against) and `--what`
   (the change or code under review), and the review surface. Exactly one mode:
   - **Diff mode** — `--base`/`--head`: review a *change*.
   - **Path mode** — `--paths <file|dir> …`: review *existing code* in full
     (proactive bug-hunting, or a repo with no reviewable diff). Directories
     expand via `git ls-files` (honouring `.gitignore`); a total-size cap
     (`--max-surface-bytes`) bounds the surface, naming anything it omits.
2. **(Optional pre-flight)** `--dry-run` to confirm *which* personas were sourced
   (e.g. from `CLAUDE.md`) and eyeball the assembled prompt. This spawns **no**
   `claude` process — it only prints the planned wiring — so it costs nothing.
   Worth doing the first time on a repo; skip it thereafter.
   ```
   kuang \
     --repo <root> --base <base> --head <head> --dry-run
   ```
3. **Run live** (each persona is a real `claude -p` subprocess — costs tokens):
   ```
   kuang \
     --repo <root> --base <base> --head <head> \
     --why "<why it matters>" --what "<what changed>" \
     --max-epochs 3 --token-budget 400000 [--model <alias>] [--semantic-dedup]
   ```
   The script prints a per-epoch synthesis, pauses at an interactive HITL gate
   (continue/stop) between epochs, and finally emits a `--- JSON ---` block for
   you to consume (with a `clusters` array alongside the raw `findings`). Exit
   code `3` means an UGLY circuit-break.
   - **`--semantic-dedup`** (opt-in) folds the same concern raised by multiple
     personas — differently worded, at different lines, or across files — into one
     **canonical cluster**, so stall/convergence and your synthesis count *issues*,
     not restatements. A cluster's severity is the **max** of its members
     (UGLY-preserving: a merge never hides ruin). It adds one cheap clustering call
     per epoch (`--cluster-model` to choose the model); the default is a
     deterministic signature dedup. Every raw finding stays visible, grouped.
   - **`--prior-findings <path>`** seeds the panel with an earlier run's ledger,
     for the hunt → fix → re-run loop. Without it every invocation starts cold and
     re-derives what you already fixed. Accepts a saved `findings.json`, a run's
     JSON output, or a raw `run.log` containing the `--- JSON ---` block. Injected
     from **epoch 1** (a confirm-the-fixes re-run may only get one), with each
     carried-in `file:line` marked **advisory** — the fixes moved the code, so
     personas adjudicate against current source. Prior-run and this-run findings
     arrive as separately labelled blocks, so a persona can tell "found before the
     fixes" from "found now". Fails loudly on an unreadable seed rather than
     starting silently cold.

## Your responsibilities as the convening session
- **Read the participation and reduce lines BEFORE the ledger.** Every run states,
  per persona per epoch, whether it *contributed*, *found nothing*, or never
  reviewed — and if it did not, which channel failed (`agent error` = the runtime
  returned no review; `spawn failed` = our own code raised; `unreadable` = it
  replied and we could not read it; `not spawned` = a dry run). It also states
  whether the semantic reduce *ran*, was *not requested*, had *nothing to reduce*,
  or **degraded** and why. A ledger from a panel that did not run in full is still
  real material, but its *silence* proves nothing: findings nobody raised are not
  findings nobody found. Never treat a degraded run as a clean pass — say so, and
  re-run before calling the surface clean.
- **Read the permissions line too, and read it as grounding.** Every run states the
  policy it used and, per persona, any granted tool the deny-list removed — such a
  tool is *absent* from the reviewer's session, not refused, so nothing else in the
  output betrays it. A panel that reviewed with no read tools speculated; its
  findings are hypotheses about a diff nobody opened, and its clean verdict is worth
  nothing. Refused tool *calls* are listed separately and are **not** a degradation
  on their own — deny-by-default means a refused `Bash` is usually the policy
  working. Adjudicate against source with extra care when either appears.
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
- **STALL** — K epochs with no new *material* findings while blockers remain open
  (with `--semantic-dedup`, a re-worded or re-located restatement of a known issue
  no longer counts as new material).

## Good / Bad / Ugly (severity → behaviour)
- **GOOD** — the absence of open blockers/uglies with scope covered (a halt target).
- **BAD** — NOTE / NON_BLOCKING / BLOCKER: bugs, weak logic, incomplete scope,
  scope-creep. Drive iteration or become tracked residuals.
- **UGLY** — ruin-class: non-linear, multiplicative, cascading, irreversible.
  A circuit-breaker **and** a non-negotiable gate: you may halt on budget with
  BADs outstanding-and-tracked; **never** with an open UGLY.
- **Unresolved severity** — a value a persona emitted that did not resolve to one
  of the four levels: unreadable (`CRITICAL`), or a *declared tie* the contract
  invites from an undecided reviewer (`BLOCKER/UGLY`). It is escalated to BLOCKER
  and reported verbatim, per persona, in the run output and under
  `unresolved_severities` in the JSON — never downgraded, and never silent. The
  tool does not pick a level it was not given: **you** adjudicate what the
  reviewer meant, and a tie that turns out to be ruin is yours to promote.

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
circuit-breaker; and an opt-in semantic **reduce** (`--semantic-dedup`) that
clusters same-concept findings (UGLY-preserving, degrades to signature dedup on
failure). Open work is in `NOTES.md` (notably a richer default persona set). The
structured-findings contract is enforced via prompt.
