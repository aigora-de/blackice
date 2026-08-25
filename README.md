# blackice

**A human-in-the-loop adversarial review panel for mission-critical code.**

`kuang` convenes a panel of adversarial reviewer *personas* — each tasked to
**break** a change, not approve it — and runs them in a **bounded iteration
loop** until a halting condition, with a ruin-class **circuit-breaker** that
stops the moment a survivability-threatening ("UGLY") finding appears.

It's the tool form of a review pattern for code where a subtle bug is expensive,
silent, or irreversible. The pattern and its prior art are in
[`two-pass-adversarial-review-pattern.md`](two-pass-adversarial-review-pattern.md).

> **What you install, import and run is `kuang`.** `pip install kuang`,
> `import kuang`, `kuang --repo …`. `blackice` is the repository this is
> developed in, and nothing else. The import name `blackice` is already taken on
> PyPI by an unrelated project that ships a top-level `blackice` package *and* a
> `blackice` console script from the same entry point — installing both merges
> the two into one directory, hands the command to whichever landed last, and
> breaks the other's uninstall, all silently. Hence **Kuang**, after the
> intrusion program that cuts through ICE.

## When to use it
Execution paths where being wrong is expensive, silent, or irreversible — money
movement, tax/regulatory records, data integrity, safety interlocks, migrations,
auth. A change that "looks obviously right" in a place you can't afford a bug.
**Not** for routine, reversible, or well-covered changes: it is deliberately
heavy and spends tokens.

## How it works
Three subpackages, three concerns:
- **`kuang/cli/`** — the **entry point you run** (`kuang`). Wires a
  backend into the engine and exposes the CLI. This is what the examples below invoke.
- **`kuang/engine/`** — the deterministic **engine**. Owns the control loop:
  halting predicates, dedup/stall detection, token/time budget, and the UGLY
  circuit-breaker. Backend-agnostic (dependency-injected seams); not run directly.
  It imports nothing from a backend, and a test enforces that.
- **`kuang/backends/claude_code/`** — a **backend**: binds the engine to the
  **Claude Code CLI** (one `claude -p` subprocess per persona per epoch) and sources
  the panel. A different runtime would be its own subpackage, swapped in by the
  entry point — the engine is unchanged.

Each **epoch** fans out the panel (independent within the epoch), adjudicates
findings against source, and dedups against the running ledger — a coarse
signature always, plus an opt-in **semantic reduce** (`--semantic-dedup`) that
folds the same concept raised by several personas (differently worded or located)
into one **canonical cluster**, so stall/convergence *and* the human summary count
issues, not raw restatements. A cluster's severity is the **max** of its members,
so a merge can never hide an UGLY. Epoch *N>1* receives all prior findings
(cross-epoch memory). **A human convenes and gates; the panel informs — it does
not decide.**

**Halting set** (OR of predicates, ruin checked first): `ESCALATE_UGLY` ·
`CONVERGED` · `BUDGET` · `EPOCH` · `STALL`. Severity ladder: **GOOD** (nothing
open) / **BAD** (bugs, weak logic, scope-creep — iterate or track) / **UGLY**
(non-linear, cascading, irreversible — circuit-break; never halt with one open).

## Install
```bash
pipx install kuang          # recommended for a CLI: its own environment, command on PATH
pip install kuang           # or into an environment you already have
pip install 'kuang[yaml]'   # only if a target repo defines its panel in panel.yaml
```
Python 3.11+. No runtime dependencies — the core is stdlib-only.

`0.2.0` is on PyPI, and it is early: it predates the epoch-2 work on whether a run
can be trusted to report what it actually did. See [`ROADMAP.md`](ROADMAP.md) —
`v0.3.0` is the first release we would recommend installing rather than reading.

**The `claude` CLI is a prerequisite, and installing this does not install it.**
Each persona is a real `claude -p` subprocess, so the Claude Code CLI has to be on
the machine; it is not a Python package and cannot be declared as a dependency.
It is looked for in this order:

1. `$CLAUDE_BIN`, if set and the path exists
2. `claude` on `PATH`
3. `~/.local/bin/claude`

If none resolve, a live run currently fails with a `FileNotFoundError` traceback
at the first spawn rather than a diagnosis ([#51](https://github.com/aigora-de/blackice/issues/51)).
`--dry-run` spawns nothing, so it works without `claude` installed and is the
cheapest way to confirm the rest of the wiring.

## Quickstart
```bash
# Pre-flight (spawns nothing, costs nothing): confirm which personas were sourced
kuang --repo <root> --base <base> --head <head> --dry-run

# Live, read-only (the safe default)
kuang --repo <root> --base <base> --head <head> \
  --why "why this matters" --what "what changed" --max-epochs 2

# Permissioned: let reviewers verify against source / run the suite (scoped)
kuang --repo <root> --base <base> --head <head> \
  --allow-tools Read Grep Glob 'Bash(git:*)' 'Bash(pytest:*)' --permission-mode default

# Path mode: review existing code (whole files/dirs), not a diff — proactive
# bug-hunting, or a repo with no reviewable diff. Dirs expand via git ls-files.
kuang --repo <root> --paths src/pkg/a.py src/pkg/b/ --max-epochs 2

# Semantic dedup: fold the same concept (raised by several personas, worded or
# located differently) into one canonical issue — sharpens stall/convergence and
# the summary. Opt-in: adds a cheap clustering call per epoch (--cluster-model to
# pick a model); the default is a deterministic signature dedup.
kuang --repo <root> --base <base> --head <head> \
  --semantic-dedup --max-epochs 2

# Cross-run memory: seed a re-run with the previous run's ledger, so the panel
# reports which findings the fixes actually closed instead of re-deriving them
# cold. Takes a saved findings.json, a run's JSON output, or a raw run.log with
# the '--- JSON ---' block in it. Both modes.
kuang --repo <root> --paths src/pkg/ \
  --prior-findings runs/2026-01-01/run.log
```
Exactly one mode per run: `--base/--head` (diff) **or** `--paths` (whole-file).

## Personas
Sourced by precedence: a repo's **`CLAUDE.md` "Resident Experts"** →
**`panel.yaml`/`panel.md`** → a **distilled default set**. A completeness-critic
and a survivability (ruin) lens are always ensured. Mandates stay open-ended (the
persona's role is its lens; we don't lead the witness). See [`SKILL.md`](SKILL.md).

## Permission model
**Deny-by-default, read-only** (`Read`/`Grep`/`Glob`; no shell or edits) —
because headless `claude -p` runs any *allowed* tool unsupervised, and HITL here
is per-epoch, not per-command. Scoped verification (`Bash(pytest:*)`, `Bash(git
diff:*)`) is opt-in via `--allow-tools`. Never bare `Bash`.

## Layout
| Path | Role |
|------|------|
| `kuang/cli/` | **entry point** — run this; wires a backend into the engine |
| `kuang/engine/` | deterministic engine: `findings` · `protocols` · `halting` · `reduce` · `loop` |
| `kuang/backends/claude_code/` | Claude Code CLI backend: `personas` · `surface` · `spawn` · `contract` · `memory` · `cluster` · `permissions` · `session` |
| `kuang/report.py` | presentation shared by the above (ledger lines, argv rendering) |
| `SKILL.md` | the skill definition (how a convening agent runs it) |
| `NOTES.md` | design notes, open decisions, backlog |
| `RELEASING.md` | how a release is cut and published (Trusted Publishing) |
| `two-pass-adversarial-review-pattern.md` | the pattern + origin case study |

## Status
Experimental; dogfooded end-to-end. Semantic dedup is implemented as an opt-in
reduce step (`--semantic-dedup`, UGLY-preserving). Open work is organised into
eight sequenced epochs in [`ROADMAP.md`](ROADMAP.md), which also states what
would make this 1.0; the design reasoning behind individual items is in
[`NOTES.md`](NOTES.md).

---
Licensed under either of **MIT** ([`LICENSE-MIT`](LICENSE-MIT)) or **Apache-2.0** ([`LICENSE-APACHE`](LICENSE-APACHE)) at your option.
Copyright © 2026 Agilit Ltd.
