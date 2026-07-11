# blackice

**A human-in-the-loop adversarial review panel for mission-critical code.**

`blackice` convenes a panel of adversarial reviewer *personas* — each tasked to
**break** a change, not approve it — and runs them in a **bounded iteration
loop** until a halting condition, with a ruin-class **circuit-breaker** that
stops the moment a survivability-threatening ("UGLY") finding appears.

It's the tool form of a review pattern for code where a subtle bug is expensive,
silent, or irreversible. The pattern and its prior art are in
[`two-pass-adversarial-review-pattern.md`](two-pass-adversarial-review-pattern.md).

## When to use it
Execution paths where being wrong is expensive, silent, or irreversible — money
movement, tax/regulatory records, data integrity, safety interlocks, migrations,
auth. A change that "looks obviously right" in a place you can't afford a bug.
**Not** for routine, reversible, or well-covered changes: it is deliberately
heavy and spends tokens.

## How it works
Three files, three concerns:
- **`blackice.py`** — the **entry point you run**. Wires a backend into the engine
  and exposes the CLI. This is what the examples below invoke.
- **`loop.py`** — the deterministic **engine**. Owns the control loop:
  halting predicates, dedup/stall detection, token/time budget, and the UGLY
  circuit-breaker. Backend-agnostic (dependency-injected seams); not run directly.
- **`claude_code_backend.py`** — a **backend**: binds the engine to the **Claude
  Code CLI** (one `claude -p` subprocess per persona per epoch) and sources the
  panel. A different runtime would be its own `*_backend.py`, swapped in by the
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

## Quickstart
```bash
# Pre-flight (spawns nothing, costs nothing): confirm which personas were sourced
python blackice.py --repo <root> --base <base> --head <head> --dry-run

# Live, read-only (the safe default)
python blackice.py --repo <root> --base <base> --head <head> \
  --why "why this matters" --what "what changed" --max-epochs 2

# Permissioned: let reviewers verify against source / run the suite (scoped)
python blackice.py --repo <root> --base <base> --head <head> \
  --allow-tools Read Grep Glob 'Bash(git:*)' 'Bash(pytest:*)' --permission-mode default

# Path mode: review existing code (whole files/dirs), not a diff — proactive
# bug-hunting, or a repo with no reviewable diff. Dirs expand via git ls-files.
python blackice.py --repo <root> --paths src/pkg/a.py src/pkg/b/ --max-epochs 2

# Semantic dedup: fold the same concept (raised by several personas, worded or
# located differently) into one canonical issue — sharpens stall/convergence and
# the summary. Opt-in: adds a cheap clustering call per epoch (--cluster-model to
# pick a model); the default is a deterministic signature dedup.
python blackice.py --repo <root> --base <base> --head <head> \
  --semantic-dedup --max-epochs 2
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
| File | Role |
|------|------|
| `blackice.py` | **entry point** — run this; wires a backend into the engine |
| `loop.py` | deterministic engine (halting, dedup, budget, circuit-breaker) |
| `claude_code_backend.py` | Claude Code CLI backend (persona sourcing, spawn/gather/gate) |
| `SKILL.md` | the skill definition (how a convening agent runs it) |
| `NOTES.md` | design notes, open decisions, backlog |
| `two-pass-adversarial-review-pattern.md` | the pattern + origin case study |

## Status
Experimental; dogfooded end-to-end. Semantic dedup is implemented as an opt-in
reduce step (`--semantic-dedup`, UGLY-preserving). Open work (richer default
personas, sandboxing) is tracked in [`NOTES.md`](NOTES.md).

---
Licensed under either of **MIT** ([`LICENSE-MIT`](LICENSE-MIT)) or **Apache-2.0** ([`LICENSE-APACHE`](LICENSE-APACHE)) at your option.
Copyright © 2026 Agilit Ltd.
