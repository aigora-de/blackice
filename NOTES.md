# blackice — design notes & backlog

Working notes kept *outside* `SKILL.md` (which stays user-facing): open decisions
and rough edges from the first cut.

## Backlog (tracked issues)

- **#1** — **Semantic dedup / synthesis (reduce step).** *Priority.* Signature
  dedup (file + line-bucket + claim-class + severity) is insufficient — the same
  concept raised by several personas with *different* file:lines does not collapse.
  Add a semantic clustering/reduce step at the orchestration layer that folds
  findings into canonical issues, feeding **both** stall/convergence detection and
  the human output. Design detail in "Semantic dedup — where it should live" below.
- **#2** — Make the completeness-critic + survivability personas first-class (next
  section).
- **#3** — Persona quality: "stay strictly in your lens" mandate clause; per-persona
  tool tailoring; model-mix for high-criticality (see follow-ups below).
- **#4** — Permission hardening: ship a `--settings` profile + sandbox (see
  "Permission model").
- **#5** — Robustness/cost: `--max-turns` cap; cross-epoch memory pruning;
  structured-output robustness (see "Other first-cut follow-ups").

## Make the completeness-critic and survivability personas explicit (priority)

Right now these two are **auto-injected** by `_ensure_specialists()` in
`claude_code_backend.py` from hardcoded `COMPLETENESS_CRITIC` / `SURVIVABILITY`
constants, and the survivability lens is *suppressed* by a keyword match when the
sourced set already looks like it has a ruin/survivability persona.

This was fine to get moving, but it hides two personas that are arguably the most
important (the "what did everyone miss?" critic and the ruin/UGLY detector). Make
them **explicit, first-class default personas** instead of special-cased
injections — e.g.:

- define them in the distilled default set (and in a shipped `panel.yaml`), so
  they are visible, tunable, and overridable like any other persona;
- replace the keyword "already has a ruin lens?" match with an explicit capability
  tag (e.g. a `role: survivability` / `role: completeness` field a persona can
  declare), so suppression is intentional, not heuristic;
- let a repo opt out or re-word them deliberately rather than by accident of naming.

## Other first-cut follow-ups

- **Structured-findings contract brittleness** — findings are parsed from a fenced
  ```json block the persona is asked to emit. If models drift from the contract,
  consider a more robust extraction (or a `stream-json` tool-call style). A
  retry-on-contract-miss is already implemented as a first line of defence.
- **Per-persona tool tailoring** — all personas share the same toolset. The "tools
  ground behaviour" lever could be sharpened (e.g. the empiricist alone gets the
  test runner).
- **Model mix** — `--model` sets one default; per-persona `model` is supported in
  `panel.yaml` but not yet exercised for blind-spot diversity on high-criticality.
- **Synthesis boundary** — the script owns the loop and emits structured output for
  the convening `main` session to synthesise. If per-epoch synthesis *by* `main`
  mid-loop is wanted, factor the loop so `main` can drive it one epoch at a time
  (engine still owning halting/dedup/budget).
- **Cross-epoch memory volume** — `prior_summary` grows with the ledger; for long
  runs, summarise/prune older resolved findings before injecting.
- **`--max-turns` cap** — bound per-persona cost/latency in the argv.

## Permission model (important)

**How permissions work driving the agent CLI headless:** interactive mode prompts
per command; headless (`-p`) has no TTY, so there are no prompts. A tool call
resolves against `--allowedTools`/`--disallowedTools`, the repo/user
`settings.json`, and `--permission-mode`. Outcomes: *allowed → runs UNSUPERVISED*,
*not allowed → auto-DENIED (never asked)*, or *bypass all*.

**The trap:** allow-listing bare `Bash` pre-approves *all* shell for an agent
reviewing a possibly-untrusted diff — unrestricted, unsupervised (`rm`,
`git push`, `curl | sh`, network egress). It also inflates cost/latency (personas
wander, run suites).

**Decision (enforced):** **deny-by-default, read-only.** Allow `Read`/`Grep`/`Glob`;
disallow `Edit`/`Write`/`NotebookEdit`/`Bash`. The key principle: **HITL in this
loop is per-EPOCH (convene/synthesise/gate), not per-command — so per-command
safety must come from POLICY, not prompts.**

**Roadmap for verification power, in order of preference:**
1. **Scoped allow-list** — `--allow-tools "Bash(pytest:*)" "Bash(git diff:*)"
   "Bash(rg:*)" "Bash(ruff:*)"`; everything else denied. Gives the empiricist real
   test-running without a blanket shell. (Supported today via `--allow-tools`.)
2. **Ship a `--settings` profile** so the skill enforces *its own* policy rather
   than inheriting the target repo's (which may be permissive).
3. **Sandbox** (no internet, read-only FS bar a scratch dir) — even scoped `Bash`
   is safer sandboxed.
4. (Advanced) route genuinely-needed escalations to the human convener rather than
   silently deny; likely overkill — default-deny + re-run with a wider allow-list
   is usually enough.

## Semantic dedup — where it should live

NOT per-persona (breaks independence; duplicated across mandates) and NOT the core
review prompt. It belongs in the **orchestration/synthesis layer** as a dedicated
*reduce* step after the fan-out: cluster the epoch's findings into canonical issues
(an LLM call — cheap model), feeding **both** stall-detection (so re-worded /
differently-located dups don't read as "new material") **and** the human-facing
output. Either the engine runs a `synthesizer/clusterer` agent per epoch, or the
convening `main` session does the semantic synthesis (the script's signature dedup
stays a coarse backstop). Leaning toward an engine-level clusterer so
stall/convergence is semantically accurate, not just the presentation.

## Validation & lessons

Exercised end-to-end (a synthetic diff with an obvious bug — a function that lost
its empty-input guard, `ZeroDivisionError`; and larger real diffs). Confirmed:
subprocess spawn, JSON-envelope parse, structured-findings extraction, parallel
fan-out, cross-epoch memory (epoch N>1 labels findings `inherited`/`still open`/
`new`/`elevation`), the **UGLY circuit-breaker**, token accounting, and
retry-on-contract-miss.

Transferable lessons:
- **Permissions cut hallucinations.** A scoped verification allow-list reduces
  wrong line numbers and false positives versus read-only, because personas
  *check* claims instead of speculating. Read-only stays the safe default;
  permissioned is the "real run" mode.
- **Retry-on-contract-miss matters.** A persona that reviews but omits the output
  contract would otherwise be discarded entirely; reformatting its raw reply
  recovers it.
- **The circuit-breaker earns its place, but adjudicate.** UGLY escalations can be
  *conditional* (predicated on a code path that may not be reachable). A human
  adjudication pass against source is still required — raw panel output is
  material, not a verdict.
- **Weak models stray beyond their lens.** Use a stronger model for real runs
  and/or a "stay strictly in your lens" mandate clause.
- **Fixed startup gotcha:** the child process may not find the agent CLI on its
  `PATH` — `_resolve_claude_bin()` resolves `$CLAUDE_BIN` → `which` → a common
  install path.
