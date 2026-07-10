# The adversarial review loop — a pattern for mission-critical code

*The reusable pattern behind `blackice`.*

---

## 0. TL;DR

For code that **must** be correct — execution paths where a subtle bug is
expensive, silent, or irreversible (money movement, tax/regulatory records, data
integrity, safety, migrations, auth) — a single review pass is not enough. Use a
structured **adversarial panel** of independent reviewer personas, each tasked to
*break* the change rather than approve it, and run it as a **bounded,
human-convened loop**:

1. **Pass 1 — over the proposed diff:** find edge cases, traps, pathologies.
   *Adjudicate every claim against the source* (reviewers over-claim).
2. **Pass 2 — over the committed diff + the follow-up issues:** confirm the fix
   is correct **and complete to its stated scope**, and that every deferred
   concern is **tracked**. Require an explicit per-reviewer verdict.

"Two passes" is just N=2 of the loop. The value is not the sign-off. It's that
(a) a *wrong* blocker gets **withdrawn on the evidence** rather than argued about,
(b) claims are **mutation-verified**, not asserted, and (c) the boundary between
"fixed now" and "tracked for later" is made explicit and agreed.

---

## 1. When to use it

- **Mission-critical / records-critical / irreversible** paths: money movement,
  tax or regulatory records, data integrity, safety interlocks, migrations,
  auth/permission logic, anything failing *silently*.
- A change that "looks obviously right" in a path where being wrong is expensive.
- **Not** for routine, easily-reversible, or well-covered changes — it is
  deliberately heavy and spends tokens.

Why one pass isn't enough: a plausible-looking change in a records-critical path
can pass tests and human eyes yet harbour a reachable edge case that fails
silently. A single reviewer (human or agent) anchors on the happy path; an
adversarial *panel*, run to convergence, is far more likely to surface the trap —
and to distinguish a real hazard from a confident misread.

## 2. The reviewer panel

Independent personas with **distinct, partly-conflicting** mandates — the tension
is the point. A default set of generalised lenses (adapt to the domain):

- **Correctness / domain expert** — does it compute the right thing; edge cases.
- **Adversary / threat modeller** — what breaks it; worst-case inputs; pathologies.
- **Compliance / constraints** — external rules the code must not violate.
- **Engineer** — code quality, hidden state, error handling, change discipline.
- **Empiricist / test-rigour** — would each test fail without the fix; coverage gaps.

Two specialists are always ensured: a **completeness-critic** ("what did nobody
check?") and a **survivability / ruin lens** (hunts non-linear, multiplicative,
cascading, irreversible failure). Each persona is told: **be adversarial; find
where it is wrong or incomplete; only approve what you cannot break.** Run one
reviewer per agent, in parallel, over the same diff. Present views **distinctly** —
surface disagreement, don't blend it.

## 3. The loop (two passes is just N=2)

A **bounded, human-convened iteration loop** over the panel, run until a halting
condition. "Two passes" is simply **N=2** — pass 1 on the *proposed* diff, pass 2
on the *committed* diff plus the filed follow-ups.

Each **epoch**: gather the surface → fan out the panel (one agent per persona, in
parallel, independent *within* the epoch) → **adjudicate** findings against source
→ **mutation-verify** load-bearing tests → classify severity → dedup vs the
running ledger (to detect *new material* findings) → evaluate the halting set.
Epoch N>1 receives *all* prior epochs' findings (cross-epoch memory); synthesis is
by the convening ("main") persona and **decisions are the human's**
(human-on-the-loop, not in every step).

**Halting set — an OR of predicates, the ruin check first:**
- **ESCALATE (Ugly)** — any ruin-class finding → **circuit-break** and escalate.
- **Converged (Good)** — no open Ugly, no open blocker, scope complete, quorum agrees.
- **Budget** — token or time ceiling (partial halt; *Bads* may remain **if tracked**).
- **Epoch** — max iterations reached.
- **Stall** — K epochs with no new *material* findings while blockers remain open.

**Good / Bad / Ugly** (severity → behaviour):
- **Good** — the absence of open blockers/uglies with scope covered (a halt target).
- **Bad** — bugs, weak logic, incomplete scope, scope-creep. Iterate, or track.
- **Ugly** — ruin-class: non-linear, multiplicative, cascading, irreversible. A
  circuit-breaker **and** a non-negotiable gate: you may halt on budget with *Bads*
  outstanding-and-tracked; **never** with an open Ugly.

Pass 2's two questions — *(a) correct & complete to scope?* and *(b) residuals
tracked?* — are just the **Converged** gate made explicit.

## 4. Non-negotiable rules (the parts that give it teeth)

- **Adjudicate every claim against source.** A blocker is a hypothesis until
  verified. Reviewers over-claim off misreads; check the code, and record the
  adjudication. Ideally the author of a refuted claim **re-verifies and withdraws
  it** on a later pass. *A panel's output is raw material, not a verdict.*
- **Mutation-verify tests.** Neuter the fix; confirm *exactly* the intended tests
  go red. Label characterisation-only tests honestly.
- **Make the scope boundary explicit.** Everything not fixed now must be a filed,
  dependency-ordered issue — "complete to scope" ≠ "complete".
- **Verify the worktree after a panel** if reviewers can mutate a shared tree
  (`git status` + `git diff HEAD`) — a concurrent reviewer's experiment can leave
  a transient state another reviewer misreads.
- **Defer scope decisions to the human** at the gates; the panel informs, it
  doesn't decide.

## 5. Tools it should orchestrate

- `git diff <base>...<branch>` / `git show <sha>` — the exact review surface.
- `git status` + `git diff HEAD` — post-panel worktree integrity check.
- a test runner — run + **mutation-verify** (neuter, re-run).
- linters / type-checkers — cheap correctness signal per reviewer.
- an issue CLI — read follow-ups from source; file tracked residuals.

## 6. The skill (implemented)

This pattern is implemented in this repository as **`blackice`**: a deterministic
engine (`loop.py`, public API `loop.run()`) that owns the loop — halting, dedup,
stall detection, token/time budget, and the **Ugly circuit-breaker** — bound to a
coding agent via a backend (`claude_code_backend.py`), which drives **one subagent
per persona per epoch** (read-only by default). The human is the convener and the
gate; the agent runtime supplies the fan-out; the script supplies the missing
control loop.

The domain-specific part is only the *reviewer persona set*, sourced by
precedence — a repo's `CLAUDE.md` experts → `panel.yaml`/`panel.md` → a distilled
default — so a project gets its own panel for free. The backend is
dependency-injected, so the same loop runs against a real agent CLI, an SDK, or a
fake for tests. See `SKILL.md`.

---

## 7. Prior art

The adversarial multi-agent review panel is well-trodden ground. Several public
agent skills implement the core idea, and there is a research literature behind
the "teeth" (mutation-verification, claim-checking against source). Before
building your own, evaluate these — this pattern's *differentiators* (below) are
narrower than the whole thing.

**Comparable public skills**
- [agent-review-panel](https://github.com/wan-huiyan/agent-review-panel) (MIT) —
  the closest match: persona reviewers → multi-round adversarial debate → a
  "supreme judge" verdict, with severity confirmation against source and a
  post-judge hallucination gate. Multi-phase, but a *single* review episode.
- [Deep Review](https://mcpmarket.com/tools/skills/deep-review-2) — Advocate /
  Skeptic / Architect subagents.
- Adversarial-review variants:
  [BMAD](https://mcpmarket.com/tools/skills/adversarial-code-review),
  [pedronauck](https://crossaitools.com/skills/pedronauck/skills/adversarial-review),
  [alirezarezvani](https://alirezarezvani.github.io/claude-skills/skills/engineering-team/adversarial-reviewer/).
- [claude-octopus](https://github.com/nyldn/claude-octopus) — multi-LLM
  deliberation with **quorum + critical-veto gates**.

**Official foundation (for building/distributing a skill)**
- [Anthropic — Equipping agents with Agent Skills](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills)
- [anthropics/skills](https://github.com/anthropics/skills) · [Agent skills docs](https://code.claude.com/docs/en/skills) · registry: agentskills.io
- The official `code-reviewer` / `simplify` skills are the lightweight,
  single-pass baseline this pattern deliberately escalates from.

**Research behind the "teeth"**
- Mutation-verification: [Meta — LLMs for mutation testing (ACH)](https://engineering.fb.com/2025/09/30/security/llms-are-the-key-to-mutation-testing-and-better-compliance/) ·
  [Test vs Mutant: Adversarial LLM Agents](https://arxiv.org/pdf/2602.08146) ·
  [AgentAssay — mutation operators for agent artifacts](https://arxiv.org/pdf/2603.02601)
- Debate / critique pipelines: [TriAdReview — triangular adversarial review](https://arxiv.org/pdf/2606.15074) ·
  [Multi-Agent LLM Debater](https://github.com/mjsushanth/Multi_Agent_LLM_Debater) ·
  Mixture-of-Agents (Generate → Cross-Critique → Rebuttal → Judge)

**What's distinctive here (worth keeping):**
1. **Two passes tied to the change lifecycle** — pass 1 on the *proposed* diff;
   pass 2 on the *committed* diff **plus the filed follow-up issues**, explicitly
   checking that deferred concerns are *tracked*. Most tools run one (multi-phase)
   episode; none gate on "residuals filed and dependency-ordered".
2. **Human at the gates, not a "supreme judge."** The panel informs; a person
   decides fix-now vs defer and owns scope.
3. **Source-adjudication + author-withdrawal** and **mutation-verification** as
   *hard rules*, not optional phases.
4. **Scope-boundary → dependency-ordered tracked issues** as a first-class output.

**Recommendation:** don't reinvent the panel. Trial the closest existing skill
first; if it covers ~80%, fork/configure rather than build, and layer these
differentiators (two-pass committed-and-tracked gate, human-gated scope, mandatory
source-adjudication + mutation-verify) on top.

*(Reference URLs surfaced via web search; sanity-check before relying on them.)*
