# blackice — roadmap

The work is organised into **eight sequenced epochs**. Each epoch is a tracked issue, a
GitHub milestone, and a tag on completion. This document says what each is for, what
order they go in, and why that order.

It is a plan, not a promise. Epoch 3 exists to produce evidence that will reshape the
epochs after it; anything past epoch 3 is a hypothesis held with proportionate confidence.

---

## What this plan is built on

The sequence is not taste. It follows from measurements taken against the code and against
an archive of 18 real runs:

- **The tool could report GOOD on a review it never performed.** A mistyped base ref made
  `git diff` exit non-zero; the return code and stderr were discarded; the panel was handed
  an empty string, found nothing, and the loop halted `CONVERGED`. Fixed ahead of the plan
  (#18), and it is the reason epoch 2 precedes everything that spends money.
- **Silent degradation is the recurring defect class.** A dead reduce step in 2 of 15 runs.
  Two absent personas in another, on a load-bearing pass. A severity string with a
  parenthetical silently recorded three levels lower. In every case the run reads normally.
- **The circuit-breaker cannot be released.** `ESCALATE_UGLY` fires while any UGLY is open;
  a finding is open unless adjudication refutes it; **adjudication is never wired**. Every
  archived run that raised an UGLY halted in epoch 1, and `CONVERGED` has never once
  occurred. The bounded loop is not unimportant — it is unreachable.
- **Lenses are less distinct than intended.** Six specialists produced 126–150 findings
  each, UGLY rates within 6.2–10.3%, and title-vocabulary overlap of ~0.37 for *every*
  pair, with no structure.
- **But lenses are not redundant.** 51.3% of canonical issues were raised by exactly one
  persona. Each lens was the sole raiser of 17.5–29% of what it touched — which is why
  panel size must be chosen rather than maximised.
- **A whole methodology lives outside the tool.** Pre-registered predictions, run archives,
  participation checks: an operator built all of it in shell because the tool does not
  record enough to be trusted on its own.

---

## The epochs

| # | Epoch | Tag | What it settles |
|---|---|---|---|
| 1 | **Foundation** | `v0.2.0` | Package structure, one implementation per concern, installable |
| 2 | **Instrument integrity** | `v0.3.0` | The run must not lie, degrade silently, or die — **first PyPI publish** |
| 3 | **Evidence** | `v0.4.0` | Eight dogfood runs, one per core code path |
| 4 | **Teeth** | `v0.5.0` | Adjudication, tracked residuals, mutation-verification — **the 1.0 gate** |
| 5 | **Panel** | `v0.6.0` | Distinct lenses, declared identity, selected for the ask |
| 6 | **Interface** | `v0.7.0` | The verb CLI and the operator surface |
| 7 | **Safety and cost** | `v0.8.0` | Make the mode people want to use safe to use |
| 8 | **Documentation** | `v0.9.0` | Say what the tool actually does |

### Why this order

**Foundation first** because it is mechanical and its cost rises with every pull request
that touches the flat modules — one rebase now, a dozen later.

**Integrity before evidence** because eight expensive runs on an instrument that can
silently degrade buy evidence nobody can trust. The archive already contains two degraded
runs that were read as real.

**Evidence before teeth and panel** because both of those epochs are shaped by what the
runs find, and because the dogfood runs are themselves the test of whether blackice works
in the role we are proposing for it.

**Teeth before panel** because the teeth unblock the loop: once a refuted UGLY stops
latching the breaker, epoch 2 of a run becomes reachable for the first time, and the panel
work is then being tuned against a system that can actually iterate.

**Interface and safety after** because both are better designed once the run artefact
carries enough to be worth reading back, and once the panel's shape has settled.

**Documentation last** because that is when there is something settled to describe — and
because three of the four differentiators the documents claim will only have become true
somewhere in epochs 4 and 5.

---

## Versions and releases

- **`v0.1.0`** tags the state that produced every measurement above: the tool as
  dogfooded, plus the surface-assembly fix. It is a baseline, not a recommendation.
- **Each epoch bumps a minor** on completion. Tags mark epoch boundaries; milestones track
  the work. There is no milestone-per-release — they would say the same thing twice.
- **The first PyPI publish is `v0.3.0`**, at the close of epoch 2. Packaging lands in
  epoch 1, so the capability arrives earlier than the act: being installable is not the
  same as being fit to hand someone. Publishing before epoch 2 would distribute failure
  modes that are invisible to anyone who did not build them.
- Every release before 1.0 is a pre-release in intent: the interface will move.

### What would make this 1.0

**1.0 is a gate, not a date.** It is claimed when the differentiators the project exists
for are not merely built but *demonstrated*:

1. Adjudication is wired: a finding can be confirmed or withdrawn against source, and the
   result is recorded in the run artefact rather than in someone's head.
2. A refuted UGLY no longer latches the breaker — shown by a run that continues past one.
3. **A run converges.** `CONVERGED` has never occurred in 18 runs. Until one does, the
   GOOD branch of Good/Bad/Ugly is untested in the field.
4. Deferred concerns leave a run as structured, tracked output — the fixed-now versus
   tracked-later boundary made explicit by the tool, not by a human's notes.

Epoch 4 is the earliest this can happen. If epoch 4 ships and no run ever converges, we do
not get 1.0 — we get a finding, which is the more interesting outcome.

Note what is deliberately *not* on this list: feature count, interface stability, and
adoption. The version number is tied to the project's own stated value claim — *"the value
is not the sign-off"* — and nothing else.

---

## Scope control

The eight epochs are a fixed frame. Work will arrive that does not fit them — from the
dogfood runs especially — and the discipline is that it is **allocated, never accumulated**.

**Every issue carries exactly one milestone.** Either one of the eight epochs, or
**After the plan** for work that is real but deliberately outside the frame. An issue with
no milestone is a triage failure, not a backlog.

When new work arrives, it goes to an epoch only if it serves that epoch's stated goal. If
it merely *could* be done at the same time, it goes to **After the plan**. Epochs are
allowed to be amended — but by decision and in writing, not by drift.

The discovery issues (#21, #22, #23) exist precisely so that open questions are settled
before they become scope. Anything a discovery produces is filed and allocated like
anything else.

---

## Non-goals

Carried from `CLAUDE.md` and the pattern document, and binding on every epoch:

- **Never a judge.** No auto-approval, no auto-merge, no tool-owned verdict. The exit-code
  work in epoch 2 distinguishes outcomes; it does not gate anything.
- **Not a general code-review tool.** blackice is deliberately heavy and wrong for routine,
  reversible, or well-covered changes.
- **Not more panel.** The prior-art section recommends forking an existing panel skill if
  one covers most of the need. Any candidate feature that merely adds reviewing capacity is
  presumptively rejected; what justifies this project is the differentiators.
- **Stdlib-only core, deny-by-default permissions, a deterministic engine.** A change that
  requires abandoning one of these must say so explicitly and argue it.

---

## Tracking

| Epoch | Issue | Milestone |
|---|---|---|
| 1 Foundation | #39 | Epoch 1 — Foundation |
| 2 Instrument integrity | #40 | Epoch 2 — Instrument integrity |
| 3 Evidence | #41 | Epoch 3 — Evidence |
| 4 Teeth | #42 | Epoch 4 — Teeth |
| 5 Panel | #43 | Epoch 5 — Panel |
| 6 Interface | #44 | Epoch 6 — Interface |
| 7 Safety and cost | #45 | Epoch 7 — Safety and cost |
| 8 Documentation | #46 | Epoch 8 — Documentation |

Labels mirror the milestones (`epoch:1-foundation` …) for filtering; the milestone is the
authoritative allocation. `discovery`, `confirmed-defect`, `silent-degradation` and
`dogfood` classify work across epochs.
