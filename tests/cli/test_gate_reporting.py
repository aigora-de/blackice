# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""A run must report what happened at the human gate (#117).

The reporting half. Channel 9 of #40's table was reported by nothing: not whether
the gate was reached, not what the operator chose, not when — and because the gate
is the sole reporter of what an epoch raised, its silence took the deciding
epoch's account down with it.

Two channels, and the split is deliberate. The **artefact** carries an
unconditional ``gate`` row per epoch, because an absent key makes no claim
(#103's precedent) and "not a tty, continued without asking" is a fact about every
run this tool has ever done in CI rather than an absence. The **console** gains
one always-on end-of-run section that walks every epoch — it runs after the loop,
so unlike the gate it cannot claim what did not happen (#72), and unlike a notice
printed at the gate it cannot appear for some epochs and never for the one a run
halts on (the asymmetry #119 refused).

These drive ``kuang.cli.main`` end to end over a throwaway repo with the
subprocess boundary stubbed — nothing spawned, no network.

**This module is the first in the suite to stub anything other than the agent
call**, and the asymmetry is stated rather than left to read as inconsistency:
``interactive_gate`` returns early when stdin is not a tty, so no fixture that
leaves stdin alone can tell "the gate said continue" from "the gate was never
asked". A CLI test's job is to exercise OUR branch, so it supplies a stdin; a
probe capture's job is to stay black-box, which is why the probe harness gets no
abort scenario and states the gap instead.

Console assertions are anchored to the line they are about (#87, #103) — the gate
synthesis prints the same four counts two sections above, so a substring search
over the whole console would be killed by nothing. Note also that ``input()``
writes its prompt with no trailing newline, so in a tty fixture the next printed
line is prefixed by it: tty tests assert on the artefact, console tests run
non-interactively.
"""

from __future__ import annotations

import io
import json
import sys

import pytest

from kuang.backends.claude_code import session as session_module
from kuang.backends.claude_code.spawn import CallResult
from kuang.cli import main

_CLAUDE_MD = """\
# A repo

# Resident Experts

## Analyst — Correctness

Does the change compute the right thing?

## Critic — Completeness

Find what everyone else missed; assume shared blind spots.

## Sentinel — Ruin

Hunt ruin-class hazards only.
"""


def _contract(verdict="YES", findings=()) -> CallResult:
    body = json.dumps({"verdict": verdict, "findings": list(findings)})
    return CallResult(f"I reviewed it.\n\n```json\n{body}\n```\n", 0)


def _finding(claim_class, *, title="a finding"):
    return {"title": title, "severity": "BLOCKER", "claim_class": claim_class,
            "file": "a.py", "line": 10, "evidence": "read it"}


_CLEAN = _contract()
_NO_JSON = CallResult("I read the diff and have nothing structured to say.", 0)
# Epoch 2 raises a DIFFERENT blocker, so the two epochs' counts differ: a fixture
# whose numbers coincide cannot tell a per-epoch account from a constant.
_RAISES = [_contract("NO", [_finding("correctness-1", title="the first")]),
           _contract("NO", [_finding("correctness-2", title="the second")])]


@pytest.fixture
def sourced_repo(changed_repo):
    (changed_repo / "CLAUDE.md").write_text(_CLAUDE_MD)
    return changed_repo


def _stub(monkeypatch, replies: dict, *, default=_CLEAN, formatter=_NO_JSON) -> None:
    seen: dict[str, int] = {}

    def _fake(self, prompt, mandate, tools, model):  # noqa: ANN001, ARG001
        if "careful synthesiser" in mandate:
            return _NO_JSON
        if "formatter" in mandate:
            return formatter
        for name, per_epoch in replies.items():
            if f"You are {name} " in mandate:
                i = seen.get(name, 0)
                seen[name] = i + 1
                return per_epoch[min(i, len(per_epoch) - 1)]
        return default

    monkeypatch.setattr(session_module.PanelSession, "_run_claude", _fake)


class _FakeTTY(io.StringIO):
    """A stdin that IS a terminal and answers the prompt.

    ``input()`` falls back to ``sys.stdin.readline()`` once stdin is not the
    original console object, so this drives the real prompt rather than stubbing
    the gate — the branch no test has ever reached.
    """

    def isatty(self) -> bool:
        return True


def _tty(monkeypatch, answer: str) -> None:
    monkeypatch.setattr(sys, "stdin", _FakeTTY(f"{answer}\n" * 8))


def _run(repo, *extra) -> int:
    """Two epochs by default, so one gate is reached and one epoch halts."""
    return main(["--repo", str(repo), "--base", "HEAD~1", "--max-epochs", "2",
                 "--stall-patience", "2", "--no-parallel", *extra])


def _artefact(out: str) -> dict:
    return json.loads(out.split("--- JSON ---")[-1])


def _console(out: str) -> str:
    return out.split("--- JSON ---")[0]


_HEADER = "epoch account:"


def _account_block(out: str) -> list[str]:
    """The end-of-run section's header and its indented lines, and nothing else.

    Anchored on the header rather than searched for across the console: the gate
    synthesis prints the same counts two sections above, so a whole-output
    assertion here would be satisfied by a line this section did not write (#87).
    """
    lines = _console(out).splitlines()
    for i, line in enumerate(lines):
        if line.startswith(_HEADER):
            block = [line]
            for nxt in lines[i + 1:]:
                if not nxt.startswith("  "):
                    break
                block.append(nxt)
            return block
    return []


# --- 1. the artefact accounts for the gate on every epoch --------------------

def test_the_artefact_records_the_gate_on_every_epoch(sourced_repo, monkeypatch,
                                                      capsys):
    """Unconditional and exhaustive, including the epoch that never reached it.

    Pinned by exact equality, like ``raised``: a key arriving in this row should
    be somebody's decision rather than a drift.
    """
    _stub(monkeypatch, {"Analyst": _RAISES})
    _run(sourced_repo)
    payload = _artefact(capsys.readouterr().out)

    assert payload["gate"] == [
        {"epoch": 1, "reached": True, "asked": False, "stopped": False},
        {"epoch": 2, "reached": False, "asked": None, "stopped": None}], \
        "the run does not account for its own gate"
    assert len(payload["gate"]) == payload["epochs"]


def test_a_run_whose_gate_was_never_reached_does_not_report_that_it_was(
        sourced_repo, monkeypatch, capsys):
    """MIRROR IMAGE: the rule must not fire on a run where nothing happened.

    A one-epoch run halts before any gate, so every row must say so. Without this
    pair, "a degraded gate is reported" is satisfiable by reporting every run as
    gated — which is this defect inverted.
    """
    _stub(monkeypatch, {"Analyst": _RAISES})
    _run(sourced_repo, "--max-epochs", "1")
    payload = _artefact(capsys.readouterr().out)

    assert payload["gate"] == [
        {"epoch": 1, "reached": False, "asked": None, "stopped": None}]


def test_a_non_interactive_run_says_it_did_not_ask(sourced_repo, monkeypatch,
                                                   capsys):
    """"Not a tty, continued without asking" is a FACT, not an absence.

    It is what every run in CI has ever done, and a record omitting it cannot be
    told from one written before the field existed. ``loop.run`` already records
    the harm in its own comments: the pre-#72 loop asked at a terminal and
    "silently repeated itself when piped".
    """
    _stub(monkeypatch, {"Analyst": _RAISES})
    _run(sourced_repo)
    payload = _artefact(capsys.readouterr().out)

    assert payload["gate"][0]["asked"] is False, \
        "a run that asked nobody does not say so"


def test_the_artefact_does_not_derive_reached_from_a_value_a_seam_can_rewrite(
        sourced_repo, monkeypatch, capsys):
    """The rejected derivation, pinned where the artefact is actually written.

    ``reached`` is published from the PRESENCE of the record the loop stored, never
    from ``EpochResult.halt``. The two coincide on every ordinary run — which is
    exactly why a mutation swapping one for the other is killed by nothing unless a
    test drives the case where they diverge.

    A gate holds the epoch mutably and the loop has already passed its own halt
    check by the time the gate is called, so a gate that writes ``halt`` leaves an
    epoch whose gate DID run looking, to a derived reporter, like one that never
    reached it. A rule firing on a healthy run is this issue's defect inverted.

    The engine half is pinned in ``tests/engine/test_gate_decisions.py``. This is
    the CLI half, and it exists because the mutation matrix found it missing: the
    row implementing the derived version survived every other test in this module.
    """
    from kuang.engine import HaltReason

    _stub(monkeypatch, {"Analyst": _RAISES})
    real_gate = session_module.PanelSession.interactive_gate

    def _meddling_gate(self, result, run):  # noqa: ANN001
        decision = real_gate(self, result, run)
        result.halt = HaltReason.STALL
        return decision

    monkeypatch.setattr(session_module.PanelSession, "interactive_gate",
                        _meddling_gate)
    _run(sourced_repo)
    payload = _artefact(capsys.readouterr().out)

    assert payload["gate"][0]["reached"] is True, \
        "the artefact derived 'reached' from a field the gate itself rewrote"
    assert payload["gate"][0]["asked"] is False, \
        "the fixture stopped exercising the real gate"


# --- 2. the operator's own decision ------------------------------------------

def test_a_human_who_stops_the_run_is_recorded_with_the_epoch(
        sourced_repo, monkeypatch, capsys):
    """``aborted`` reaches an artefact for the first time.

    The run must name the epoch a human intervened at — "an archived run cannot be
    asked at which epoch a human intervened or why" is #117's own consequence.
    """
    _stub(monkeypatch, {"Analyst": _RAISES})
    _tty(monkeypatch, "s")
    rc = _run(sourced_repo, "--max-epochs", "3")
    payload = _artefact(capsys.readouterr().out)

    assert payload["halt_reason"] == "aborted"
    assert payload["epochs"] == 1
    assert payload["gate"] == [
        {"epoch": 1, "reached": True, "asked": True, "stopped": True}]
    # #32 owns what an exit code MEANS; this observes the 0 it returns today so a
    # change to it is somebody's decision rather than a silent drift.
    assert rc == 0


def test_a_human_who_continues_is_distinguishable_from_one_never_asked(
        sourced_repo, monkeypatch, capsys):
    """VACUITY's partner: ``asked`` must not be able to report a constant.

    Without this, "a non-interactive run says it did not ask" is satisfied by a
    field that is always False.
    """
    _stub(monkeypatch, {"Analyst": _RAISES})
    _tty(monkeypatch, "c")
    _run(sourced_repo)
    payload = _artefact(capsys.readouterr().out)

    assert payload["gate"][0] == {"epoch": 1, "reached": True, "asked": True,
                                  "stopped": False}


def test_a_stopped_single_epoch_run_is_not_read_as_one_that_halted(
        sourced_repo, monkeypatch, capsys):
    """Both hold ``epochs == 1``, and they are different runs.

    ``epochs == 1`` must never be read as "no gate": a run stopped at the first
    gate reached it, and a run that halted at epoch 1 did not.
    """
    _stub(monkeypatch, {"Analyst": _RAISES})
    _tty(monkeypatch, "s")
    _run(sourced_repo, "--max-epochs", "3")
    stopped = _artefact(capsys.readouterr().out)

    _stub(monkeypatch, {"Analyst": _RAISES})
    _run(sourced_repo, "--max-epochs", "1")
    halted = _artefact(capsys.readouterr().out)

    assert stopped["epochs"] == halted["epochs"] == 1
    assert stopped["gate"][0]["reached"] is True
    assert halted["gate"][0]["reached"] is False


def test_a_gate_that_returns_a_truthy_non_boolean_publishes_a_boolean(
        sourced_repo, monkeypatch, capsys):
    """``loop.run`` halts on TRUTHINESS, so a seam can hand it any value.

    ``counted_vote``'s hardening one field along: what the artefact publishes must
    be the type it claims, not whatever a seam returned.
    """
    from kuang.engine import GateDecision

    _stub(monkeypatch, {"Analyst": _RAISES})
    monkeypatch.setattr(session_module.PanelSession, "interactive_gate",
                        lambda self, result, run: GateDecision(stop="yes"))
    _run(sourced_repo, "--max-epochs", "3")
    payload = _artefact(capsys.readouterr().out)

    assert payload["halt_reason"] == "aborted"
    assert payload["gate"][0]["stopped"] is True, \
        "a non-boolean reached the artefact"


# --- 3. the console account of every epoch -----------------------------------

def test_the_console_accounts_for_every_epoch_including_the_one_that_halted(
        sourced_repo, monkeypatch, capsys):
    """#117's §3: the epoch a run stops on got no console account at all.

    The counts differ between the two epochs deliberately — a section reporting a
    constant would pass a fixture whose numbers coincide.
    """
    _stub(monkeypatch, {"Analyst": _RAISES})
    _run(sourced_repo)
    block = _account_block(capsys.readouterr().out)

    assert block, "the console says nothing about what each epoch raised"
    assert block[0].startswith(f"{_HEADER} 2 epoch(s)"), \
        f"the header does not count the epochs: {block[0]!r}"
    first, second = block[1], block[2]
    assert first.startswith("  epoch 1: new findings: 1 | new material issues: 1 "
                            "| open blockers: 1 | open uglies: 0"), \
        f"epoch 1's account is wrong: {first!r}"
    assert second.startswith("  epoch 2: new findings: 1 | new material issues: 1 "
                             "| open blockers: 2 | open uglies: 0"), \
        f"the halting epoch gets no account of what it raised: {second!r}"
    assert second.endswith("the epoch halted, so the gate was never reached"), \
        f"the halting epoch's line claims a gate outcome: {second!r}"
    assert first.endswith("continued without asking a human"), \
        f"epoch 1's gate outcome is wrong: {first!r}"


def test_the_gate_synthesis_says_nothing_about_its_own_decision(
        sourced_repo, monkeypatch, capsys):
    """GUARD on a REJECTED option: passes on ``main`` and must keep passing.

    Printing the gate's account where the gate already prints was refused for
    #119's reason: ``human_gate`` runs only between epochs, so such a notice would
    appear for some epochs and never for the one a run halts on, with nothing
    telling a reader which. The end-of-run section is the answer instead. This is
    not a regression test — it pins the channel decision, and a mutation moving
    the account to the gate is killed by exactly this.
    """
    _stub(monkeypatch, {"Analyst": _RAISES})
    _run(sourced_repo)
    out = _console(capsys.readouterr().out)
    lines = out.splitlines()
    start = next(i for i, ln in enumerate(lines)
                 if ln.startswith("=== epoch 1 synthesis ==="))
    end = next(i for i, ln in enumerate(lines) if ln.startswith("=== HALT"))
    synthesis = "\n".join(lines[start:end])

    assert "gate:" not in synthesis
    assert "asking a human" not in synthesis
    assert "never reached" not in synthesis


def test_the_console_says_when_a_human_stopped_the_run(sourced_repo, monkeypatch,
                                                       capsys):
    """The stopped case on the CONSOLE, which no artefact assertion can cover.

    Found missing by the mutation matrix: rendering a human who STOPPED the run as
    one who merely continued was killed by nothing, because every console
    assertion in this module drives a non-interactive run and every tty test reads
    the artefact instead.

    Asserting on the console is safe in a tty fixture despite ``input()`` writing
    its prompt without a trailing newline: the next thing printed is the halt line,
    whose own leading newline closes the prompt's line, so no later line is
    prefixed by it.
    """
    _stub(monkeypatch, {"Analyst": _RAISES})
    _tty(monkeypatch, "s")
    _run(sourced_repo, "--max-epochs", "3")
    block = _account_block(capsys.readouterr().out)

    assert block, "the console gives no epoch account for a run a human stopped"
    assert block[1].endswith("a human STOPPED the run"), \
        f"a human who stopped the run is not rendered as one: {block[1]!r}"


# --- 3b. a value the gate cannot be read as is not a measurement (#131) ------

def _gate_returning(monkeypatch, **kwargs):
    """Wire a gate that hands back exactly what a foreign seam might."""
    from kuang.engine import GateDecision

    monkeypatch.setattr(session_module.PanelSession, "interactive_gate",
                        lambda self, result, run: GateDecision(**kwargs))


def test_a_string_answer_is_not_published_as_a_human_saying_yes(
        sourced_repo, monkeypatch, capsys):
    """REGRESSION for #131: ``asked='no'`` rendered as "a human continued the run".

    The rule is #26's one field along — a verdict is the word or it is not a
    verdict. ``asked`` is a REPORT value nothing acts on, so a value the tool
    cannot read is not a measurement and must publish as ``None``. That is
    deliberately different from ``stopped``, which is coerced, because ``stop`` is
    a CONTROL value the loop already acted on by truthiness: there the record must
    say what the loop did.
    """
    _stub(monkeypatch, {"Analyst": _RAISES})
    _gate_returning(monkeypatch, stop=False, asked="no")
    _run(sourced_repo)
    out = capsys.readouterr().out
    row = _artefact(out)["gate"][0]

    assert row["asked"] is None, \
        f"a value the gate cannot be read as reached the artefact: {row['asked']!r}"
    block = _account_block(out)
    assert "a human continued the run" not in block[1], \
        f"the console claims a human acted on a value that is not one: {block[1]!r}"
    assert block[1].endswith("the gate did not say whether a human was asked"), \
        f"the unreadable case is not rendered as unsaid: {block[1]!r}"


def test_a_numeric_answer_is_not_published_as_a_boolean(
        sourced_repo, monkeypatch, capsys):
    """``1 == True`` in Python, which is why identity is the only honest test.

    A coercion (``bool(...)``) would publish ``1`` as ``True`` and ``0`` as
    ``False``, inventing a measurement from a value the gate never meant as one —
    and ``in (True, False, None)`` cannot reject either, which is how the probe
    rule written to catch this missed it.
    """
    _stub(monkeypatch, {"Analyst": _RAISES})
    _gate_returning(monkeypatch, stop=False, asked=1)
    _run(sourced_repo)
    assert _artefact(capsys.readouterr().out)["gate"][0]["asked"] is None, \
        "a numeric answer was resolved to a human's decision"


def test_a_gate_value_that_cannot_be_serialised_does_not_destroy_the_artefact(
        sourced_repo, monkeypatch, capsys):
    """The artefact is stdout-only: `json.dumps` raising loses the whole run.

    A paid-for panel's entire record — ledger, participation, surface, tokens —
    exists nowhere else, so a seam-supplied value must not be able to take it.
    """
    _stub(monkeypatch, {"Analyst": _RAISES})
    _gate_returning(monkeypatch, stop=False, asked=object())
    _run(sourced_repo)
    payload = _artefact(capsys.readouterr().out)

    assert payload["gate"][0]["asked"] is None
    assert payload["epochs"] == 2, "the run's own record did not survive"


# --- 3c. a dry run took no measurement (#133) --------------------------------

def test_a_dry_run_does_not_claim_what_its_epoch_raised(sourced_repo, monkeypatch,
                                                        capsys):
    """REGRESSION for #133: the section said `new findings: 0` for a dry run.

    Nothing was spawned, so nothing was reviewed, and *a dry run took no
    measurement while a real epoch that raised nothing took one and it came back
    zero*. `new findings: 0` makes those two indistinguishable — and 12 of the 15
    golden invocations are dry runs, so an operator grepping an archive for epochs
    that turned nothing up would count runs that never looked.

    `nothing was spawned` is `cli/__init__.py:1007`'s wording, deliberately: that
    line exists for this same reason one section along, where "no tool call was
    refused" would have been *true* and read as its opposite.
    """
    _stub(monkeypatch, {"Analyst": _RAISES})
    _run(sourced_repo, "--dry-run")
    block = _account_block(capsys.readouterr().out)

    assert block, "the dry run prints no epoch account at all"
    assert "nothing was spawned" in block[1], \
        f"the dry run claims a measurement it never took: {block[1]!r}"
    assert "new findings:" not in block[1], \
        f"the counts survived on a run that reviewed nothing: {block[1]!r}"


def test_an_epoch_that_really_raised_nothing_still_reports_its_zeros(
        sourced_repo, monkeypatch, capsys):
    """GREEN on `main`, and it guards the fix against over-reaching.

    The pair to the test above, and the reason the discriminator is `--dry-run`
    rather than "the counts are zero". A real epoch whose panel ran and found
    nothing **did** take the measurement, and zero is its honest result — so the
    counts must still print. Without this, "suppress the counts where nothing was
    raised" is satisfiable by suppressing them whenever they are zero, which would
    destroy the very distinction #133 exists to draw.
    """
    _stub(monkeypatch, {"Analyst": [_contract("NO")]})
    _run(sourced_repo, "--max-epochs", "1")
    block = _account_block(capsys.readouterr().out)

    assert block[1].startswith("  epoch 1: new findings: 0 | new material issues: 0"), \
        f"a real epoch that found nothing stopped saying so: {block[1]!r}"
    assert "nothing was spawned" not in block[1], \
        "a spawned epoch is described as one that never ran"


def test_a_dry_run_keeps_the_half_of_the_line_that_is_true(sourced_repo,
                                                           monkeypatch, capsys):
    """GREEN on `main`: the gate clause is correct on a dry run and must survive.

    A dry run halts at epoch 1, so it never reaches the gate, and *"the epoch
    halted, so the gate was never reached"* is simply true. The fix removes the
    half of the line that has no subject; a fix that threw away the whole line
    would take a true statement with it and leave the channel silent again — the
    defect #117 closed, reopened from the other side.
    """
    _stub(monkeypatch, {"Analyst": _RAISES})
    _run(sourced_repo, "--dry-run")
    block = _account_block(capsys.readouterr().out)

    assert block[1].endswith("the epoch halted, so the gate was never reached"), \
        f"the true half of the line did not survive: {block[1]!r}"


# --- 4. the counts the halt was actually taken on ----------------------------

def test_the_artefact_records_the_two_counts_the_halt_was_taken_on(
        sourced_repo, monkeypatch, capsys):
    """The circuit-breaker reads ``open_uglies``; CONVERGED and STALL read
    ``open_blockers`` — both per epoch, and reported by nothing but the gate.

    The artefact's top-level counts are a different measurement: finding-level,
    over the ledger as it ENDS. Verified by execution while planning #117 — a
    two-epoch run publishes 2 while epoch 1 decided on 1 — so the halting epoch's
    own inputs were unrecoverable from an archived run. #33's thesis on two fields
    #33 never named.
    """
    _stub(monkeypatch, {"Analyst": _RAISES})
    _run(sourced_repo)
    payload = _artefact(capsys.readouterr().out)
    first, second = payload["raised"]["epochs"]

    assert (first["open_blockers"], first["open_uglies"]) == (1, 0)
    assert (second["open_blockers"], second["open_uglies"]) == (2, 0), \
        "the halting epoch's own halt inputs are still unrecorded"
    assert payload["open_blockers"] == 2, \
        "the top-level count is the end-of-run one and must be left alone"
