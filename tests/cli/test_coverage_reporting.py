# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""A good verdict must say what agreement it rests on, and what lenses looked (#82).

The fifth instance of the doctrine ``test_unresolved_severity_reporting`` (#24),
``test_verdict_reporting`` (#26), ``test_participation_reporting`` (#30),
``test_permission_reporting`` (#67) and ``test_turn_reporting`` (#70) established,
and the sibling of #72: there, a good verdict for a review that never happened;
here, for one that happened and was worth nothing. A panel of **one** whose sole
reviewer called no tool meets every conjunct of ``CONVERGED`` — no open ugly, no
open blocker, quorum ``1 >= 1`` — and the run prints the good verdict with nothing
beside it saying the agreement behind it was one voice that did not look.

**Two facts, reported in two places, and neither derived from the other.** That is
the issue's central demand, and the reason this file has two halves:

* **the voting quorum** — a threshold over the votes cast by whoever ran. Qualified
  beside the verdict, and *only* beside a verdict: under a ``no_review`` halt the
  halt line already says nothing was reviewed, and "0 of 7" beneath it would state
  one fact twice;
* **lens coverage** — which required lens was covered by whom, and on what
  evidence. A fact about the *panel*, so it is reported always-on, in a dry run
  too, for the reason ``review surface:`` is (#74): pre-flight confirmation is the
  dry run's only job, and "your ruin lens is a substring match on one persona" is
  the most useful thing a dry run can say about a panel.

**Qualify, never gate.** Every run below that converges still converges. A halt
reason is for a state the loop cannot usefully continue from
(``HaltReason.NO_REVIEW``); a degenerate-but-real review *can* continue, and
personas are a parameter — a deliberate one-reviewer run is the operator's call.
``test_engine`` holds the engine to that; this file holds the report to it.

These drive ``kuang.cli.main`` end to end over a throwaway repo with the subprocess
boundary stubbed. The turn counts asserted on are the live captures in
``backends/claude_code/test_spawn.py``; the rule they feed is measured (#70).
"""

from __future__ import annotations

import json

import pytest

from kuang.backends.claude_code import session as session_module
from kuang.backends.claude_code.spawn import CallResult
from kuang.cli import main

# One expert whose grounding happens to mention completeness AND ruin, so
# ``_ensure_specialists`` injects neither default and the panel is ONE. This is
# #82's own route, and it needs nothing unusual of an operator.
_SOLE = """\
# A repo

# Resident Experts

## Sole Reviewer — Everything

Find what everyone else missed, and hunt cascading ruin-class hazards.
"""

# Three experts, as ``test_turn_reporting`` uses: "Completeness" and "Ruin" appear
# in the bodies, so no default specialist is injected and both required lenses are
# covered by KEYWORD MATCH rather than by construction.
_MATCHED = """\
# A repo

# Resident Experts

## Analyst — Correctness

Does the change compute the right thing?

## Critic — Completeness

Find what everyone else missed; assume shared blind spots.

## Sentinel — Ruin

Hunt ruin-class hazards only.
"""

# Exactly TWO experts, one matching each keyword set, so neither default is
# injected and the panel is two. This is the boundary the rule turns on: two
# grounded voices are the smallest real agreement, and one is an opinion. Without
# a panel of exactly this size the ``< 2`` rule is unpinned at its upper end — a
# mutation to ``< 3`` survived the six rows below until this was added, which is
# how the gap was found rather than argued.
_PAIR = """\
# A repo

# Resident Experts

## Critic — Completeness

Find what everyone else missed; assume shared blind spots.

## Sentinel — Ruin

Hunt ruin-class hazards only.
"""

# One expert matching neither keyword set, so BOTH defaults are injected: the
# panel is three, and each required lens is covered by construction rather than by
# a substring. The exact half of the claim, against ``_MATCHED``'s heuristic half.
_INJECTED = """\
# A repo

# Resident Experts

## Analyst — Correctness

Does the change compute the right thing?
"""

# The measured healthy range is 6-9 turns; the one-turn call is the defect, and 0
# is the runtime declining to say (#70).
_GROUNDED, _NO_TOOL, _UNREPORTED = 7, 1, 0

# The mandate substring that reaches an INJECTED specialist. Its grounding is a
# constant in ``personas.py`` and does not open "You are <name>", so a default is
# addressed the way the replay harness addresses it.
_SURVIVABILITY = "ruin-class hazards"


# Which mandate substring addresses which persona, per panel. A sourced persona's
# grounding opens "You are <name> — <role>."; an injected default's does not, so
# the two are addressed differently and the table says which panel is which.
_NEEDLES = {
    _SOLE: ("You are Sole Reviewer",),
    _PAIR: ("You are Critic", "You are Sentinel"),
    _MATCHED: ("You are Analyst", "You are Critic", "You are Sentinel"),
}


def _contract(verdict="YES", findings=(), *, turns=_GROUNDED) -> CallResult:
    body = json.dumps({"verdict": verdict, "findings": list(findings)})
    return CallResult(f"I reviewed it.\n\n```json\n{body}\n```", 0, None, (), turns)


def _repo(changed_repo, claude_md: str | None):
    if claude_md is not None:
        (changed_repo / "CLAUDE.md").write_text(claude_md)
    return changed_repo


def _stub(monkeypatch, replies: dict, *, default=None) -> None:
    """``replies`` maps a MANDATE SUBSTRING to the reply for that persona."""
    default = default if default is not None else _contract()

    def _fake(self, prompt, mandate, tools, model):  # noqa: ANN001, ARG001
        if "formatter" in mandate:
            return CallResult("no json", 0)
        for needle, reply in replies.items():
            if needle in mandate:
                return reply
        return default

    monkeypatch.setattr(session_module.PanelSession, "_run_claude", _fake)


def _run(repo, *extra) -> int:
    return main(["--repo", str(repo), "--base", "HEAD~1", "--max-epochs", "1",
                 "--no-parallel", *extra])


def _artefact(out: str) -> dict:
    return json.loads(out.split("--- JSON ---")[-1])


def _verdict_line(out: str) -> str | None:
    return next((ln for ln in out.splitlines()
                 if ln.startswith("verdict rests on:")), None)


def _degenerate_line(out: str) -> str | None:
    return next((ln for ln in out.splitlines()
                 if "DEGENERATE VERDICT" in ln), None)


def _coverage_block(out: str) -> list[str]:
    """The ``panel coverage:`` section: its header and its own indented lines."""
    lines = out.splitlines()
    start = next((i for i, ln in enumerate(lines)
                  if ln.startswith("panel coverage")), None)
    if start is None:
        return []
    block = [lines[start]]
    for ln in lines[start + 1:]:
        if ln.strip() and not ln.startswith(" "):
            break
        block.append(ln)
    return [ln for ln in block if ln.strip()]


# --- the defect, and the one principle behind it -----------------------------

def test_a_panel_of_one_that_opened_nothing_says_so_beside_the_verdict(
        changed_repo, capsys, monkeypatch):
    """THE defect (#82). Every other channel reports this run as healthy.

    Nothing errored, nothing was refused, the surface was whole, the persona
    contributed and the panel was unanimous. The run still halts ``converged`` and
    still exits 0 — that is deliberate, and #32 owns the exit code — but the
    verdict now states the agreement it rests on, and marks it degenerate.
    """
    repo = _repo(changed_repo, _SOLE)
    _stub(monkeypatch, {}, default=_contract(turns=_NO_TOOL))
    code = _run(repo)
    out = capsys.readouterr().out
    payload = _artefact(out)

    assert payload["halt_reason"] == "converged", "qualify, never gate"
    assert code == 0, "the exit code is #32's, not this issue's"
    line = _verdict_line(out)
    assert line is not None, "converged was printed with no agreement beside it"
    assert "1 of 1 persona(s) voting YES" in line, line
    assert "quorum 1" in line, line
    assert "0 opened the source" in line, line
    assert _degenerate_line(out) is not None, "a one-voice agreement is unmarked"


@pytest.mark.parametrize("claude_md,turns,degenerate", [
    (_SOLE, (_NO_TOOL,), True),
    (_SOLE, (_GROUNDED,), True),
    (_PAIR, (_GROUNDED, _NO_TOOL), True),
    (_PAIR, (_GROUNDED, _GROUNDED), False),
    (_PAIR, (_GROUNDED, _UNREPORTED), False),
    (_MATCHED, (_NO_TOOL, _NO_TOOL, _NO_TOOL), True),
    (_MATCHED, (_GROUNDED, _GROUNDED, _UNREPORTED), False),
    (_MATCHED, (_GROUNDED, _GROUNDED, _GROUNDED), False),
    (_MATCHED, (_UNREPORTED, _UNREPORTED, _UNREPORTED), False),
], ids=["one_voice_that_opened_nothing",
        "one_voice_that_looked",
        "two_voices_only_one_of_which_looked",
        "two_looked_the_smallest_real_agreement",
        "one_looked_and_one_turn_count_unreported",
        "three_voices_none_of_which_looked",
        "two_looked_and_one_turn_count_unreported",
        "three_looked",
        "no_turn_count_was_reported_at_all"])
def test_the_degeneracy_rule_is_one_principle_over_what_the_run_knows(
        changed_repo, capsys, monkeypatch, claude_md, turns, degenerate):
    """One rule, two-ended: at most one voice KNOWN to have looked and agreed.

    The counted votes partition by what the run knows about ``turns`` (#70's three
    states, never collapsed): ``> 1`` looked, ``== 1`` did not, ``== 0`` was not
    reported. Degenerate iff ``opened + unreported < 2`` — every vote the run
    cannot rule out as ungrounded gets the benefit of the doubt, because a backend
    that reported nothing must not have a degradation attributed to it.

    ``< 2`` is not a threshold anybody baselined; it is the definition of the word
    *agreement*, and one voice is an opinion. The last row is why the rule takes
    ``unreported`` on its side: three silent turn counts is a run that measured
    nothing, and it must read as neither health nor harm.

    **Both ends are pinned, and by the smallest panel that can pin them.** The two
    ``_PAIR`` rows straddle the boundary exactly — one grounded voice is degenerate,
    two are not — so the rule cannot be loosened to ``< 3`` or tightened to ``< 1``
    without a row going red. The three-persona rows alone left the upper end free.
    """
    repo = _repo(changed_repo, claude_md)
    names = _NEEDLES[claude_md]
    _stub(monkeypatch, {n: _contract(turns=t) for n, t in zip(names, turns)})
    _run(repo)
    out = capsys.readouterr().out

    assert _artefact(out)["halt_reason"] == "converged", "the premise of the test"
    assert _artefact(out)["agreement"]["degenerate"] is degenerate
    assert (_degenerate_line(out) is not None) is degenerate, _verdict_line(out)


def test_a_full_grounded_unanimous_panel_reads_clean(changed_repo, capsys,
                                                     monkeypatch):
    """The MIRROR IMAGE: a rule that fires on a healthy run is the same defect.

    The default seven, every one of them grounded and voting YES. The verdict is
    stated — that is the point of stating it always — and it carries no mark, the
    coverage section carries no warning, and no lens is heuristic.
    """
    repo = _repo(changed_repo, None)
    _stub(monkeypatch, {})
    _run(repo)
    out = capsys.readouterr().out

    assert _artefact(out)["halt_reason"] == "converged"
    assert "7 of 7 persona(s) voting YES" in _verdict_line(out)
    assert "7 opened the source" in _verdict_line(out)
    assert _degenerate_line(out) is None, "a healthy panel was marked degenerate"
    assert "DEGENERATE" not in out and "WARNING" not in "\n".join(_coverage_block(out))
    assert "KEYWORD MATCH" not in out, "the default specialists are injected, exact"


def test_not_every_run_reports_a_degenerate_panel(changed_repo, capsys,
                                                  monkeypatch):
    """VACUITY PROBE, and the sibling of the test above.

    "A degenerate verdict is marked" passes trivially if every run is marked. Two
    runs of the same panel differing only in what the reviewers did must differ in
    exactly this mark and nowhere else in the verdict line's shape.
    """
    repo = _repo(changed_repo, _MATCHED)
    _stub(monkeypatch, {})
    _run(repo)
    clean = capsys.readouterr().out

    _stub(monkeypatch, {}, default=_contract(turns=_NO_TOOL))
    _run(repo)
    marked = capsys.readouterr().out

    assert _degenerate_line(clean) is None and _degenerate_line(marked) is not None
    assert _verdict_line(clean) != _verdict_line(marked)
    assert "3 of 3 persona(s) voting YES" in _verdict_line(clean)
    assert "3 of 3 persona(s) voting YES" in _verdict_line(marked), (
        "the agreement is the same size; only its grounding differs")


def test_the_three_turn_states_are_named_apart_on_the_verdict_line(
        changed_repo, capsys, monkeypatch):
    """A count the runtime gave, one it gave as 1, and one it did not give.

    Collapsing any two of them is #70's defect one field along, so each clause is
    printed separately and only when it has something to report.
    """
    repo = _repo(changed_repo, _MATCHED)
    _stub(monkeypatch, {"You are Critic": _contract(turns=_NO_TOOL),
                        "You are Sentinel": _contract(turns=_UNREPORTED)})
    _run(repo)
    line = _verdict_line(capsys.readouterr().out)

    assert "1 opened the source" in line, line
    assert "1 called no tool" in line, line
    assert "1 turn count(s) unreported" in line, line


# --- the qualifier belongs to a verdict, and only to a verdict ---------------

def test_a_dry_run_prints_no_verdict_line(changed_repo, capsys, monkeypatch):
    """Since #72 a dry run halts ``no_review``: there is no verdict to qualify.

    The halt line already says nothing was reviewed, and "0 of 7" beneath it would
    state one fact twice. Requested on the record at #72's close-out, and pinned
    here so it cannot be undone by accident.
    """
    repo = _repo(changed_repo, _MATCHED)
    _stub(monkeypatch, {})
    _run(repo, "--dry-run")
    out = capsys.readouterr().out

    assert _artefact(out)["halt_reason"] == "no_review"
    assert _verdict_line(out) is None, "a quorum line under a no_review halt"
    assert _coverage_block(out), "coverage is a fact about the PANEL, always-on"


def test_a_run_that_did_not_converge_prints_no_verdict_line(changed_repo, capsys,
                                                            monkeypatch):
    """A blocker keeps the run off the good verdict, so there is nothing to rest on.

    The line qualifies ``converged``; a run that halted on the epoch ceiling with
    an open blocker made no claim for it to qualify.
    """
    repo = _repo(changed_repo, _MATCHED)
    _stub(monkeypatch, {}, default=_contract(findings=[
        {"title": "off-by-one", "severity": "BLOCKER", "claim_class": "logic",
         "file": "a.py", "line": 1, "evidence": "read it"}]))
    _run(repo)
    out = capsys.readouterr().out

    assert _artefact(out)["halt_reason"] == "epoch"
    assert _verdict_line(out) is None
    assert _artefact(out)["agreement"]["yes_votes"] == 3, (
        "the agreement is still recorded — an absent key makes no claim")
    assert _artefact(out)["agreement"]["degenerate"] is None, (
        "three voices agreed, and the run did not converge on it: the counts are "
        "facts about the votes, the qualifier is about a verdict there isn't one")


# --- coverage: which lens, covered by whom, on what evidence ------------------

def test_a_keyword_matched_lens_is_not_mistaken_for_a_declared_one(
        changed_repo, capsys, monkeypatch):
    """The heuristic half, marked as such (#2).

    ``_ensure_specialists`` suppresses a default when a sourced persona's name or
    grounding contains a capability keyword. That is a claim on the strength of a
    substring: this persona may or may not be a ruin lens, and nothing checked.
    Until #2 makes a lens declarable the run must say which kind of claim it is
    making, or the heuristic half reads as the exact half.
    """
    repo = _repo(changed_repo, _MATCHED)
    _stub(monkeypatch, {})
    _run(repo)
    block = "\n".join(_coverage_block(capsys.readouterr().out))

    assert "completeness — KEYWORD MATCH (Critic)" in block, block
    assert "ruin — KEYWORD MATCH (Sentinel)" in block, block
    assert block.count("not declared (#2)") == 2, "the heuristic is unattributed"


def test_an_injected_lens_is_exact_and_says_so(changed_repo, capsys, monkeypatch):
    """The exact half: nothing matched, so the default was added by construction."""
    repo = _repo(changed_repo, _INJECTED)
    _stub(monkeypatch, {})
    _run(repo)
    block = "\n".join(_coverage_block(capsys.readouterr().out))

    lenses = [ln for ln in block.splitlines() if not ln.startswith("panel")]
    assert "completeness — injected (completeness-critic)" in block, block
    assert "ruin — injected (survivability)" in block, block
    assert "KEYWORD MATCH" not in block
    assert not any("#2" in ln for ln in lenses), (
        "the exact half must not carry the heuristic half's caveat; the header "
        "names #2 because provenance VARIES, and these two lines are the kind "
        "that does not")


def test_one_persona_covering_both_lenses_is_named(changed_repo, capsys,
                                                   monkeypatch):
    """#82's route, stated as the coverage fact it is.

    A single sourced expert suppressed both injections, so the panel has no lens
    independent of any other. Exactly knowable, and not derivable from the quorum:
    a three-persona panel meeting quorum can have the same defect.

    Stated as counts rather than as "both", so a third required lens is a row in
    ``REQUIRED_LENSES`` and not a rewording here.
    """
    repo = _repo(changed_repo, _SOLE)
    _stub(monkeypatch, {})
    _run(repo)
    block = "\n".join(_coverage_block(capsys.readouterr().out))

    assert "WARNING: 2 required lens(es) rest on 1 persona(s)" in block, block
    assert "no lens independent of any other" in block, "the mark is unexplained"
    assert block.count("Sole Reviewer") == 2, "and it is named against each lens"


@pytest.mark.parametrize("claude_md,needle", [
    (_MATCHED, "You are Sentinel"),
    (_INJECTED, _SURVIVABILITY),
], ids=["a_keyword_matched_lens", "an_injected_lens"])
def test_coverage_reports_whether_the_covering_lens_actually_looked(
        changed_repo, capsys, monkeypatch, claude_md, needle):
    """Rostered is not reviewed, and reviewed is not grounded.

    A lens that is present but never opened the source has covered nothing, which
    is the dimension #70 added and the sourcing-time check cannot see. Reported
    through the same renderer the participation line uses, so the two cannot
    disagree about one persona.

    Both rows, because how a lens got onto the roster says nothing about what it
    then did: an injected default is exact about *which* lens it is and makes no
    claim at all about whether it looked.
    """
    repo = _repo(changed_repo, claude_md)
    _stub(monkeypatch, {needle: _contract(turns=_NO_TOOL)})
    _run(repo)
    block = _coverage_block(capsys.readouterr().out)

    ruin = next(ln for ln in block if ln.strip().startswith("ruin —"))
    done = next(ln for ln in block if ln.strip().startswith("completeness —"))
    assert "CALLED NO TOOL" in ruin, ruin
    assert "CALLED NO TOOL" not in done, done


def test_coverage_is_always_on_and_names_the_required_lenses(changed_repo, capsys,
                                                             monkeypatch):
    """The claim no absent section can make (#30's asymmetry, #74's application).

    The header names the closed set of required lenses and states the boundary the
    measurement forced: presence is guaranteed at sourcing — measured, a lens can
    never be absent, because ``_ensure_specialists`` appends the default on every
    branch — so the run reports HOW each was covered, not WHETHER.
    """
    repo = _repo(changed_repo, _MATCHED)
    _stub(monkeypatch, {})
    _run(repo)
    header = _coverage_block(capsys.readouterr().out)[0]

    assert "completeness" in header and "ruin" in header, header
    assert "presence is guaranteed at sourcing" in header, header


# --- the artefact ------------------------------------------------------------

def test_the_artefact_carries_both_facts_separately(changed_repo, capsys,
                                                    monkeypatch):
    """Named apart wherever either appears — the issue's fourth criterion.

    No single number may stand for both halves: a run can have full coverage and
    fail quorum, or meet quorum with no coverage at all, and the second is this
    issue. Counts only — per-persona verdicts and ``claim_class`` stay #33's.
    """
    repo = _repo(changed_repo, _SOLE)
    _stub(monkeypatch, {}, default=_contract(turns=_NO_TOOL))
    _run(repo)
    payload = _artefact(capsys.readouterr().out)

    assert payload["agreement"] == {
        "quorum": 1, "yes_votes": 1, "roster": 1, "reviewed": 1,
        "opened_source": 0, "called_no_tool": 1, "turns_unreported": 0,
        "degenerate": True}
    assert payload["coverage"] == [
        {"lens": "completeness", "persona": "Sole Reviewer", "how": "keyword",
         "reviewed": True, "opened_source": False},
        {"lens": "ruin", "persona": "Sole Reviewer", "how": "keyword",
         "reviewed": True, "opened_source": False}]
    assert "verdict" not in json.dumps(payload["coverage"]), "#33 is not this issue"


def test_an_unreported_turn_count_reaches_the_artefact_as_neither(
        changed_repo, capsys, monkeypatch):
    """``opened_source`` is three-state, never collapsed to two (#70).

    ``null`` is the runtime declining to say, and it must not be written as
    ``false`` — that would report a degradation nobody measured.
    """
    repo = _repo(changed_repo, _MATCHED)
    _stub(monkeypatch, {"You are Sentinel": _contract(turns=_UNREPORTED)})
    _run(repo)
    payload = _artefact(capsys.readouterr().out)

    ruin = next(c for c in payload["coverage"] if c["lens"] == "ruin")
    assert ruin["opened_source"] is None, ruin
    assert payload["agreement"]["turns_unreported"] == 1


def test_a_dry_run_records_coverage_and_an_empty_agreement(changed_repo, capsys,
                                                           monkeypatch):
    """Nothing reviewed, so nothing agreed — and neither key may be absent.

    An absent key makes no claim, and a reader coming to an artefact cold cannot
    otherwise tell a panel that agreed from one that never ran.
    """
    repo = _repo(changed_repo, _INJECTED)
    _stub(monkeypatch, {})
    _run(repo, "--dry-run")
    payload = _artefact(capsys.readouterr().out)

    assert payload["agreement"]["yes_votes"] == 0
    assert payload["agreement"]["reviewed"] == 0
    assert payload["agreement"]["degenerate"] is None, (
        "a dry run reached no verdict, so there is none to call degenerate — and "
        "11 of the 13 golden invocations are dry runs, so an arithmetic `true` "
        "here would be what an archive mostly contained")
    assert [c["how"] for c in payload["coverage"]] == ["injected", "injected"]
    assert all(c["reviewed"] is False for c in payload["coverage"])
    assert all(c["opened_source"] is None for c in payload["coverage"])
