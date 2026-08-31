# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""What the next epoch's panel is told about a call that degraded (#71).

``session.on_epoch`` rewrites ``prior_summary`` from the whole ledger after every
epoch and ``build_prompt`` injects it into every persona's prompt from epoch 2, so
a finding a degraded call produced is handed to personas that did not degrade — and
it compounds, because epoch 3 is shaped by epochs 1 and 2 together and
``--prior-findings`` (#13) carries the result into a later run.

**The rule, and it is one principle: prior context marks the lines this run knows
were not grounded in the source, and the prompt says what to do with each.**
Nothing is excluded — #71's own acceptance says *unmarked*, not *excluded*, which
is this epoch's doctrine (#24, #26, #67, #70, #69, #73) one field along.

Two facts establish "not grounded" exactly, and both are set at the source:

* ``Finding.about_run`` (#73) — the finding is the instrument's own diagnosis, not
  a claim about the source at all;
* ``status.reviewed and called_no_tool(turns)`` (#70) — the reviewer answered from
  the prompt alone and never opened a file.

Neither is a predicate over model-produced text, which is the constraint #73
measured: excluding ``claim_class == "meta"`` turns a run that halts
``escalate_ugly`` on a persona-declared ``meta`` UGLY into one that halts
``converged`` with none. ``claim_class`` and a finding's *title* are both written
by the persona, so neither may decide what a mark says.

What this deliberately does **not** cover, and the sibling that owns each: a
reduced surface (#69 — the reviewer looked, at less, which is a different failure
from not looking), a refused tool (#67 — a reviewer denied ``Bash`` may still have
read every file, so "not grounded" is not knowable from a refusal), and a finding
*fabricated* by the reformat retry (#63 — it carries whatever the model wrote and
no flag our code sets can name it).
"""

from __future__ import annotations

import pytest

from kuang.backends.claude_code.memory import epoch_summary, ungrounded_keys
from kuang.backends.claude_code.contract import build_prompt
from kuang.engine import (EpochResult, Finding, PanelConfig, PersonaReport,
                          PersonaStatus, ReviewRun, ReviewSpec, Severity, run)
from kuang.engine.fakes import FakeEnsemble
from kuang.engine.halting import HaltingSet

_HEALTHY_TURNS = 7        # measured 6-9 on agent CLI 2.1.246 (probes/README)
_NO_TOOL_TURNS = 1        # the whole of #70's rule


def _report(persona: str, findings=(), *, turns: int = _HEALTHY_TURNS,
            status: PersonaStatus = PersonaStatus.CONTRIBUTED) -> PersonaReport:
    return PersonaReport(persona=persona, findings=list(findings), verdict="NO",
                         status=status, turns=turns)


def _review_run(*epochs: list[PersonaReport]) -> ReviewRun:
    """Assemble a run the way ``engine.loop`` does: the ledger is FIRST SIGHTING.

    A deliberate mirror of ``loop.py``'s ledger insertion, kept small and stated
    so the unit tests below can vary one fact at a time.
    ``test_the_mark_follows_the_call_and_not_the_persona`` drives the REAL engine
    instead, so the property does not rest on this helper being faithful.
    """
    review = ReviewRun()
    for index, reports in enumerate(epochs, 1):
        for report in reports:
            for f in report.findings:
                if f.counts_open and f.key not in review.ledger:
                    review.ledger[f.key] = f
        review.epochs.append(EpochResult(index=index, reports=list(reports),
                                         new_findings=[], open_blockers=0,
                                         open_uglies=0))
    return review


def _summary(review: ReviewRun) -> str:
    """Exactly what ``session.on_epoch`` renders into the next epoch's prompt."""
    return epoch_summary(review.ledger.values(), ungrounded_keys(review))


# --- the two facts that mark a line ------------------------------------------

def test_an_instrument_diagnosis_is_marked_as_being_about_the_run():
    """The next panel must not be handed ``agent error:`` as review material."""
    review = _review_run([_report("P1", [
        Finding("P1", "agent error: error_max_turns", Severity.NOTE, "meta",
                about_run=True)])])

    assert _summary(review) == (
        "- [NOTE/open] (P1) agent error: error_max_turns @ - [about the run]")


def test_a_finding_from_a_reviewer_that_called_no_tool_is_marked_ungrounded():
    """The purest case of the defect: a claim answered from the prompt alone."""
    review = _review_run([_report("P1", [
        Finding("P1", "off-by-one in the bound", Severity.BLOCKER, "logic",
                "a.py", 3)], turns=_NO_TOOL_TURNS)])

    assert _summary(review) == (
        "- [BLOCKER/open] (P1) off-by-one in the bound @ a.py:3 [ungrounded]")


def test_a_healthy_finding_carries_no_mark():
    """The mirror image of the defect: a rule that fires on a healthy run."""
    review = _review_run([_report("P1", [
        Finding("P1", "off-by-one in the bound", Severity.BLOCKER, "logic",
                "a.py", 3)])])

    assert _summary(review) == (
        "- [BLOCKER/open] (P1) off-by-one in the bound @ a.py:3")


def test_both_marks_can_land_on_one_line():
    """A parser diagnosis from a contributing persona that opened nothing."""
    review = _review_run([_report("P1", [
        Finding("P1", "output contract echoed: 1 template block ignored",
                Severity.NOTE, "meta", about_run=True)], turns=_NO_TOOL_TURNS)])

    assert _summary(review).endswith(" @ - [about the run] [ungrounded]")


# --- nothing is excluded ------------------------------------------------------

def test_a_marked_finding_still_reaches_the_next_epoch_verbatim():
    """REJECTED design: exclude rather than mark.

    #71's acceptance says a degraded call's finding may not enter prior context
    *unmarked* — not that it may not enter. Dropping it would make a failed
    persona's diagnosis unrecoverable from the context the next panel reads, which
    is the discard #24, #26, #67, #70, #69 and #73 all refused.
    """
    review = _review_run([
        _report("P1", [Finding("P1", "agent error: boom", Severity.NOTE, "meta",
                               about_run=True)]),
        _report("P2", [Finding("P2", "guessed at it", Severity.BLOCKER, "logic",
                               "b.py", 9)], turns=_NO_TOOL_TURNS),
    ])

    summary = _summary(review)

    assert "agent error: boom" in summary
    assert "guessed at it" in summary
    assert len(summary.splitlines()) == 2


# --- the mark is never inferred from model-produced text ----------------------

def test_a_persona_writing_claim_class_meta_is_not_marked():
    """REJECTED design, and it is a measurement rather than a preference (#73).

    ``claim_class`` is a model-controlled string and ``meta`` is an ordinary word
    in Python review — a finding about a metaclass or about metadata is naturally
    labelled that way. Matching on it here would let a persona's own wording
    decide that its ruin-class claim is not a claim about the change.
    """
    review = _review_run([_report("P1", [
        Finding("P1", "the metaclass registry leaks", Severity.UGLY, "meta",
                "m.py", 12)])])

    assert _summary(review) == (
        "- [UGLY/open] (P1) the metaclass registry leaks @ m.py:12")


def test_a_persona_writing_a_diagnosis_shaped_title_is_not_marked():
    """The same refusal one field along: #30 already rejected matching a title."""
    review = _review_run([_report("P1", [
        Finding("P1", "agent error: the retry handler swallows it",
                Severity.BLOCKER, "logic", "spawn.py", 40)])])

    assert "[about the run]" not in _summary(review)


# --- which call, exactly ------------------------------------------------------

def test_a_call_that_produced_no_review_is_not_also_called_ungrounded():
    """One failure, one heading — ``_turn_note``'s rule, reused not reinvented.

    A call that errored has no review to be ungrounded, so its own diagnosis is
    marked ``[about the run]`` and nothing else, even at one turn.
    """
    review = _review_run([_report("P1", [
        Finding("P1", "agent error: boom", Severity.NOTE, "meta", about_run=True)],
        turns=_NO_TOOL_TURNS, status=PersonaStatus.AGENT_ERROR)])

    assert _summary(review).endswith(" [about the run]")


def test_a_turn_count_the_runtime_never_reported_marks_nothing():
    """0 means the envelope did not say. A run must not report what it did not measure."""
    review = _review_run([_report("P1", [
        Finding("P1", "off-by-one", Severity.BLOCKER, "logic", "a.py", 3)],
        turns=0)])

    assert "[ungrounded]" not in _summary(review)


def test_one_degraded_reviewer_does_not_mark_the_rest_of_the_panel():
    """The mark is per call, not per epoch — that is what makes it informative."""
    review = _review_run([
        _report("P1", [Finding("P1", "guessed", Severity.BLOCKER, "logic",
                               "a.py", 3)], turns=_NO_TOOL_TURNS),
        _report("P2", [Finding("P2", "read it", Severity.BLOCKER, "logic",
                               "b.py", 5)]),
    ])

    marked = [ln for ln in _summary(review).splitlines() if "[ungrounded]" in ln]

    assert len(marked) == 1
    assert "guessed" in marked[0]


def test_the_mark_follows_the_call_and_not_the_persona(monkeypatch):
    """A persona healthy in epoch 1 and degraded in epoch 2, through the REAL loop.

    ``f.persona`` alone is the wrong key: the ledger stores first sighting and a
    ``Finding`` carries no epoch, so the join has to recover the epoch from
    ``EpochResult.reports``. Driven through ``kuang.engine.run`` so the ledger and
    the epoch records are the engine's own rather than a helper's.
    """
    script = {
        1: {"P1": _report("P1", [Finding("P1", "read it", Severity.BLOCKER,
                                         "logic", "a.py", 3)])},
        2: {"P1": _report("P1", [Finding("P1", "guessed at it", Severity.BLOCKER,
                                         "logic", "b.py", 5)],
                          turns=_NO_TOOL_TURNS)},
    }
    review = run(ReviewSpec(why="w", what="x"),
                 HaltingSet(max_epochs=2, require_scope_complete=False),
                 PanelConfig(personas=[("P1", "m")]),
                 spawn=FakeEnsemble(script), gather=lambda epoch: "surface",
                 parallel=False)

    by_title = {ln.split(") ")[1].split(" @ ")[0]: ln
                for ln in _summary(review).splitlines()}

    assert "[ungrounded]" not in by_title["read it"]
    assert "[ungrounded]" in by_title["guessed at it"]


def test_a_finding_re_raised_by_a_degraded_call_still_reads_as_first_seen():
    """The ledger keeps the FIRST sighting, so the mark must name that call.

    Two personas raise the same signature in two epochs, the second having called
    no tool. The entry the next panel reads is epoch 1's, raised by a reviewer that
    did open the source — marking it would say the ledger holds something it does
    not.
    """
    first_seen = Finding("P1", "off-by-one", Severity.BLOCKER, "logic", "a.py", 3)
    re_raised = Finding("P2", "off by one, again", Severity.BLOCKER, "logic",
                        "a.py", 3)
    assert first_seen.key == re_raised.key, "the fixture must exercise one key"
    review = _review_run([_report("P1", [first_seen])],
                         [_report("P2", [re_raised], turns=_NO_TOOL_TURNS)])

    assert "[ungrounded]" not in _summary(review)


def test_a_refuted_finding_from_a_degraded_call_does_not_mark_its_survivor():
    """A refuted claim never entered the ledger, so it cannot have produced its entry.

    The join applies the engine's own ``counts_open`` filter for this: without it,
    a withdrawn claim from a degraded call in epoch 1 would mark the healthy
    finding that shares its signature in epoch 2.
    """
    withdrawn = Finding("P1", "guessed", Severity.BLOCKER, "logic", "b.py", 9,
                        verified=False)
    survivor = Finding("P2", "read it", Severity.BLOCKER, "logic", "b.py", 9)
    assert withdrawn.key == survivor.key, "the fixture must exercise one key"
    review = _review_run([_report("P1", [withdrawn], turns=_NO_TOOL_TURNS)],
                         [_report("P2", [survivor])])

    assert _summary(review) == "- [BLOCKER/open] (P2) read it @ b.py:9"


# --- the property, stated both ways ------------------------------------------

def test_a_contaminated_run_cannot_be_read_as_a_clean_one():
    review = _review_run([
        _report("P1", [Finding("P1", "agent error: boom", Severity.NOTE, "meta",
                               about_run=True)],
                status=PersonaStatus.AGENT_ERROR),
        _report("P2", [Finding("P2", "guessed at it", Severity.BLOCKER, "logic",
                               "b.py", 9)], turns=_NO_TOOL_TURNS),
        _report("P3", [Finding("P3", "read it", Severity.BLOCKER, "logic",
                               "c.py", 1)]),
    ])

    assert sum("[" in ln.split(" @ ")[1] for ln in _summary(review).splitlines()) == 2


def test_a_clean_run_is_not_read_as_a_contaminated_one():
    """The sibling that forbids the test above passing by marking everything."""
    review = _review_run([
        _report("P1", [Finding("P1", "read it", Severity.BLOCKER, "logic",
                               "a.py", 3)]),
        _report("P2", [Finding("P2", "read it too", Severity.NOTE, "style",
                               "b.py", 9)]),
        _report("P3", [], status=PersonaStatus.FOUND_NOTHING),
    ])

    summary = _summary(review)

    assert "[about the run]" not in summary
    assert "[ungrounded]" not in summary


# --- a mark is decoration unless the prompt says what it means ---------------

@pytest.mark.parametrize("fragment", [
    "NOT grounded in the source",
    "[about the run]",
    "do not restate it as a finding",
    "[ungrounded]",
    "called no tool",
    "check it against the source",
    "An UNTAGGED line is not a CHECKED line",
    "what surface a reviewer was given",
    "which tools it was refused",
], ids=["the-claim", "about-run-tag", "about-run-action", "ungrounded-tag",
        "ungrounded-cause", "ungrounded-action", "untagged-is-not-checked",
        "under-reports-the-surface", "under-reports-refusals"])
def test_the_prompt_says_what_each_mark_means_and_what_to_do(fragment):
    """#24's lesson: tolerating more without changing what the tool ASKS FOR
    leaves the cause in place. The two marks carry two different instructions.

    The last three rows are what the marks do NOT say, and they are here because
    tagging some lines is what makes an untagged one read as vouched-for — a
    hazard this change introduces rather than one it inherits. An untagged line is
    one nothing is known against, not one anybody checked: a finding *fabricated*
    by the reformat retry (#63) carries no tag and never can, and the tags are
    silent about a reduced surface (#69) and a refused tool (#67).
    """
    # The prior block carries NO tag, so every fragment below has to come from
    # the legend: asserting a tag that the fixture itself supplies is a test that
    # passes on its own data, which is what killed this row's first version.
    prompt = build_prompt(ReviewSpec(why="w", what="x"), "s", epoch=2,
                          prior="- [NOTE/open] (P1) something @ a.py:1")

    assert fragment in prompt


def test_a_first_epoch_prompt_carries_no_legend():
    """Epoch 1 has no prior context, so it must not be told how to read one."""
    prompt = build_prompt(ReviewSpec(why="w", what="x"), "s", epoch=1, prior="")

    assert "[ungrounded]" not in prompt
    assert "PRIOR EPOCHS" not in prompt


def test_an_epoch_with_nothing_to_remember_gets_no_legend():
    """A panel that found nothing in epoch 1 must not be handed a key to tags
    that are not there — a run must not report a contamination it did not suffer,
    which is the question #67, #70 and #69 each had to answer for a dry run.
    """
    prompt = build_prompt(ReviewSpec(why="w", what="x"), "s", epoch=2, prior="")

    assert "[ungrounded]" not in prompt
    assert "PRIOR EPOCHS" not in prompt
