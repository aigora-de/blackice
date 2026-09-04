# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""What counts as a vote (#26).

Quorum is one of ``CONVERGED``'s conjuncts, so the set of strings that can
produce a good verdict is the set of strings that can make the run lie. It used
to be every string beginning "YES" — which is how the literal contract
placeholder ``"YES | NO"`` voted for convergence.

The engine takes any ``SpawnPersona``, so this hardens independently of the
Claude backend's parser: these tests hand the loop verdicts directly. Both sides
are pinned deliberately — a rule that made convergence *unreachable* would pass
every test in the first half of this file.
"""

from __future__ import annotations

import pytest

from kuang.engine import (HaltingSet, HaltReason, PanelConfig, PersonaReport,
                          PersonaStatus, ReviewSpec, run)

PANEL = PanelConfig(personas=[(n, "mandate") for n in ("correctness", "adversary")])


def _run(verdict):
    def spawn(persona, mandate, surface, epoch):  # noqa: ANN001, ARG001
        return PersonaReport(persona=persona, verdict=verdict)

    return run(ReviewSpec(why="w", what="x"), HaltingSet(max_epochs=1), PANEL,
               spawn=spawn, gather=lambda e: "surface", parallel=False,
               scope_complete=lambda r: True)


@pytest.mark.parametrize("verdict", [
    "YES | NO",                    # REGRESSION for #26: the contract placeholder
    "YES/NO",
    "YES, approve",
    "YES — nothing blocking",
    "SOUND-WITH-CONCERNS",
    "yes please",
    "MAYBE",
    "",
    None,
    {"decision": "YES"},           # not a string at all (#25 keeps it from raising)
])
def test_only_the_word_yes_can_produce_a_good_verdict(verdict):
    """Nothing prefix-shaped remains: a vote is the word, or it is not a vote."""
    review_run = _run(verdict)

    assert review_run.halt_reason is not HaltReason.CONVERGED


@pytest.mark.parametrize("verdict", ["YES", "yes", "  YES  "])
def test_convergence_stays_reachable_for_a_panel_that_voted(verdict):
    """The other side of the guard, and the reason it is not simply ``== "YES"``.

    Written with the literal rather than ``AFFIRMATIVE_VERDICT``: asserting
    against the constant would follow it if the constant moved, which is #24's
    mutation-2 lesson.
    """
    review_run = _run(verdict)

    assert review_run.halt_reason is HaltReason.CONVERGED


# --- the quorum a run must be able to STATE, not merely apply (#82) ----------

def test_effective_quorum_defaults_to_unanimity_among_the_roster():
    """The number a run must print beside ``converged``, and where it comes from.

    ``quorum`` is ``None`` on every CLI run today — there is no ``--quorum`` flag —
    so the default is the whole answer, and the loop computed it in a local
    variable nothing outside could see. It is a property so the CLI reads the same
    rule the gate applies rather than restating it, which is #26's lesson about
    ``AFFIRMATIVE_VERDICT`` one field along.
    """
    assert PanelConfig(personas=[("a", "m"), ("b", "m")]).effective_quorum == 2
    assert PanelConfig(personas=[("a", "m")]).effective_quorum == 1


def test_effective_quorum_honours_an_explicit_threshold():
    """Personas are a parameter, and so is the agreement required of them."""
    panel = PanelConfig(personas=[("a", "m"), ("b", "m"), ("c", "m")], quorum=2)

    assert panel.effective_quorum == 2


def test_effective_quorum_over_an_empty_panel_is_zero():
    """``0 >= 0`` is the arithmetic behind #72's empty-panel case, stated once.

    Not a rounded-up 1: the property reports what the gate uses, and softening it
    here would make the printed number disagree with the applied one.
    """
    assert PanelConfig(personas=[]).effective_quorum == 0


@pytest.mark.parametrize("verdict,status,counted", [
    ("YES", PersonaStatus.CONTRIBUTED, True),
    ("YES", PersonaStatus.FOUND_NOTHING, True),
    ("YES", PersonaStatus.UNREPORTED, True),
    ("YES", PersonaStatus.NOT_SPAWNED, False),
    ("YES", PersonaStatus.AGENT_ERROR, False),
    ("NO", PersonaStatus.CONTRIBUTED, False),
    ("YES | NO", PersonaStatus.CONTRIBUTED, False),
    (None, PersonaStatus.CONTRIBUTED, False),
], ids=["contributed_yes", "found_nothing_yes", "unreported_yes",
        "not_spawned_yes", "agent_error_yes", "contributed_no",
        "the_contract_placeholder", "no_verdict_at_all"])
def test_counted_vote_is_the_one_predicate_behind_the_tally(verdict, status,
                                                            counted):
    """The rule the loop applies, exposed so a reporter cannot restate it wrongly.

    #82 has to report how many votes a verdict rests on and how many of those were
    grounded, which needs the identity of the voters and not just their number. A
    second copy of this predicate in the CLI would be a place for the printed
    number and the gated one to drift apart — which is exactly what #72 found the
    denominator had already done.
    """
    report = PersonaReport(persona="p", verdict=verdict, status=status)

    assert report.counted_vote is counted
