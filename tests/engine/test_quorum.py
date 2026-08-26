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
                          ReviewSpec, run)

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
