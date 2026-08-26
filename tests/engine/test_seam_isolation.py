# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""One persona must not be able to end the run (#25).

The engine treats ``spawn`` as a fallible black box, but until now an exception
crossing that seam propagated out of ``pool.map`` and killed the loop — after
every persona had been spawned and paid for. A total parser fixes the shapes we
know about; isolating the seam makes it a property.

The second half of this file is the part that keeps the first half honest. Turning
"the run dies" into "the run continues" must not turn it into "the run converges
with half a panel", which is #30's persona-dropout failure wearing a fix. It
cannot, because a report that carries no verdict is not a YES and quorum is
therefore unmet — and the last test pins the other side of that, so the guard
cannot be satisfied by making convergence impossible for everyone.
"""

from __future__ import annotations

from kuang.engine import (Finding, HaltingSet, HaltReason, PanelConfig,
                          PersonaReport, ReviewSpec, Severity, run)

PANEL = PanelConfig(personas=[(n, "mandate") for n in
                              ("correctness", "adversary", "engineer", "empiricist")])


def _spec() -> ReviewSpec:
    return ReviewSpec(why="mission-critical", what="the diff")


def _healthy(persona: str) -> PersonaReport:
    # A distinct claim_class per persona, because ``Finding.key`` does not include
    # the persona: four identical findings would dedup to one ledger entry and the
    # survivors' work would look lost when it was merely merged.
    return PersonaReport(persona=persona, verdict="YES", findings=[
        Finding(persona, f"{persona}'s finding", Severity.NON_BLOCKING,
                f"{persona}-lens", "a.py", 1)])


def _one_persona_raises(persona, mandate, surface, epoch):  # noqa: ANN001, ARG001
    if persona == "adversary":
        raise ValueError("invalid literal for int() with base 10: '~120'")
    return _healthy(persona)


def _run(spawn, *, parallel: bool, max_epochs: int = 1):
    return run(_spec(), HaltingSet(max_epochs=max_epochs), PANEL,
               spawn=spawn, gather=lambda epoch: "diff", parallel=parallel)


def test_a_raising_persona_does_not_end_the_run():
    """The original repro: this raised ValueError out of the loop, both ways."""
    for parallel in (True, False):
        review_run = _run(_one_persona_raises, parallel=parallel)

        assert review_run.halt_reason is HaltReason.EPOCH


def test_the_other_personas_work_survives():
    review_run = _run(_one_persona_raises, parallel=False)

    titles = {f.title for f in review_run.ledger.values()}
    assert {"correctness's finding", "engineer's finding", "empiricist's finding"} <= titles


def test_the_failure_is_recorded_as_a_finding_not_swallowed():
    review_run = _run(_one_persona_raises, parallel=False)

    failures = [f for f in review_run.ledger.values() if f.claim_class == "meta"]
    assert len(failures) == 1
    assert "ValueError" in failures[0].title or "ValueError" in failures[0].evidence
    assert "~120" in failures[0].title + failures[0].evidence


def test_a_crashed_persona_cannot_produce_a_good_verdict():
    """Survivable, not silent: three YES votes are not a quorum of four."""
    def spawn(persona, mandate, surface, epoch):  # noqa: ANN001, ARG001
        if persona == "adversary":
            raise RuntimeError("the backend fell over")
        return PersonaReport(persona=persona, verdict="YES")

    review_run = _run(spawn, parallel=False)

    assert not review_run.open_blockers          # nothing is blocking...
    assert review_run.halt_reason is not HaltReason.CONVERGED   # ...and it still is not GOOD


def test_a_whole_panel_that_found_nothing_still_converges():
    """The other side of the guard: convergence must remain reachable."""
    def spawn(persona, mandate, surface, epoch):  # noqa: ANN001, ARG001
        return PersonaReport(persona=persona, verdict="YES")

    review_run = _run(spawn, parallel=False)

    assert review_run.halt_reason is HaltReason.CONVERGED


def test_a_non_string_verdict_does_not_end_the_run():
    """The quorum count sits OUTSIDE the seam guard, so it hardens separately.

    ``parse_findings`` coerces the verdict, so this cannot arrive from the Claude
    backend — but the engine takes any ``SpawnPersona``, and a backend that hands
    back a dict used to raise ``AttributeError`` in ``.upper()`` and end the run
    after the panel had been paid for.
    """
    def spawn(persona, mandate, surface, epoch):  # noqa: ANN001, ARG001
        return PersonaReport(persona=persona, verdict={"decision": "YES"})

    review_run = _run(spawn, parallel=False)

    assert review_run.halt_reason is HaltReason.EPOCH   # not a YES, so not converged


def test_a_keyboard_interrupt_still_stops_the_run():
    """The seam guard catches Exception, deliberately — not BaseException."""
    def spawn(persona, mandate, surface, epoch):  # noqa: ANN001, ARG001
        raise KeyboardInterrupt

    try:
        _run(spawn, parallel=False)
    except KeyboardInterrupt:
        return
    raise AssertionError("the run swallowed a KeyboardInterrupt")
