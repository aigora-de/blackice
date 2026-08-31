# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""A finding about the RUN is not a finding about the change under review (#73).

A tooling failure becomes a ``meta`` finding — ``agent error: …`` (#29),
``persona failed: …`` (#25), and the contract parser's five — and those findings
enter the ledger like any other, so the reduce step turns them into canonical
issues and the operator's headline count adds one defect in the code to one
failure of the instrument reading it.

**The fact is set by the code that produces the finding, never matched out of
``claim_class``.** That is not a style preference. ``claim_class`` is a
model-controlled string, built in ``parse_findings`` from the persona's own JSON,
and ``meta`` is an ordinary word in Python review — a finding about a metaclass or
about metadata is naturally labelled that way. Measured before this was designed:
excluding ``claim_class == "meta"`` from the ledger turns a run that should halt
``escalate_ugly`` with one open UGLY into one that halts ``converged`` with none.
A consumer-side string match on model data **unlatches the circuit-breaker**,
which is #24's and #26's doctrine one field along.

The same measurement settled the shape of the change: for every meta finding the
instrument actually produces, halting does **not** move, because they are all
``Severity.NOTE`` and every gate keys on ``>= BLOCKER``. So this is a change to
what a run *reports*, and the tests below pin that it stays one.
"""

from __future__ import annotations

import pytest

from kuang.engine import (Cluster, Finding, HaltingSet, PanelConfig,
                          PersonaReport, PersonaStatus, ReviewRun, ReviewSpec,
                          Severity, run)

BLOCKER, NOTE, UGLY = Severity.BLOCKER, Severity.NOTE, Severity.UGLY


def _f(title, sev=BLOCKER, cls="logic", line=1, *, about_run=False):
    return Finding(persona="a", title=title, severity=sev, claim_class=cls,
                   file="a.py", line=line, evidence="e", about_run=about_run)


def _spawn_from(per_epoch):
    """per_epoch: {epoch: [Finding, ...]} — every persona reports the same set."""
    def spawn(name, mandate, surface, epoch):  # noqa: ANN001, ARG001
        findings = [Finding(**{**f.__dict__, "persona": name})
                    for f in per_epoch.get(epoch, [])]
        return PersonaReport(persona=name, verdict="YES", findings=findings,
                             status=(PersonaStatus.CONTRIBUTED if findings
                                     else PersonaStatus.FOUND_NOTHING))
    return spawn


def _run(per_epoch, halting=None, reduce=None):
    kwargs = {"reduce": reduce} if reduce else {}
    return run(ReviewSpec(why="w", what="x"),
               halting or HaltingSet(max_epochs=3, stall_patience=1),
               PanelConfig(personas=[("a", "m")]),
               spawn=_spawn_from(per_epoch), gather=lambda e: "SURFACE",
               parallel=False, **kwargs)


# --- the vocabulary -----------------------------------------------------------

def test_a_finding_is_about_the_change_by_default():
    """The default is the common case, and it is never inferred from the text."""
    assert _f("off-by-one").about_run is False


def test_the_engine_marks_its_own_failure_finding():
    """``persona failed: …`` is the engine's own (#25), so the engine sets it."""
    def _boom(name, mandate, surface, epoch):  # noqa: ANN001, ARG001
        raise RuntimeError("boom")

    result = run(ReviewSpec(why="w", what="x"), HaltingSet(max_epochs=1),
                 PanelConfig(personas=[("a", "m")]), spawn=_boom,
                 gather=lambda e: "S", parallel=False)
    finding = result.epochs[0].reports[0].findings[0]

    assert "persona failed" in finding.title
    assert finding.about_run is True


def test_the_ledger_splits_into_the_change_and_the_run():
    result = _run({1: [_f("real blocker"),
                       _f("agent error: boom", NOTE, "meta", 9, about_run=True)]})

    assert [f.title for f in result.change_findings] == ["real blocker"]
    assert [f.title for f in result.run_findings] == ["agent error: boom"]
    assert len(result.ledger) == 2, "and nothing was dropped to achieve it"


def test_a_cluster_is_about_the_run_if_any_member_is():
    """``max``-preserving, like ``Cluster.severity``: a mixed cluster is surfaced.

    Hiding it would let a reduce step launder an instrument failure into the
    change's issue count by merging it with a real finding.
    """
    change, instrument = _f("real"), _f("agent error", NOTE, "meta", about_run=True)

    assert Cluster(members=(change,)).about_run is False
    assert Cluster(members=(instrument,)).about_run is True
    assert Cluster(members=(change, instrument)).about_run is True


def test_the_flag_does_not_change_a_findings_ledger_identity():
    """Dedup keys must not shift, or every prior run's ledger stops matching."""
    assert _f("t", about_run=True).key == _f("t", about_run=False).key


# --- the measurement this design exists because of ----------------------------

def test_a_persona_declared_meta_claim_class_is_still_about_the_change():
    """The row that refutes a string match, and the reason for the whole design.

    ``claim_class`` comes from the persona's JSON. A reviewer may label a finding
    about a metaclass, or about metadata, ``meta`` — and a consumer-side match
    would drop a ruin-class claim out of the ledger and unlatch the breaker.
    """
    result = _run({1: [_f("unbounded loss on retry", UGLY, "meta")]},
                  HaltingSet(max_epochs=3))

    assert result.halt_reason.value == "escalate_ugly", "the breaker must latch"
    assert len(result.open_uglies) == 1
    assert [f.title for f in result.change_findings] == ["unbounded loss on retry"]
    assert result.run_findings == []


@pytest.mark.parametrize("name,per_epoch,halting", [
    ("stall", {1: [_f("real blocker")],
               2: [_f("agent error", NOTE, "meta", 20, about_run=True)],
               3: [_f("persona failed", NOTE, "meta", 30, about_run=True)]},
     HaltingSet(max_epochs=5, stall_patience=1)),
    ("converged", {1: []}, HaltingSet(max_epochs=3)),
    ("breaker", {1: [_f("ruin", UGLY, "ruin")]}, HaltingSet(max_epochs=3)),
])
def test_instrument_findings_do_not_move_the_halt(name, per_epoch, halting):
    """Measured before the design and pinned here: adding an instrument failure
    changes what a run REPORTS and nothing it decides.

    Every meta finding is ``Severity.NOTE`` and every gate keys on
    ``>= BLOCKER``, so the chain from the ledger to the halt predicate exists and
    carries no value. If this ever goes red, #73 stopped being a counting change.
    """
    without = _run({e: [f for f in fs if not f.about_run]
                    for e, fs in per_epoch.items()}, halting)
    with_meta = _run({e: list(fs) + [_f(f"agent error {e}", NOTE, "meta",
                                        90 + e, about_run=True)]
                      for e, fs in per_epoch.items()}, halting)

    assert with_meta.halt_reason is without.halt_reason
    assert len(with_meta.epochs) == len(without.epochs)
    assert len(with_meta.open_uglies) == len(without.open_uglies)
    assert len(with_meta.open_blockers) == len(without.open_blockers)
    assert with_meta.epochs[-1].open_blockers == without.epochs[-1].open_blockers
    assert with_meta.epochs[-1].open_uglies == without.epochs[-1].open_uglies
    assert len(with_meta.run_findings) > 0, "and the run really did carry them"
