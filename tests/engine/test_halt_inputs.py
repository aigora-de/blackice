# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""The number the stall counter is taken on must be readable (#33, folded #114).

The epoch synthesis prints the count of new *findings*; the loop halts on the
count of *material* new clusters. They are different numbers, and until this
issue only the first could be read by anything outside ``loop.run`` — the second
was a local variable, and ``EpochResult.new_clusters`` (which it is computed
from) was written and read by nothing in the package.

These tests hold the **engine** half: the predicate lives on the object both the
loop and a reporter read, so the number a run states and the number its gate
applies cannot drift apart. That is ``PanelConfig.effective_quorum``'s and
``PersonaReport.counted_vote``'s precedent, applied one field along.

Each test asserts the count **and** the halt the count produced, deliberately: a
reported number nobody compares to a decision is the defect this issue is about.

Stdlib + ``FakeEnsemble`` only — no network, no ``claude`` subprocess.
"""

from __future__ import annotations

from kuang.engine import (Cluster, Finding, HaltingSet, HaltReason, PanelConfig,
                          PersonaReport, ReviewSpec, Severity, run)
from kuang.engine.fakes import FakeEnsemble
from kuang.engine.reduce import _identity_reduce


def _f(claim_class, severity, *, line, title="t"):
    """One finding. The line moves between epochs, so the signature differs."""
    return Finding(persona="p", title=title, severity=severity,
                   claim_class=claim_class, file="r.py", line=line)


def _merging_reduce(findings):
    """A reduce double that MERGES: one cluster per ``claim_class``.

    A scenario that does not merge cannot express this axis at all — under the
    identity default every new finding is a new cluster, so the two numbers can
    never differ and the test would pass on the defect.
    """
    groups: dict[str, list[Finding]] = {}
    for f in findings:
        groups.setdefault(f.claim_class, []).append(f)
    return [Cluster(members=tuple(v), title=v[0].title) for v in groups.values()]


def _two_epochs(epoch1, epoch2, *, reduce=_merging_reduce, patience=1):
    script = {1: {"p": PersonaReport(persona="p", verdict="NO", findings=[epoch1])},
              2: {"p": PersonaReport(persona="p", verdict="NO", findings=[epoch2])}}
    return run(
        ReviewSpec(why="w", what="x"),
        HaltingSet(max_epochs=5, stall_patience=patience,
                   require_scope_complete=False),
        PanelConfig(personas=[("p", "m")]),
        spawn=FakeEnsemble(script), gather=lambda e: "s",
        reduce=reduce, parallel=False)


def test_a_reworded_finding_shows_one_new_finding_and_no_new_material():
    """#114's own acceptance criterion, and the one the fold must not drop.

    Epoch 2 re-words the same concept at a moved line: a NEW ledger signature (so
    ``new_findings`` is 1) that the reduce folds into the concept epoch 1 already
    raised (so no cluster is new, and nothing material resets the counter).

    Without this the third field is satisfiable by emitting a value nobody
    compares. MUTATION: ``material_new_clusters`` returning ``new_clusters``
    unfiltered leaves this green but breaks the severity test below; returning
    the new *findings* instead turns this red.
    """
    review = _two_epochs(_f("b", Severity.BLOCKER, line=1230, title="original"),
                         _f("b", Severity.BLOCKER, line=1240, title="reworded"))
    second = review.epochs[1]

    assert len(second.new_findings) == 1, "the re-wording is a new signature"
    assert second.new_clusters == [], "it merged into the concept epoch 1 raised"
    assert second.material_new_clusters == [], "so nothing material was raised"
    # The number and the decision it produced, together: epoch 2's OWN halt, not
    # the run's, because a later empty epoch would stall the run either way and
    # an assertion on ``review.halt_reason`` would pass without this epoch's count
    # doing any work.
    assert second.halt is HaltReason.STALL


def test_an_epoch_that_raises_new_material_reports_it():
    """The pair, and the vacuity guard: not everything reads zero.

    A second concept in epoch 2 IS material, so the counter resets and the run
    does not stall. Without this test, ``material_new_clusters`` could return
    ``[]`` unconditionally and the test above would still pass.
    """
    review = _two_epochs(_f("b", Severity.BLOCKER, line=1230),
                         _f("c", Severity.BLOCKER, line=1240))
    second = review.epochs[1]

    assert len(second.new_findings) == 1
    assert len(second.new_clusters) == 1, "a second concept is a new cluster"
    assert len(second.material_new_clusters) == 1, "and it is above the blocker line"
    assert second.halt is None, "material was raised, so the epoch did not stall"


def test_a_new_cluster_below_the_blocker_line_is_not_material():
    """The mirror image: 'new' and 'material' are not the same number either.

    Two mechanisms make the printed count differ from the acted-on one — the
    reduce merging a finding into a seen concept (the test above) and the
    severity filter (this one). A record that reported only the endpoints would
    make an over-merging clusterer indistinguishable from a panel raising
    nothing that matters. MUTATION: drop the ``>= BLOCKER`` filter -> the counter
    resets on a NOTE and the run no longer stalls.
    """
    review = _two_epochs(_f("b", Severity.BLOCKER, line=1230),
                         _f("c", Severity.NOTE, line=1240))
    second = review.epochs[1]

    assert len(second.new_clusters) == 1, "the NOTE is a new concept"
    assert second.material_new_clusters == [], "but not one the counter reads"
    assert second.halt is HaltReason.STALL


def test_under_the_identity_default_the_two_numbers_cannot_differ():
    """Why the fixture above must merge, stated as a test rather than a comment.

    With one cluster per signature every new finding is a new cluster, so a
    fixture built on the default reduce cannot express the divergence — #111's
    lesson about a fixture too small to straddle the bound, one axis along.
    """
    review = _two_epochs(_f("b", Severity.BLOCKER, line=1230),
                         _f("b", Severity.BLOCKER, line=1240),
                         reduce=_identity_reduce)
    second = review.epochs[1]

    assert len(second.new_findings) == 1
    assert len(second.material_new_clusters) == 1, "the re-wording reads as new"
    assert second.halt is None, "so the epoch did not stall"
