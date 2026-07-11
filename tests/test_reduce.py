# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""Engine tests for the semantic reduce step (issue #1).

These exercise the ``Reduce`` seam and the cluster-level control logic in
``loop.py`` in isolation — stdlib + ``FakeEnsemble`` only, no network, no
``claude`` subprocess. The reduce *doubles* used here are deterministic test
stand-ins (group-by-``claim_class``); the LLM clusterer is tested separately in
``test_clusterer.py``.

Load-bearing tests carry an inline MUTATION note: how to neuter the fix and see
the test go red (e.g. revert new-material detection to raw ``Finding.key``, or
compute ``Cluster.severity`` with ``min`` instead of ``max``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from loop import (
    Cluster,
    Finding,
    FakeEnsemble,
    HaltingSet,
    HaltReason,
    PanelConfig,
    PersonaReport,
    ReviewRun,
    ReviewSpec,
    Severity,
    _identity_reduce,
    run,
)

FIXTURE = Path(__file__).parent / "fixtures" / "generic_ledger.json"


# --- reduce test-doubles ----------------------------------------------------

def group_by_claim_class(findings):
    """Deterministic reduce double: one cluster per ``claim_class``.

    Over the fixture (where ``claim_class`` == the ground-truth concept id) this
    reproduces the intended clustering exactly — so the engine's cluster logic
    can be tested without an LLM.
    """
    groups: dict[str, list[Finding]] = {}
    for f in findings:
        groups.setdefault(f.claim_class, []).append(f)
    return [Cluster(members=tuple(v), title=v[0].title) for v in groups.values()]


def _f(claim_class, severity, *, file="m.py", line=10, persona="p", title="t",
       verified=None):
    return Finding(persona=persona, title=title, severity=severity,
                   claim_class=claim_class, file=file, line=line, verified=verified)


def _one_epoch(findings, *, reduce, halting=None, personas=None):
    """Run a single-persona, single-epoch loop that emits ``findings``."""
    persona = "p"
    script = {1: {persona: PersonaReport(persona=persona, verdict="NO",
                                         findings=list(findings))}}
    halting = halting or HaltingSet(max_epochs=1, require_scope_complete=False)
    return run(
        ReviewSpec(why="w", what="x"), halting,
        PanelConfig(personas=[(persona, "m")]),
        spawn=FakeEnsemble(script), gather=lambda e: "s",
        reduce=reduce, parallel=False)


# --- the identity default reproduces today's behaviour ----------------------

def test_identity_default_is_one_cluster_per_signature():
    """Default reduce: each distinct-signature finding is its own cluster."""
    findings = [
        _f("a", Severity.UGLY, line=10),
        _f("a", Severity.BLOCKER, line=10),   # differs by severity -> distinct key
        _f("b", Severity.NOTE, line=90),
    ]
    review = _one_epoch(findings, reduce=_identity_reduce)
    # Raw ledger preserved (wrap, don't replace) and one cluster per key.
    assert len(review.ledger) == 3
    assert len(review.clusters) == 3
    assert {c.key for c in review.clusters} == set(review.ledger)


def test_demo_still_converges_under_default_reduce():
    """The end-to-end demo (default reduce) is unchanged: converges in 2 epochs."""
    from loop import _demo
    review = _demo()
    assert review.halt_reason is HaltReason.CONVERGED
    assert len(review.epochs) == 2


# --- intra-epoch collapse + UGLY preservation -------------------------------

def test_same_concept_across_lines_collapses_to_one_cluster():
    """Four same-concept findings (distinct keys) fold to ONE new cluster.

    Spans NON_BLOCKING->UGLY, so the cluster is UGLY and trips the breaker.
    MUTATION: compute ``Cluster.severity`` with ``min`` -> cluster reads
    NON_BLOCKING -> no ESCALATE_UGLY (halt becomes EPOCH). MUTATION: detect new
    material by raw ``Finding.key`` -> four new material findings, not one.
    """
    findings = [
        _f("drop", Severity.NON_BLOCKING, file="a.py", line=499),
        _f("drop", Severity.BLOCKER, file="a.py", line=517),
        _f("drop", Severity.UGLY, file="b.py", line=1281),
        _f("drop", Severity.UGLY, file="a.py", line=499),
    ]
    review = _one_epoch(findings, reduce=group_by_claim_class,
                        halting=HaltingSet(max_epochs=3, require_scope_complete=False))
    assert len(review.ledger) == 4                 # all raw findings preserved
    epoch = review.epochs[0]
    assert len(epoch.new_clusters) == 1            # collapsed to one concept
    (cluster,) = review.clusters
    assert cluster.severity is Severity.UGLY       # max over members
    assert review.halt_reason is HaltReason.ESCALATE_UGLY
    assert epoch.open_uglies == 1                  # cluster-level count


# --- cross-epoch dup no longer reads as new material (the core bug) ----------

def test_reworded_dup_next_epoch_does_not_reset_stall():
    """A re-worded dup in epoch 2 merges into the epoch-1 cluster -> STALL.

    Both findings are the same concept (same ``claim_class``) at different lines,
    so signature dedup gives them different keys. Under the reduce, epoch 2 adds
    no new cluster -> no material -> STALL fires (blocker still open).
    MUTATION: swap the reduce for the identity default -> epoch 2's dup is a new
    key with BLOCKER severity -> stall resets -> run continues to EPOCH at 5.
    """
    p = "p"
    script = {
        1: {p: PersonaReport(persona=p, verdict="NO", findings=[
            _f("b", Severity.BLOCKER, file="r.py", line=1230, title="original")])},
        2: {p: PersonaReport(persona=p, verdict="NO", findings=[
            _f("b", Severity.BLOCKER, file="r.py", line=1240, title="reworded")])},
    }
    review = run(
        ReviewSpec(why="w", what="x"),
        HaltingSet(max_epochs=5, stall_patience=1, require_scope_complete=False),
        PanelConfig(personas=[(p, "m")]),
        spawn=FakeEnsemble(script), gather=lambda e: "s",
        reduce=group_by_claim_class, parallel=False)
    assert review.halt_reason is HaltReason.STALL
    assert len(review.epochs) == 2
    assert len(review.ledger) == 2                 # both raw findings kept
    assert len(review.clusters) == 1               # but one canonical issue


def test_reworded_dup_under_identity_default_does_not_stall():
    """Control for the mutation above: identity default lets the dup read as new."""
    p = "p"
    script = {
        1: {p: PersonaReport(persona=p, verdict="NO", findings=[
            _f("b", Severity.BLOCKER, file="r.py", line=1230)])},
        2: {p: PersonaReport(persona=p, verdict="NO", findings=[
            _f("b", Severity.BLOCKER, file="r.py", line=1240)])},
    }
    review = run(
        ReviewSpec(why="w", what="x"),
        HaltingSet(max_epochs=2, stall_patience=1, require_scope_complete=False),
        PanelConfig(personas=[(p, "m")]),
        spawn=FakeEnsemble(script), gather=lambda e: "s",
        reduce=_identity_reduce, parallel=False)
    # No STALL: the dup counted as new material each epoch; halts on EPOCH cap.
    assert review.halt_reason is HaltReason.EPOCH


# --- cluster severity: max, but breaker keys off OPEN members ----------------

def test_cluster_severity_is_max_over_members():
    c = Cluster(members=(_f("x", Severity.NOTE), _f("x", Severity.UGLY),
                         _f("x", Severity.BLOCKER)))
    assert c.severity is Severity.UGLY


def test_resolved_ugly_does_not_count_as_open_ugly():
    """A cluster of {resolved UGLY, open NOTE} must NOT trip the breaker.

    ``open_severity`` is the max over *open* members only, so a withdrawn ruin
    finding does not keep the circuit-breaker latched.
    """
    cluster = Cluster(members=(
        _f("x", Severity.UGLY, verified=False),   # refuted -> not open
        _f("x", Severity.NOTE, verified=None),    # open
    ))
    assert cluster.severity is Severity.UGLY          # raw max unchanged
    assert cluster.open_severity is Severity.NOTE     # over open members only
    run_ = ReviewRun(clusters=[cluster])
    assert run_.open_ugly_clusters == []
    assert run_.open_blocker_clusters == []


# --- zero-ness equivalence: cluster counts vs finding counts -----------------

@pytest.mark.parametrize("findings", [
    [],
    [_f("a", Severity.NOTE)],
    [_f("a", Severity.BLOCKER), _f("a", Severity.NOTE)],           # merged blocker
    [_f("a", Severity.UGLY), _f("a", Severity.BLOCKER)],           # merged ugly+blocker
    [_f("a", Severity.UGLY, verified=False), _f("a", Severity.NOTE)],  # resolved ugly
    [_f("a", Severity.BLOCKER), _f("b", Severity.UGLY)],           # separate concepts
])
def test_cluster_open_counts_match_finding_open_counts_in_zeroness(findings):
    """The zero-ness invariants that let the halt gate move to cluster level.

    * UGLY (unconditional): an open UGLY cluster exists iff an open UGLY finding
      does — so the circuit-breaker (checked first) fires identically.
    * BLOCKER (given no open UGLY, which is exactly when convergence is judged): an
      open BLOCKER cluster exists iff an open BLOCKER finding does. A blocker merged
      into an UGLY cluster is legitimately dominated — the breaker halts first.

    MUTATION: define ``open_severity`` over ALL members (not just open) -> the
    resolved-ugly parametrisation breaks the UGLY invariant.
    """
    clusters = group_by_claim_class(findings)
    review = ReviewRun(ledger={f.key: f for f in findings}, clusters=clusters)

    finding_ugly = any(f.severity.is_ugly and f.counts_open for f in findings)
    finding_blocker = any(f.severity is Severity.BLOCKER and f.counts_open
                          for f in findings)
    assert bool(review.open_ugly_clusters) == finding_ugly
    if not finding_ugly:  # the state in which convergence actually reads blockers
        assert bool(review.open_blocker_clusters) == finding_blocker


# --- partition integrity: never drop a finding ------------------------------

def _fixture_findings():
    data = json.loads(FIXTURE.read_text())
    return [Finding(persona=d["persona"], title=d["title"],
                    severity=Severity[d["severity"]], claim_class=d["claim_class"],
                    file=d["file"], line=d["line"], evidence=d.get("evidence", ""))
            for d in data["findings"]]


@pytest.mark.parametrize("reduce", [_identity_reduce, group_by_claim_class])
def test_reduce_is_a_partition(reduce):
    """Every input finding appears in exactly one output cluster."""
    findings = _fixture_findings()
    clusters = reduce(findings)
    members = [m for c in clusters for m in c.members]
    assert len(members) == len(findings)
    assert set(id(m) for m in members) == set(id(f) for f in findings)


# --- the genericised corpus collapses N -> M, UGLY survives -----------------

def test_fixture_collapses_to_canonical_issues():
    """13 raw findings -> 5 canonical issues; every UGLY is preserved."""
    data = json.loads(FIXTURE.read_text())
    findings = _fixture_findings()
    canonical = data["canonical"]

    # Signature dedup alone leaves them all distinct (the problem being fixed).
    assert len({f.key for f in findings}) == len(findings) == 13

    clusters = {c.members[0].claim_class: c for c in group_by_claim_class(findings)}
    assert set(clusters) == set(canonical)                       # 5 concepts
    for concept, expected in canonical.items():
        c = clusters[concept]
        assert len(c.members) == expected["size"]
        assert c.severity is Severity[expected["severity"]]      # UGLY survives

    ugly_clusters = [c for c in clusters.values() if c.severity.is_ugly]
    assert len(ugly_clusters) == 3
    # No UGLY finding is hidden in a non-UGLY cluster.
    for f in findings:
        if f.severity.is_ugly:
            assert clusters[f.claim_class].severity.is_ugly
