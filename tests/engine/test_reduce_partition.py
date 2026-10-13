# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""A reduce that does not partition the ledger cannot switch the circuit-breaker off (#150).

#61's rule — "a reduce must partition the findings" — was doctrine in the ``Reduce``
protocol and enforced only in one backend's clusterer. The engine took every halt that
reads open severity (the breaker, ``CONVERGED`` and ``STALL``) from cluster counts, and
the clusters were whatever ``reduce`` returned. Measured on ``f17c256``: ``reduce -> []``
with open UGLYs in the ledger halted ``epoch`` on a panel voting NO, and ``converged`` —
the good verdict, exit 0 — on one voting YES. Nothing was recorded.

Two mechanisms, by the maintainer's ruling of 2026-10-13, and they kill different rows:

* **(a) the engine validates the partition**, inside #148's reduce guard: every ledger
  finding in exactly one cluster, and no cluster empty. A non-partition is a failed
  reduce — a ``SeamFailure`` with an engine-authored, count-only detail — and the
  identity reduce follows: "the state before the call", never a repair.
* **(b) the halting zero-tests read the LEDGER.** "Is any UGLY open" and "is any
  BLOCKER open" are exactly knowable there and depend on no seam. Under a partition
  they agree with the cluster counts by construction; (a) guarantees the partition
  inside the loop, so (b) is reached only by the predicate tests below, which hand it
  the state (a) refuses to build.

No shipped reducer is known to trigger this: the identity makes singletons and the
backend's clusterer repairs to a strict partition. The case is #89's premise.

**Which are regressions, measured on main with this file in place** (it imports nothing
main lacks, so no shim). Tests green on main say so in their own docstrings.

Stdlib + stub seams only: no network, no ``claude`` subprocess.
"""

from __future__ import annotations

from itertools import product

import pytest

from kuang.engine import (Cluster, EpochResult, Finding, HaltingSet, HaltReason,
                          PanelConfig, PersonaReport, PersonaStatus, ReviewRun,
                          ReviewSpec, Severity, run)
from kuang.engine.halting import _evaluate_halt
from kuang.engine.reduce import _identity_reduce

PANEL = PanelConfig(personas=[("first", "mandate"), ("second", "mandate")])

# A string no engine-authored detail can contain, carried in every title.
MARKER = "MODEL-TEXT-MARKER"


def _contained(call):
    """Call ``call()``; an ``Exception`` escaping it is a plain test failure.

    ``tests/engine/test_unrenderable_exception.py``'s pattern.
    """
    try:
        return call()
    except Exception as exc:  # noqa: BLE001
        name = str.__str__(type.__dict__["__name__"].__get__(type(exc), type))
        raise AssertionError(f"escaped the guard: {name}") from None


def _spawn_for(severity, *, verdict="NO", repeat=False):
    """Each persona raises one finding per epoch: distinct per epoch, or the SAME one
    every epoch where ``repeat`` (which is what lets ``STALL`` be reached)."""
    def spawn(persona, mandate, surface, epoch):  # noqa: ARG001
        e = 1 if repeat else epoch
        return PersonaReport(
            persona=persona, verdict=verdict, tokens=9, status=PersonaStatus.CONTRIBUTED,
            findings=[Finding(persona, f"{MARKER} {persona} at {e}", severity,
                              f"{persona}-lens-{e}", "a.py", e * 100)])
    return spawn


def _run(reduce, *, severity=Severity.UGLY, verdict="NO", max_epochs=2,
         stall_patience=5, repeat=False, checkpoint=None):
    return _contained(lambda: run(
        ReviewSpec(why="w", what="x"),
        HaltingSet(max_epochs=max_epochs, stall_patience=stall_patience), PANEL,
        spawn=_spawn_for(severity, verdict=verdict, repeat=repeat),
        gather=lambda epoch: "s", reduce=reduce, checkpoint=checkpoint, parallel=False))


def _seams(review_run):
    return [[f.seam for f in e.seam_failures] for e in review_run.epochs]


def _returns_nothing(findings):
    return []


def _drops_the_uglies(findings):
    return [c for c in _identity_reduce(findings) if c.severity is not Severity.UGLY]


def _drops_one(findings):
    return _identity_reduce(findings)[1:]


def _duplicates_one(findings):
    return _identity_reduce(findings) + _identity_reduce(findings)[:1]


def _twice_in_one_cluster(findings):
    first, *rest = _identity_reduce(findings)
    return [Cluster(members=first.members * 2, title=first.title), *rest]


def _adds_an_empty_cluster(findings):
    return _identity_reduce(findings) + [Cluster(members=(), title="")]


def _merges_everything(findings):
    return [Cluster(members=tuple(findings), title="merged")] if findings else []


NON_PARTITIONS = {
    "returns-nothing": _returns_nothing,
    "drops-the-uglies": _drops_the_uglies,
    "drops-one": _drops_one,
    "duplicates-one": _duplicates_one,
    "twice-in-one-cluster": _twice_in_one_cluster,
    "adds-an-empty-cluster": _adds_an_empty_cluster,
}


# --- (a) at the loop: a non-partition is a failed reduce ---------------------------

@pytest.mark.parametrize("reduce", NON_PARTITIONS.values(), ids=NON_PARTITIONS.keys())
def test_a_non_partition_is_a_failed_reduce_and_the_breaker_fires(reduce):
    """REGRESSION for #150, one per shape of non-partition.

    On main ``returns-nothing`` and ``drops-the-uglies`` halted ``epoch`` with four open
    UGLYs in the ledger; the other four fired the breaker but recorded nothing, with a
    cluster count that was not the number of canonical issues. Here every one is a
    recorded failure, and the identity reduce makes the counts the ledger's.
    """
    review_run = _run(reduce)

    assert review_run.halt_reason is HaltReason.ESCALATE_UGLY
    assert _seams(review_run) == [["reduce"]]
    assert review_run.epochs[0].open_uglies == 2
    assert [c.members for c in review_run.clusters] == [
        (f,) for f in review_run.ledger.values()]


def test_a_non_partition_cannot_converge_an_ugly_run():
    """REGRESSION for #150, the row the issue body did not measure: with the panel
    voting YES, ``reduce -> []`` halted ``converged`` on main with two open UGLYs."""
    review_run = _run(_returns_nothing, verdict="YES", max_epochs=3)

    assert review_run.halt_reason is HaltReason.ESCALATE_UGLY


def test_a_non_partition_cannot_converge_over_open_blockers():
    """REGRESSION for #150: ``CONVERGED``'s ``open_blockers == 0`` was a cluster count,
    so ``reduce -> []`` converged at epoch 1 on main with two open BLOCKERs. Here the run
    is the identity's: three epochs, halted ``epoch``."""
    review_run = _run(_returns_nothing, severity=Severity.BLOCKER, verdict="YES",
                      max_epochs=3)

    assert review_run.halt_reason is HaltReason.EPOCH
    assert len(review_run.epochs) == 3
    assert _seams(review_run) == [["reduce"]] * 3


def test_a_non_partition_cannot_mask_a_stall():
    """REGRESSION for #150, found re-measuring it: ``STALL`` also read a cluster count,
    so a run that should stall at epoch 3 bought three more epochs on main."""
    review_run = _run(_returns_nothing, severity=Severity.BLOCKER, max_epochs=6,
                      stall_patience=2, repeat=True)

    assert review_run.halt_reason is HaltReason.STALL
    assert len(review_run.epochs) == 3


def test_a_non_partition_counts_new_clusters_as_the_identity_does():
    """REGRESSION for #150: on main ``reduce -> []`` gave the stall counter
    ``material_new_clusters`` of [0, 0, 0] where the identity gives [2, 2, 2]. Falling
    back means a failed epoch counts exactly as a run without a clusterer does."""
    failed = _run(_returns_nothing, severity=Severity.BLOCKER, max_epochs=3)
    healthy = _run(_identity_reduce, severity=Severity.BLOCKER, max_epochs=3)

    for runs in (failed, healthy):
        assert [len(e.material_new_clusters) for e in runs.epochs] == [2, 2, 2]
        assert [len(e.new_clusters) for e in runs.epochs] == [2, 2, 2]


def test_an_empty_cluster_is_a_non_partition_and_inflates_nothing():
    """REGRESSION for #150, on the maintainer's ruling that an empty cluster is a
    non-partition. It moves no halt, but on main it counted as a NEW cluster in every
    epoch (``all`` over no members is ``True``) and as an issue in the change."""
    review_run = _run(_adds_an_empty_cluster, severity=Severity.BLOCKER, max_epochs=3)

    assert [len(e.new_clusters) for e in review_run.epochs] == [2, 2, 2]
    assert all(c.members for c in review_run.clusters)
    assert _seams(review_run) == [["reduce"]] * 3


# --- the detail: engine-authored, counts only --------------------------------------

@pytest.mark.parametrize("reduce, detail", [
    (_returns_nothing, "2 of 2 findings in no cluster, 0 placed more than once, "
                       "0 empty clusters"),
    (_drops_one, "1 of 2 findings in no cluster, 0 placed more than once, "
                 "0 empty clusters"),
    (_duplicates_one, "0 of 2 findings in no cluster, 1 placed more than once, "
                      "0 empty clusters"),
    (_twice_in_one_cluster, "0 of 2 findings in no cluster, 1 placed more than once, "
                            "0 empty clusters"),
    (_adds_an_empty_cluster, "0 of 2 findings in no cluster, 0 placed more than once, "
                             "1 empty clusters"),
], ids=["nothing", "drops-one", "duplicates-one", "twice-in-one", "empty"])
def test_the_failure_says_how_the_partition_failed_in_counts(reduce, detail):
    """REGRESSION for #150: on main nothing was recorded at all.

    Built from counts, never from titles: a title is model-authored text on its way to
    a published artefact (#118, #121). Every title here carries ``MARKER``.
    """
    review_run = _run(reduce)
    (failure,) = review_run.epochs[0].seam_failures

    assert failure.detail == f"ValueError: reduce did not partition the ledger: {detail}"
    assert failure.detail_chars == len(failure.detail)
    assert MARKER not in failure.detail


# --- the pair: a genuine partition is never a failure, and moves nothing -----------

def test_a_reduce_that_genuinely_merges_is_kept_and_recorded_as_nothing():
    """GREEN ON MAIN, by design: the twin of every row above. A clusterer that merges is
    what ``--semantic-dedup`` is for, and a partition check that refused one would be
    the rule firing on a healthy run (#148's lesson: its first repair did exactly that).
    """
    review_run = _run(_merges_everything, severity=Severity.BLOCKER, verdict="YES",
                      max_epochs=3)

    assert _seams(review_run) == [[], [], []]
    assert review_run.halt_reason is HaltReason.EPOCH
    assert len(review_run.clusters) == 1
    assert [e.open_blockers for e in review_run.epochs] == [1, 1, 1]
    assert [len(e.material_new_clusters) for e in review_run.epochs] == [1, 0, 0]


def test_a_merged_ugly_still_trips_the_breaker_once():
    """GREEN ON MAIN: the UGLY-preserving merge (``Cluster.severity`` is the max)."""
    review_run = _run(_merges_everything)

    assert review_run.halt_reason is HaltReason.ESCALATE_UGLY
    assert review_run.epochs[0].open_uglies == 1
    assert _seams(review_run) == [[]]


def _partitions(items):
    """Every set partition of ``items``, in a fixed order (no randomness in a test)."""
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for part in _partitions(rest):
        yield [[first], *part]
        for i in range(len(part)):
            yield [*part[:i], [first, *part[i]], *part[i + 1:]]


SEVERITY_MIXES = {
    "ugly-blocker-notes": (Severity.UGLY, Severity.BLOCKER, Severity.NOTE, Severity.NOTE),
    "blockers-and-minors": (Severity.BLOCKER, Severity.BLOCKER, Severity.NON_BLOCKING,
                            Severity.NOTE),
    "minors-only": (Severity.NON_BLOCKING, Severity.NOTE, Severity.NOTE, Severity.NOTE),
}


def _one_epoch(severities, reduce):
    """Four findings from one persona in one epoch, the panel voting YES, so every one
    of ESCALATE_UGLY, CONVERGED and EPOCH is reachable."""
    def spawn(persona, mandate, surface, epoch):  # noqa: ARG001
        return PersonaReport(
            persona=persona, verdict="YES", tokens=9, status=PersonaStatus.CONTRIBUTED,
            findings=[Finding(persona, f"{MARKER} {i}", sev, f"lens-{i}", "a.py", i)
                      for i, sev in enumerate(severities)])
    return _contained(lambda: run(
        ReviewSpec(why="w", what="x"), HaltingSet(max_epochs=1), PanelConfig([("p", "m")]),
        spawn=spawn, gather=lambda epoch: "s", reduce=reduce, parallel=False))


@pytest.mark.parametrize("severities", SEVERITY_MIXES.values(), ids=SEVERITY_MIXES.keys())
def test_every_partition_halts_as_the_identity_does_and_records_nothing(severities):
    """GREEN ON MAIN, by design: the property's healthy half, over EVERY set partition of
    a four-finding ledger (fifteen). The halt is the identity's, nothing is recorded, and
    the published counts agree with the ledger's zero-tests (#33)."""
    expected = _one_epoch(severities, _identity_reduce).halt_reason
    shapes = list(_partitions(list(range(len(severities)))))
    assert len(shapes) == 15

    for shape in shapes:
        def reduce(findings, shape=shape):
            return [Cluster(members=tuple(findings[i] for i in group)) for group in shape]
        review_run = _one_epoch(severities, reduce)
        (epoch,) = review_run.epochs

        assert review_run.halt_reason is expected, shape
        assert epoch.seam_failures == [], shape
        assert (epoch.open_uglies > 0) == bool(review_run.open_uglies), shape
        # Conditional, and the condition is the breaker: a cluster merging an open UGLY
        # with a BLOCKER counts as UGLY, so the BLOCKER counts agree only while no UGLY
        # is open — and while one is, the breaker halts before either is read.
        if not review_run.open_uglies:
            assert (epoch.open_blockers > 0) == bool(review_run.open_blockers), shape


@pytest.mark.parametrize("severities", SEVERITY_MIXES.values(), ids=SEVERITY_MIXES.keys())
def test_every_single_fault_non_partition_is_recorded_and_halts_as_the_identity(severities):
    """REGRESSION for #150, the property's other half: drop any ONE finding, place any
    one twice, or add an empty cluster, and the reduce is a recorded failure whose run
    halts exactly as the identity's. On main a dropped UGLY converged."""
    expected = _one_epoch(severities, _identity_reduce).halt_reason
    faults = [("drop", i) for i in range(4)] + [("twice", i) for i in range(4)]
    faults.append(("empty", None))

    for kind, i in faults:
        def reduce(findings, kind=kind, i=i):
            clusters = [Cluster(members=(f,)) for f in findings]
            if kind == "drop":
                del clusters[i]
            elif kind == "twice":
                clusters.append(Cluster(members=(findings[i],)))
            else:
                clusters.append(Cluster(members=()))
            return clusters
        review_run = _one_epoch(severities, reduce)
        (epoch,) = review_run.epochs

        assert review_run.halt_reason is expected, (kind, i)
        assert [f.seam for f in epoch.seam_failures] == ["reduce"], (kind, i)
        assert (epoch.open_uglies > 0) == bool(review_run.open_uglies), (kind, i)
        # Conditional, and the condition is the breaker: a cluster merging an open UGLY
        # with a BLOCKER counts as UGLY, so the BLOCKER counts agree only while no UGLY
        # is open — and while one is, the breaker halts before either is read.
        if not review_run.open_uglies:
            assert (epoch.open_blockers > 0) == bool(review_run.open_blockers), (kind, i)


@pytest.mark.parametrize("reduce", [_identity_reduce, _merges_everything,
                                    *NON_PARTITIONS.values()],
                         ids=["identity", "merges", *NON_PARTITIONS.keys()])
def test_the_published_counts_decide_as_the_ledger_does_at_every_epoch(reduce):
    """#33's rule, both ways and at EVERY epoch: the per-epoch counts the artefact
    publishes as the halt's inputs (#117) must be zero exactly when the ledger's open
    UGLYs and BLOCKERs are, whatever ``reduce`` returned. Snapshotted through the
    ``checkpoint`` seam, which sees each epoch as it was decided.

    Green on main for ``identity``, ``merges``, and the non-partitions that drop no
    severity class; red for ``returns-nothing`` and ``drops-the-uglies``.
    """
    seen = []

    def checkpoint(review_run):
        epoch = review_run.epochs[-1]
        seen.append(((epoch.open_uglies > 0) == bool(review_run.open_uglies),
                     (epoch.open_blockers > 0) == bool(review_run.open_blockers)))

    for severity in (Severity.UGLY, Severity.BLOCKER):
        _run(reduce, severity=severity, max_epochs=3, checkpoint=checkpoint)

    assert seen and all(ugly and blocker for ugly, blocker in seen)


# --- (b) at the predicate: the zero-tests read the ledger ---------------------------

def _finding(severity, verified=None, n=0):
    return Finding("p", f"{MARKER} {n}", severity, f"lens-{n}", "a.py", n, verified=verified)


def _decide(ledger_findings, *, open_uglies=0, open_blockers=0, quorum_met=True,
            stall_epochs=0, stall_patience=5):
    """The predicate over a ledger and the cluster counts given — which (a) never lets
    disagree inside the loop, so this is the only place (b) can be shown to act."""
    review_run = ReviewRun(ledger={f.key: f for f in ledger_findings})
    result = EpochResult(index=1, reports=[], new_findings=[],
                         open_blockers=open_blockers, open_uglies=open_uglies)
    return _evaluate_halt(result, review_run,
                          HaltingSet(max_epochs=9, stall_patience=stall_patience),
                          epochs_done=1, stall_epochs=stall_epochs, tokens_spent=0,
                          elapsed_s=0.0, scope_complete=True, quorum_met=quorum_met,
                          no_persona_reviewed=False)


def test_the_breaker_reads_the_ledger_whatever_the_cluster_count():
    """REGRESSION for #150's mechanism (b): on main the breaker read only the cluster
    count, so an open UGLY in the ledger with a count of zero did not trip it."""
    assert _decide([_finding(Severity.UGLY)]) is HaltReason.ESCALATE_UGLY


def test_convergence_reads_the_ledger_whatever_the_cluster_count():
    """REGRESSION for #150 (b): an open BLOCKER with a cluster count of zero converged."""
    assert _decide([_finding(Severity.BLOCKER)]) is None


def test_a_stall_reads_the_ledger_whatever_the_cluster_count():
    """REGRESSION for #150 (b): ``STALL`` needs blockers open, and read a count."""
    assert _decide([_finding(Severity.BLOCKER)], quorum_met=False,
                   stall_epochs=1, stall_patience=1) is HaltReason.STALL


def test_a_withdrawn_finding_in_the_ledger_opens_nothing():
    """GREEN ON MAIN, by design: the pair to the three above. A refuted UGLY or BLOCKER
    is not open, so it neither trips the breaker nor blocks convergence. It guards the
    ledger read's other end: one that ignored ``counts_open`` would fire on a withdrawn
    claim, which is the rule firing on a healthy run."""
    withdrawn = [_finding(Severity.UGLY, verified=False, n=1),
                 _finding(Severity.BLOCKER, verified=False, n=2)]

    assert _decide(withdrawn) is HaltReason.CONVERGED


def test_a_healthy_ledger_decides_as_its_cluster_counts_do():
    """GREEN ON MAIN, by design: wherever the ledger and the counts agree — which is
    every loop-built state — the decision is today's, for each zero-test."""
    ugly, blocker = _finding(Severity.UGLY, n=1), _finding(Severity.BLOCKER, n=2)
    for uglies, blockers in product((0, 1), (0, 1)):
        ledger = [ugly] * uglies + [blocker] * blockers
        decided = _decide(ledger, open_uglies=uglies, open_blockers=blockers)
        expected = (HaltReason.ESCALATE_UGLY if uglies else
                    None if blockers else HaltReason.CONVERGED)
        assert decided is expected, (uglies, blockers)
