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
* **(b) the halting zero-tests read the LEDGER**, as a snapshot taken after the insert
  and before ``reduce`` or any halting input runs, and the predicate takes two booleans.
  "Is any UGLY open" and "is any BLOCKER open" are exactly knowable there, and no value
  ``reduce`` returns reaches them. Under a partition they agree with the cluster counts
  by construction. (b) is NOT only a backstop for (a). The epoch's record is mutable,
  and ``scope_complete`` is handed it before the halt is taken: the first pre-merge
  review measured a scope check that zeroed the counts converging an UGLY run on main,
  and the second measured one writing through the ledger's own findings doing the same
  to this fix's second shape, which read the ledger live. The ``scope_complete`` and
  ``reduce`` side-effect tests below are (b)'s loop-level regressions.

The ruin row itself — an UGLY run halting ``converged`` on a non-partition — is killed
by neither mechanism alone in the mutation matrix, and that is the design rather than a
gap: (a) replaces the clusters, so the counts agree with the ledger, and (b) reads the
ledger whatever the counts say. Each of the two is shown load-bearing by the rows only
it kills.

No shipped reducer is known to trigger this: the identity makes singletons and the
backend's clusterer repairs to a strict partition. The case is #89's premise.

**Which are regressions, measured on main with this file in place.** It imports nothing
main lacks, but ``_evaluate_halt`` now takes two booleans where main took the epoch's
record and the run, so ``test_unguarded_seams.py``'s ``_halt`` was measured on main
through a minimal SHIM handing main the record it read. Tests green on main say so in
their own docstrings.

Stdlib + stub seams only: no network, no ``claude`` subprocess.
"""

from __future__ import annotations

import pytest

from kuang.engine import (Cluster, Finding, GateDecision, HaltingSet, HaltReason,
                          PanelConfig, PersonaReport, PersonaStatus, ReviewRun,
                          ReviewSpec, Severity, run)
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
    (_returns_nothing, "findings in no cluster: 2 of 2; "
                       "findings placed more than once: 0; empty clusters: 0"),
    (_drops_one, "findings in no cluster: 1 of 2; "
                 "findings placed more than once: 0; empty clusters: 0"),
    (_duplicates_one, "findings in no cluster: 0 of 2; "
                      "findings placed more than once: 1; empty clusters: 0"),
    (_twice_in_one_cluster, "findings in no cluster: 0 of 2; "
                            "findings placed more than once: 1; empty clusters: 0"),
    (_adds_an_empty_cluster, "findings in no cluster: 0 of 2; "
                             "findings placed more than once: 0; empty clusters: 1"),
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


# --- (b) the snapshot: taken from the ledger, before any seam can write it ---------

def _finding(severity, verified=None, n=0):
    return Finding("p", f"{MARKER} {n}", severity, f"lens-{n}", "a.py", n, verified=verified)


def _writes_through(kind):
    """A ``scope_complete`` that writes through the frozen ``Finding`` objects its record
    aliases — the ledger's own — and then answers yes. ``Finding`` is frozen without
    slots, so its ``__dict__`` is writable and ``object.__setattr__`` bypasses the freeze.
    """
    def scope(result):
        for f in result.new_findings:
            if kind == "refute":
                f.__dict__["verified"] = False
            elif kind == "downgrade":
                object.__setattr__(f, "severity", Severity.NOTE)
            else:
                f.__dict__["severity"] = "corrupt"
        return True
    return scope


@pytest.mark.parametrize("kind", ["refute", "downgrade", "corrupt"])
def test_a_seam_writing_through_the_ledgers_findings_cannot_mask_the_breaker(kind):
    """The halt is GREEN ON MAIN, by design: it guards the REJECTED second shape of #150, which
    read the ledger LIVE when the predicate ran. The ledger assertion is RED ON MAIN, for
    #155: there the halt held but the write landed. The second pre-merge review measured that
    shape converging an UGLY run for ``refute`` and ``downgrade``, and raising out of
    ``run`` for ``corrupt`` (``'str' object has no attribute 'is_ugly'``, #148's invariant
    broken). Main decided on cluster counts taken before ``scope_complete``; the snapshot
    restores that order and keeps the ledger as the source."""
    review_run = _contained(lambda: run(
        ReviewSpec(why="w", what="x"), HaltingSet(max_epochs=3), PANEL,
        spawn=_spawn_for(Severity.UGLY, verdict="YES"), gather=lambda epoch: "s",
        scope_complete=_writes_through(kind), parallel=False))

    assert review_run.halt_reason is HaltReason.ESCALATE_UGLY
    # Since #155 the write does not land at all: the seam was handed a copy. Red on main,
    # where the halt held but the ledger kept what the seam wrote.
    assert all((f.severity, f.verified) == (Severity.UGLY, True)
               for f in review_run.ledger.values())


def _downgrades_its_input(findings):
    for f in findings:
        object.__setattr__(f, "severity", Severity.NOTE)
    return _identity_reduce(findings)


@pytest.mark.parametrize("verdict", ["YES", "NO"])
def test_a_reduce_that_downgrades_its_input_cannot_mask_an_ugly(verdict):
    """REGRESSION for #150: on main a ``reduce`` that rewrote its input's severity in place
    converged (YES) or ran out its epochs (NO) with the panel's UGLYs downgraded, because
    the cluster counts were taken AFTER it ran. The snapshot is taken before ``reduce`` is
    called, so the breaker fires on the epoch the UGLY lands. Since #155 the write does not
    land either: ``reduce`` is handed copies, and the ledger keeps what the panel raised."""
    review_run = _run(_downgrades_its_input, verdict=verdict, max_epochs=3)

    assert review_run.halt_reason is HaltReason.ESCALATE_UGLY
    assert len(review_run.epochs) == 1
    assert all(f.severity is Severity.UGLY for f in review_run.ledger.values())


def test_the_ledger_read_counts_only_open_findings():
    """GREEN ON MAIN, by design: the snapshot's other end. A withdrawn UGLY or BLOCKER is
    not open, so it neither trips the breaker nor blocks convergence; a read that ignored
    ``counts_open`` would fire on a withdrawn claim, a rule firing on a healthy run."""
    ugly, blocker = _finding(Severity.UGLY, n=1), _finding(Severity.BLOCKER, n=2)
    withdrawn = [_finding(Severity.UGLY, verified=False, n=3),
                 _finding(Severity.BLOCKER, verified=False, n=4)]
    review_run = ReviewRun(ledger={f.key: f for f in [ugly, blocker, *withdrawn]})

    assert review_run.open_uglies == [ugly]
    assert review_run.open_blockers == [blocker]


def test_the_ledger_read_cannot_raise_on_a_corrupted_severity():
    """A PROPERTY test of the read the snapshot is taken from, not of ``run``: red on
    main, where ``open_uglies`` called ``severity.is_ugly``, but main's loop never read
    ``open_uglies``. Since #155 no seam can write a finding the ledger holds, so this guards
    the read itself rather than a reachable state: it pins that the snapshot adds no raise."""
    corrupted = _finding(Severity.BLOCKER)
    corrupted.__dict__["severity"] = "corrupt"
    review_run = ReviewRun(ledger={"k": corrupted})

    assert _contained(lambda: review_run.open_uglies) == []
    assert _contained(lambda: review_run.open_blockers) == []


# --- (b) at the loop: a seam handed the record cannot steer the halt through it -----

def _zeroes(field):
    """A ``scope_complete`` that writes the record it is handed, then answers yes."""
    def scope(result):
        setattr(result, field, 0)
        return True
    return scope


def _scoped_run(severity, field, *, verdict="YES", max_epochs=3, stall_patience=5,
                repeat=False):
    return _contained(lambda: run(
        ReviewSpec(why="w", what="x"),
        HaltingSet(max_epochs=max_epochs, stall_patience=stall_patience), PANEL,
        spawn=_spawn_for(severity, verdict=verdict, repeat=repeat),
        gather=lambda epoch: "s", scope_complete=_zeroes(field), parallel=False))


def test_a_seam_that_zeroes_the_ugly_count_cannot_converge_the_run():
    """REGRESSION for #150 (b), found by the pre-merge review: on main the breaker read
    ``result.open_uglies``, which ``scope_complete`` is handed mutably, so a scope check
    that zeroed it converged a run with open UGLYs in the ledger. (a) cannot see this:
    the clusters partition, and the count was overwritten after they were taken."""
    review_run = _scoped_run(Severity.UGLY, "open_uglies")

    assert review_run.halt_reason is HaltReason.ESCALATE_UGLY


def test_a_seam_that_zeroes_the_blocker_count_cannot_converge_the_run():
    """REGRESSION for #150 (b): the same door for ``CONVERGED`` — main converged at
    epoch 1 with an open BLOCKER; here the run is the identity's, halted ``epoch``."""
    review_run = _scoped_run(Severity.BLOCKER, "open_blockers")

    assert review_run.halt_reason is HaltReason.EPOCH
    assert len(review_run.epochs) == 3


def test_a_seam_that_zeroes_the_blocker_count_cannot_mask_a_stall():
    """REGRESSION for #150 (b): and for ``STALL`` — main ran to ``epoch`` at 6."""
    review_run = _scoped_run(Severity.BLOCKER, "open_blockers", verdict="NO",
                             max_epochs=6, stall_patience=2, repeat=True)

    assert review_run.halt_reason is HaltReason.STALL
    assert len(review_run.epochs) == 3


class _DuckEntry:
    """Not a ``Finding``: what a gate writing into ``dict[str, Finding]`` can leave there."""

    key = "duck"
    title = "a withdrawn entry"
    counts_open = False


def test_a_withdrawn_entry_a_gate_left_in_the_ledger_is_never_read_for_severity():
    """The halt is GREEN ON MAIN, by design; the ledger assertion is RED ON MAIN, for #155.
    It guards a regression this fix's third shape introduced,
    found by the third pre-merge review. Main's cluster counts read the severity only of
    OPEN members, so a gate that left a withdrawn, duck-typed entry in the ledger raised
    nothing; the snapshot read ``severity`` before ``counts_open`` and raised out of
    ``run`` (#148's invariant). Since #155 the gate's write does not reach the ledger at
    all, so the read order is guarded directly by the property test below."""
    def gate(result, review_run):
        review_run.ledger["duck"] = _DuckEntry()
        return GateDecision(stop=False)

    review_run = _contained(lambda: run(
        ReviewSpec(why="w", what="x"), HaltingSet(max_epochs=2, stall_patience=5), PANEL,
        spawn=_spawn_for(Severity.BLOCKER), gather=lambda epoch: "s", human_gate=gate,
        parallel=False))

    assert review_run.halt_reason is HaltReason.EPOCH
    assert "duck" not in review_run.ledger


def test_the_ledger_read_never_reads_the_severity_of_a_withdrawn_entry():
    """The read order, as a property of ``ReviewRun`` rather than of a gate's reach: a
    withdrawn entry with no severity at all is skipped, not read. GREEN ON MAIN, by
    design, for the reason the test above is."""
    review_run = ReviewRun(ledger={"duck": _DuckEntry()})

    assert _contained(lambda: review_run.open_uglies) == []
    assert _contained(lambda: review_run.open_blockers) == []
