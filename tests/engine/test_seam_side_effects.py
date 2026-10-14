# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""No injected seam can change the run through an object it is handed (#155).

#129 settled the rule for what a seam RETURNS: the engine never reads an unvalidated value
off a return outside the guard. Nothing applied a rule to what a seam DOES to its
arguments, and every seam that received a record received the live object:
``checkpoint`` and ``human_gate`` the ``ReviewRun``, ``scope_complete`` and ``human_gate``
the epoch's ``EpochResult``, ``reduce`` the ledger's own ``Finding``s, and ``adjudicate``
the finding the ledger entry was then rebuilt from. ``Finding`` is frozen without slots,
so ``__dict__`` and ``object.__setattr__`` both write through the freeze.

**The rule, stated once beside #129's:** a seam's only channel into the run is its return
value. Every record a seam is handed is a copy the engine built, so nothing it does to that
object reaches the engine's decisions or its record. It covers the gate too, by the
maintainer's ruling of 2026-10-14: retiring a fixed finding is a channel into the run, and
it is #157's, on a return value.

Measured on ``954cdc1`` with this file's fixture: every numbered row below reproduced, each
beside a control whose seam only reads. One regression per row, in the issue's order, the
checkpoint row first because #31 makes ``checkpoint`` a durable write. Tests GREEN on main
say so in their own docstrings.

Stdlib + stub seams only: no network, no ``claude`` subprocess.
"""

from __future__ import annotations

import pytest

from kuang.engine import (Finding, GateDecision, HaltingSet, HaltReason, PanelConfig,
                          PersonaReport, PersonaStatus, ReviewRun, ReviewSpec, Severity, run)
import kuang.engine.loop as loop_module
from kuang.engine.reduce import _identity_reduce

PANEL = PanelConfig(personas=[("p", "mandate")])
U, B = Severity.UGLY, Severity.BLOCKER


def _contained(call):
    """Call ``call()``; an ``Exception`` escaping it is a plain test failure.

    ``tests/engine/test_unrenderable_exception.py``'s pattern.
    """
    try:
        return call()
    except Exception as exc:  # noqa: BLE001
        name = str.__str__(type.__dict__["__name__"].__get__(type(exc), type))
        raise AssertionError(f"escaped the guard: {name}") from None


def _spawn_for(severity, verdict, repeat):
    """One finding per epoch: distinct per epoch, or the SAME one every epoch where
    ``repeat``. The masking rows (3', 8) need the same finding raised again, and the
    CONVERGED rows a panel voting YES with quorum met (one persona, quorum one): a fixture
    that cannot express either is how #150's issue body under-measured."""
    def spawn(persona, mandate, surface, epoch):  # noqa: ARG001
        e = 1 if repeat else epoch
        return PersonaReport(persona=persona, verdict=verdict, tokens=9,
                             status=PersonaStatus.CONTRIBUTED,
                             findings=[Finding(persona, f"defect {e}", severity, f"lens-{e}",
                                               "a.py", e * 100)])
    return spawn


def _raising():
    raise RuntimeError("the meter is down")


def _run(severity, verdict="NO", *, repeat=False, **seams):
    return _contained(lambda: run(
        ReviewSpec(why="w", what="x"),
        HaltingSet(max_epochs=3, stall_patience=1 if repeat else 5), PANEL,
        spawn=_spawn_for(severity, verdict, repeat), gather=lambda epoch: "s",
        parallel=False, **seams))


def _open(review_run, severity):
    return [f for f in review_run.ledger.values()
            if f.verified is not False and f.severity is severity]


def _seams(review_run):
    return [[f.seam for f in e.seam_failures] for e in review_run.epochs]


def _halts(review_run):
    return [e.halt for e in review_run.epochs]


# --- row 1: checkpoint rewrites the halt (FIRST: #31 makes checkpoint durable) -----------

def _rewrites_the_halt(review_run):
    review_run.epochs[-1].halt = HaltReason.CONVERGED


def test_row_1_a_checkpoint_cannot_rewrite_the_halt():
    """REGRESSION, row 1: on main an UGLY run with the panel voting YES halted
    ``converged``, exit 0, with the UGLY open in the ledger and nothing recorded."""
    review_run = _run(U, "YES", checkpoint=_rewrites_the_halt)

    assert review_run.halt_reason is HaltReason.ESCALATE_UGLY
    assert _halts(review_run) == [HaltReason.ESCALATE_UGLY]
    assert len(_open(review_run, U)) == 1


def test_row_1_control_a_checkpoint_that_reads():
    """GREEN ON MAIN, by design: the twin. A checkpoint that only reads changes nothing."""
    review_run = _run(U, "YES", checkpoint=lambda r: len(r.ledger))

    assert review_run.halt_reason is HaltReason.ESCALATE_UGLY
    assert _halts(review_run) == [HaltReason.ESCALATE_UGLY]


# --- row 2: scope_complete zeroes the published count -----------------------------------

def _zeroes_the_ugly_count(result):
    result.open_uglies = 0
    return True


def test_row_2_a_scope_check_cannot_rewrite_the_published_count():
    """REGRESSION, row 2, for the RECORD. The halt is GREEN ON MAIN since #150 (it reads a
    ledger snapshot), but the artefact published the overwritten ``open_uglies: 0`` beside
    ``escalate_ugly``, so #117's record of the halt's inputs was seam-writable."""
    review_run = _run(U, "YES", scope_complete=_zeroes_the_ugly_count)

    assert review_run.halt_reason is HaltReason.ESCALATE_UGLY
    assert [e.open_uglies for e in review_run.epochs] == [1]


# --- row 3': a write at epoch k masks a BLOCKER at k+1 ----------------------------------

def _refutes_at_epoch_1(result):
    if result.index == 1:
        for f in result.new_findings:
            f.__dict__["verified"] = False
    return True


def test_row_3_a_scope_check_cannot_mask_a_blocker_at_the_next_epoch():
    """REGRESSION, row 3': the same BLOCKER raised every epoch and the panel voting YES. On
    main a refutation written in place at epoch 1 converged the run at epoch 2."""
    review_run = _run(B, "YES", repeat=True, scope_complete=_refutes_at_epoch_1)

    assert review_run.halt_reason is HaltReason.STALL
    assert len(review_run.epochs) == 2
    assert len(_open(review_run, B)) == 1


def test_row_3_control_the_same_run_with_no_write():
    """GREEN ON MAIN, by design: what row 3' must equal."""
    review_run = _run(B, "YES", repeat=True)

    assert review_run.halt_reason is HaltReason.STALL
    assert len(review_run.epochs) == 2


# --- row 4: a corrupted severity raises at the next dedup -------------------------------

def _corrupts_at_epoch_1(result):
    if result.index == 1:
        for f in result.new_findings:
            f.__dict__["severity"] = "corrupt"
    return True


def test_row_4_a_scope_check_cannot_discard_the_run_through_a_finding():
    """REGRESSION, row 4: on main a ``str`` written into a BLOCKER's severity at epoch 1
    raised ``AttributeError`` at the next epoch's dedup (``key_components``), discarding a
    paid-for run: #148's invariant broken through a side effect."""
    review_run = _run(B, scope_complete=_corrupts_at_epoch_1)

    assert review_run.halt_reason is HaltReason.EPOCH
    assert len(_open(review_run, B)) == 3


# --- row 5: a gate aliasing a ledger key is blamed on reduce ----------------------------

def _aliases_a_key(result, review_run):  # noqa: ARG001
    key = next(iter(review_run.ledger))
    review_run.ledger["alias"] = review_run.ledger[key]
    return GateDecision(stop=False)


def test_row_5_a_gate_cannot_corrupt_the_ledger_and_have_reduce_blamed():
    """REGRESSION, row 5: on main the alias reached the ledger and #150's partition check
    recorded the engine's OWN identity reduce as a failed ``reduce`` at epochs 2 and 3."""
    review_run = _run(B, human_gate=_aliases_a_key)

    assert _seams(review_run) == [[], [], []]
    assert "alias" not in review_run.ledger
    assert len(review_run.ledger) == 3


# --- row 6: adjudicate rewrites the finding it is handed --------------------------------

def _downgrades(answer):
    def adjudicate(finding, surface):  # noqa: ARG001
        object.__setattr__(finding, "severity", Severity.NOTE)
        return answer
    return adjudicate


def test_row_6_an_adjudicator_cannot_downgrade_what_it_verifies():
    """REGRESSION, row 6: the loop rebuilt the ledger entry from the handed object's
    ``__dict__`` after the call, so a downgrade written there was what the ledger held, and
    an UGLY run with the panel voting YES converged. A refutation is a verdict; rewriting a
    severity is not."""
    review_run = _run(U, "YES", adjudicate=_downgrades(True))

    assert review_run.halt_reason is HaltReason.ESCALATE_UGLY
    (held,) = review_run.ledger.values()
    assert (held.severity, held.verified) == (U, True)


def test_row_6_an_adjudicator_that_fails_cannot_downgrade_either():
    """REGRESSION, row 6': a verdict that is no answer leaves the finding unadjudicated, and
    on main it was the MUTATED finding that was kept, so the run converged with the
    failure recorded."""
    review_run = _run(U, "YES", adjudicate=_downgrades(None))

    assert review_run.halt_reason is HaltReason.ESCALATE_UGLY
    assert _seams(review_run) == [["adjudicate"]]
    (held,) = review_run.ledger.values()
    assert (held.severity, held.verified) == (U, None)


def test_row_6_control_an_adjudicator_that_reads():
    """GREEN ON MAIN, by design: the twin."""
    review_run = _run(U, "YES", adjudicate=lambda f, s: f.severity is U)

    assert review_run.halt_reason is HaltReason.ESCALATE_UGLY


# --- row 7: the gate writes the per-epoch halt ------------------------------------------

def _writes_the_halt(result, review_run):  # noqa: ARG001
    result.halt = HaltReason.CONVERGED
    return GateDecision(stop=False)


def test_row_7_a_gate_cannot_rewrite_the_published_per_epoch_halt():
    """REGRESSION, row 7: on main the per-epoch halt was published as ``[converged,
    converged, epoch]`` for a run that halted ``epoch`` without converging."""
    review_run = _run(B, human_gate=_writes_the_halt)

    assert _halts(review_run) == [None, None, HaltReason.EPOCH]
    assert all(e.gate is not None for e in review_run.epochs[:2])


# --- row 8: the gate refutes the ledger in place ----------------------------------------

def _refutes_the_ledger(result, review_run):  # noqa: ARG001
    for f in review_run.ledger.values():
        f.__dict__["verified"] = False
    return GateDecision(stop=False)


def test_row_8_a_gate_cannot_refute_the_ledger_in_place():
    """REGRESSION, row 8: on main the run converged at epoch 2 with nothing recorded. This
    is the mechanism ``fakes._demo``'s gate used to retire a fixed BLOCKER; retiring one is
    #157's, on a return value."""
    review_run = _run(B, "YES", repeat=True, human_gate=_refutes_the_ledger)

    assert review_run.halt_reason is HaltReason.STALL
    assert len(_open(review_run, B)) == 1


# --- row 9': checkpoint clears the ledger -----------------------------------------------

def test_row_9_a_checkpoint_cannot_erase_the_escalations_own_finding():
    """REGRESSION, row 9': on main the run halted ``escalate_ugly`` beside an EMPTY ledger."""
    review_run = _run(U, checkpoint=lambda r: r.ledger.clear())

    assert review_run.halt_reason is HaltReason.ESCALATE_UGLY
    assert len(_open(review_run, U)) == 1


# --- rows 10 and 11: #148's own record erased -------------------------------------------

def _clears_the_failures(result):
    result.seam_failures.clear()
    return True


def test_row_10_a_scope_check_cannot_erase_the_failure_record():
    """REGRESSION, row 10: a run whose meter was down for every epoch read as one whose
    seams all answered (``[[], [], []]``)."""
    review_run = _run(B, scope_complete=_clears_the_failures, budget_spent=_raising)

    assert _seams(review_run) == [["budget_spent"]] * 3


def test_row_11_a_checkpoint_cannot_erase_the_failure_record():
    """REGRESSION, row 11: the same through ``checkpoint``."""
    review_run = _run(B, checkpoint=lambda r: r.epochs[-1].seam_failures.clear(),
                      budget_spent=_raising)

    assert _seams(review_run) == [["budget_spent"]] * 3


def test_rows_10_and_11_control_the_failure_is_recorded():
    """GREEN ON MAIN, by design: the record rows 10 and 11 must equal."""
    review_run = _run(B, budget_spent=_raising)

    assert _seams(review_run) == [["budget_spent"]] * 3


# --- the property, and its twin ---------------------------------------------------------
#
# P1: whatever a seam writes into what it is handed, the run equals the run whose seams
# only read. P2: a seam that only reads is handed a record EQUAL to the live one at that
# moment, with the same aliasing, and disjoint from it.

def _vandalise(obj, how, seen=None):
    """Write ``how`` into everything reachable from ``obj``, through every door there is."""
    seen = set() if seen is None else seen
    if id(obj) in seen:
        return
    seen.add(id(obj))
    if isinstance(obj, Finding):
        if how == "refute":
            obj.__dict__["verified"] = False
        elif how == "downgrade":
            object.__setattr__(obj, "severity", Severity.NOTE)
        elif how == "corrupt":
            obj.__dict__["severity"] = "corrupt"
        return
    if isinstance(obj, (list, tuple)):
        for item in list(obj):
            _vandalise(item, how, seen)
        if how == "clear" and isinstance(obj, list):
            obj.clear()
        return
    if isinstance(obj, dict):
        for item in list(obj.values()):
            _vandalise(item, how, seen)
        if how == "clear":
            obj.clear()
        return
    if hasattr(obj, "__dataclass_fields__"):
        for name in obj.__dataclass_fields__:
            _vandalise(obj.__dict__.get(name), how, seen)
        if how == "halt":
            if "halt" in obj.__dict__:
                obj.halt = HaltReason.CONVERGED
            if "halt_reason" in obj.__dict__:
                obj.halt_reason = HaltReason.CONVERGED
        if how == "counts" and "open_uglies" in obj.__dict__:
            obj.open_uglies = obj.open_blockers = 0


WRITES = ["halt", "counts", "refute", "downgrade", "corrupt", "clear"]


def _seam_writing(seam, how):
    """``seam`` answering exactly as ``_reading``'s does, after writing ``how``."""
    def write(*handed):
        for h in handed:
            _vandalise(h, how)
    if seam == "adjudicate":
        return lambda f, s: (write(f), True)[1]
    if seam == "reduce":
        def reduce(findings):
            # The answer is taken BEFORE the write, so what is returned is a partition of
            # what was handed: this row is about the write, not about the return value.
            answer = _identity_reduce([Finding(**f.__dict__) for f in findings])
            write(findings)
            return answer
        return reduce
    if seam == "scope_complete":
        return lambda r: (write(r), True)[1]
    if seam == "checkpoint":
        return lambda r: write(r)
    return lambda r, rr: (write(r, rr), GateDecision(stop=False))[1]


_READING = {
    "adjudicate": lambda f, s: True,
    "reduce": lambda findings: _identity_reduce(list(findings)),
    "scope_complete": lambda r: True,
    "checkpoint": lambda r: None,
    "human_gate": lambda r, rr: GateDecision(stop=False),
}

# Each shape exercises a different door: a BLOCKER panel voting NO runs all three epochs
# and reaches every seam, the gate included; an UGLY panel voting YES is the breaker's run;
# a repeated BLOCKER voting YES is the masking shape.
SHAPES = {"blocker-no": dict(severity=B), "ugly-yes": dict(severity=U, verdict="YES"),
          "repeat-yes": dict(severity=B, verdict="YES", repeat=True)}


def _shaped(shape, **seams):
    return _run(**SHAPES[shape], budget_spent=_raising, **seams)


@pytest.mark.parametrize("shape", sorted(SHAPES))
@pytest.mark.parametrize("how", WRITES)
@pytest.mark.parametrize("seam", sorted(_READING))
def test_no_write_into_what_a_seam_is_handed_reaches_the_run(seam, how, shape):
    """P1. REGRESSION for most cells, GREEN ON MAIN for some: a cell whose seam is never
    reached by that shape (the gate on an epoch-1 escalation) or whose write main's order
    already absorbed. The pair below forbids the vacuous reading, where every seam is
    handed nothing and so every write is trivially lost."""
    baseline = _shaped(shape, **_READING)
    written = _shaped(shape, **{**_READING, seam: _seam_writing(seam, how)})

    assert written == baseline


@pytest.fixture
def live(monkeypatch):
    """The engine's own ``ReviewRun``, captured as it is constructed, so a seam can be
    compared against the live record at the moment it is called."""
    made: list[ReviewRun] = []

    def construct():
        made.append(ReviewRun())
        return made[-1]
    monkeypatch.setattr(loop_module, "ReviewRun", construct)
    return made


def _graph(root):
    """The aliasing graph: every record and container reachable from ``root``, numbered by
    first visit, so two structures alias alike iff the sequences are equal."""
    order: dict[int, int] = {}
    out: list[int] = []

    def walk(obj):
        if not (isinstance(obj, (list, tuple, dict)) or hasattr(obj, "__dataclass_fields__")):
            return
        if isinstance(obj, type):
            return
        first = id(obj) not in order
        out.append(order.setdefault(id(obj), len(order)))
        if not first:
            return
        items = (obj.values() if isinstance(obj, dict) else obj
                 if isinstance(obj, (list, tuple))
                 else [obj.__dict__[n] for n in obj.__dataclass_fields__])
        for item in items:
            walk(item)
    walk(root)
    return out


def _ids(root):
    """Every record and container reachable from ``root``, by identity."""
    seen: set[int] = set()

    def walk(obj):
        if id(obj) in seen or isinstance(obj, type):
            return
        if isinstance(obj, (list, tuple, dict)) or hasattr(obj, "__dataclass_fields__"):
            seen.add(id(obj))
            items = (obj.values() if isinstance(obj, dict) else obj
                     if isinstance(obj, (list, tuple))
                     else [obj.__dict__[n] for n in obj.__dataclass_fields__])
            for item in items:
                walk(item)
    walk(root)
    return seen


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_a_seam_that_reads_is_handed_an_equal_record_with_equal_aliasing(live, shape):
    """P2, the twin: GREEN ON MAIN for equality and aliasing (main handed the live record
    itself), RED ON MAIN for disjointness. A copy that loses anything (a field, a report, the
    one-object-two-places structure a ledger finding has with ``new_findings``) turns this
    red, and so does handing an empty record, which is the vacuity probe P1 needs."""
    checked: list[str] = []
    problems: list[str] = []

    def compare(seam, handed, actual):
        # Recorded, not asserted: an assertion raised inside a seam is contained by the
        # engine's own guard and would surface only as a seam failure.
        if handed != actual:
            problems.append(f"{seam} was handed a record unequal to the live one")
        if _graph(handed) != _graph(actual):
            problems.append(f"{seam}: aliasing differs")
        if _ids(handed) & _ids(actual):
            problems.append(f"{seam} was handed part of the live record")
        checked.append(seam)

    scoped: list = []
    adjudicated: list = []

    def adjudicate(finding, surface):  # noqa: ARG001
        adjudicated.append(finding)
        return True

    def reduce(findings):
        compare("reduce", findings, list(live[0].ledger.values()))
        return _identity_reduce(list(findings))

    def scope(result):
        # The live result is not reachable until it is appended, so it is compared at the
        # checkpoint below, with the one field written between the two calls: the halt.
        scoped.append(result)
        return True

    def checkpoint(r):
        compare("checkpoint", r, live[0])
        newest = live[0].epochs[-1]
        handed = scoped[-1]
        handed.halt = newest.halt
        compare("scope_complete", handed, newest)

    def gate(result, r):
        compare("human_gate", r, live[0])
        if result is not r.epochs[-1]:
            problems.append("the gate's result is not its run's newest epoch")
        return GateDecision(stop=False)

    review_run = _shaped(shape, adjudicate=adjudicate, reduce=reduce, scope_complete=scope,
                         checkpoint=checkpoint, human_gate=gate)

    assert problems == []
    for handed in adjudicated:
        held = review_run.ledger[handed.key]
        assert handed == Finding(**{**held.__dict__, "verified": None})
        assert handed is not held
    assert adjudicated and {"checkpoint", "reduce", "scope_complete"} <= set(checked)
    if shape == "blocker-no":
        assert "human_gate" in checked


def test_each_seam_call_is_handed_its_own_copy(live):
    """A copy shared between seams would let ``checkpoint`` write what the gate then reads.
    The gate must see the live record whatever the checkpoint before it wrote. GREEN ON MAIN,
    by design: main handed both the live record, which trivially equals itself. It guards
    a copy made once per epoch and shared, which the mutation matrix turns red."""
    problems: list[str] = []

    def checkpoint(r):
        r.ledger.clear()
        r.epochs[-1].seam_failures.clear()

    def gate(result, r):  # noqa: ARG001
        if r != live[0]:
            problems.append("the gate saw what the checkpoint wrote")
        return GateDecision(stop=False)

    _shaped("blocker-no", checkpoint=checkpoint, human_gate=gate)

    assert problems == []


# --- a malformed spawn: the axis the first shape's fixtures could not express ----------
#
# Found by #155's pre-merge reviews: every fixture above builds well-formed records. The
# first shape handed what it could not copy over AS IT WAS (live); the second refused it
# at every seam, so one odd value refused every later copy of the run. By the
# maintainer's ruling of 2026-10-14 a ``spawn`` report is now copied ONCE, as it enters
# the run, and one the engine cannot copy is that persona's contained spawn failure (#25),
# whose finding carries the highest severity the report claimed. ``p`` is the malformed
# persona; ``q``, where present, is well-formed.

class _Tokens(int):
    """An ``int`` subclass, as a third-party backend might count with."""


def _malformed_spawn(severity, verdict, *, cut_finding=False, cut_report=False,
                     denied=None, tokens=9, repeat=False, healthy_q=None):
    def spawn(persona, mandate, surface, epoch):  # noqa: ARG001
        e = 1 if repeat else epoch
        f = Finding(persona, f"defect {e}", severity, f"lens-{e}", "a.py", e * 100)
        if persona == "q":
            return PersonaReport(persona="q", verdict=healthy_q, tokens=9,
                                 status=PersonaStatus.CONTRIBUTED,
                                 findings=[f] if repeat else [])
        if cut_finding:
            object.__delattr__(f, "evidence")   # still READS, through the class default
        report = PersonaReport(persona=persona, verdict=verdict, tokens=tokens,
                               status=PersonaStatus.CONTRIBUTED, findings=[f])
        if cut_report:
            del report.__dict__["turns"]
        if denied is not None:
            report.denied_tools = denied
        return report
    return spawn


def _malformed_run(spawn, *, repeat=False, panel=PANEL, **seams):
    return _contained(lambda: run(
        ReviewSpec(why="w", what="x"),
        HaltingSet(max_epochs=3, stall_patience=1 if repeat else 5), panel,
        spawn=spawn, gather=lambda epoch: "s", parallel=False, **seams))


PAIR = PanelConfig(personas=[("p", "mandate"), ("q", "mandate")])
PAIR_QUORUM_ONE = PanelConfig(personas=[("p", "mandate"), ("q", "mandate")], quorum=1)


def _refused_at_entry(review_run, persona="p"):
    return [r for e in review_run.epochs for r in e.reports
            if r.persona == persona and r.status is PersonaStatus.SPAWN_FAILED
            and "could not copy" in r.findings[0].title]


def _copy_refusals(review_run):
    return [f.detail for e in review_run.epochs for f in e.seam_failures
            if "could not copy" in f.detail]


@pytest.mark.parametrize("cut", [False, True], ids=["control", "finding-missing-a-field"])
def test_a_malformed_finding_never_reaches_adjudicate_and_its_ugly_still_escalates(cut):
    """REGRESSION for the first review's UGLY: an UGLY ``spawn`` finding missing one
    instance-dict field was handed to ``adjudicate`` live, the downgrade it wrote reached
    the ledger, and the run halted ``converged``. Now the report is refused as it enters,
    and the failure's finding is UGLY, so the breaker fires. The control is RED ON MAIN
    too: it is row 6, a downgrade written into a well-formed finding."""
    review_run = _malformed_run(_malformed_spawn(U, "YES", cut_finding=cut),
                                adjudicate=_downgrades(True))

    assert review_run.halt_reason is HaltReason.ESCALATE_UGLY
    assert [f.severity for f in review_run.ledger.values()] == [U]
    assert bool(_refused_at_entry(review_run)) is cut


def test_an_ugly_in_a_refused_report_cannot_be_voted_past():
    """REGRESSION for the ingress refusal's own first shape (measured before it shipped):
    a report refused only for an ``int`` subclass in ``tokens`` lost its UGLY, and a panel
    with a quorum below its size voted the run ``converged``; main escalated. The failure's
    finding carries the report's highest claimed severity."""
    review_run = _malformed_run(_malformed_spawn(U, "YES", tokens=_Tokens(5), healthy_q="YES"),
                                panel=PAIR_QUORUM_ONE)

    assert review_run.halt_reason is HaltReason.ESCALATE_UGLY
    assert _refused_at_entry(review_run)


def test_one_odd_value_costs_one_report_not_the_rest_of_the_run():
    """REGRESSION for the second review's BAD: an ``int`` subclass in one report's
    ``tokens`` stayed in the record and refused EVERY later ``checkpoint`` and gate copy,
    so the gate was never reached again. Refused at entry, it costs that report only.
    GREEN ON MAIN, by design: main copied nothing, so refused nothing."""
    gates: list[int] = []
    checkpoints: list[int] = []

    def gate(result, review_run):  # noqa: ARG001
        gates.append(result.index)
        return GateDecision(stop=False)

    review_run = _malformed_run(
        _malformed_spawn(B, "NO", tokens=_Tokens(5), healthy_q="NO"), panel=PAIR,
        human_gate=gate, checkpoint=lambda r: checkpoints.append(len(r.epochs)))

    assert gates == [1, 2] and checkpoints == [1, 2, 3]
    assert _copy_refusals(review_run) == []
    assert all(e.gate_failure is None for e in review_run.epochs)


def test_a_malformed_finding_cannot_end_the_run_through_adjudicate():
    """REGRESSION for the first review (R4a, #148): handed the incomplete finding live, an
    adjudicator deleting ``persona`` made the loop's rebuild raise out of ``run``."""
    def deletes(finding, surface):  # noqa: ARG001
        object.__delattr__(finding, "persona")
        return True

    review_run = _malformed_run(_malformed_spawn(B, "NO", cut_finding=True), adjudicate=deletes)

    assert review_run.halt_reason is HaltReason.NO_REVIEW
    assert _refused_at_entry(review_run)


def _refutes_through_reports(result, review_run):  # noqa: ARG001
    for report in result.reports:
        for f in report.findings:
            f.__dict__["verified"] = False
    return GateDecision(stop=False)


@pytest.mark.parametrize("cut", [False, True], ids=["control", "report-missing-a-field"])
def test_a_malformed_report_cannot_expose_the_ledgers_findings_to_the_gate(cut):
    """REGRESSION for the first review (R2): a report missing a field was handed to the gate
    live, and through ``report.findings`` it reached the ledger's OWN findings, so a
    refutation written there converged a run with the same BLOCKER raised every epoch.
    The control is RED ON MAIN too: it is row 8, through the reports."""
    review_run = _malformed_run(
        _malformed_spawn(B, "YES", cut_report=cut, repeat=True, healthy_q="YES"),
        repeat=True, panel=PAIR, human_gate=_refutes_through_reports)

    assert review_run.halt_reason is HaltReason.STALL
    assert len([f for f in _open(review_run, B) if not f.about_run]) == 1
    # The refused report's failure carries the BLOCKER it claimed, and is open too.
    assert bool([f for f in _open(review_run, B) if f.about_run]) is cut


def test_a_malformed_report_cannot_end_the_run_through_the_gate():
    """REGRESSION for the first review (R4b, #148): the gate deleted ``severity`` from a
    ledger finding it reached through a live report, and the next snapshot raised."""
    def deletes(result, review_run):  # noqa: ARG001
        for report in result.reports:
            for f in report.findings:
                object.__delattr__(f, "severity")
        return GateDecision(stop=False)

    review_run = _malformed_run(_malformed_spawn(B, "NO", cut_report=True, repeat=True,
                                                 healthy_q="NO"),
                                panel=PAIR, human_gate=deletes)

    assert review_run.halt_reason is HaltReason.EPOCH
    assert len([f for f in _open(review_run, B) if not f.about_run]) == 1


def test_a_mutable_value_spawn_supplied_never_enters_the_run():
    """REGRESSION for the first review (R3b): a ``set`` in ``denied_tools`` was shared with
    the live record, so a checkpoint could rewrite what the run publishes. Now the report
    is refused as it enters, and no value it carried is in the record."""
    def writes(review_run):
        for e in review_run.epochs:
            for report in e.reports:
                report.denied_tools.append("Write")

    review_run = _malformed_run(_malformed_spawn(B, "NO", denied={"Bash"}), checkpoint=writes)

    assert _refused_at_entry(review_run)
    assert all(type(r.denied_tools) is list and r.denied_tools == []
               for e in review_run.epochs for r in e.reports)


def test_a_backend_that_writes_a_report_after_returning_it_writes_nothing_the_run_holds():
    """REGRESSION: a backend kept a reference to the report it returned at epoch 1 and
    rewrote its finding at epoch 2. On main the run held that object, so the published
    epoch-1 record and the ledger changed after the fact; the report is now copied as it
    enters."""
    returned: list[PersonaReport] = []

    def spawn(persona, mandate, surface, epoch):  # noqa: ARG001
        for earlier in returned:
            for f in earlier.findings:
                object.__setattr__(f, "severity", Severity.NOTE)
            earlier.findings.clear()
        report = PersonaReport(persona=persona, verdict="NO", status=PersonaStatus.CONTRIBUTED,
                               findings=[Finding(persona, f"d{epoch}", B, f"c{epoch}", "a.py",
                                                 epoch * 100)])
        returned.append(report)
        return report

    review_run = _malformed_run(spawn)

    assert [len(e.reports[0].findings) for e in review_run.epochs] == [1, 1, 1]
    assert len(_open(review_run, B)) == 3


@pytest.mark.parametrize("value", [None, "a report", 7],
                         ids=["none", "str", "int"])
def test_a_spawn_that_returns_no_report_is_a_contained_spawn_failure(value):
    """REGRESSION: on main a ``spawn`` returning something that is not a report raised out
    of ``run`` at the first read of it, after the panel was paid for (#148). The entry copy
    refuses it, and it is that persona's spawn failure."""
    review_run = _malformed_run(lambda *a: value, panel=PAIR)

    assert review_run.halt_reason is HaltReason.NO_REVIEW
    assert all(r.status is PersonaStatus.SPAWN_FAILED for r in review_run.epochs[0].reports)


def test_a_spawn_that_returns_another_record_is_not_taken_for_a_report():
    """The copy rebuilds whatever record it is given, so a ``Finding`` returned as a report
    is refused by type after the copy, not read as one."""
    review_run = _malformed_run(lambda *a: Finding("p", "t", U, "c"))

    assert review_run.epochs[0].reports[0].status is PersonaStatus.SPAWN_FAILED


def test_a_foreign_asked_is_read_as_no_answer_and_refuses_no_later_copy():
    """REGRESSION for the second review's BAD: ``asked`` is stored un-coerced by #131's
    ruling, so a gate answering ``asked=object()`` once refused every later copy of the
    run. The copy hands a seam ``None``, which is how the publisher resolves it; the
    record itself keeps what #131 says it keeps. GREEN ON MAIN, by design: main copied
    nothing, so refused nothing."""
    odd = object()
    asked_seen: list[object] = []

    def gate(result, review_run):  # noqa: ARG001
        return GateDecision(stop=False, asked=odd if result.index == 1 else True)

    def checkpoint(review_run):
        asked_seen.extend(e.gate.asked for e in review_run.epochs if e.gate is not None)

    review_run = _malformed_run(_malformed_spawn(B, "NO"), human_gate=gate,
                                checkpoint=checkpoint)

    assert _copy_refusals(review_run) == []
    assert all(e.gate_failure is None for e in review_run.epochs)
    assert asked_seen == [None, None, True]
    assert review_run.epochs[0].gate.asked is odd


# --- refusing runs nothing the refused value carries (the second review) ---------------

def test_refusing_a_value_cannot_run_its_metaclass_and_mask_an_ugly():
    """REGRESSION for the second pre-merge review's UGLY, introduced by the first repair:
    the refusal named the value's type through ``type_name``, which evaluates the
    metaclass's ``__name__``. A payload whose ``__name__`` downgraded the UGLY ``spawn``
    still held ran INSIDE the copy, and the breaker never fired; main escalated on the same
    input, because nothing there named the type. The refusal now names nothing it refuses.
    GREEN ON MAIN, by design, for that reason."""
    kept: list[Finding] = []
    calls: list[str] = []

    class _Meta(type):
        @property
        def __name__(cls):  # noqa: N805
            calls.append("__name__")
            for f in kept:
                object.__setattr__(f, "severity", Severity.NOTE)
            return "Inert"

    class _Payload(metaclass=_Meta):
        pass

    def spawn(persona, mandate, surface, epoch):  # noqa: ARG001
        f = Finding(persona, "t", U, "c", "a.py", 1, evidence=_Payload())
        kept.append(f)
        return PersonaReport(persona=persona, verdict="YES", status=PersonaStatus.CONTRIBUTED,
                             findings=[f])

    review_run = _malformed_run(spawn, adjudicate=lambda f, s: True)

    assert calls == []
    assert review_run.halt_reason is HaltReason.ESCALATE_UGLY


def test_a_foreign_key_in_an_instance_dict_has_no_code_run_by_the_copy():
    """REGRESSION for the second review: a key whose hash matches a field name had its
    ``__eq__`` run by the copy's field lookup. Keys are now checked to be exact ``str``
    before any lookup, and the report is refused as it enters."""
    calls: list[str] = []
    armed: list[bool] = []   # inserting the key compares it with "evidence" in SPAWN itself

    class _Key:
        def __hash__(self):
            return hash("evidence")

        def __eq__(self, other):
            if armed:
                calls.append("__eq__")
            return False

    def spawn(persona, mandate, surface, epoch):  # noqa: ARG001
        f = Finding(persona, "t", B, f"c{epoch}", "a.py", epoch)
        armed.clear()
        f.__dict__[_Key()] = "x"
        armed.append(True)
        return PersonaReport(persona=persona, verdict="NO", status=PersonaStatus.CONTRIBUTED,
                             findings=[f])

    review_run = _malformed_run(spawn, adjudicate=lambda f, s: True)

    assert calls == []
    assert review_run.halt_reason is HaltReason.NO_REVIEW
    assert _refused_at_entry(review_run)


# --- a refused report's UGLY survives however it is held (the third review) ----------

class _List(list):
    pass


class _InstanceDict(dict):
    pass


def _refused_ugly(shape):
    """``p`` returns one well-formed UGLY in a report the engine refuses; ``q`` votes YES."""
    def spawn(persona, mandate, surface, epoch):  # noqa: ARG001
        if persona == "q":
            return PersonaReport(persona="q", verdict="YES", status=PersonaStatus.CONTRIBUTED)
        f = Finding(persona, "t", U, "c", "a.py", 1)
        report = PersonaReport(persona=persona, verdict="YES", status=PersonaStatus.CONTRIBUTED,
                               findings=[f])
        if shape == "list-subclass":
            report.findings = _List([f])
        elif shape == "tuple":
            report.findings = (f,)
            report.denied_tools = {"Bash"}
        elif shape == "dict-subclass":
            object.__setattr__(f, "__dict__", _InstanceDict(f.__dict__))
            report.denied_tools = {"Bash"}
        else:
            report.tokens = _Tokens(5)
        return report
    return spawn


@pytest.mark.parametrize("shape", ["tokens", "list-subclass", "tuple", "dict-subclass"])
def test_a_refused_reports_ugly_escalates_however_its_findings_are_held(shape):
    """REGRESSION for the third review's UGLY: the claimed severity was read only from an
    exact ``list`` of findings with exact instance dicts, and returned NOTE otherwise, so a
    quorum-below-panel run CONVERGED on an UGLY main escalated (all but ``tokens``)."""
    review_run = _malformed_run(_refused_ugly(shape), panel=PAIR_QUORUM_ONE)

    assert review_run.halt_reason is HaltReason.ESCALATE_UGLY
    assert _refused_at_entry(review_run)


def test_an_adjudicator_cannot_refute_a_refused_reports_ugly():
    """REGRESSION for the third review's second UGLY: the refused report's failure is an
    UGLY naming no file, and was handed to ``adjudicate``; one checking claims against
    source refuted it and the run converged. An instrument finding is never adjudicated:
    on main none is above NOTE, so none ever was."""
    seen: list[bool] = []

    def adjudicate(finding, surface):  # noqa: ARG001
        seen.append(finding.about_run)
        return finding.file is not None

    review_run = _malformed_run(_refused_ugly("tokens"), panel=PAIR_QUORUM_ONE,
                                adjudicate=adjudicate)

    assert review_run.halt_reason is HaltReason.ESCALATE_UGLY
    assert True not in seen
