# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""A guard that renders its own diagnosis must not be ended by it (#142).

#85, #25 and #129 put a guard round each injected seam so that a fallible black
box cannot discard a run already spawned and paid for. Each guard then rendered
the exception INSIDE its own ``except`` block — ``f"{type(exc).__name__}: {exc}"``,
and ``repr(exc)`` for spawn's evidence — so an exception that could not be
rendered raised from inside the handler and took the whole ``ReviewRun`` with it:
the container ended by the thing it contained.

No trigger in the shipped backend is known to reach this since #141. The case is
#89's premise: the engine may assume nothing about what a seam raises.

**The class is wider than "``__str__`` raises"**, and every shape below was
measured escaping on ``main`` at ``7c82e66``: a ``__str__`` that returns a
non-``str``; a plain exception whose ARGUMENT has a broken ``__repr__`` (the
default ``__str__`` falls back to it); a metaclass ``__name__`` that raises; a
``__name__`` assigned a ``str`` subclass whose ``__format__`` raises (the C-level
slot hands the subclass back); and, at spawn only, a ``repr`` that returns a
``str`` subclass whose ``__len__`` raises inside ``bounded_diagnosis``.

**What each test is.** The seam rows and the helper's hostile rows are the
regression. Every test green on ``main`` is a GUARD and says so in its own
docstring — measured with a shim reproducing today's expressions: the
``KeyboardInterrupt`` rows (Ctrl-C must still stop a run); the healthy-exception
rows, the captured spawn record and the metaclass-renamed exception (the mirror
image of the defect — a healthy exception rendering differently from today is the
fix firing on a healthy run); the hostile-subclass MESSAGE row; and the
``repr-returns-hostile`` rows at gather and the gate, which are safe on main only
by accident.
"""

from __future__ import annotations

import pytest

from kuang.engine import (GateDecision, HaltingSet, HaltReason, PanelConfig,
                          PersonaReport, PersonaStatus, ReviewSpec, run)
from kuang.engine.findings import render_exception, repr_exception, type_name

# --- the hostile shapes -----------------------------------------------------


class StrRaises(Exception):
    def __str__(self):
        raise ValueError("str exploded")


class StrReturnsInt(Exception):
    def __str__(self):
        return 42


class BrokenRepr:
    def __repr__(self):
        raise ValueError("repr exploded")


class Meta(type):
    @property
    def __name__(cls):
        raise ValueError("name exploded")


class MetaName(Exception, metaclass=Meta):
    pass


class HostileStr(str):
    """Every method downstream code calls on a diagnosis, made to raise."""

    def __str__(self):
        raise ValueError("str exploded")

    def __format__(self, spec):
        raise ValueError("format exploded")

    def __len__(self):
        raise ValueError("len exploded")

    def __getitem__(self, key):
        raise ValueError("getitem exploded")

    def split(self, *args, **kwargs):
        raise ValueError("split exploded")


class AssignedName(Exception):
    pass


AssignedName.__name__ = HostileStr("AssignedName")


class ReprReturnsHostile(Exception):
    def __repr__(self):
        return HostileStr("ReprReturnsHostile('kept')")


class StrReturnsHostile(Exception):
    def __str__(self):
        return HostileStr("the message survived")


class SecondaryHostile(Exception, metaclass=Meta):
    """A failure whose own rendering is hostile: recursing into it reopens #142."""

    def __str__(self):
        raise ValueError("secondary exploded")


class RaisesHostileFailure(Exception):
    def __str__(self):
        raise SecondaryHostile()

    def __repr__(self):
        raise SecondaryHostile()


class NothingRenders(Exception, metaclass=Meta):
    """Name, message and repr all hostile, so the FALLBACK's own name lookup is
    the one under test. Without it a ``type(exc).__name__`` fallback survived
    the mutation matrix: the metaclass shape alone renders, and never reaches
    the fallback."""

    def __str__(self):
        raise ValueError("str exploded")

    def __repr__(self):
        raise ValueError("repr exploded")


class StrRaisesInterrupt(Exception):
    def __str__(self):
        raise KeyboardInterrupt


SHAPES = {
    "str-raises": StrRaises,
    "str-returns-int": StrReturnsInt,
    "arg-repr-raises": lambda: Exception(BrokenRepr()),
    "metaclass-name": MetaName,
    "assigned-name": AssignedName,
    "repr-returns-hostile": ReprReturnsHostile,
    "failure-is-hostile": RaisesHostileFailure,
    "nothing-renders": NothingRenders,
}

# --- driving the engine -----------------------------------------------------


def _contained(call):
    """Call ``call()``; an ``Exception`` escaping it is a plain test failure.

    Not decoration. Measured on main, an escaping hostile exception does not
    merely fail a test: pytest's own report renders it, and the run ends in an
    INTERNALERROR — the defect reproduced one layer up, in the test runner. The
    escape is named by the C-level route and nothing else of it is rendered;
    ``from None`` keeps pytest from rendering the context chain.
    """
    try:
        return call()
    except Exception as exc:  # noqa: BLE001
        name = str.__str__(type.__dict__["__name__"].__get__(type(exc), type))
        raise AssertionError(f"escaped the guard: {name}") from None


PANEL = PanelConfig(personas=[("first", "mandate"), ("second", "mandate")])


def _drive(seam: str, make):
    """Run two epochs with ``seam`` raising ``make()``; gather fails at epoch 2.

    Epoch 2 for ``gather`` because its guard re-raises when no epoch has
    completed (#18's doctrine): containment is only observable after one has.
    """
    def gather(epoch):
        if seam == "gather" and epoch == 2:
            raise make()
        return "diff"

    def spawn(persona, mandate, surface, epoch):  # noqa: ARG001
        if seam == "spawn" and persona == "first":
            raise make()
        return PersonaReport(persona=persona, verdict="NO")

    def gate(result, review_run):  # noqa: ARG001
        if seam == "gate":
            raise make()
        return GateDecision(stop=False, asked=True)

    return _contained(lambda: run(
        ReviewSpec(why="w", what="x"), HaltingSet(max_epochs=2), PANEL,
        spawn=spawn, gather=gather, human_gate=gate, parallel=False))


@pytest.mark.parametrize("shape", SHAPES)
def test_gather_contains_an_unrenderable_exception(shape):
    """The ``repr-returns-hostile`` row is a GUARD, green on main: gather never
    calls ``repr``, and an f-string WITH literal parts yields an exact ``str``.
    Safe by accident — a bare ``f"{exc}"`` keeps the subclass — so it is pinned."""
    review_run = _drive("gather", SHAPES[shape])

    assert review_run.halt_reason is HaltReason.SURFACE_LOST
    assert len(review_run.epochs) == 1
    failure = review_run.surface_failure
    assert type(failure.detail) is str
    assert failure.detail_chars == len(failure.detail)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("parallel", [False, True])
def test_spawn_contains_an_unrenderable_exception(shape, parallel):
    def spawn(persona, mandate, surface, epoch):  # noqa: ARG001
        if persona == "first":
            raise SHAPES[shape]()
        return PersonaReport(persona=persona, verdict="NO")

    review_run = _contained(lambda: run(
        ReviewSpec(why="w", what="x"), HaltingSet(max_epochs=2), PANEL,
        spawn=spawn, gather=lambda epoch: "diff", parallel=parallel))

    assert review_run.halt_reason is HaltReason.EPOCH
    report = review_run.epochs[0].reports[0]
    assert report.status is PersonaStatus.SPAWN_FAILED
    (finding,) = report.findings
    assert type(finding.title) is str
    assert type(finding.evidence) is str


@pytest.mark.parametrize("shape", SHAPES)
def test_the_gate_contains_an_unrenderable_exception(shape):
    """The ``repr-returns-hostile`` row is a GUARD, green on main, for the reason
    the gather test gives."""
    review_run = _drive("gate", SHAPES[shape])

    assert review_run.halt_reason is HaltReason.EPOCH
    assert len(review_run.epochs) == 2
    failure = review_run.epochs[0].gate_failure
    assert type(failure.detail) is str
    assert failure.detail_chars == len(failure.detail)


@pytest.mark.parametrize("seam", ["gather", "spawn", "gate"])
def test_ctrl_c_raised_while_rendering_still_stops_the_run(seam):
    """GUARD, green on main. A ``KeyboardInterrupt`` from ``__str__`` cannot be
    told from an operator's Ctrl-C, and every guard's doctrine is ``Exception``,
    never ``BaseException``. Kills a helper that catches ``BaseException``."""
    with pytest.raises(KeyboardInterrupt):
        _drive(seam, StrRaisesInterrupt)


def test_ctrl_c_raised_while_rendering_crosses_the_thread_pool():
    """GUARD, green on main. Spawn's guard runs inside a worker thread, so the
    serial row above does not show the interrupt surviving ``pool.map``."""
    def spawn(persona, mandate, surface, epoch):  # noqa: ARG001
        if persona == "first":
            raise StrRaisesInterrupt()
        return PersonaReport(persona=persona, verdict="NO")

    with pytest.raises(KeyboardInterrupt):
        run(ReviewSpec(why="w", what="x"), HaltingSet(max_epochs=2), PANEL,
            spawn=spawn, gather=lambda epoch: "diff", parallel=True)


def test_the_records_name_the_type_that_could_not_be_rendered():
    """What each seam's record says, end to end: never blank, never generic."""
    expected = "StrRaises: <unrenderable: ValueError raised while rendering it>"

    assert _drive("gather", StrRaises).surface_failure.detail == expected
    assert _drive("gate", StrRaises).epochs[0].gate_failure.detail == expected
    (finding,) = _drive("spawn", StrRaises).epochs[0].reports[0].findings
    assert finding.title == f"persona failed: {expected}"
    # Only ``__str__`` is hostile, so the repr renders, and is kept as rendered.
    assert finding.evidence == "StrRaises()"


def test_a_gate_returning_an_object_with_a_hostile_type_name_is_reported_as_that():
    """``loop.py``'s "gate returned X" runs inside the gate's own ``try``. Measured
    on main, a hostile name there did not escape; it was recorded as the SECONDARY
    fault, ``ValueError: name exploded``, as though the gate had raised it."""
    class Returned(metaclass=Meta):
        pass

    review_run = _contained(lambda: run(
        ReviewSpec(why="w", what="x"), HaltingSet(max_epochs=2), PANEL,
        spawn=lambda p, m, s, e: PersonaReport(persona=p, verdict="NO"),
        gather=lambda epoch: "diff",
        human_gate=lambda result, review_run: Returned(),
        parallel=False))

    assert (review_run.epochs[0].gate_failure.detail
            == "gate returned Returned, not GateDecision")


# --- the helper -------------------------------------------------------------


class OwnFormat(Exception):
    """Healthy, and rendered through its own ``__format__``: the f-string the
    guards used called ``format``, not ``str``, so the helper must too."""

    def __format__(self, spec):
        return "formatted by the exception itself"


HEALTHY = [
    RuntimeError("RuntimeError: boom"),
    ValueError("invalid literal for int() with base 10: '~120'"),
    OSError(32, "Broken pipe"),
    UnicodeEncodeError("ascii", "—", 0, 1, "ordinal not in range(128)"),
    EOFError(),
    KeyError("missing"),
    Exception("multi\nline\tmessage"),
    OwnFormat("message"),
]


@pytest.mark.parametrize("exc", HEALTHY, ids=lambda e: type(e).__name__)
def test_a_healthy_exception_renders_exactly_as_it_did(exc):
    """GUARD, green on main: the mirror image of the defect. Each expected value
    is the expression the guards used before #142, so any byte of difference is
    the fix changing the healthy path."""
    assert _contained(lambda: render_exception(exc)) == \
        f"{type(exc).__name__}: {exc}"
    assert _contained(lambda: repr_exception(exc)) == repr(exc)
    assert _contained(lambda: type_name(exc)) == type(exc).__name__


def test_the_captured_spawn_record_does_not_move():
    """GUARD, green on main. The record ``captures/replay/after-141/
    spawn_failed.txt`` carries, to the byte."""
    exc = RuntimeError("RuntimeError: boom")

    assert f"persona failed: {_contained(lambda: render_exception(exc))}" == \
        "persona failed: RuntimeError: RuntimeError: boom"
    assert _contained(lambda: repr_exception(exc)) == "RuntimeError('RuntimeError: boom')"
    assert len(_contained(lambda: repr_exception(exc))) == 34


@pytest.mark.parametrize("shape", SHAPES)
def test_the_helper_always_returns_an_exact_str(shape):
    exc = SHAPES[shape]()

    for helper in (render_exception, repr_exception, type_name):
        rendered = _contained(lambda: helper(exc))  # noqa: B023
        assert type(rendered) is str


def test_the_fallback_says_what_failed_and_what_raised():
    """Every part of the fallback is pinned: the type that could not be rendered,
    that rendering failed, and the type of what was raised doing it."""
    assert _contained(lambda: render_exception(StrRaises())) == \
        "StrRaises: <unrenderable: ValueError raised while rendering it>"
    assert _contained(lambda: repr_exception(Exception(BrokenRepr()))) == \
        "Exception: <unrenderable: ValueError raised while rendering it>"
    assert _contained(lambda: render_exception(StrReturnsInt())) == \
        "StrReturnsInt: <unrenderable: TypeError raised while rendering it>"


def test_the_secondary_exception_contributes_only_its_type_name():
    """Never its message: it can be hostile too, and rendering it reopens the
    defect one level down. Its name comes by the same C-level route."""
    rendered = _contained(lambda: render_exception(RaisesHostileFailure()))

    assert rendered == ("RaisesHostileFailure: "
                        "<unrenderable: SecondaryHostile raised while rendering it>")
    assert "exploded" not in _contained(lambda: render_exception(StrRaises()))


def test_the_type_name_survives_a_metaclass():
    """Kills a ``type(exc).__name__`` fallback: the metaclass property raises."""
    assert _contained(lambda: type_name(MetaName())) == "MetaName"
    assert _contained(lambda: render_exception(MetaName())) == "MetaName: "


def test_the_fallback_names_the_type_when_nothing_else_renders():
    """Kills a ``type(exc).__name__`` FALLBACK, which is what the issue first
    proposed: it raises where the message already could not be rendered."""
    expected = "NothingRenders: <unrenderable: ValueError raised while rendering it>"

    assert _contained(lambda: render_exception(NothingRenders())) == expected
    assert _contained(lambda: repr_exception(NothingRenders())) == expected


def test_a_type_name_assigned_a_str_subclass_is_coerced():
    """The C-level slot accepts a ``str`` subclass and hands it back unchanged."""
    name = _contained(lambda: type_name(AssignedName()))

    assert type(name) is str
    assert name == "AssignedName"


class FormatsHostile(str):
    """A name that formats cleanly — to a ``str`` subclass that is hostile."""

    def __format__(self, spec):
        return HostileStr("FormatsHostileName")


class FormatsHostileName(Exception):
    pass


FormatsHostileName.__name__ = FormatsHostile("FormatsHostileName")


def test_a_type_name_formatted_to_a_str_subclass_is_coerced():
    """The first route — the name as the guards rendered it — succeeds here, and
    ``format`` passes the subclass through: kills dropping its coercion."""
    name = _contained(lambda: type_name(FormatsHostileName()))

    assert type(name) is str
    assert name == "FormatsHostileName"


class Renamer(type):
    @property
    def __name__(cls):
        return "Renamed"


class RenamedByItsMetaclass(Exception, metaclass=Renamer):
    pass


def test_a_name_a_metaclass_supplies_healthily_is_kept():
    """GUARD, green on main: the mirror image of the defect. A metaclass property
    that RETURNS is healthy, and today's f-string used its value. Consulting the
    C-level slot first — the first shape of this fix, caught by the pre-merge
    lens pass — rendered ``RenamedByItsMetaclass`` instead."""
    exc = RenamedByItsMetaclass("m")

    assert _contained(lambda: render_exception(exc)) == \
        f"{type(exc).__name__}: {exc}" == "Renamed: m"
    assert _contained(lambda: type_name(exc)) == "Renamed"


def test_a_message_returned_as_a_hostile_str_subclass_is_kept():
    """GUARD, green on main: today's f-string formats the EXCEPTION, never the
    string it returns. The helper must keep that: the text is real and is kept,
    and only the subclass is shed. Kills dropping the coercion between
    ``format`` and the interpolation, which calls the hostile ``__format__``
    and turns a renderable message into the fallback."""
    assert _contained(lambda: render_exception(StrReturnsHostile())) == \
        "StrReturnsHostile: the message survived"
    assert _contained(lambda: repr_exception(ReprReturnsHostile())) == \
        "ReprReturnsHostile('kept')"


def test_the_fallback_is_deterministic():
    """Kills an ``object.__repr__`` fallback, which embeds a memory address and
    would break byte-identical captures ("the engine is deterministic")."""
    first = _contained(lambda: repr_exception(Exception(BrokenRepr())))
    second = _contained(lambda: repr_exception(Exception(BrokenRepr())))

    assert first == second
    assert "0x" not in first


def test_ctrl_c_raised_while_rendering_escapes_the_helper():
    """GUARD, green on main: the helper catches ``Exception``, never more."""
    with pytest.raises(KeyboardInterrupt):
        _contained(lambda: render_exception(StrRaisesInterrupt()))
