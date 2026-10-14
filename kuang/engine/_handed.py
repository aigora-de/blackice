# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""What the engine hands an injected seam: a copy it built, never the live record (#155).

#129 settled the rule for a seam's RETURN value: the engine never reads an unvalidated
value off it outside the guard. This module is the same rule applied to its SIDE EFFECTS.
A seam's only channel into the run is its return value, so every record a seam is handed
is a copy, and nothing it does to that object reaches the engine's decisions or its
record. Measured on ``954cdc1`` before this existed: a ``checkpoint`` rewriting
``epochs[-1].halt`` converged an UGLY run with nothing recorded, and a ``scope_complete``
or ``checkpoint`` clearing ``seam_failures`` erased #148's own record of a meter that was
down at every epoch.

**Built by the engine, from known fields, running no foreign code.** ``copy.deepcopy``
would call ``__deepcopy__`` and ``__reduce_ex__`` on every object in the graph, including
whatever ``spawn`` returned, which is foreign code outside any guard (#129's subclass
lesson). So:

* a record is matched by ``issubclass(type(value), base)``, never ``isinstance``, which can
  consult a foreign ``__class__``;
* its fields are read from an instance dict that must be an EXACT ``dict`` keyed by EXACT
  ``str``, through the BASE class's own descriptor, so no property, ``__getattr__``,
  ``__getattribute__``, dict-subclass hook or foreign key's ``__eq__`` is called;
* it is rebuilt as the BASE class without calling ``__init__``, so a subclass's extra
  state is dropped, as #129's ``GateDecision`` is re-made;
* a container is copied at an exact ``list``, ``tuple`` or ``dict`` type;
* every other value must be an immutable scalar of an exact type (``str``, ``int``,
  ``float``, ``bool``, ``None``, or one of the engine's own enums), so sharing it shares
  nothing a seam can write.

Nothing about a refused value is rendered into the refusal: its detail is a fixed string
naming the engine's own record class, so refusing runs no code the value carries.

**Copied, or refused: never handed live.** Anything else (a record missing a field, a set,
a container subclass, a foreign object, a structure nested deeper than ``MAX_DEPTH``) raises
``UncopiableRecord``. The first shape of this module handed such a value over AS IT WAS,
which is to say live, and the pre-merge review measured that reopening every door it
shuts: a ``spawn`` finding missing one field was handed to ``adjudicate`` live, and an UGLY
run converged with nothing recorded. Every call site is inside the seam's guard, so a
refusal is that seam's recorded failure and its UNKNOWN outcome (#148), which is the
conservative one for every seam: an unadjudicated finding stays open, the reduce is the
identity, an unknown scope cannot support ``CONVERGED``, nothing is persisted, and the gate
continues. The failure's detail names the COPY, so the record does not blame a callable that
never ran. The depth bound is the engine's own constant, so where a copy is refused does not
depend on the interpreter's recursion limit.

**Faithful, aliasing included.** One memo, keyed by identity, per copy: an object reached
twice is copied once, so a ledger ``Finding`` that is also in ``new_findings`` and a
cluster's ``members`` is still one object in the copy. Nothing shipped relies on that, as
every match is by ``key``, but a copy that changed it would hand a reading seam a different
structure from today's. A tuple enters the memo only once its members are built, so a cycle
running through a tuple would be copied unaliased; no record can form one.

**At entry, too (the maintainer's ruling of 2026-10-14).** ``spawn``'s report is copied
ONCE, as it enters the run, so the record holds only values the engine built, and a
backend that kept a reference to a report it returned and writes it later writes nothing
the run holds. A report the engine refuses there is that persona's contained spawn failure
(#25), whose finding carries the highest severity the report claimed
(``claimed_severity``), so an UGLY in it still trips the breaker. Refused at every SEAM
instead, which was this module's second shape, one odd value stayed in the record and
refused every later copy of the run, so the gate was never reached again. The one field the
engine itself keeps a foreign value in, ``GateDecision.asked`` (#131), is copied as the
``None`` the publisher reads it as, for the same reason.

**What it does not cover, stated.** A seam that writes to the WORLD rather than to the run
(the repository, the filesystem, the network) is the permission policy's (#76, #79, #96). A
refused report's OTHER findings are lost with it, as a spawn that raises loses its findings
(#25): what survives is the failure, at the highest severity claimed. A seam that keeps its
COPY and writes it later writes nothing the engine reads. ``adjudicate`` is
still handed the live ``surface``: it is opaque to the engine (``ReviewSurface = object``),
the run does not store it, and nothing the engine decides reads it.
"""

from __future__ import annotations

from dataclasses import fields
from typing import TypeVar

from .findings import (Cluster, EpochResult, Finding, GateFailure, PersonaReport,
                       PersonaStatus, ReviewRun, SeamFailure, Severity, Suppression,
                       SurfaceFailure)
from .halting import HaltReason
from .protocols import GateDecision

_T = TypeVar("_T")

# Every record a seam can be handed, or reach from one it is handed.
_RECORDS: tuple[type, ...] = (ReviewRun, EpochResult, PersonaReport, Finding, Cluster,
                              Suppression, SeamFailure, GateFailure, SurfaceFailure,
                              GateDecision)
_FIELDS = {base: tuple(f.name for f in fields(base)) for base in _RECORDS}
_INSTANCE_DICT = {base: base.__dict__["__dict__"] for base in _RECORDS}
# Shared, never copied: exact types whose instances nothing can write. Compared by
# identity, never hashed: a foreign metaclass can define ``__hash__``.
_SCALARS: tuple[type, ...] = (str, int, float, bool, type(None), Severity, HaltReason,
                              PersonaStatus)
# Far above any record the engine builds (a run is eight levels deep to a finding's field).
MAX_DEPTH = 64


class UncopiableRecord(Exception):
    """The engine could not build the copy a seam is handed, so the seam is not called."""


def handed(value: _T) -> _T:
    """A copy of ``value`` sharing nothing a seam can write with the live record.

    Raises ``UncopiableRecord`` where no such copy exists. The caller is always inside the
    seam's guard, and the diagnosis says it was the copy, not the seam, that failed.
    """
    try:
        return _copied(value, {}, 0)
    except UncopiableRecord:
        raise
    except Exception:  # noqa: BLE001
        # Reached only by a defect in this module, or by a backend thread writing what it
        # returned while it is being copied ("dict changed size during iteration"). Either
        # way it is recorded as the COPY failing, never against a seam that was not
        # called, and with a fixed string, so nothing the value carries is rendered.
        raise _refused("an unexpected error") from None


def _refused(why: str) -> UncopiableRecord:
    return UncopiableRecord(f"the engine could not copy a record: {why}")


def _copied(value, memo: dict[int, object], depth: int):
    kind = type(value)
    if any(kind is scalar for scalar in _SCALARS):
        return value
    found = memo.get(id(value))
    if found is not None:
        return found
    if depth > MAX_DEPTH:
        raise _refused(f"nested deeper than {MAX_DEPTH}")
    depth += 1
    if kind is list:
        out: list = []
        memo[id(value)] = out
        out.extend(_copied(item, memo, depth) for item in value)
        return out
    if kind is dict:
        table: dict = {}
        memo[id(value)] = table
        for k, v in value.items():
            table[_copied(k, memo, depth)] = _copied(v, memo, depth)
        return table
    if kind is tuple:
        memo[id(value)] = copied = tuple(_copied(item, memo, depth) for item in value)
        return copied
    base = next((b for b in _RECORDS if issubclass(kind, b)), None)
    if base is None:
        # No type NAME, deliberately: ``type_name`` evaluates ``type(v).__name__``
        # through the metaclass (#142's fidelity rule), which is foreign code. The
        # second pre-merge review measured a metaclass ``__name__`` downgrading an UGLY
        # from inside this very message, and the breaker never fired.
        raise _refused("a value is neither a record nor an immutable value")
    state = _INSTANCE_DICT[base].__get__(value, base)
    if type(state) is not dict:
        raise _refused(f"a {base.__name__}'s attributes are not a plain dict")
    # Every key an exact ``str`` BEFORE any lookup: a foreign key whose hash matches a
    # field name would otherwise have its ``__eq__`` run by ``name in state``. The name
    # interpolated here and below is the engine's OWN class's, never the value's.
    if not all(type(key) is str for key in state):
        raise _refused(f"a {base.__name__}'s attributes have a key that is not a str")
    names = _FIELDS[base]
    missing = [name for name in names if name not in state]
    if missing:
        raise _refused(f"a {base.__name__} is missing {len(missing)} field(s)")
    record = object.__new__(base)
    memo[id(value)] = record
    for name in names:
        field_value = state[name]
        if base is GateDecision and name == "asked" and not _is_answer(field_value):
            # The one field the engine keeps a foreign value in, by #131's ruling: a
            # value the tool cannot read is not a measurement, and the publisher
            # resolves it by IDENTITY to "no answer". The copy hands a seam exactly that
            # reading rather than the object, which nothing in the engine reads, so a
            # gate's odd answer at one epoch cannot refuse every later copy of the run.
            field_value = None
        object.__setattr__(record, name, _copied(field_value, memo, depth))
    return record


def claimed_severity(returned: object) -> Severity:
    """The highest severity a report the engine REFUSED claimed, failing CLOSED.

    A refused report is a contained spawn failure (#25), and its findings never enter the
    run. Without this, an UGLY in a report refused only for an odd value elsewhere (an
    ``int`` subclass in ``tokens``, a ``set`` of denied tools) was lost, and a panel with a
    quorum below its size could vote the run ``converged`` where main had escalated. So
    the failure's own finding carries the highest severity the report claimed.

    **Fails closed, by the maintainer's ruling of 2026-10-14.** A claim whose severity
    cannot be read is UGLY, never NOTE: whether an UGLY is open is then UNKNOWN, and #148's
    rule is that an unknown is never read as a measurement, so the breaker fires rather
    than a quorum voting past it. The third pre-merge review measured the first shape of
    this function returning NOTE for a ``findings`` that was a list subclass or a tuple and
    for a finding whose instance dict was a ``dict`` subclass, each converging a run main
    escalated. A report that claims nothing (no findings, or a return that is not a report
    at all) is NOTE. No healthy run reaches this: the shipped backend's output is never
    refused (measured over 42 reports).

    Read with the copy's own discipline, running nothing the value carries: types by
    ``type()`` and ``issubclass``, instance dicts through the BASE class's descriptor,
    sequences through ``list.__iter__`` / ``tuple.__iter__`` and dicts through
    ``dict.items`` (the underlying storage, so a subclass's own hooks are not called),
    keys compared only once they are exact ``str``, and a severity by identity.
    """
    try:
        if not issubclass(type(returned), PersonaReport):
            return Severity.NOTE
        state = _INSTANCE_DICT[PersonaReport].__get__(returned, PersonaReport)
        found, findings = _stored(state, "findings")
        if not found:
            return Severity.UGLY
        if issubclass(type(findings), list):
            claims = list(list.__iter__(findings))
        elif issubclass(type(findings), tuple):
            claims = list(tuple.__iter__(findings))
        else:
            return Severity.UGLY
        highest = Severity.NOTE
        for f in claims:
            if not issubclass(type(f), Finding):
                return Severity.UGLY
            found, severity = _stored(_INSTANCE_DICT[Finding].__get__(f, Finding), "severity")
            if not found or not any(severity is s for s in Severity):
                return Severity.UGLY
            if severity > highest:
                highest = severity
        return highest
    except Exception:  # noqa: BLE001
        # Reached only by a defect here, or by a backend thread writing what it returned
        # while it is read ("dict changed size during iteration"). Closed, as above.
        return Severity.UGLY


def _stored(state: object, name: str) -> tuple[bool, object]:
    """``state[name]`` from an instance dict, read through ``dict.items``: no hook runs."""
    if not issubclass(type(state), dict):
        return False, None
    for key, value in dict.items(state):
        if type(key) is str and key == name:
            return True, value
    return False, None


def _is_answer(value: object) -> bool:
    """``True``, ``False`` or ``None``, by identity: the three values ``asked`` can mean."""
    return value is True or value is False or value is None
