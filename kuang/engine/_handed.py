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
* its fields are read from an instance dict that must be an EXACT ``dict``, through the
  BASE class's own descriptor, so no property, ``__getattr__``, ``__getattribute__`` or
  dict-subclass hook a subclass defines is called;
* it is rebuilt as the BASE class without calling ``__init__``, so a subclass's extra
  state is dropped, as #129's ``GateDecision`` is re-made;
* a container is copied at an exact ``list``, ``tuple`` or ``dict`` type;
* every other value must be an immutable scalar of an exact type (``str``, ``int``,
  ``float``, ``bool``, ``None``, or one of the engine's own enums), so sharing it shares
  nothing a seam can write.

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

**What it does not cover, stated.** A seam that writes to the WORLD rather than to the run
(the repository, the filesystem, the network) is the permission policy's (#76, #79, #96). A
value ``spawn`` returned is a return value (#25): a backend that keeps a reference to a
report it returned and writes it later is that door, not this one, and so is a finding so
malformed that the engine's own reads of it raise before any seam is called. A seam that
keeps its COPY and writes it later writes nothing the engine reads. ``adjudicate`` is
still handed the live ``surface``: it is opaque to the engine (``ReviewSurface = object``),
the run does not store it, and nothing the engine decides reads it.
"""

from __future__ import annotations

from dataclasses import fields
from typing import TypeVar

from .findings import (Cluster, EpochResult, Finding, GateFailure, PersonaReport,
                       PersonaStatus, ReviewRun, SeamFailure, Severity, Suppression,
                       SurfaceFailure, render_exception, type_name)
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
    except Exception as exc:  # noqa: BLE001
        raise UncopiableRecord(
            f"the engine could not copy what it hands this seam: {render_exception(exc)}"
        ) from None


def _refused(why: str) -> UncopiableRecord:
    return UncopiableRecord(f"the engine could not copy what it hands this seam: {why}")


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
        raise _refused(f"a {type_name(value)} is neither a record nor an immutable value")
    state = _INSTANCE_DICT[base].__get__(value, base)
    if type(state) is not dict:
        raise _refused(f"a {base.__name__}'s attributes are not a plain dict")
    names = _FIELDS[base]
    missing = [name for name in names if name not in state]
    if missing:
        raise _refused(f"a {base.__name__} is missing {len(missing)} field(s)")
    record = object.__new__(base)
    memo[id(value)] = record
    for name in names:
        object.__setattr__(record, name, _copied(state[name], memo, depth))
    return record
