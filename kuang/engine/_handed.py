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
* its fields are read from the instance dict through the BASE class's own descriptor, so no
  property, ``__getattr__`` or ``__getattribute__`` a subclass defines is called;
* it is rebuilt as the BASE class without calling ``__init__``, so a subclass's extra
  state is dropped, as #129's ``GateDecision`` is re-made;
* a container is copied only at an exact ``list``, ``tuple`` or ``dict`` type, and anything
  else is handed as it is. Every value a record's fields hold that is not a record or a
  container is an immutable scalar (``str``, ``int``, ``None``, an enum), so sharing it
  shares nothing a seam can write.

**Faithful, aliasing included.** One memo, keyed by identity, per copy: an object reached
twice is copied once, so a ledger ``Finding`` that is also in ``new_findings`` and a
cluster's ``members`` is still one object in the copy. Nothing shipped relies on that, as
every match is by ``key``, but a copy that changed it would hand a reading seam a different
structure from today's.

**What it does not cover, stated.** A record whose instance dict is missing a field can be
reached only through a foreign subclass or a deleted attribute, and it cannot be copied
faithfully, so it is handed as it is. A seam that writes to the WORLD rather than to the run
(the repository, the filesystem, the network) is the permission policy's (#76, #79, #96).
A value ``spawn`` returned is a return value (#25), and a backend that keeps a reference to
a report it returned and writes it later is that door, not this one. A seam that keeps its
COPY and writes it later writes nothing the engine reads.
"""

from __future__ import annotations

from dataclasses import fields
from typing import TypeVar

from .findings import (Cluster, EpochResult, Finding, GateFailure, PersonaReport,
                       ReviewRun, SeamFailure, Suppression, SurfaceFailure)
from .protocols import GateDecision

_T = TypeVar("_T")

# Every record a seam can be handed, or reach from one it is handed.
_RECORDS: tuple[type, ...] = (ReviewRun, EpochResult, PersonaReport, Finding, Cluster,
                              Suppression, SeamFailure, GateFailure, SurfaceFailure,
                              GateDecision)
_FIELDS = {base: tuple(f.name for f in fields(base)) for base in _RECORDS}
_INSTANCE_DICT = {base: base.__dict__["__dict__"] for base in _RECORDS}


def handed(value: _T) -> _T:
    """A copy of ``value`` that shares nothing a seam can write with the live record."""
    return _copied(value, {})


def _copied(value, memo: dict[int, object]):
    found = memo.get(id(value))
    if found is not None:
        return found
    kind = type(value)
    if kind is list:
        out: list = []
        memo[id(value)] = out
        out.extend(_copied(item, memo) for item in value)
        return out
    if kind is dict:
        table: dict = {}
        memo[id(value)] = table
        for k, v in value.items():
            table[k] = _copied(v, memo)
        return table
    if kind is tuple:
        memo[id(value)] = copied = tuple(_copied(item, memo) for item in value)
        return copied
    base = next((b for b in _RECORDS if issubclass(kind, b)), None)
    if base is None:
        return value
    state = _INSTANCE_DICT[base].__get__(value, base)
    names = _FIELDS[base]
    if any(name not in state for name in names):
        return value
    record = object.__new__(base)
    memo[id(value)] = record
    for name in names:
        object.__setattr__(record, name, _copied(state[name], memo))
    return record
