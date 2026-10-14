# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""The copy a seam is handed: faithful, disjoint, and built running no foreign code (#155).

``kuang.engine._handed`` does not exist on main, so this module was measured there with a
minimal import shim (``handed = lambda v: v``, the live object, which is main's behaviour).
Tests GREEN under the shim say so in their own docstrings.

Stdlib only: no network, no ``claude`` subprocess.
"""

from __future__ import annotations

import pytest

from kuang.engine import (Cluster, EpochResult, Finding, GateDecision, PersonaReport,
                          PersonaStatus, ReviewRun, SeamFailure, Severity, Suppression)
from kuang.engine._handed import handed


def _finding(n=1, severity=Severity.BLOCKER):
    return Finding("p", f"t{n}", severity, f"c{n}", "a.py", n * 10)


def _run():
    """A record with every kind of field: one finding held in four places."""
    f, g = _finding(1), _finding(2, Severity.UGLY)
    report = PersonaReport("p", findings=[f, g], verdict="NO", tokens=3,
                           status=PersonaStatus.CONTRIBUTED, denied_tools=["Bash"])
    cluster = Cluster(members=(f,), title="c")
    result = EpochResult(index=1, reports=[report], new_findings=[f, g], open_blockers=1,
                         open_uglies=1, new_clusters=[cluster, Cluster(members=(g,))],
                         suppressed=[Suppression(finding=_finding(1), holder=f)],
                         gate=GateDecision(stop=False, asked=True),
                         seam_failures=[SeamFailure("budget_spent", "down", 4)])
    return ReviewRun(epochs=[result], ledger={f.key: f, g.key: g},
                     clusters=[cluster, Cluster(members=(g,))])


def _reachable(root):
    seen: dict[int, object] = {}

    def walk(obj):
        if id(obj) in seen:
            return
        if isinstance(obj, (list, tuple, dict)):
            seen[id(obj)] = obj
            for item in (obj.values() if isinstance(obj, dict) else obj):
                walk(item)
        elif hasattr(obj, "__dataclass_fields__") and not isinstance(obj, type):
            seen[id(obj)] = obj
            for name in obj.__dataclass_fields__:
                walk(obj.__dict__[name])
    walk(root)
    return seen


def test_the_copy_is_equal_to_the_record():
    """GREEN UNDER THE SHIM, by design: the identity is trivially equal. The fidelity half
    of the pair: a copy that drops a field, a report or a failure turns this red."""
    live = _run()

    assert handed(live) == live


def test_the_copy_shares_no_record_or_container_with_the_live_one():
    """REGRESSION for #155: the shim hands the live object, so everything is shared."""
    live = _run()

    assert not set(_reachable(handed(live))) & set(_reachable(live))


def test_the_copy_keeps_one_object_in_four_places_as_one_object():
    """A ledger finding is also in ``new_findings``, ``reports[].findings`` and a cluster's
    ``members``. The copy keeps that: a reading seam sees today's structure. GREEN UNDER
    THE SHIM, by design (the live object has it); a copy without the memo turns it red."""
    copy = handed(_run())
    held = copy.ledger[_finding(1).key]
    epoch = copy.epochs[0]

    assert epoch.new_findings[0] is held
    assert epoch.reports[0].findings[0] is held
    assert epoch.new_clusters[0].members[0] is held
    assert copy.clusters[0] is epoch.new_clusters[0]
    assert epoch.suppressed[0].holder is held


def test_a_write_through_the_freeze_into_the_copy_does_not_reach_the_record():
    """REGRESSION for #155: ``Finding`` is frozen without slots, so both doors write."""
    live = _run()
    copy = handed(live)
    for f in copy.ledger.values():
        f.__dict__["verified"] = False
        object.__setattr__(f, "severity", Severity.NOTE)
    copy.epochs[0].seam_failures.clear()
    copy.epochs[0].halt = "rewritten"

    assert live == _run()


class _Hostile(Finding):
    """A ``Finding`` subclass whose every hook raises: nothing here may be called."""

    @property
    def __class__(self):  # noqa: D401
        raise AssertionError("__class__ consulted")

    def __deepcopy__(self, memo):
        raise AssertionError("__deepcopy__ called")

    def __reduce_ex__(self, protocol):
        raise AssertionError("__reduce_ex__ called")

    def __getattribute__(self, name):
        if name == "__dict__":
            raise AssertionError("__dict__ read through the subclass")
        return super().__getattribute__(name)


def test_a_hostile_subclass_is_copied_without_calling_any_of_its_hooks():
    """REGRESSION for #155's rejected design: ``copy.deepcopy`` calls ``__deepcopy__``, and
    ``isinstance`` consults ``__class__``. The copy is the BASE class, as #129 re-makes a
    ``GateDecision``: a subclass's extra behaviour is dropped."""
    hostile = _Hostile("p", "t", Severity.UGLY, "c", "a.py", 1)

    copy = handed([hostile])[0]

    assert type(copy) is Finding
    assert (copy.title, copy.severity) == ("t", Severity.UGLY)


class _Report(PersonaReport):
    @property
    def turns(self):  # noqa: D401
        raise AssertionError("a property was called")

    @turns.setter
    def turns(self, value):
        self.__dict__["turns"] = value


def test_a_report_subclass_with_a_raising_property_is_still_copied():
    """The field is read from the instance dict through the base class, never through the
    subclass's attribute machinery, so the copy cannot raise on what ``spawn`` returned.
    Red under the shim only because the shim hands the subclass itself."""
    report = _Report("p", turns=4)

    copy = handed(report)

    assert type(copy) is PersonaReport and copy.turns == 4


class _Odd(list):
    pass


def test_only_exact_containers_are_copied_and_anything_else_is_handed_as_is():
    """A ``list`` subclass could run code on iteration, so it is not walked: handed as it
    is, and stated in the module as what the copy does not cover. GREEN UNDER THE SHIM,
    by design: it guards a copy that walks foreign containers."""
    odd = _Odd([1])
    report = PersonaReport("p", denied_tools=odd)

    assert handed(report).denied_tools is odd


def test_a_record_missing_a_field_is_handed_as_is_rather_than_misrepresented():
    """Reachable only through a deleted attribute or a foreign subclass: it cannot be
    copied faithfully, so it is not copied at all. GREEN UNDER THE SHIM, by design: it
    guards a copy that would raise, or invent a field, on it."""
    f = _finding()
    del f.__dict__["evidence"]

    assert handed([f])[0] is f


@pytest.mark.parametrize("value", [None, 0, "s", Severity.UGLY, (1, "a")])
def test_a_scalar_is_returned_unchanged(value):
    """GREEN UNDER THE SHIM, by design: nothing a seam can write is shared by sharing it."""
    assert handed(value) == value
