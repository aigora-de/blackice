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
from kuang.engine._handed import MAX_DEPTH, UncopiableRecord, claimed_severity, handed


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


def test_a_hostile_subclass_is_refused_without_calling_any_of_its_hooks():
    """REGRESSION for #155's rejected design (``copy.deepcopy`` calls ``__deepcopy__``, and
    ``isinstance`` consults ``__class__``) and for the fourth review: a subclass is refused
    by its exact type, so not one of its hooks runs, and it is never handed live."""
    hostile = _Hostile("p", "t", Severity.UGLY, "c", "a.py", 1)

    assert "neither a record nor" in _refused([hostile])


class _Report(PersonaReport):
    @property
    def turns(self):  # noqa: D401
        raise AssertionError("a property was called")

    @turns.setter
    def turns(self, value):
        self.__dict__["turns"] = value


class _Shadow(PersonaReport):
    """Presents a NO vote through a property, over an instance dict holding YES."""

    @property
    def verdict(self):  # noqa: D401
        return "NO"

    @verdict.setter
    def verdict(self, value):
        self.__dict__["verdict"] = value


@pytest.mark.parametrize("make", [lambda: _Report("p", turns=4),
                                  lambda: _Shadow("p", verdict="YES")],
                         ids=["raising-property", "shadowing-property"])
def test_a_report_subclass_is_refused_never_copied_from_its_dict(make):
    """REGRESSION for the fourth review's UGLY: the previous shape copied a subclass's BASE
    fields from its instance dict, so a subclass presenting a NO vote (or an UGLY) over a
    dict holding YES (or nothing) was handed on as the stale value, and the run converged
    where main did not. No copy of a subclass is faithful, so it is refused."""
    assert "neither a record nor" in _refused(make())


class _Odd(list):
    pass


class _Spy(dict):
    """An instance dict that is not a plain ``dict``: its hooks are foreign code."""

    def __contains__(self, key):
        raise AssertionError("a dict-subclass hook ran inside the copy")

    def __getitem__(self, key):
        raise AssertionError("a dict-subclass hook ran inside the copy")


def _refused(value):
    with pytest.raises(UncopiableRecord) as caught:
        handed(value)
    return str(caught.value)


@pytest.mark.parametrize("value", [
    PersonaReport("p", denied_tools=_Odd(["Bash"])),
    PersonaReport("p", denied_tools={"Bash"}),
    PersonaReport("p", unresolved_severities=bytearray(b"x")),
], ids=["list-subclass", "set", "bytearray"])
def test_a_value_that_is_neither_copyable_nor_immutable_is_refused(value):
    """REGRESSION for #155's review: the first shape handed these over AS THEY WERE, so a
    ``set`` or a container subclass ``spawn`` supplied was shared with the live record and a
    gate or checkpoint could rewrite what the run publishes. Under the shim (main's live
    object) nothing is refused."""
    assert "could not copy" in _refused(value)


def test_a_record_missing_a_field_is_refused_never_handed_live():
    """REGRESSION for the review's UGLY: a ``spawn`` finding missing one instance-dict field
    (it still READS, through the class default) was handed to ``adjudicate`` live, so a
    downgrade written there reached the ledger and an UGLY run converged."""
    f = _finding()
    object.__delattr__(f, "evidence")

    assert "missing 1 field" in _refused([f])


def test_an_instance_dict_that_is_not_a_plain_dict_is_refused_without_calling_it():
    """REGRESSION for the review: a dict-subclass ``__dict__`` had its ``__contains__`` and
    ``__getitem__`` called 18 times in one copy, which is foreign code inside the copy."""
    report = PersonaReport("p")
    report.__dict__ = _Spy(report.__dict__)

    assert "not a plain dict" in _refused(report)


def test_a_structure_nested_past_the_bound_is_refused_at_the_engines_own_depth():
    """REGRESSION for the review: a ``spawn`` value nested past the interpreter's recursion
    limit raised ``RecursionError``, recorded against whichever seam was being copied for.
    The bound is the engine's constant, so where a copy is refused is deterministic."""
    deep: list = []
    cur = deep
    for _ in range(MAX_DEPTH + 5):
        cur.append([])
        cur = cur[0]

    assert f"deeper than {MAX_DEPTH}" in _refused(PersonaReport("p", denied_tools=deep))


def test_a_structure_inside_the_bound_is_copied():
    """The bound's other end: a list nested to just inside it is a value, and is copied.
    GREEN UNDER THE SHIM, by design (the shim refuses nothing)."""
    deep: list = []
    cur = deep
    for _ in range(MAX_DEPTH - 10):
        cur.append([])
        cur = cur[0]
    report = PersonaReport("p", denied_tools=deep)

    assert handed(report) == report


@pytest.mark.parametrize("value", [None, 0, "s", Severity.UGLY, (1, "a")])
def test_a_scalar_is_returned_unchanged(value):
    """GREEN UNDER THE SHIM, by design: nothing a seam can write is shared by sharing it."""
    assert handed(value) == value


# --- what a REFUSED report claimed (#155's third review) -------------------------------
#
# The failure a refused report becomes carries the highest severity it claimed, read
# running nothing the report carries, and FAILING CLOSED (UGLY) where a claim's severity
# cannot be read: the maintainer's ruling of 2026-10-14.

class _Seq(list):
    def __iter__(self):
        raise AssertionError("a list subclass's __iter__ ran")


class _Dict(dict):
    def items(self):
        raise AssertionError("a dict subclass's items ran")

    def get(self, *a):
        raise AssertionError("a dict subclass's get ran")


def _report_of(*findings, holder=list):
    report = PersonaReport("p")
    report.__dict__["findings"] = holder(findings)
    return report


@pytest.mark.parametrize("holder", [list, tuple], ids=["list", "tuple"])
def test_a_refused_reports_ugly_is_read_through_a_list_or_tuple(holder):
    """REGRESSION for the third review's UGLY (the ``list`` case is the CONTROL, green on
    the shape before it): the first shape read only an exact ``list`` and returned NOTE for
    a tuple, so a quorum-below-panel run converged on an UGLY main escalated."""
    report = _report_of(_finding(1, Severity.NOTE), _finding(2, Severity.UGLY), holder=holder)

    assert claimed_severity(report) == (Severity.UGLY, True)


class _Presents(list):
    """Presents an UGLY through ``__iter__`` over storage holding none."""

    def __iter__(self):
        return iter([_finding(9, Severity.UGLY)])


@pytest.mark.parametrize("holder", [_Seq, _Presents], ids=["hook-raises", "hook-presents"])
def test_a_container_subclass_of_claims_fails_closed_without_its_hook_running(holder):
    """REGRESSION for the fifth review's UGLY: the previous shape read a list subclass's
    STORAGE, which is not what main iterates (its own ``__iter__``), so an UGLY presented
    that way over empty storage was read as nothing claimed. Closed; ``_Seq``'s hook
    raises, so a read that ran it would fail here."""
    report = PersonaReport("p")
    report.__dict__["findings"] = holder()

    assert claimed_severity(report) == (Severity.UGLY, False)


@pytest.mark.parametrize("severity", [Severity.NOTE, Severity.UGLY], ids=["note", "ugly"])
def test_a_finding_whose_instance_dict_is_a_dict_subclass_fails_closed(severity):
    """REGRESSION for the third review's UGLY (the dict-subclass shape lost an UGLY) and the
    sixth review's BLOCKER: main's adjudicated rebuild reads ``{**f.__dict__}``, which runs
    a dict subclass's own ``keys`` and ``__getitem__``, so storage is not what main acts
    on. Closed whatever the storage holds, and none of ``_Dict``'s hooks runs."""
    f = _finding(1, severity)
    object.__setattr__(f, "__dict__", _Dict(f.__dict__))

    assert claimed_severity(_report_of(f)) == (Severity.UGLY, False)


def test_the_highest_readable_claim_is_carried():
    """The max, not the first or the last: a BLOCKER beside NOTEs blocks convergence. A
    CONTROL: green on the shape before the third review's repair, as on this one."""
    report = _report_of(_finding(1, Severity.NOTE), _finding(2, Severity.BLOCKER),
                        _finding(3, Severity.NON_BLOCKING))

    assert claimed_severity(report) == (Severity.BLOCKER, True)


@pytest.mark.parametrize("returned", [None, "text", 7, _report_of()],
                         ids=["none", "str", "int", "a-report-claiming-nothing"])
def test_a_return_that_claims_nothing_is_note(returned):
    """The fail-closed rule's other end: nothing claimed is nothing unknown. Without this
    pair a function returning UGLY for everything would pass the tests above. GREEN UNDER
    THE SHIM, by design: the shim claims NOTE for everything, which is this end."""
    assert claimed_severity(returned) == (Severity.NOTE, True)


_FOREIGN_CALLS: list[str] = []


class _IntSeverity(int):
    """RECORDS its calls rather than raising: ``claimed_severity`` catches every exception
    and fails closed, so a hook that raised would be invisible behind a closed result."""

    def __gt__(self, other):
        _FOREIGN_CALLS.append("__gt__")
        return True

    def __lt__(self, other):
        _FOREIGN_CALLS.append("__lt__")
        return False

    def __eq__(self, other):
        _FOREIGN_CALLS.append("__eq__")
        return False

    __hash__ = int.__hash__


class _CollidingKey:
    armed = False   # inserting it compares it with "severity" in the TEST's own setup

    def __hash__(self):
        return hash("severity")

    def __eq__(self, other):
        if self.armed:
            raise AssertionError("a foreign key's __eq__ ran")
        return False


def _with_state(f, **state):
    f.__dict__.update(state)
    return f


@pytest.mark.parametrize("make", [
    lambda: _report_of(_with_state(_finding(), severity="UGLY")),
    lambda: _report_of(_with_state(_finding(), severity=_IntSeverity(4))),
    lambda: _report_of(_finding()).__dict__.__setitem__("findings", {_finding()}) or None,
    lambda: _report_of(object()),
], ids=["severity-a-str", "severity-an-int-subclass", "findings-a-set", "a-non-finding-claim"])
def test_a_claim_that_cannot_be_read_fails_closed(make):
    """Fail closed, by the maintainer's ruling: a claim whose severity cannot be read leaves
    "is an UGLY open?" unknown, and the breaker fires. The int-subclass case is the third
    review's surviving mutation M7 (an ``isinstance`` check would run its ``__gt__``)."""
    report = make()
    if report is None:
        report = PersonaReport("p")
        report.__dict__["findings"] = {_finding()}
    _FOREIGN_CALLS.clear()

    assert claimed_severity(report) == (Severity.UGLY, False)
    assert _FOREIGN_CALLS == [], "a severity's own comparison ran"


class _ShadowedFinding(Finding):
    """Presents an UGLY through a property over an instance dict holding NOTE."""

    @property
    def severity(self):  # noqa: D401
        return Severity.UGLY


@pytest.mark.parametrize("make", [
    lambda: _Shadow("p", findings=[_finding(1, Severity.NOTE)]),
    lambda: _report_of(_with_state(object.__new__(_ShadowedFinding), **_finding(1, Severity.NOTE).__dict__)),
], ids=["report-subclass", "finding-subclass"])
def test_a_subclass_claim_fails_closed_because_what_it_presents_cannot_be_read(make):
    """REGRESSION for the fourth review's UGLY: its dict says NOTE, it presents UGLY, and
    the engine will not run the property that presents it. Unknown, so closed."""
    assert claimed_severity(make()) == (Severity.UGLY, False)


def test_a_report_with_no_findings_attribute_fails_closed():
    """Nothing to read is not the same as nothing claimed: closed."""
    report = PersonaReport("p")
    del report.__dict__["findings"]

    assert claimed_severity(report) == (Severity.UGLY, False)


def test_a_colliding_foreign_key_beside_the_severity_has_no_code_run():
    """The third review's surviving mutation M6: a key whose hash matches ``severity`` had
    its ``__eq__`` run by a ``.get``. A dict with a key that is not an exact ``str`` is not
    read at all: closed, and the key's ``__eq__`` never runs."""
    f = _finding(1, Severity.UGLY)
    key = _CollidingKey()
    f.__dict__[key] = "x"
    key.armed = True

    assert claimed_severity(_report_of(f)) == (Severity.UGLY, False)


class _DuckReport:
    """Not a ``PersonaReport``, but main reads ``.findings`` off it all the same."""

    def __init__(self):
        self.findings = [_finding(1, Severity.UGLY)]


@pytest.mark.parametrize("make", [_DuckReport, lambda: _finding(1, Severity.UGLY),
                                  lambda: [_finding(1, Severity.UGLY)]],
                         ids=["duck-report", "a-finding", "a-list"])
def test_anything_but_an_exact_report_or_an_immutable_value_fails_closed(make):
    """REGRESSION for the sixth review's UGLY: a duck-typed report was "not a report", so
    it claimed NOTE, while main read its ``findings`` by attribute lookup and escalated.
    Only an immutable scalar claims nothing; anything else is closed."""
    assert claimed_severity(make()) == (Severity.UGLY, False)


# --- the bound is on the height under a value, whichever path reaches it first ----------

def _chain(n, tail=None):
    out: list = [] if tail is None else [tail]
    for _ in range(n - 1):
        out = [out]
    return out


def test_a_value_admitted_through_a_short_path_is_bounded_on_a_long_one():
    """REGRESSION for the eighth review's BLOCKER: the bound was checked on a value's FIRST
    visit only, so a list reached first through a short path was admitted, and then a
    copy walking a longer path first (``adjudicate``'s, ``reduce``'s) refused it at every
    epoch. Whichever path is walked first, the answer is now the same."""
    shared = _chain(40)
    short = _finding(1, Severity.NOTE)
    short.__dict__["evidence"] = shared
    long = _finding(2, Severity.BLOCKER)
    long.__dict__["evidence"] = _chain(40, shared)

    def outcome(value):
        try:
            handed(value)
            return "admitted"
        except UncopiableRecord:
            return "refused"

    assert outcome([short, long]) == outcome([long, short]) == outcome([long]) == "refused"


def test_a_value_that_contains_itself_is_refused():
    """No record can contain itself, so only a hostile ``spawn`` value can: it has no
    height to bound, and it is refused rather than copied."""
    loop: list = []
    loop.append(loop)

    assert "contains itself" in _refused(PersonaReport("p", denied_tools=loop))


def test_a_value_reached_twice_within_the_bound_is_still_one_object():
    """The bound's other end, and the aliasing it must not cost: a value reached twice,
    both times within the bound, is copied once and admitted. GREEN UNDER THE SHIM."""
    shared = _chain(5)
    copy = handed([shared, [[shared]]])

    assert copy[0] is copy[1][0][0]


def test_a_value_reached_twice_at_the_bound_is_admitted_both_times():
    """The revisit check's exact boundary: a list exactly ``MAX_DEPTH`` high, reached twice
    at the same depth, is admitted on the first visit, so it must be on the second (an
    off-by-one on the revisit refused it there, and only there). GREEN UNDER THE SHIM."""
    tall = _chain(MAX_DEPTH)
    handed([tall])

    copy = handed([tall, tall])

    assert copy[0] is copy[1]
