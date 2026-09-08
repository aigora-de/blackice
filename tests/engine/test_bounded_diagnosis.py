# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""A bounded diagnostic string must say when it was cut (#111).

``surface.py`` already states the rule, and implements it:

> once the cap is reached, remaining files are dropped and named in an explicit
> OMITTED notice — **never a silent truncation**. A lone file that by itself
> exceeds the cap is truncated in place **with a marker**

Seven ``[:400]`` sites broke it. The measured harm is not that a string was
shortened — it is that the cut lands **mid-token**, so the record ends on
``src/modul`` and an operator cannot tell a path that failed to resolve from a
fragment of one. The record does not merely omit; it presents a plausible-looking
value that was never real. That is why these tests are about the **boundary**
rather than about the presence of a notice.

**What each test is.** ``test_a_cut_string_says_so`` and
``test_the_marker_is_unmistakable_at_the_boundary`` are the regression. The rest
are the guards that keep it honest, and they are the point of the exercise:

* ``test_a_short_string_is_returned_unchanged`` and
  ``test_a_string_exactly_at_the_bound_is_not_marked`` are the **mirror image** of
  the defect. A rule that fires on a healthy value is the same failure wearing a
  fix, and it is easy to write here — ``return text[:limit] + MARKER`` passes
  every regression test above and is wrong.
* ``test_the_marker_rides_on_top_of_the_bound`` pins the decision, against the
  option that was rejected: the marker must not eat into the 400. Adopting
  ``surface.py``'s precedent (``rendered[:budget] + marker``, and
  ``SurfaceRecord``'s "the assembler's own notices ride on top of it") keeps 400
  meaning 400 characters of diagnosis. The alternative makes the bound mean two
  things and silently couples how much diagnosis survives to how the marker is
  worded.

**The fixture straddles the bound deliberately** (#87). This whole change is about
a string boundary, so an input that does not cross it tests nothing: an ordering
mutation survived #85's matrix for exactly that reason. ``limit``, ``limit + 1``
and a realistic over-long message are all exercised — and the over-long one is the
message the issue measured, whose 400th character falls inside a path name rather
than between two of them. A fixture that cut cleanly at a comma would pass every
test here and demonstrate nothing.

**Red on main:** the whole file, as a collection error — ``bounded_diagnosis`` does
not exist. Nine tests, none of which can pass before the change, which is a weaker
statement than it looks: a file that cannot import is not evidence that its
assertions are load-bearing. The mutation matrix in the PR is, and every test here
is killed by at least one mutation of the shipped helper.
"""

from __future__ import annotations

import pytest

from kuang.engine import DIAGNOSIS_BOUND, bounded_diagnosis

# The shape the issue measured: an ordinary path list, long enough that the 400
# bound lands inside a path name rather than between two of them.
_PATHS = ", ".join(f"src/module_{i}/component_handler_{i}.py" for i in range(1, 21))
_REFUSAL = (f"SurfaceError: no reviewable files in the requested paths: {_PATHS}. "
            "Check they exist, are tracked by git, and are not gitignored.")


# --- the defect ----------------------------------------------------------------

def test_a_cut_string_says_so():
    """The regression: a string over the bound comes back marked.

    The marker names the bound **and** the true length. Both are exactly knowable
    — ``len(text)`` is not an estimate — and the pair is what tells an operator
    whether they lost twelve characters or twelve hundred, which is the difference
    between a diagnosis worth reading and one worth re-running for.
    """
    out = bounded_diagnosis(_REFUSAL)

    assert "truncated" in out
    assert str(DIAGNOSIS_BOUND) in out, "the marker names the bound"
    assert str(len(_REFUSAL)) in out, "the marker names what the string actually was"


def test_the_marker_is_unmistakable_at_the_boundary():
    """The measured harm: a fragment must not be readable as a whole value.

    The 400th character of this message falls inside a path name, so before the
    fix the record ended on something that reads exactly like a path the tool
    failed to resolve. What an operator reads last must be the marker, never the
    fragment.
    """
    assert len(_REFUSAL) > DIAGNOSIS_BOUND
    assert _REFUSAL[DIAGNOSIS_BOUND - 1] not in " ,", (
        "the fixture must cut mid-token, or it cannot express the defect")

    out = bounded_diagnosis(_REFUSAL)

    assert not out.endswith(".py"), "the record must not end on a path-shaped fragment"
    assert out.endswith("]"), "the marker is what the string ends on"
    assert out.startswith(_REFUSAL[:DIAGNOSIS_BOUND]), "the diagnosis is kept, not moved"
    # The sentinel is the boundary, and it is asserted here rather than left to the
    # ``split("…")`` calls elsewhere that would catch its loss only by accident. A
    # marker separated from the fragment by an ordinary space puts the notice and
    # the content in the same typographic register, which is the ambiguity this
    # whole test is named for.
    assert out[DIAGNOSIS_BOUND] == "…", "the sentinel abuts the fragment it cuts"


# --- the mirror image: a rule that fires on a healthy value --------------------

def test_a_short_string_is_returned_unchanged():
    """A string that was never cut carries nothing at all.

    Not "carries no marker" — **unchanged**, byte for byte. Marking an uncut value
    is the mirror image of this defect: the record would claim a loss that never
    happened, which is the instrument lying about itself in the opposite direction.
    """
    short = "SurfaceError: git diff main...HEAD is empty."

    assert bounded_diagnosis(short) == short


def test_a_string_exactly_at_the_bound_is_not_marked():
    """The boundary itself, from both sides.

    ``limit`` characters fit and nothing was lost; ``limit + 1`` did not and one
    character was. An off-by-one here would mark a string that is whole, or leave
    the shortest possible cut silent.
    """
    exact = "x" * DIAGNOSIS_BOUND

    assert bounded_diagnosis(exact) == exact
    assert "truncated" in bounded_diagnosis(exact + "y")


# --- the decision, pinned against the option that was rejected -----------------

def test_the_marker_rides_on_top_of_the_bound():
    """400 means 400 characters of diagnosis, not 400 of diagnosis-plus-notice.

    ``surface.py`` sets the precedent one module over — ``rendered[:budget] +
    marker`` — and ``SurfaceRecord``'s docstring states the rule: "the cap bounds
    the rendered file content, and the assembler's own OMITTED and missing-path
    notices ride on top of it. Reporting the bounded figure instead would report a
    number nobody was given."

    The rejected alternative spends the last thirty-odd characters of every
    diagnosis on the marker, and couples how much survives to how the marker is
    worded: reword it, silently lose more. This test is what goes red if anyone
    implements that instead.
    """
    out = bounded_diagnosis(_REFUSAL)

    assert out[:DIAGNOSIS_BOUND] == _REFUSAL[:DIAGNOSIS_BOUND], (
        "the first 400 characters are 400 characters of the diagnosis")
    assert len(out) > DIAGNOSIS_BOUND, "the marker is not paid for out of the bound"


@pytest.mark.parametrize("limit", [1, 10, DIAGNOSIS_BOUND, 4000])
def test_the_bound_is_a_parameter_and_the_rule_holds_at_every_size(limit):
    """The rule is one rule, not a table: it must hold wherever the bound is put.

    ``DIAGNOSIS_BOUND`` is 400 today and is deliberately unchanged by #111 — it has
    never been baselined, and the issue's scope says so. What #111 does change is
    that a bound nobody has measured now **reports** what it cost instead of
    swallowing it. Parameterising the helper is what stops the next caller writing
    a second literal, and this pins that the marker's arithmetic does not quietly
    assume 400.
    """
    assert bounded_diagnosis("x" * limit, limit) == "x" * limit
    out = bounded_diagnosis("x" * (limit + 1), limit)
    assert out.startswith("x" * limit)
    assert str(limit) in out and str(limit + 1) in out
