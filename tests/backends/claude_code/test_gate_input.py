# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""The gate's INPUT boundary: an answer the tool cannot read is not a decision (#132).

`interactive_gate` resolved every answer with ``ans.startswith("s")``, so the one
input a *human* supplies was parsed from prose. That fails in both directions and
the issue names only one of them:

* ``abort``, ``no``, ``n``, ``quit``, ``exit``, a bare Enter and a stray escape
  sequence all became ``stop=False, asked=True`` — published as
  ``{"asked": true, "stopped": false}`` and printed as *"a human continued the
  run"*, which is the opposite of what the operator typed;
* ``skip``, ``sure`` and ``scope`` all began with ``s`` and became
  ``stop=True, asked=True`` — a run **halted** and archived as *"a human STOPPED
  the run"*. Measured, and not described on the issue when it was filed.

This is #24's and #26's doctrine — *a value that is not the word is not the value*
— applied at the input boundary, where #131 applied it at the output boundary.

**The ruling this module pins** (made on #132, cited by #129): when the gate does
not resolve a human's answer the run **continues** and records
``GateDecision(stop=False, asked=None)``. ``asked=False`` stays reserved for the
**measured** non-interactive case, so no capture and no existing test moves. The
run never claims a human decided, and `_gate_outcome` already renders the value.

**This is the first test in ``tests/backends/`` to drive the gate at all.**
``_FakeTTY`` is duplicated from ``tests/cli/test_gate_reporting.py`` rather than
promoted to ``tests/conftest.py``: promotion would edit a module this issue does
not otherwise touch, which the change-discipline rule refuses. The tidy-up is
filed rather than smuggled in here.

Every stdin fixture is **finite**. That is load-bearing rather than tidy: the
mutation implementing the rejected unbounded re-prompt terminates against a finite
stream and is killed by a red test, where against an endless one it would hang the
suite and report nothing.

Stdlib only — no subprocess, no network. The gate is pure I/O over ``sys.stdin``.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

from kuang.backends.claude_code import PanelSession, Persona
from kuang.engine import EpochResult, GateDecision, ReviewRun, ReviewSpec

_PROMPT = "gate — [c]ontinue / [s]top? "


class _FakeTTY(io.StringIO):
    """A stdin that IS a terminal and answers the prompt.

    ``input()`` falls back to ``sys.stdin.readline()`` once stdin is not the
    original console object, so this drives the real prompt rather than stubbing
    the gate. An **empty** one raises a genuine ``EOFError`` from ``input()``,
    which is how the EOF case is exercised without patching anything.
    """

    def isatty(self) -> bool:
        return True


@pytest.fixture
def session(tmp_path):
    return PanelSession(
        repo_root=Path(tmp_path), spec=ReviewSpec(why="w", what="x"),
        personas={"p": Persona("p", "be adversarial")},
        base="main", head="HEAD", claude_bin="/fake/claude")


def _epoch() -> EpochResult:
    """A completed epoch that raised nothing: the gate reads counts off it."""
    return EpochResult(index=1, reports=[], new_findings=[],
                       open_blockers=0, open_uglies=0)


def _drive(session, monkeypatch, capsys, answer: str, *, repeats: int = 8):
    """Answer the real prompt ``repeats`` times over, and return (decision, output).

    Finite by construction — see the module docstring.
    """
    monkeypatch.setattr(sys, "stdin", _FakeTTY(f"{answer}\n" * repeats))
    decision = session.interactive_gate(_epoch(), ReviewRun())
    return decision, capsys.readouterr().out


# --- 1. the two-token vocabulary ---------------------------------------------

@pytest.mark.parametrize("answer", ["s", "stop", "S", " s ", " STOP "])
def test_a_stop_answer_stops_the_run(session, monkeypatch, capsys, answer):
    """GUARD: passes on ``main`` and must keep passing.

    The fix narrows a substring test to an exact two-token vocabulary, and the
    way to get that wrong is to narrow it too far — to bare ``"s"``, or to drop
    the ``.strip().lower()`` that makes ``" STOP "`` the same answer as ``s``.
    This is not a regression test; it pins the half of the behaviour that was
    always right.
    """
    decision, _ = _drive(session, monkeypatch, capsys, answer)

    assert decision == GateDecision(stop=True, asked=True), \
        f"{answer!r} is a human stopping the run"


@pytest.mark.parametrize("answer", ["c", "continue", "C", " c ", " CONTINUE "])
def test_a_continue_answer_is_recorded_as_a_human_who_continued(
        session, monkeypatch, capsys, answer):
    """GUARD, and the unit-level MIRROR IMAGE of this whole change.

    A fix that recorded silence for answers it *can* read would be this defect
    inverted — the run would stop claiming a human decided at the exact moment
    one did. Without this pair, "an unreadable answer records ``asked=None``" is
    satisfiable by recording ``asked=None`` for everything.
    """
    decision, _ = _drive(session, monkeypatch, capsys, answer)

    assert decision == GateDecision(stop=False, asked=True), \
        f"{answer!r} is a human choosing to continue"


# --- 2. an answer the tool cannot read is not a decision ----------------------

@pytest.mark.parametrize("answer", ["abort", "no", "n", "quit", "exit", "",
                                    "\x1b[A", "y", "yes", "?"])
def test_an_answer_the_gate_cannot_read_is_not_a_human_continuing(
        session, monkeypatch, capsys, answer):
    """REGRESSION for #132, the direction the issue describes.

    On ``main`` every one of these returns ``stop=False, asked=True``, which the
    artefact publishes as ``{"asked": true, "stopped": false}`` and the console
    prints as *"a human continued the run"*. An operator who typed ``abort`` at
    epoch 1 of a five-epoch run is archived as having chosen to continue.

    ``asked=None`` and NOT ``asked=False``: ``False`` is the measured claim that
    no human could be reached, which is a different fact and is what a run with
    no terminal reports. #131's rule one boundary out — a value the tool cannot
    read is not a measurement, it is silence, and the tri-state has a member for
    that.
    """
    decision, _ = _drive(session, monkeypatch, capsys, answer)

    assert decision.asked is None, \
        f"{answer!r} was resolved to a claim about what a human chose"
    assert decision.stop is False, \
        f"{answer!r} was resolved to a decision to stop"


@pytest.mark.parametrize("answer", ["skip", "sure", "scope", "start", "submit"])
def test_an_answer_the_gate_cannot_read_does_not_stop_the_run_either(
        session, monkeypatch, capsys, answer):
    """REGRESSION for #132, the direction the issue does NOT describe.

    Measured while planning: ``startswith("s")`` fails **both** ways. Every answer
    here begins with ``s``, so on ``main`` each returns ``stop=True, asked=True``
    — the run halts, ``HaltReason.ABORTED`` is set, and the artefact records that
    a human stopped it. An operator typing ``skip`` ends a paid-for panel and the
    archive says they meant to.

    Filed as evidence on #132 rather than as a separate issue: it is the same
    line, the same cause and the same fix, and the issue's title names only the
    complement.
    """
    decision, _ = _drive(session, monkeypatch, capsys, answer)

    assert decision.stop is False, \
        f"{answer!r} halted the run because it happens to begin with 's'"
    assert decision.asked is None, \
        f"{answer!r} was resolved to a claim about what a human chose"


def test_the_gate_does_not_guess_what_an_operator_meant(session, monkeypatch,
                                                        capsys):
    """GUARD on a REJECTED design: mapping ``n``/``no``/``abort`` to *stop*.

    Green on ``main`` and must stay green. The obvious "fix" for #132 is to widen
    the guess-set so ``abort`` means stop — which is the same defect in a new
    direction, because the next operator types ``halt`` or ``kill`` or ``x``. The
    tool does not infer intent from prose; it reads two tokens and asks again.
    A mutation adding that mapping is killed by exactly this test.
    """
    for answer in ("n", "no", "abort", "halt", "kill"):
        decision, _ = _drive(session, monkeypatch, capsys, answer)
        assert decision.stop is False, \
            f"the gate guessed that {answer!r} meant stop"


# --- 3. what ends the prompt --------------------------------------------------

def test_an_eof_at_the_gate_continues_and_records_no_human_answer(
        session, monkeypatch, capsys):
    """REGRESSION: on ``main`` this raises ``EOFError`` out of the gate.

    A closed or exhausted stdin that still reports a terminal is not exotic — it
    is what a CI runner with an allocated pty, or a harness driving the CLI,
    produces. Today that exception propagates through ``loop.run`` and discards
    every completed epoch of a paid-for panel (#129 owns the containment; this
    owns not raising in the first place).

    The empty ``_FakeTTY`` makes ``input()`` raise for real rather than by
    patching, so this exercises the actual failure rather than a model of it.
    """
    monkeypatch.setattr(sys, "stdin", _FakeTTY(""))

    decision = session.interactive_gate(_epoch(), ReviewRun())

    assert decision == GateDecision(stop=False, asked=None), \
        "an EOF at the gate is not a human's decision"


def test_the_gate_stops_re_prompting_at_its_bound(session, monkeypatch, capsys):
    """REGRESSION: the re-prompt must terminate, and the bound is what ends it.

    An unbounded loop against a pty streaming an escape sequence never returns,
    so the run never reaches its artefact — the same loss #129 exists to prevent,
    reached by a different route. The bound is a **termination bound, not a
    tolerance**: this asserts the constant, never the literal, because the exact
    number is not load-bearing and a threshold nobody baselined would be.

    ``GATE_ATTEMPTS`` is imported inside the test deliberately. At module level it
    would fail collection on ``main``, turning every green-on-main guard in this
    file red and destroying the red-on-main measurement this change is held to.
    """
    from kuang.backends.claude_code.session import GATE_ATTEMPTS

    decision, out = _drive(session, monkeypatch, capsys, "abort", repeats=99)

    assert out.count(_PROMPT) == GATE_ATTEMPTS, \
        f"the gate read the prompt {out.count(_PROMPT)} times, not {GATE_ATTEMPTS}"
    assert decision == GateDecision(stop=False, asked=None)


def test_a_correction_after_a_mistyped_answer_is_the_answer_that_counts(
        session, monkeypatch, capsys):
    """REGRESSION: the operator keeps control, which is the point of re-prompting.

    On ``main`` the gate reads exactly one line, so an operator who fat-fingers
    ``n`` and then types ``s`` has already been recorded as continuing before
    they corrected themselves. Re-prompting is not politeness: it is what makes
    the recorded decision the one the human actually made.
    """
    monkeypatch.setattr(sys, "stdin", _FakeTTY("n\ns\n"))

    decision = session.interactive_gate(_epoch(), ReviewRun())

    assert decision == GateDecision(stop=True, asked=True), \
        "the correction was ignored and the mistype was archived as the decision"


# --- 4. what this deliberately does NOT change --------------------------------

def test_a_non_interactive_gate_still_measures_that_nobody_could_be_asked(
        session, monkeypatch, capsys):
    """GUARD: ``asked=False`` is reserved, and every run in CI depends on it.

    Green on ``main`` and the reason this change moves no capture. *"Not a tty,
    continued without asking"* is a FACT, not an absence, and it is a **different**
    fact from "asked and could not be read". A fix that collapsed the two would
    lose the distinction #117 paid to create — and would move all 34 probe
    captures, none of which can reach a terminal.

    The prompt must also never be printed: a run with no terminal that prints a
    question nobody can answer is claiming something that did not happen.
    """
    monkeypatch.setattr(sys, "stdin", io.StringIO("s\n"))

    decision = session.interactive_gate(_epoch(), ReviewRun())
    out = capsys.readouterr().out

    assert decision == GateDecision(stop=False, asked=False), \
        "the measured non-interactive case changed"
    assert _PROMPT not in out, "a run with no terminal asked a question anyway"


def test_the_gate_never_prints_what_the_operator_typed(session, monkeypatch,
                                                       capsys):
    """GUARD on a REJECTED design: echoing the answer back in the re-prompt.

    Green on ``main`` (which prints nothing) and it must stay green. Recording or
    echoing the raw answer would put an unbounded operator-authored string into
    the run's output — which PR #130 refused when it removed ``GateDecision.note``,
    and which #118/#121 are open on. The console half matters as much as the
    artefact: console output is what gets pasted into issues, and an operator at
    a gate may have pasted anything into that line.

    If the run must say why it could not read the answer, it says so
    **categorically**.
    """
    secret = "zzq-not-a-token-9471"

    _, out = _drive(session, monkeypatch, capsys, secret)

    assert secret not in out, "the gate echoed what the operator typed"
