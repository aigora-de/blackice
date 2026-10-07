# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""A run whose gate failed must publish that, and must still publish itself (#129).

The engine's half is ``tests/engine/test_unguarded_gate.py``: the seam is guarded, a
decision is substituted, and a ``GateFailure`` is recorded on the epoch. This is the
published half. Without it the artefact says ``asked: null`` and nothing more, which
is the same thing it says for a gate that answered nothing and for one that could not
read the answer (#132) — three states, one record, and no key telling them apart.

One new top-level key, ``gate_failures``, always present and ``[]`` where nothing
failed. **A list rather than a singleton**, which is the one place it does not copy
``surface_lost``: a lost surface ends the run so one record is the whole story, while
a gate failure does not and the triggers are environmental, so a gate that fails once
usually fails at every epoch after. The ``gate[]`` rows are deliberately **untouched**
— five tests pin them by exact equality, and #117's settled shape is not this issue's
to move; #139 owns the console wording of this state and is cited rather than
pre-empted.

These drive ``kuang.cli.main`` end to end over a throwaway repo with the subprocess
boundary stubbed — nothing spawned, no network. The gate is stubbed at
``PanelSession.interactive_gate``, which is the seam the engine calls: a failure
there is the shipped backend's own failure mode, not a synthetic one.

**Which are regressions, measured on ``main`` with this file in place rather than
assumed.** Recorded in the matrix in the PR body; the ones that are not regressions
say so in their own docstrings.
"""

from __future__ import annotations

import json

import pytest

from kuang.backends.claude_code import session as session_module
from kuang.backends.claude_code.spawn import CallResult
from kuang.cli import main

_CLAUDE_MD = """\
# A repo

# Resident Experts

## Analyst — Correctness

Does the change compute the right thing?

## Critic — Completeness

Find what everyone else missed; assume shared blind spots.

## Sentinel — Ruin

Hunt ruin-class hazards only.
"""


@pytest.fixture
def sourced_repo(changed_repo):
    (changed_repo / "CLAUDE.md").write_text(_CLAUDE_MD)
    return changed_repo


def _contract(epoch: int) -> CallResult:
    """One open BLOCKER, distinct per epoch so a later epoch is reached at all.

    An identical finding deduplicates into the ledger and STALLs the run on epoch 2,
    which would stop it before a second gate failure could exist to report.
    """
    body = json.dumps({"verdict": "NO", "findings": [
        {"title": f"off-by-one in the retry bound (epoch {epoch})",
         "severity": "BLOCKER", "claim_class": f"correctness-{epoch}",
         "file": "a.py", "line": 1}]})
    return CallResult(f"I reviewed it.\n\n```json\n{body}\n```", 18, num_turns=7)


class _GateError(RuntimeError):
    """What a backend's gate raises when it cannot prompt at all."""


def _stub(monkeypatch, *, gate=None) -> None:
    """Stub the subprocess boundary, and optionally replace the gate seam.

    ``gate`` is a two-argument callable or a sentinel value: passing a callable
    makes the seam RAISE or compute, and ``_returning(value)`` makes it hand back
    something that is not a ``GateDecision``. Left as None, the real
    ``interactive_gate`` runs and returns early because stdin is not a terminal.
    """
    epochs = {"n": 0}
    real_gather = session_module.PanelSession.gather

    def _fake_call(self, prompt, mandate, tools, model):  # noqa: ANN001, ARG001
        if "careful synthesiser" in mandate or "formatter" in mandate:
            return CallResult("nothing structured to say.", 0)
        return _contract(epochs["n"])

    def _gather(self, epoch: int) -> str:  # noqa: ANN001
        epochs["n"] = epoch
        return real_gather(self, epoch)

    monkeypatch.setattr(session_module.PanelSession, "_run_claude", _fake_call)
    monkeypatch.setattr(session_module.PanelSession, "gather", _gather)
    if gate is not None:
        monkeypatch.setattr(session_module.PanelSession, "interactive_gate", gate)


def _raising_gate(self, result, run):  # noqa: ANN001, ARG001
    """The live trigger, unstubbed in spirit: the gate prints before it prompts.

    ``interactive_gate`` prints the epoch synthesis above the prompt and outside
    every guard, so an ASCII stdout or a closed pipe raises there. Raised directly
    here rather than by breaking stdout, because a test that broke the capture
    stream would be measuring pytest's plumbing rather than the engine's guard.
    """
    raise _GateError("cannot prompt: stdout is closed")


def _returning(value):
    def gate(self, result, run):  # noqa: ANN001, ARG001
        return value
    return gate


def _run(repo, *extra) -> int:
    """Two epochs by default, so one gate is reached and one epoch halts."""
    return main(["--repo", str(repo), "--base", "HEAD~1", "--max-epochs", "2",
                 "--stall-patience", "2", "--no-parallel", *extra])


def _artefact(out: str) -> dict:
    return json.loads(out.split("--- JSON ---")[-1])


def _account_block(out: str) -> list[str]:
    """The end-of-run account's header and its indented lines, and nothing else.

    Duplicated from ``tests/cli/test_gate_reporting.py`` rather than promoted to
    ``tests/conftest.py``: promotion would edit a module this issue does not
    otherwise touch, which the change-discipline rule refuses. Other sections also
    print lines beginning "  epoch 1:", so the block has to be sliced rather than
    grepped — the first shape of this assertion matched four of them.
    """
    lines = out.split("--- JSON ---")[0].splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("epoch account:"))
    block = [lines[start]]
    for line in lines[start + 1:]:
        if not line.startswith("  "):
            break
        block.append(line)
    return block


# --- the defect: a failed gate is published ------------------------------------

def test_a_raising_gate_is_published_with_the_epoch_and_the_diagnosis(sourced_repo,
                                                                     capsys,
                                                                     monkeypatch):
    """REGRESSION for #129: the run raised out of ``main`` and printed no artefact."""
    _stub(monkeypatch, gate=_raising_gate)
    rc = _run(sourced_repo)
    payload = _artefact(capsys.readouterr().out)

    detail = "_GateError: cannot prompt: stdout is closed"
    assert rc == 0
    assert payload["epochs"] == 2, "both epochs completed and are published"
    # Exact equality over the whole row, like ``gate[]``: a key arriving here should
    # be somebody's decision rather than a drift. ``detail_chars`` is written as
    # ``len(detail)`` rather than a literal — the rule is that they agree where
    # nothing was cut, and a literal would pin an arithmetic nobody can read.
    assert payload["gate_failures"] == [
        {"epoch": 1, "detail": detail, "detail_chars": len(detail)}], \
        "the run does not say its gate failed"


def test_the_detail_chars_half_says_the_diagnosis_was_not_cut(sourced_repo, capsys,
                                                             monkeypatch):
    """#111's structural half, so a reader asks a number rather than reading prose.

    "Was this cut" is ``detail_chars > 400``, exactly knowable and unaffected by any
    rewording of the marker the string carries. The same pair ``surface_lost``
    publishes, and the same reason it is present when nothing was cut.
    """
    _stub(monkeypatch, gate=_raising_gate)
    _run(sourced_repo)
    row = _artefact(capsys.readouterr().out)["gate_failures"][0]

    assert row["detail_chars"] == len(row["detail"]) <= 400
    assert "truncated" not in row["detail"]


def test_a_gate_that_fails_at_each_epoch_publishes_each_failure(sourced_repo, capsys,
                                                                monkeypatch):
    """The multiplicity a run-level singleton would silently drop.

    Three epochs, so two gates are reached and both fail: a singleton would publish
    the first and say nothing about the second, which reads as a human-gated run
    rather than the ungated one it became. The third epoch halts on the ceiling and
    never reaches the gate, so it has nothing to report — ``loop.run`` breaks before
    the gate, which is #117's rule and not an off-by-one here.
    """
    _stub(monkeypatch, gate=_raising_gate)
    _run(sourced_repo, "--max-epochs", "3", "--stall-patience", "3")
    payload = _artefact(capsys.readouterr().out)

    assert payload["epochs"] == 3
    assert [row["epoch"] for row in payload["gate_failures"]] == [1, 2]


def test_a_foreign_return_value_is_published_as_a_failure_by_type(sourced_repo,
                                                                 capsys,
                                                                 monkeypatch):
    """A duck-typed ``.stop`` is the case an attribute-shaped guard lets through.

    Measured on ``c1e0a80``: the engine survived this one and stored the object in a
    field typed ``GateDecision | None``, so the artefact published ``reached: true``
    off a value nothing had checked. It is NOT an artefact-destruction defect —
    ``_asked`` resolves by identity and ``stopped`` is coerced, so #131 already
    stopped a foreign value being serialised. What this pins is that the unchecked
    claim is now reported instead.
    """
    class _Ducked:
        stop = False
        asked = "yes"

    _stub(monkeypatch, gate=_returning(_Ducked()))
    _run(sourced_repo)
    payload = _artefact(capsys.readouterr().out)

    detail = "gate returned _Ducked, not GateDecision"
    assert payload["gate_failures"] == [
        {"epoch": 1, "detail": detail, "detail_chars": len(detail)}]
    assert payload["gate"][0]["asked"] is None


def test_a_gate_returning_none_is_published_as_a_failure(sourced_repo, capsys,
                                                         monkeypatch):
    """``None`` crashed at the ``.stop`` read, AFTER the record had been stored.

    The record was written before the value was read, so an epoch whose gate RAN
    was left rendering as "the epoch halted, so the gate was never reached". The
    assignment moved below the guard for that reason, and this is the published
    half of it: the epoch says its gate was reached AND says what went wrong.
    """
    _stub(monkeypatch, gate=_returning(None))
    _run(sourced_repo)
    payload = _artefact(capsys.readouterr().out)

    assert payload["gate"][0]["reached"] is True
    assert payload["gate_failures"][0]["detail"] == (
        "gate returned NoneType, not GateDecision")


def test_the_artefact_never_carries_what_a_foreign_object_renders(sourced_repo,
                                                                 capsys,
                                                                 monkeypatch):
    """``repr`` of a foreign object is unbounded text from outside this codebase.

    The artefact is published, so a caller's or a model's ``__repr__`` would be
    publishing whatever it returns — bounding it would keep 400 characters of it
    rather than none. The type's name is the exactly-knowable fact, which is #131's
    rule one field along. The sibling of
    ``test_the_artefact_never_carries_what_the_operator_typed``.
    """
    class _Loud:
        stop = False

        def __repr__(self) -> str:
            return "DO-NOT-PUBLISH-" + "x" * 2000

    _stub(monkeypatch, gate=_returning(_Loud()))
    _run(sourced_repo)
    out = capsys.readouterr().out

    assert "DO-NOT-PUBLISH" not in out, "not in the artefact and not on the console"
    assert _artefact(out)["gate_failures"][0]["detail"] == (
        "gate returned _Loud, not GateDecision")


# --- what a failed gate must NOT move ------------------------------------------

def test_the_gate_rows_keep_the_shape_settled_by_117(sourced_repo, capsys,
                                                     monkeypatch):
    """Pinned by exact equality, because #117's rows are not this issue's to move.

    ``reached`` stays true on the epoch whose gate failed: the gate WAS reached and
    it did run, and publishing false for it would be the falser of the two claims.
    ``asked`` is ``null`` rather than ``false`` — ``false`` is the measured
    non-interactive case, and a seam that failed measured nothing.
    """
    _stub(monkeypatch, gate=_raising_gate)
    _run(sourced_repo)
    payload = _artefact(capsys.readouterr().out)

    assert payload["gate"] == [
        {"epoch": 1, "reached": True, "asked": None, "stopped": False},
        {"epoch": 2, "reached": False, "asked": None, "stopped": None}], \
        "a key arriving in this row should be somebody's decision, not a drift"


def test_every_always_on_section_survives_a_failed_gate(sourced_repo, capsys,
                                                        monkeypatch):
    """The measured cost of the defect, section by section.

    Each of these is always-on and was discarded whole when the gate raised. Named
    individually rather than counted, so a section that stops being printed is
    caught by its own assertion rather than by an arithmetic satisfied elsewhere —
    ``test_lost_surface_reporting.py``'s discipline at the sibling seam.
    """
    _stub(monkeypatch, gate=_raising_gate)
    _run(sourced_repo)
    out = capsys.readouterr().out

    assert "review surface: diff" in out                            # #74
    assert "panel participation: 3 persona(s) x 2 epoch(s)" in out   # #30
    assert "panel coverage:" in out                                 # #82
    assert "panel permissions: mode=" in out                        # #67
    assert "semantic reduce:" in out                                # #30
    assert "epoch account:" in out                                  # #117
    assert "--- JSON ---" in out


def test_the_console_account_is_left_to_139(sourced_repo, capsys, monkeypatch):
    """The end-of-run line for this state is #117's settled surface, and it is coarse.

    A substituted decision records ``asked=None``, so ``_gate_outcome`` already
    renders the epoch as *"continued; the gate did not say whether a human was
    asked"* — true, and unable to distinguish a seam that failed from a gate that
    said nothing. Sharpening it moves a string a test pins and a capture would show,
    which is **#139**'s subject. Asserted here so that landing #139 has to change
    this line deliberately rather than discover it.
    """
    _stub(monkeypatch, gate=_raising_gate)
    _run(sourced_repo)
    out = capsys.readouterr().out.split("--- JSON ---")[0]

    # Pinned as the EXACT line, not as the absence of a phrase. The first shape of
    # this test asserted ``"gate failed" not in out``, and that substring exists
    # nowhere in ``kuang/`` — it passed on main, passes here, and would keep passing
    # under any wording #139 chose, so it pinned nothing. An exact line is what makes
    # #139 change it deliberately.
    assert _account_block(out)[1] == (
        "  epoch 1: new findings: 1 | new material issues: 1 | open blockers: 1 | "
        "open uglies: 0 — continued; the gate did not say whether a human was asked"), \
        "the end-of-run account of a failed gate is #139's surface, not this issue's"


def test_a_healthy_run_says_so_in_the_artefact(sourced_repo, capsys, monkeypatch):
    """The key is always present, and ``[]`` is the claim.

    An absent key makes no claim, and a reader coming to an artefact cold cannot
    otherwise tell a run whose gate held from one written before the field existed.
    The rule ``surface_lost`` and ``agreement`` follow.

    **The MIRROR IMAGE as well**: "a failed gate is published" is satisfied by
    publishing a failure every epoch, which would turn every run into a degraded
    one. This is the partner that forbids it, and it is the vacuity probe's target.

    **Red on main only for the missing key** — what it pins about the gate rows
    passes before and after. It is a guard, not a regression.
    """
    _stub(monkeypatch)
    _run(sourced_repo)
    payload = _artefact(capsys.readouterr().out)

    assert "gate_failures" in payload
    assert payload["gate_failures"] == []
    assert payload["gate"][0]["asked"] is False, (
        "the measured non-interactive case, which a failure must not be read as")
