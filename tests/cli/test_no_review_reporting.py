# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""A run that reviewed nothing must not report a good verdict (#72).

The purest member of the class this epoch is for: the instrument reporting a
review that never happened. A dry run spawns nothing by design, and every one of
``CONVERGED``'s conjuncts is satisfied by **absence** — no open blocker, no open
ugly, and a quorum of votes the backend fabricated for calls nobody made. So the
pre-flight check printed a clean bill of health, in the same output that said,
four sections further down, that seven personas were never spawned.

These drive ``kuang.cli.main`` end to end with the subprocess boundary stubbed, so
the halt line, the exit code and the ``--- JSON ---`` artefact are the real ones.

Read this file as three claims, and the second two are what keep the first honest:

* a dry run does not report a good verdict, and says which reason it stopped for;
* it still reports everything it legitimately knows — four issues put pre-flight
  material there and none of it is measurement, so none of it is lost;
* a **real** panel that ran and agreed still converges, with the same halt reason,
  the same epoch count and the same exit code as before this change. A rule that
  fires on a healthy run is the mirror image of the defect it was written for.

**Which of these are regression tests, said plainly.** Only the first two go red on
``main``: they are the defect. The other six pass before this change and after it,
because what they pin is what must **not** move — the trap the fix must not spring,
the pre-flight material the rejected option would have deleted, the exit code #32
owns, and the healthy run. They are guards, not regressions, and they earn their
place in the mutation matrix rather than on today's diff: each is killed by a
mutation implementing a design that was considered and rejected.
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

_NO_JSON = CallResult("I read the diff and have nothing structured to say.", 0)


def _contract(verdict="YES", findings=()) -> CallResult:
    body = json.dumps({"verdict": verdict, "findings": list(findings)})
    return CallResult(f"I reviewed it.\n\n```json\n{body}\n```", 0, num_turns=7)


@pytest.fixture
def sourced_repo(changed_repo):
    """``changed_repo`` with a CLAUDE.md the panel is sourced from."""
    (changed_repo / "CLAUDE.md").write_text(_CLAUDE_MD)
    return changed_repo


def _stub(monkeypatch, *, reply=None) -> None:
    def _fake(self, prompt, mandate, tools, model):  # noqa: ANN001, ARG001
        if "formatter" in mandate:
            return _NO_JSON
        return reply if reply is not None else _contract()

    monkeypatch.setattr(session_module.PanelSession, "_run_claude", _fake)


def _run(repo, *extra) -> int:
    return main(["--repo", str(repo), "--base", "HEAD~1", "--max-epochs", "1",
                 "--no-parallel", *extra])


def _artefact(out: str) -> dict:
    return json.loads(out.split("--- JSON ---")[-1])


# --- the defect ---------------------------------------------------------------

def test_a_dry_run_does_not_report_a_good_verdict(sourced_repo, capsys, monkeypatch):
    """REGRESSION for #72: this printed ``=== HALT: converged after 1 epoch(s) ===``."""
    _stub(monkeypatch)
    _run(sourced_repo, "--dry-run")
    out = capsys.readouterr().out

    assert "=== HALT: no_review after 1 epoch(s) ===" in out
    assert _artefact(out)["halt_reason"] == "no_review"
    assert "converged" not in out, "no good verdict, anywhere in the output"


def test_a_dry_run_agrees_with_its_own_participation_record(sourced_repo, capsys,
                                                            monkeypatch):
    """The two halves of the artefact that used to contradict each other.

    ``halt_reason`` and the participation block were both already in the same JSON
    object — one saying the panel agreed, the other that nobody was spawned — and
    nothing reconciled them. A reader should not have to.
    """
    _stub(monkeypatch)
    _run(sourced_repo, "--dry-run")
    payload = _artefact(capsys.readouterr().out)

    assert payload["halt_reason"] == "no_review"
    assert all(r["status"] == "not_spawned" for r in payload["participation"])


def test_a_multi_epoch_dry_run_still_stops_after_one_epoch(sourced_repo, capsys,
                                                           monkeypatch):
    """What stops epoch 2, now that a fabricated quorum no longer does.

    The trap this issue records: refusing the vote alone makes ``CONVERGED``
    unreachable, and the loop then runs the whole epoch budget re-printing wiring
    — asking the operator at each gate at a terminal, and silently repeating
    itself when piped. The halt reason is the half that prevents it, and one epoch
    is what it produced before, so an operator sees no new behaviour.
    """
    _stub(monkeypatch)
    _run(sourced_repo, "--dry-run", "--max-epochs", "3")
    out = capsys.readouterr().out

    assert _artefact(out)["epochs"] == 1
    assert "=== epoch 1 synthesis ===" not in out, "the gate was never reached"


def test_a_dry_run_still_exits_zero(sourced_repo, capsys, monkeypatch):
    """Where this issue stops and #32 starts, asserted rather than assumed.

    A pre-flight check that succeeded returns success. Whether an exit code should
    distinguish halt outcomes AT ALL is #32, which owns the whole mapping — and
    ``no_review`` is a strong candidate there, being an outcome with nothing
    *examined* rather than nothing *open*. Changing one reason's code here would
    settle a general question as a side effect of a reporting fix.
    """
    _stub(monkeypatch)
    exit_code = _run(sourced_repo, "--dry-run")
    capsys.readouterr()

    assert exit_code == 0


# --- what a dry run must still say --------------------------------------------

def test_a_dry_run_still_reports_all_its_pre_flight_material(sourced_repo, capsys,
                                                             monkeypatch):
    """The standing guard on the option this issue rejected.

    Stopping before the loop entirely was the cheapest fix when #72 was filed, one
    day after #30. It is not any more: four issues deliberately made the dry run
    carry pre-flight material that needs no subprocess to be true — the argv it
    would spawn (#19), what it *would* have reviewed in files and bytes (#69, #74),
    the tools the policy removes (#67) — and every one of those sections is printed
    from the report block after the loop. Deleting them to fix the halt line would
    trade a defect for four regressions, so this asserts they all survive together
    rather than leaving each to its own issue's test.
    """
    _stub(monkeypatch)
    _run(sourced_repo, "--dry-run")
    out = capsys.readouterr().out
    payload = _artefact(out)

    assert "[dry-run] persona=Analyst" in out
    assert "  argv= " in out
    assert "review surface: diff — HEAD~1...HEAD" in out
    assert payload["surface"][0]["files"] == 1
    assert "panel participation: 3 persona(s) x 1 epoch(s)" in out
    assert "panel permissions: mode=plan" in out
    assert "nothing was spawned" in out


def test_a_dry_run_still_claims_no_measurement_it_did_not_take(sourced_repo, capsys,
                                                               monkeypatch):
    """The other side of the section above, and the reason it is not a free pass.

    Reporting more is not better: #67, #70 and #71 each had to stop the dry run
    claiming something it had not measured. A new halt reason must not become a
    licence to fill the gap with the rest.
    """
    _stub(monkeypatch)
    _run(sourced_repo, "--dry-run")
    out = capsys.readouterr().out
    payload = _artefact(out)

    assert all(r["turns"] == 0 for r in payload["participation"])
    assert "CALLED NO TOOL" not in out
    assert payload["permissions"]["refusals"] == []
    assert payload["findings"] == []


# --- the mirror image: a real run is untouched --------------------------------

def test_a_real_unanimous_panel_still_converges(sourced_repo, capsys, monkeypatch):
    """The mirror image of the defect, and the pair to every test above.

    Every persona ran, reviewed, and voted YES. If this run reported anything but
    the good verdict it earned, the fix would be the same instrument failure the
    other way round — and every "a dry run does not converge" test above would pass
    trivially in a world where nothing ever converges.
    """
    _stub(monkeypatch)
    exit_code = _run(sourced_repo)
    payload = _artefact(capsys.readouterr().out)

    assert payload["halt_reason"] == "converged"
    assert payload["epochs"] == 1
    assert exit_code == 0
    assert all(r["status"] == "found_nothing" for r in payload["participation"])


def test_a_panel_with_one_persona_unspawned_is_not_a_no_review_run(sourced_repo,
                                                                   capsys,
                                                                   monkeypatch):
    """A mixed panel is the case only the vote rule reaches.

    ``no_review`` is about a run in which the panel produced no review at all. A
    panel where *some* personas failed still reviewed, so it halts on an ordinary
    reason — and the failed one simply does not help the survivors reach quorum,
    which is the half of the fix that has nothing to do with dry runs.
    """
    def _fake(self, prompt, mandate, tools, model):  # noqa: ANN001, ARG001
        if "formatter" in mandate:
            return _NO_JSON
        if "You are Sentinel " in mandate:
            return CallResult("", 0, error="agent error: the runtime returned nothing")
        return _contract()

    monkeypatch.setattr(session_module.PanelSession, "_run_claude", _fake)
    _run(sourced_repo)
    payload = _artefact(capsys.readouterr().out)

    assert payload["halt_reason"] not in ("no_review", "converged")
    assert sorted(r["status"] for r in payload["participation"]) == [
        "agent_error", "found_nothing", "found_nothing"]
