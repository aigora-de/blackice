# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""A run whose surface is lost mid-flight must still report what it paid for (#85).

``gather`` runs once per epoch. Applying a fix at the human gate that removes a
named path — the intended workflow — meant epoch 2 could not reassemble the surface,
and the whole of ``engine.run`` was wrapped in one ``except SurfaceError``: the run
exited 2 having printed a single ``error:`` line, and the findings, participation,
coverage, permissions, surface composition and artefact from the epoch that DID
complete were discarded. The panel was already spawned; the money was already spent.

Filed against a version that printed four sections; it now discards seven and the
whole ``--- JSON ---`` artefact. The defect did not change — the cost of it grew.

These drive ``kuang.cli.main`` end to end with the subprocess boundary stubbed, so
the halt line, the exit code and the artefact are the real ones. The trigger is the
real one too: the gate deletes the file, which is what an operator applying a fix
does.

**Which are regressions, said plainly, and measured rather than assumed.** Run
against ``main`` with this file in place, five of the eight go red: the four under
"the defect" that assert on output, because nothing is printed at all, and
``test_a_healthy_run_says_so_in_the_artefact``, because the key does not exist yet —
that last one is a guard, not a regression, and what it pins does not move.

The other three pass on ``main``, and one passes **for the opposite reason**:
``test_a_lost_surface_keeps_the_exit_code_it_has_today`` sees 2 today from the
unreported error path, and 2 afterwards from an explicit mapping. It is here because
reporting the run without that mapping would have made it 0 — a test that passes
before and after, and is killed by three separate mutations in between.
"""

from __future__ import annotations

import json
import subprocess

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
def sourced_repo(git_repo, commit_all):
    """A repo with one reviewable file and a CLAUDE.md the panel is sourced from."""
    (git_repo / "a.py").write_text("def f():\n    return 1\n")
    (git_repo / "CLAUDE.md").write_text(_CLAUDE_MD)
    commit_all(git_repo, "init")
    return git_repo


def _contract(epoch: int) -> CallResult:
    """One open BLOCKER, distinct per epoch so a second epoch is reached at all."""
    body = json.dumps({"verdict": "NO", "findings": [
        {"title": f"off-by-one in the retry bound (epoch {epoch})",
         "severity": "BLOCKER", "claim_class": f"correctness-{epoch}",
         "file": "a.py", "line": 1}]})
    return CallResult(f"I reviewed it.\n\n```json\n{body}\n```", 18, num_turns=7)


def _stub(monkeypatch, repo, *, remove_at_gate: bool) -> None:
    """Stub the subprocess, and optionally apply the fix that removes the surface.

    The gate is where a human applies fixes, so the removal belongs there rather
    than in a patched ``gather``: it reproduces the operator's action, and leaves
    the backend's own refusal — and its message — entirely real.
    """
    epochs = {"n": 0}

    def _fake_call(self, prompt, mandate, tools, model):  # noqa: ANN001, ARG001
        return _contract(epochs["n"])

    real_gate = session_module.PanelSession.interactive_gate

    def _gate(self, result, run):  # noqa: ANN001
        decision = real_gate(self, result, run)
        if remove_at_gate:
            subprocess.run(["git", "-C", str(repo), "rm", "-q", "-f", "a.py"],
                           check=True, capture_output=True, text=True)
        return decision

    def _gather(self, epoch: int) -> str:
        epochs["n"] = epoch
        return real_gather(self, epoch)

    real_gather = session_module.PanelSession.gather
    monkeypatch.setattr(session_module.PanelSession, "_run_claude", _fake_call)
    monkeypatch.setattr(session_module.PanelSession, "interactive_gate", _gate)
    monkeypatch.setattr(session_module.PanelSession, "gather", _gather)


def _run(repo, *extra) -> int:
    return main(["--repo", str(repo), "--paths", "a.py", "--max-epochs", "3",
                 "--no-parallel", *extra])


def _artefact(out: str) -> dict:
    return json.loads(out.split("--- JSON ---")[-1])


# --- the defect ---------------------------------------------------------------

def test_a_lost_surface_still_reports_the_epoch_that_completed(sourced_repo, capsys,
                                                               monkeypatch):
    """REGRESSION for #85: this printed one ``error:`` line and nothing else."""
    _stub(monkeypatch, sourced_repo, remove_at_gate=True)
    _run(sourced_repo)
    out = capsys.readouterr().out

    assert "=== HALT: surface_lost after 1 epoch(s) ===" in out
    assert "[BLOCKER] (Analyst) off-by-one in the retry bound (epoch 1)" in out
    assert "converged" not in out, "no good verdict for a run that lost its surface"


def test_every_always_on_section_survives_the_loss(sourced_repo, capsys, monkeypatch):
    """The measured cost of the defect, section by section.

    Each of these is always-on and was discarded whole. Named individually rather
    than counted, so a section that stops being printed is caught by the assertion
    for that section rather than by an arithmetic that could be satisfied elsewhere.
    """
    _stub(monkeypatch, sourced_repo, remove_at_gate=True)
    _run(sourced_repo)
    out = capsys.readouterr().out

    assert "review surface: paths — a.py" in out          # #74
    assert "panel participation: 3 persona(s) x 1 epoch(s)" in out  # #30
    assert "panel coverage:" in out                       # #82
    assert "panel permissions: mode=" in out              # #67
    assert "semantic reduce:" in out                      # #30
    assert "--- JSON ---" in out


def test_the_output_says_which_epoch_could_not_be_reassembled(sourced_repo, capsys,
                                                              monkeypatch):
    """The halt line counts epochs that COMPLETED, so it says 1 and cannot say this.

    Both channels carry it: the operator's console, above the findings, because why
    the run stopped is read before what it found; and the artefact, for a run read
    back cold. The backend's own message rides along — it names the paths, which is
    what an operator needs to correct and what the engine could never produce.
    """
    _stub(monkeypatch, sourced_repo, remove_at_gate=True)
    _run(sourced_repo)
    out = capsys.readouterr().out
    payload = _artefact(out)

    assert ("surface lost: epoch 2 could not be reassembled — SurfaceError: "
            "no reviewable files in the requested paths: a.py") in out
    assert payload["halt_reason"] == "surface_lost"
    assert payload["surface_lost"]["epoch"] == 2
    assert "no reviewable files" in payload["surface_lost"]["detail"]
    report = out.split("=== HALT:")[-1]
    assert report.index("surface lost:") < report.index("[BLOCKER]"), (
        "within the final report — the epoch-1 synthesis at the gate prints "
        "findings of its own, long before the run knows the surface is gone")


def test_a_lost_surface_keeps_the_exit_code_it_has_today(sourced_repo, capsys,
                                                         monkeypatch):
    """2, unchanged — and #32 owns whether it stays that way.

    The report path returns ``3`` for an UGLY and ``0`` for everything else, so
    reporting this run without mapping it would exit **0**: a run that lost its
    surface reported as a success, to a caller and to a shell ``&&``. The code this
    path already returns is preserved instead. That invents nothing, and #32 is
    where the mapping is settled.

    Green on ``main`` and green after, for two different reasons — the unreported
    error path returns 2, and so does the mapping. Measured, it is killed by the
    mutation dropping the mapping, by the one returning the breaker's 3, and by the
    one reusing ``NO_REVIEW``, so it is load-bearing rather than vacuous.
    """
    _stub(monkeypatch, sourced_repo, remove_at_gate=True)
    rc = _run(sourced_repo)
    capsys.readouterr()

    assert rc == 2


def test_the_other_road_a_base_ref_that_stops_resolving(sourced_repo, capsys,
                                                        monkeypatch):
    """#85 names two roads to the same loss; this is the one path mode is not.

    Diff mode against a branch the operator then deletes at the gate. A different
    backend assembler (``gather_diff``, not ``build_path_surface``) and a different
    message — git's own three-line diagnosis, which the engine collapses to one so
    its continuation lines cannot print unprefixed inside the report block.
    """
    subprocess.run(["git", "-C", str(sourced_repo), "branch", "base-ref", "HEAD"],
                   check=True, capture_output=True, text=True)
    (sourced_repo / "a.py").write_text("def f():\n    return 2\n")
    subprocess.run(["git", "-C", str(sourced_repo), "commit", "-qam", "change"],
                   check=True, capture_output=True, text=True)

    epochs = {"n": 0}
    real_gather = session_module.PanelSession.gather
    real_gate = session_module.PanelSession.interactive_gate

    def _gather(self, epoch: int) -> str:
        epochs["n"] = epoch
        return real_gather(self, epoch)

    def _gate(self, result, run):  # noqa: ANN001
        decision = real_gate(self, result, run)
        subprocess.run(["git", "-C", str(sourced_repo), "branch", "-qD", "base-ref"],
                       check=True, capture_output=True, text=True)
        return decision

    monkeypatch.setattr(session_module.PanelSession, "_run_claude",
                        lambda self, p, m, t, mo: _contract(epochs["n"]))
    monkeypatch.setattr(session_module.PanelSession, "gather", _gather)
    monkeypatch.setattr(session_module.PanelSession, "interactive_gate", _gate)

    rc = main(["--repo", str(sourced_repo), "--base", "base-ref", "--max-epochs", "3",
               "--no-parallel"])
    out = capsys.readouterr().out
    report = out.split("=== HALT:")[-1]

    assert rc == 2
    assert "=== HALT: surface_lost after 1 epoch(s) ===" in out
    assert "unknown revision" in _artefact(out)["surface_lost"]["detail"]
    lost = [ln for ln in report.splitlines() if ln.startswith("surface lost:")]
    assert len(lost) == 1
    assert "Use '--' to separate paths" in lost[0], (
        "git's continuation lines belong ON the line, not under it")
    assert not any(ln.startswith("Use '--'") for ln in report.splitlines()), (
        "an unprefixed continuation line reads as a section of the report")


# --- the boundary: what this change deliberately does NOT touch ----------------

def test_a_first_epoch_with_no_surface_is_unchanged(sourced_repo, capsys, monkeypatch):
    """GUARD for #18: no surface, no review, no verdict, exit 2.

    Nothing has been spent and there is nothing to report, so the operator error is
    reported on stderr and no halt line is printed for a panel that never ran. The
    fix sits on this same path, and one that quietly made the first epoch report an
    empty run would invert the doctrine the whole surface guard rests on.

    ``tests/backends/claude_code/test_gather.py`` already drives the CLI on this
    path, in **diff** mode, asserting ``rc != 0`` and no halt line. This adds path
    mode, the exact code, and the absence of the artefact — and it sits beside the
    lost-surface tests, so the asymmetry the fix turns on is visible in one file
    rather than inferred across two. The rest of that coverage is inherited and this
    change neither widens nor fills it.
    """
    _stub(monkeypatch, sourced_repo, remove_at_gate=False)
    subprocess.run(["git", "-C", str(sourced_repo), "rm", "-q", "-f", "a.py"],
                   check=True, capture_output=True, text=True)
    rc = _run(sourced_repo)
    captured = capsys.readouterr()

    assert rc == 2
    assert "no reviewable files in the requested paths: a.py" in captured.err
    assert "=== HALT:" not in captured.out
    assert "--- JSON ---" not in captured.out


def test_a_healthy_run_reports_no_loss_on_the_console(sourced_repo, capsys,
                                                      monkeypatch):
    """MIRROR IMAGE: a rule that fires on a healthy run is the defect reversed."""
    _stub(monkeypatch, sourced_repo, remove_at_gate=False)
    rc = _run(sourced_repo)
    out = capsys.readouterr().out

    assert "surface lost:" not in out
    assert "surface_lost" not in _artefact(out)["halt_reason"]
    assert rc == 0, "the exit code a run without a lost surface already had"


def test_a_healthy_run_says_so_in_the_artefact(sourced_repo, capsys, monkeypatch):
    """The key is always present, and ``null`` is the claim.

    An absent key makes no claim, and a reader coming to an artefact cold cannot
    otherwise tell a run that kept its surface from one written before the field
    existed. The same rule ``agreement`` and ``panel.discarded`` follow.
    """
    _stub(monkeypatch, sourced_repo, remove_at_gate=False)
    _run(sourced_repo)
    payload = _artefact(capsys.readouterr().out)

    assert "surface_lost" in payload
    assert payload["surface_lost"] is None
