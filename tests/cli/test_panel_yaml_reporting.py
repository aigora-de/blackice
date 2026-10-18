# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""An ill-typed ``panel.yaml``, end to end, and the grant each persona really had (#115).

The field contract is pinned in ``tests/backends/claude_code/test_panel_yaml_contract``.
This module pins what an operator sees:

* ``--panel`` naming an ill-typed file is refused at rc 2 with one ``error:`` line,
  nothing on stdout and nothing spawned — run WITHOUT ``--dry-run``, so "before any
  spend" is measured, not assumed;
* an ill-typed implicit ``panel.yaml`` is reported as a discard and the run goes on;
* two personas with different grants are each SPAWNED with their own, which a
  duplicate name broke (S13′), PAIRED WITH the duplicate itself refused;
* the console's ``[panel] tools=`` line agrees with the artefact persona by persona
  (O2, folded into #115). It printed ``personas[0].tools`` alone, so a panel whose
  personas differed was reported as all having the first one's grant — measured
  hiding a declared ``Bash`` behind ``['Read', 'Grep', 'Glob']``. PAIRED WITH a
  uniform panel, whose line is byte-identical to before.

Every test needs the real parser and skips where the optional extra is absent.

**Which are regressions, measured on main with this file in place.** Recorded in the PR
body; any test green on main says so in its own docstring.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("yaml")

from kuang.backends.claude_code import session as session_module  # noqa: E402
from kuang.backends.claude_code.spawn import CallResult  # noqa: E402
from kuang.cli import main  # noqa: E402

_CLEAN = CallResult('I reviewed it.\n\n```json\n{"verdict": "YES", "findings": []}\n```',
                    18, num_turns=7)

_DISTINCT = """\
personas:
  - name: Auditor
    grounding: Audit it.
    tools: [Read]
  - name: Adversary
    grounding: Break it.
    tools: [Grep]
    model: other-model
"""


def _stub(monkeypatch) -> dict:
    """Stub the reviewer and record each call's mandate, tools and model."""
    state = {"calls": []}

    def _fake_call(self, prompt, mandate, tools, model):  # noqa: ANN001, ARG001
        state["calls"].append((mandate, list(tools), model))
        return _CLEAN

    monkeypatch.setattr(session_module.PanelSession, "_run_claude", _fake_call)
    return state


def _argv(repo, *extra) -> list[str]:
    return ["--repo", str(repo), "--base", "HEAD~1", "--max-epochs", "1",
            "--no-parallel", *extra]


def _artefact(out: str) -> dict:
    return json.loads(out.split("--- JSON ---")[-1])


def test_an_ill_typed_explicit_panel_is_refused_before_anything_is_spent(
        changed_repo, tmp_path, capsys, monkeypatch):
    """REGRESSION for #115 (W1): rc 0, with ``--allowedTools R e a d`` and nothing said."""
    state = _stub(monkeypatch)
    panel = tmp_path / "x.yaml"
    panel.write_text("personas:\n  - name: Auditor\n    tools: Read\n")

    rc = main(_argv(changed_repo, "--panel", str(panel)))

    captured = capsys.readouterr()
    assert rc == 2
    assert state["calls"] == [], "a refusal comes before any spend"
    assert captured.out == "", "a refusal comes before the first [panel] line"
    assert captured.err.splitlines() == [
        f"error: --panel {panel}: it could not be read or parsed "
        f"(personas[0].tools: str, not a list of strings)"]


def test_an_ill_typed_implicit_panel_is_reported_and_the_run_goes_on(
        changed_repo, capsys, monkeypatch):
    """REGRESSION for #115 (G1): a traceback at rc 1, nothing on stdout."""
    state = _stub(monkeypatch)
    (changed_repo / "panel.yaml").write_text(
        "personas:\n  - name: Auditor\n    grounding: 42\n")

    rc = main(_argv(changed_repo))

    out = capsys.readouterr().out
    assert rc == 0
    assert state["calls"], "the default panel ran"
    assert ("[panel]   panel.yaml: it could not be read or parsed — "
            "personas[0].grounding: int, not a string") in out.splitlines()
    assert _artefact(out)["panel"]["discarded"] == [
        {"origin": "panel.yaml", "reason": "malformed",
         "detail": "personas[0].grounding: int, not a string"}]


def test_two_distinct_personas_are_each_spawned_with_their_own_grant(
        changed_repo, tmp_path, capsys, monkeypatch):
    """PAIR, green on main: each persona's tools and model reach its own call.

    The invariant is asserted here on its own, not only against the duplicate below,
    so a mutation that handed every persona the same grant cannot pass by changing
    both sides alike.
    """
    state = _stub(monkeypatch)
    panel = tmp_path / "x.yaml"
    panel.write_text(_DISTINCT)

    rc = main(_argv(changed_repo, "--panel", str(panel)))

    capsys.readouterr()
    assert rc == 0
    by_mandate = {m: (t, model) for m, t, model in state["calls"]}
    assert by_mandate["Audit it."] == (["Read"], None)
    assert by_mandate["Break it."] == (["Grep"], "other-model")


def test_a_duplicate_name_is_refused_rather_than_run_with_the_others_grant(
        changed_repo, tmp_path, capsys, monkeypatch):
    """REGRESSION for #115 (S13′): rc 0, the first persona spawned with ``--allowedTools
    Grep`` (the second's), while the console and the artefact said ``Read``."""
    state = _stub(monkeypatch)
    panel = tmp_path / "x.yaml"
    panel.write_text(_DISTINCT.replace("Adversary", "Auditor"))

    rc = main(_argv(changed_repo, "--panel", str(panel)))

    captured = capsys.readouterr()
    assert rc == 2
    assert state["calls"] == []
    assert captured.out == ""
    assert captured.err.splitlines() == [
        f"error: --panel {panel}: it could not be read or parsed "
        f"(personas[1].name: duplicates personas[0].name)"]


def test_the_console_lists_each_persona_grant_when_they_differ(
        changed_repo, tmp_path, capsys, monkeypatch):
    """REGRESSION for O2: ``[panel] tools=['Read']`` for a panel with four grants.

    The injected specialists keep the default grant, so a panel declaring its own
    tools always differs from them — and the old line said otherwise for both. The
    console must agree with the artefact's ``permissions.personas[].granted``, which
    was already correct, persona by persona and in the same order.
    """
    _stub(monkeypatch)
    panel = tmp_path / "x.yaml"
    panel.write_text(_DISTINCT)

    main(_argv(changed_repo, "--panel", str(panel)))

    lines = capsys.readouterr().out.splitlines()
    granted = [(r["persona"], r["granted"])
               for r in _artefact("\n".join(lines))["permissions"]["personas"]]
    at = lines.index("[panel] tools differ by persona; mode=plan")
    assert lines[at + 1:at + 1 + len(granted)] == [
        f"[panel]   {name}: tools={tools}" for name, tools in granted]
    assert granted[:2] == [("Auditor", ["Read"]), ("Adversary", ["Grep"])]
    assert not any(line.startswith("[panel] tools=") for line in lines)


def test_a_uniform_panel_prints_the_line_it_always_did(
        changed_repo, tmp_path, capsys, monkeypatch):
    """PAIR, green on main: one grant shared by every persona is one true line.

    A rule that fires on a healthy run is the mirror image of the defect, and every
    capture on record has a uniform panel, so its console must not move.
    """
    _stub(monkeypatch)
    panel = tmp_path / "x.yaml"
    panel.write_text("personas:\n  - name: Auditor\n    grounding: Audit it.\n")

    main(_argv(changed_repo, "--panel", str(panel)))

    lines = capsys.readouterr().out.splitlines()
    assert "[panel] tools=['Read', 'Grep', 'Glob'] mode=plan" in lines
    assert not any("differ by persona" in line for line in lines)
