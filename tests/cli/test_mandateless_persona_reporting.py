# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""A persona convened with no mandate must be reported, not convened silently (#103).

The last purely-sourcing member of #40's first exit criterion, and the first found
by *looking* for one. The class test is **distinguishability**: before this, two
runs differing only in whether half the panel had a brief were **byte-identical**
in their complete output — console, ``panel``, ``coverage``, ``participation`` and
``surface`` alike. No existing field could carry it, so the fix adds a channel
rather than strengthening one.

What an empty mandate costs, measured rather than asserted: a persona's whole
identity at the subprocess boundary is ``--append-system-prompt <grounding>``, and
its *name* never reaches the call at all. Two mandate-less personas therefore spawn
a **byte-identical** argv — the panel is duplicated, not merely under-briefed, and
the unanimity behind ``converged`` counts one voice twice.

Three boundaries the tests below pin, because each is a decision and not a detail:

* **no mandate is absent, null, or empty after strip** — exactly knowable, no
  threshold. A one-word mandate is NOT reported: that is persona *quality*
  (#2/#34), and a minimum length would be a threshold on a value nobody has
  baselined;
* **report, never refuse** — on every tier, ``--panel`` included. #16's asymmetry
  refuses an explicit instruction that cannot be *carried out*; here it is carried
  out exactly as written, and nothing is substituted;
* **the injected specialists are never reported**, because they carry a mandate by
  construction — a rule that fires on them would be the mirror image of the defect.

Driven through ``kuang.cli.main`` end to end with nothing spawned. The YAML tier is
the only one that can reach this state (``parse_claude_md_experts`` always builds
at least ``"You are <name> — <role>."``), and ``pyyaml`` is an optional extra this
venv does not carry, so the parser is stubbed exactly as
``backends/claude_code/test_personas.py`` stubs it.
"""

from __future__ import annotations

import json
import sys
import types

import pytest

from kuang.cli import main

_GOOD_EXPERTS = """\
# A repo

# Resident Experts

## Analyst — Correctness

Does the change compute the right thing?

## Sentinel — Ruin

Hunt ruin-class hazards only.
"""


def _stub_yaml(monkeypatch, payload):
    """A ``yaml`` module whose ``safe_load`` returns ``payload``.

    The real parser produces exactly these shapes: an absent ``grounding`` key is
    absent from the mapping, and ``grounding:`` with nothing after it is ``None``.
    Both were confirmed against pyyaml 6.0.3 before this file was written, because
    a stub asserting a shape nobody measured tests the stub.
    """
    stub = types.ModuleType("yaml")
    stub.safe_load = lambda _text: payload
    monkeypatch.setitem(sys.modules, "yaml", stub)


def _panel(monkeypatch, repo, personas):
    """A ``panel.yaml`` declaring ``personas``, parsed by the stub above."""
    (repo / "panel.yaml").write_text("personas: …\n")
    _stub_yaml(monkeypatch, {"personas": personas})


def _run(repo, *extra) -> int:
    return main(["--repo", str(repo), "--base", "HEAD~1", "--max-epochs", "1",
                 "--no-parallel", "--dry-run", *extra])


def _artefact(out: str) -> dict:
    return json.loads(out.split("--- JSON ---")[-1])


def _warning(out: str) -> str:
    """The mandate-less warning LINE, or "" if the run printed none.

    Anchored to the line rather than searched for in the whole output, because
    the roster line two lines above it prints the same names in the same
    comma-joined form — measured: a mutation dropping the names from the warning
    entirely was killed by NOTHING, because ``"reviewer_1, reviewer_2" in out``
    was satisfied by the roster. An assertion a defect cannot break is decoration.
    """
    return next((ln for ln in out.splitlines() if "NO MANDATE" in ln), "")


# The A/B pair the issue's decisive measurement used: identical but for whether
# two of the personas carry a brief. B's mandates deliberately match no lens
# keyword, so `coverage` cannot tell the two apart and is not asked to (#82 asks
# which lens is covered; this asks whether a persona has a brief at all).
_WITHOUT_MANDATES = [{"name": "reviewer_1"},
                     {"name": "reviewer_2"},
                     {"name": "sentinel", "grounding": "Hunt ruin-class hazards."}]
_WITH_MANDATES = [{"name": "reviewer_1",
                   "grounding": "Check every arithmetic boundary and off-by-one."},
                  {"name": "reviewer_2",
                   "grounding": "Assume all inputs are hostile and malformed."},
                  {"name": "sentinel", "grounding": "Hunt ruin-class hazards."}]


# --- the class test ----------------------------------------------------------

def test_a_mandateless_panel_is_distinguishable_from_one_with_mandates(
        changed_repo, capsys, monkeypatch):
    """The defect, stated as the criterion states it.

    Two runs, identical but for the mandates in ``panel.yaml``. On main their
    stdout and their whole artefact were byte-identical.
    """
    _panel(monkeypatch, changed_repo, _WITH_MANDATES)
    _run(changed_repo)
    briefed = capsys.readouterr().out

    _panel(monkeypatch, changed_repo, _WITHOUT_MANDATES)
    _run(changed_repo)
    mandateless = capsys.readouterr().out

    assert briefed != mandateless
    assert _artefact(briefed)["panel"] != _artefact(mandateless)["panel"]


def test_the_report_reaches_both_the_console_and_the_artefact(
        changed_repo, capsys, monkeypatch):
    """Named, with a count, in the stream an operator watches AND the one a later
    run reads back cold.

    The console half is pinned deliberately: this is an output-legibility defect,
    so a fix pinned only at the sourcing seam is pinned in the wrong place (#111).
    """
    _panel(monkeypatch, changed_repo, _WITHOUT_MANDATES)
    _run(changed_repo)
    out = capsys.readouterr().out

    warning = _warning(out)
    assert "2 of 4 persona(s) were convened with NO MANDATE" in warning
    assert "personas sharing one spawn an identical call" in warning
    assert warning.endswith(": reviewer_1, reviewer_2")
    assert _artefact(out)["panel"]["no_mandate"] == ["reviewer_1", "reviewer_2"]


def test_the_warning_lands_before_anything_is_spent(
        changed_repo, capsys, monkeypatch):
    """It is worth nothing after the panel has been paid for.

    Asserted by position, not by claim: the warning must precede every line the
    run emits about reviewing.
    """
    _panel(monkeypatch, changed_repo, _WITHOUT_MANDATES)
    _run(changed_repo)
    out = capsys.readouterr().out

    assert out.index("NO MANDATE") < out.index("review surface:")
    assert out.index("NO MANDATE") < out.index("--- JSON ---")


def test_every_mandateless_persona_is_counted_and_named(
        changed_repo, capsys, monkeypatch):
    """Two at once, because a denominator nobody pins is a denominator nobody checks.

    A single mandate-less run cannot tell a real count from a hard-coded 1 — the
    gap a mutation found at #16 and again at #82.
    """
    _panel(monkeypatch, changed_repo,
           [{"name": "one"}, {"name": "two"}, {"name": "three"},
            {"name": "sentinel", "grounding": "Hunt ruin-class hazards."}])
    _run(changed_repo)
    out = capsys.readouterr().out

    warning = _warning(out)
    assert "3 of 5 persona(s) were convened with NO MANDATE" in warning
    assert warning.endswith(": one, two, three")
    assert _artefact(out)["panel"]["no_mandate"] == ["one", "two", "three"]


# --- the mirror image --------------------------------------------------------

@pytest.mark.parametrize(
    ("claude_md", "personas"),
    [(None, None), (_GOOD_EXPERTS, None), (None, _WITH_MANDATES)],
    ids=["the-default-panel", "a-CLAUDE.md-panel", "a-panel.yaml-with-mandates"])
def test_a_panel_whose_every_persona_has_a_mandate_is_not_warned_at(
        changed_repo, capsys, monkeypatch, claude_md, personas):
    """A rule that fires on a healthy run is the defect again, inverted.

    Paired with the tests above, which forbid passing this one vacuously by
    reporting nothing anywhere. All three sources, because "no persona is ever
    reported" and "this source cannot reach the state" are different claims.
    """
    if claude_md:
        (changed_repo / "CLAUDE.md").write_text(claude_md)
    if personas:
        _panel(monkeypatch, changed_repo, personas)
    _run(changed_repo)
    out = capsys.readouterr().out

    assert "NO MANDATE" not in out
    assert _artefact(out)["panel"]["no_mandate"] == []


def test_the_injected_specialists_are_never_reported(
        changed_repo, capsys, monkeypatch):
    """The vacuity probe: every *sourced* persona mandate-less at once.

    Everything the operator declared is reported, and the two personas the tool
    itself appends are not — they carry a mandate by construction, so reporting
    them would be over-reporting a degradation the run did not suffer.
    """
    _panel(monkeypatch, changed_repo, [{"name": "one"}, {"name": "two"}])
    _run(changed_repo)
    out = capsys.readouterr().out

    payload = _artefact(out)["panel"]
    assert payload["personas"] == ["one", "two",
                                   "completeness-critic", "survivability"]
    assert payload["no_mandate"] == ["one", "two"]
    assert _warning(out).endswith("2 of 4 persona(s) were convened with NO MANDATE "
                                  "— an empty brief is not a lens and personas "
                                  "sharing one spawn an identical call: one, two")


# --- what counts as no mandate, and what deliberately does not ---------------

def test_a_grounding_key_with_no_value_is_reported_rather_than_crashing(
        changed_repo, capsys, monkeypatch):
    """``grounding:`` with nothing after it, which YAML reads as ``None``.

    Folded into this issue because it is the same state — a persona the operator
    left without a brief — reached one character differently. On main it did not
    degrade silently, it **crashed**: the ``None`` survives ``_load_declared``'s
    ``try`` and dies in ``_ensure_specialists`` outside it, with an unhandled
    ``TypeError`` and a traceback (measured against pyyaml 6.0.3, exit 1).

    Absent and null are one statement — the operator wrote no mandate — and are
    normalised together at the boundary. A ``grounding`` that is present and of
    the WRONG TYPE (an int, a list) still crashes there, along with a null
    ``name``; that is a different defect (an operator input fault that tracebacks
    rather than refusing, #59's class) and it is filed separately rather than
    fixed here by an ``or`` that would catch the list and miss the int.
    """
    _panel(monkeypatch, changed_repo,
           [{"name": "reviewer_1", "grounding": None},
            {"name": "sentinel", "grounding": "Hunt ruin-class hazards."}])

    assert _run(changed_repo) == 0
    out = capsys.readouterr().out

    warning = _warning(out)
    assert "Traceback" not in out
    assert "1 of 3 persona(s) were convened with NO MANDATE" in warning
    # One persona shares an empty brief with nobody, so the duplicate-call clause
    # must not appear: a warning that is false in a case is the defect it reports.
    assert "identical call" not in warning
    assert warning.endswith(": reviewer_1")
    assert _artefact(out)["panel"]["no_mandate"] == ["reviewer_1"]


def test_a_whitespace_only_mandate_is_no_mandate(
        changed_repo, capsys, monkeypatch):
    """``grounding: "   "`` is a brief in nothing but shape.

    The rule is *empty after strip*, which is exactly knowable and needs no
    baseline — unlike a length, which nobody here has one for.
    """
    _panel(monkeypatch, changed_repo,
           [{"name": "reviewer_1", "grounding": "   \n\t "},
            {"name": "sentinel", "grounding": "Hunt ruin-class hazards."}])
    _run(changed_repo)
    out = capsys.readouterr().out

    assert _artefact(out)["panel"]["no_mandate"] == ["reviewer_1"]


def test_a_one_word_mandate_is_not_reported(changed_repo, capsys, monkeypatch):
    """The under-report, pinned in a test rather than left to the prose.

    ``grounding: TODO`` is a persona nobody usefully briefed, and this fix does
    **not** catch it: whether a mandate is any good is #2/#34's question, and the
    rule here answers only whether one exists. Asserted so that a later change
    which quietly widens the rule to a length threshold goes red here first.
    """
    _panel(monkeypatch, changed_repo,
           [{"name": "reviewer_1", "grounding": "TODO"},
            {"name": "sentinel", "grounding": "Hunt ruin-class hazards."}])
    _run(changed_repo)
    out = capsys.readouterr().out

    assert "NO MANDATE" not in out
    assert _artefact(out)["panel"]["no_mandate"] == []


# --- --panel: report, and do not refuse --------------------------------------

def test_an_explicit_panel_with_a_mandateless_persona_reports_and_continues(
        changed_repo, capsys, monkeypatch):
    """The rejected option, asserted rather than left to the docstring.

    #16's asymmetry is that an explicit instruction is never silently
    **overridden**, and refusal is the remedy for substitution: an empty
    ``--panel`` argument (#106), a file that is not there, a file that names
    nobody. None of those can be carried out. A file that names personas with no
    brief **is** carried out, exactly as written — every persona the operator
    named is convened with the mandate they wrote. Refusing would be the tool
    ruling that they did not mean it, and the panel informs rather than decides.

    So it reports, on the same channel every other tier reports on, before the
    spend and in the artefact — and it exits 0 rather than 2.
    """
    named = changed_repo / "docs" / "panel.yaml"
    named.parent.mkdir()
    named.write_text("personas: …\n")
    _stub_yaml(monkeypatch, {"personas": _WITHOUT_MANDATES})

    assert _run(changed_repo, "--panel", str(named)) == 0
    captured = capsys.readouterr()

    assert "error:" not in captured.err
    warning = _warning(captured.out)
    assert "2 of 4 persona(s) were convened with NO MANDATE" in warning
    assert warning.endswith(": reviewer_1, reviewer_2")
    payload = _artefact(captured.out)["panel"]
    assert payload["source"] == "--panel"
    assert payload["no_mandate"] == ["reviewer_1", "reviewer_2"]
